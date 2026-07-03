const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const net = require('net');
const { spawn } = require('child_process');
const http = require('http');

const isPackaged = app.isPackaged;

// Resource root: packaged app reads from process.resourcesPath, dev mode reads from the repo root.
const resRoot = isPackaged ? process.resourcesPath : path.join(__dirname, '..');

const pythonExe = isPackaged
  ? path.join(resRoot, 'pyruntime', 'python.exe')
  : path.join(resRoot, '.venv', 'Scripts', 'python.exe');

const appPyDir = resRoot; // app.py + static/ both live directly under resRoot in both modes

let backendProc = null;
let backendPort = null;
let mainWindow = null;

function findFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

function waitForServer(port, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const req = http.get({ host: '127.0.0.1', port, path: '/', timeout: 1500 }, (res) => {
        res.resume();
        resolve();
      });
      req.on('error', () => {
        if (Date.now() > deadline) return reject(new Error('Backend did not start in time'));
        setTimeout(attempt, 300);
      });
      req.on('timeout', () => {
        req.destroy();
        if (Date.now() > deadline) return reject(new Error('Backend did not start in time'));
        setTimeout(attempt, 300);
      });
    };
    attempt();
  });
}

function startBackend(port) {
  const proc = spawn(
    pythonExe,
    ['-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', String(port)],
    { cwd: appPyDir, windowsHide: true }
  );
  proc.stdout.on('data', (d) => process.stdout.write(`[backend] ${d}`));
  proc.stderr.on('data', (d) => process.stderr.write(`[backend] ${d}`));
  proc.on('exit', (code) => {
    console.log(`[backend] exited with code ${code}`);
    backendProc = null;
  });
  return proc;
}

function stopBackend() {
  return new Promise((resolve) => {
    if (!backendProc) return resolve();
    const proc = backendProc;
    backendProc = null;
    proc.once('exit', () => resolve());
    // uvicorn shuts down gracefully on SIGINT/SIGTERM; taskkill is the reliable
    // way to signal a child process tree on Windows.
    spawn('taskkill', ['/pid', String(proc.pid), '/t', '/f'], { windowsHide: true })
      .on('error', () => resolve())
      .on('exit', () => resolve());
  });
}

async function createWindow() {
  backendPort = await findFreePort();
  backendProc = startBackend(backendPort);
  await waitForServer(backendPort);

  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadURL(`http://127.0.0.1:${backendPort}/`);
}

app.whenReady().then(createWindow);

app.on('window-all-closed', async () => {
  await stopBackend();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', async (e) => {
  if (backendProc) {
    e.preventDefault();
    await stopBackend();
    app.quit();
  }
});

// --- IPC: system/GPU status, proxied straight through to the backend ---
ipcMain.handle('system:gpu-status', async () => {
  return new Promise((resolve) => {
    http
      .get({ host: '127.0.0.1', port: backendPort, path: '/api/system/gpu-status', timeout: 5000 }, (res) => {
        let body = '';
        res.on('data', (c) => (body += c));
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch {
            resolve({ error: 'invalid response' });
          }
        });
      })
      .on('error', (e) => resolve({ error: e.message }))
      .on('timeout', function () {
        this.destroy();
        resolve({ error: 'timeout' });
      });
  });
});

ipcMain.handle('system:install-gpu', async (event) => {
  const send = (payload) => event.sender.send('system:install-gpu-progress', payload);
  try {
    send({ phase: 'stopping-backend' });
    await stopBackend();

    send({ phase: 'installing' });
    await new Promise((resolve, reject) => {
      const pip = spawn(
        pythonExe,
        [
          '-m', 'pip', 'install',
          'torch', 'torchaudio',
          '--index-url', 'https://download.pytorch.org/whl/cu128',
          '--force-reinstall',
        ],
        { windowsHide: true }
      );
      pip.stdout.on('data', (d) => send({ phase: 'installing', log: d.toString() }));
      pip.stderr.on('data', (d) => send({ phase: 'installing', log: d.toString() }));
      pip.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`pip exited with code ${code}`))));
      pip.on('error', reject);
    });

    send({ phase: 'restarting-backend' });
    backendProc = startBackend(backendPort);
    await waitForServer(backendPort);

    send({ phase: 'done' });
    return { ok: true };
  } catch (err) {
    send({ phase: 'error', message: err.message });
    // Best-effort: bring the backend back up even if the GPU install failed.
    if (!backendProc) {
      backendProc = startBackend(backendPort);
      await waitForServer(backendPort).catch(() => {});
    }
    return { ok: false, error: err.message };
  }
});
