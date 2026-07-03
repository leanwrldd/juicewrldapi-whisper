const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getGpuStatus: () => ipcRenderer.invoke('system:gpu-status'),
  installGpu: () => ipcRenderer.invoke('system:install-gpu'),
  onInstallGpuProgress: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('system:install-gpu-progress', listener);
    return () => ipcRenderer.removeListener('system:install-gpu-progress', listener);
  },
});
