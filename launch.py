#!/usr/bin/env python3
"""
WRLD Sync launcher.

Sets up the virtual environment and dependencies if needed, then starts the
server and opens it in your browser. Safe to run repeatedly — subsequent
runs skip setup steps that are already done.

Usage:
    python launch.py
    python launch.py --port 8080
    python launch.py --no-browser
    python launch.py --auto-update       # pull updates without asking
    python launch.py --no-update-check   # skip the update check entirely
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _enable_windows_vt() -> bool:
    """Older/plain cmd.exe windows don't interpret \\033[ ANSI codes unless
    ENABLE_VIRTUAL_TERMINAL_PROCESSING is explicitly turned on for the console
    handle -- without this, users see literal escape-code garbage instead of
    colored text. Returns whether it succeeded."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if not kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING):
            return False
        return True
    except Exception:
        return False


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not (sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"):
        return False
    if platform.system() == "Windows" and not _enable_windows_vt():
        return False
    return True


USE_COLOR = _supports_color()


def paint(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(codes) + text + C.RESET


def step(text: str) -> None:
    print(f"\n{paint('->', C.BOLD, C.CYAN)} {paint(text, C.BOLD)}")


def ok(text: str) -> None:
    print(f"  {paint('OK', C.GREEN, C.BOLD)} {text}")


def warn(text: str) -> None:
    print(f"  {paint('!', C.YELLOW, C.BOLD)} {paint(text, C.YELLOW)}")


def err(text: str) -> None:
    print(f"  {paint('X', C.RED, C.BOLD)} {paint(text, C.RED)}")


def fail(text: str) -> None:
    err(text)
    sys.exit(1)


def confirm(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    try:
        reply = input(f"  {paint('?', C.MAGENTA, C.BOLD)} {prompt} {suffix} ").strip().lower()
    except (EOFError, OSError):
        return default_yes
    if not reply:
        return default_yes
    return reply in ("y", "yes")


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m{seconds:02d}s"


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        fail(
            f"Python 3.10+ is required (found {platform.python_version()}). "
            "Install a newer Python from https://python.org/downloads/ and try again."
        )
    ok(f"Python {platform.python_version()}")


def _git(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)


GIT_REMOTE_URL = "https://github.com/leanwrldd/juicewrldapi-whisper.git"


def has_git() -> bool:
    return shutil.which("git") is not None


def ensure_git() -> bool:
    """Returns True if git is available (already installed, or freshly installed via winget)."""
    if has_git():
        return True
    if platform.system() != "Windows":
        warn("git not found. Install it with your OS package manager to enable auto-updates.")
        return False
    if not shutil.which("winget"):
        warn("git not found, and winget isn't available to install it automatically. "
             "Install Git for Windows manually: https://git-scm.com/download/win")
        return False

    warn("git not found — installing via winget (one-time)...")
    t0 = time.time()
    result = subprocess.run(
        ["winget", "install", "--id", "Git.Git", "-e",
         "--accept-package-agreements", "--accept-source-agreements"],
    )
    if result.returncode != 0:
        warn("Automatic git install failed. Install it manually: https://git-scm.com/download/win")
        return False

    _refresh_path_from_registry()
    if has_git():
        ok(f"git installed in {human_time(time.time() - t0)}.")
        return True
    warn(f"git installed in {human_time(time.time() - t0)}, but not detected on PATH yet in this "
         "window — restart this launcher once (a fresh terminal will have it).")
    return False


def _looks_like_this_project(path: Path) -> bool:
    return (path / "app.py").exists() and (path / "requirements.txt").exists() and (path / "launch.py").exists()


def offer_git_conversion(auto_yes: bool) -> None:
    """If this folder isn't a git checkout but is clearly a copy of this
    project (e.g. downloaded as a GitHub ZIP), offer to replace it with a
    proper git clone in place so future runs can actually auto-update."""
    if not _looks_like_this_project(ROOT):
        return
    if not ensure_git():
        return

    warn("This folder isn't a git checkout (likely downloaded as a ZIP), so it can never "
         "auto-update as-is. It can be converted into a real git checkout in place.")
    # auto_yes (only true with --auto-update) skips the prompt entirely. Otherwise
    # always ask -- but default the answer to Yes, since this is a safe, reversible
    # operation (the old folder is backed up, never deleted) and most users just
    # press Enter rather than typing an explicit y/n.
    if not auto_yes and not confirm("Convert this folder into a git checkout now?", default_yes=True):
        ok("Skipping — update checks will keep saying 'not a git checkout'.")
        return

    tmp_dir = ROOT.parent / f"{ROOT.name}__git_clone_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    step("Cloning a fresh git checkout")
    t0 = time.time()
    result = subprocess.run(["git", "clone", "--quiet", GIT_REMOTE_URL, str(tmp_dir)])
    if result.returncode != 0:
        warn("Clone failed — leaving the current folder untouched.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    ok(f"Cloned in {human_time(time.time() - t0)}.")

    # Carry over local preferences that only exist on disk, not in git.
    for name in (".env", ".model_pref", ".model_pref_align", ".model_pref_verify", ".device_pref"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, tmp_dir / name)

    backup_dir = ROOT.parent / f"{ROOT.name}__pre-git-backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)

    try:
        ROOT.rename(backup_dir)
    except OSError as e:
        warn(f"Couldn't move the old folder aside ({e}).")
        ok(f"Your new git checkout is ready at: {tmp_dir} — run start.bat from there instead.")
        sys.exit(0)

    try:
        tmp_dir.rename(ROOT)
    except OSError as e:
        backup_dir.rename(ROOT)  # put the original back, nothing lost
        warn(f"Couldn't move the new clone into place ({e}) — restored the original folder. "
             f"The git clone is still available at: {tmp_dir}")
        return

    ok(f"This folder is now a proper git checkout. The old copy is backed up at: {backup_dir}")
    step("Restarting with the git checkout")
    os.execv(sys.executable, [sys.executable, str(ROOT / "launch.py"), *sys.argv[1:]])


def check_for_updates(auto_update: bool, skip: bool) -> None:
    step("Checking for updates")
    if skip:
        ok("Skipped (--no-update-check).")
        return

    if not (ROOT / ".git").exists():
        ok("Not a git checkout, skipping update check.")
        offer_git_conversion(auto_update)
        return

    try:
        if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
            warn("git not found on PATH, skipping update check.")
            return
    except FileNotFoundError:
        warn("git not found on PATH, skipping update check.")
        return

    status = _git(["status", "--porcelain"])
    if status.returncode != 0:
        warn("Couldn't read git status, skipping update check.")
        return
    if status.stdout.strip():
        warn("You have local changes — skipping auto-update to avoid conflicts.")
        return

    try:
        fetch = _git(["fetch", "--quiet", "origin"], timeout=15)
    except subprocess.TimeoutExpired:
        warn("Timed out reaching GitHub — continuing with the current version.")
        return
    if fetch.returncode != 0:
        warn("Couldn't reach GitHub to check for updates (offline?). Continuing with the current version.")
        return

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    local = _git(["rev-parse", "HEAD"]).stdout.strip()
    remote_result = _git(["rev-parse", f"origin/{branch}"])
    if remote_result.returncode != 0 or not remote_result.stdout.strip():
        ok(f"No 'origin/{branch}' to compare against, skipping.")
        return
    remote = remote_result.stdout.strip()

    if local == remote:
        ok("You're up to date.")
        return

    behind = _git(["rev-list", "--count", f"{local}..{remote}"]).stdout.strip()
    warn(f"Update available: {behind} new commit(s) on '{branch}'.")

    do_pull = auto_update or confirm("Download and apply the update now?")
    if not do_pull:
        ok("Skipping update for this run.")
        return

    t0 = time.time()
    pull = subprocess.run(["git", "pull", "--ff-only", "origin", branch], cwd=ROOT)
    if pull.returncode != 0:
        warn("git pull failed — continuing with the current version. You may need to update manually.")
        return
    ok(f"Updated in {human_time(time.time() - t0)}.")

    step("Restarting with the updated code")
    os.execv(sys.executable, [sys.executable, str(ROOT / "launch.py"), *sys.argv[1:]])


def ensure_venv() -> None:
    step("Checking virtual environment")
    if venv_python().exists():
        ok(f".venv already set up ({venv_python()})")
        return
    warn(".venv not found, creating it...")
    t0 = time.time()
    result = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if result.returncode != 0:
        fail("Failed to create the virtual environment.")
    ok(f"Created .venv in {human_time(time.time() - t0)}")


def requirements_satisfied() -> bool:
    """Cheap check: has pip already installed everything from requirements.txt?
    Avoids re-running the (slow) pip install/resolve on every single launch."""
    marker = VENV_DIR / ".requirements_installed"
    if not marker.exists():
        return False
    try:
        return marker.read_text().strip() == _requirements_fingerprint()
    except OSError:
        return False


def _requirements_fingerprint() -> str:
    import hashlib
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def has_nvidia_gpu() -> bool:
    try:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def gpu_compute_capability() -> float | None:
    """Compute capability (e.g. 6.1 for a GT 1030) of GPU 0, or None if it
    can't be determined. The cu128 PyTorch build only ships kernels for
    Turing and newer (sm_75+), so older cards need to be steered to CPU
    instead of downloading a multi-hundred-MB wheel that'll just silently
    fall back to CPU anyway at inference time."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        first_line = result.stdout.strip().splitlines()[0].strip()
        return float(first_line)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


# Lowest compute capability the cu128 PyTorch wheels ship kernels for (Turing+).
MIN_SUPPORTED_COMPUTE_CAP = 7.5
# Lowest compute capability the older cu118 wheels ship kernels for (Maxwell+) --
# below this, no current PyTorch build has kernels for the card at all.
MIN_LEGACY_COMPUTE_CAP = 5.0


def torch_cuda_build() -> str | None:
    """The CUDA version torch reports (e.g. '12.8'), or None if torch isn't
    installed yet or is a CPU-only wheel."""
    result = subprocess.run(
        [str(venv_python()), "-c", "import torch; print(torch.version.cuda or '')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def ensure_gpu_torch() -> None:
    """PyPI's default 'torch' wheel (pulled in transitively by openai-whisper)
    is CPU-only. If there's an NVIDIA GPU, install the CUDA build instead —
    before requirements.txt installs anything, so pip sees torch already
    satisfied and doesn't downgrade it back to CPU."""
    step("Checking for GPU acceleration")
    if not venv_python().exists():
        ok("Skipping (.venv not created yet).")
        return
    if not has_nvidia_gpu():
        ok("No NVIDIA GPU detected — using CPU.")
        return
    cap = gpu_compute_capability()
    too_old_for_cu128 = cap is not None and cap < MIN_SUPPORTED_COMPUTE_CAP
    too_old_for_anything = cap is not None and cap < MIN_LEGACY_COMPUTE_CAP

    build = torch_cuda_build()
    if build:
        # A cu12x build already installed on a card too old for it (e.g. cu128 on a
        # Pascal GT 1030) will silently fall back to CPU at inference time -- reinstall
        # with the older cu118 build instead, which still ships kernels for it.
        if build.startswith("12") and too_old_for_cu128 and not too_old_for_anything:
            warn(f"GPU-accelerated PyTorch (CUDA {build}) is installed, but its compute "
                 f"capability requirement is too new for this GPU ({cap}) — reinstalling "
                 "with an older CUDA 11.8 build that supports it (one-time, larger download, "
                 "several minutes)...")
            t0 = time.time()
            result = subprocess.run(
                [str(venv_python()), "-m", "pip", "install", "--force-reinstall",
                 "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"],
                cwd=ROOT,
            )
            if result.returncode != 0:
                warn(f"Reinstall failed — keeping the existing CUDA {build} build "
                     "(this GPU will run on CPU until that's resolved).")
                return
            ok(f"Reinstalled with the CUDA 11.8 build in {human_time(time.time() - t0)}.")
            return
        ok(f"GPU-accelerated PyTorch already installed (CUDA {build}).")
        return

    index_url = "https://download.pytorch.org/whl/cu128"
    if too_old_for_cu128:
        if too_old_for_anything:
            warn(f"NVIDIA GPU detected, but its compute capability ({cap}) is too old for any "
                 "current PyTorch GPU build — staying on CPU-only PyTorch.")
            return
        warn(f"NVIDIA GPU detected with an older compute capability ({cap}) than the default "
             f"GPU-accelerated build supports ({MIN_SUPPORTED_COMPUTE_CAP}+) — installing an "
             "older CUDA 11.8 build instead, which still ships kernels for this GPU "
             "(one-time, larger download, several minutes)...")
        index_url = "https://download.pytorch.org/whl/cu118"
    else:
        warn("NVIDIA GPU detected — installing GPU-accelerated PyTorch instead of "
             "the CPU-only default (one-time, larger download, several minutes)...")

    t0 = time.time()
    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "torch", "torchaudio",
         "--index-url", index_url],
        cwd=ROOT,
    )
    if result.returncode != 0:
        warn("GPU PyTorch install failed — continuing with CPU-only PyTorch.")
        return
    ok(f"GPU-accelerated PyTorch installed in {human_time(time.time() - t0)}.")


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _refresh_path_from_registry() -> None:
    """Winget (and installers in general) update PATH in the registry, but an
    already-running process keeps its own cached copy in os.environ -- pull
    the current machine + user PATH values and merge them in, the same way
    Windows would on a fresh process/shell without needing a full restart."""
    if platform.system() != "Windows":
        return
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment") as key:
            system_path = winreg.QueryValueEx(key, "Path")[0]
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            user_path = winreg.QueryValueEx(key, "Path")[0]
        os.environ["PATH"] = system_path + os.pathsep + user_path
    except OSError:
        pass  # best-effort; if this fails we just fall back to telling the user to restart


def ensure_ffmpeg() -> None:
    """Whisper shells out to ffmpeg to decode audio; without it, syncing fails
    deep inside a background task with an opaque '[WinError 2] The system
    cannot find the file specified' instead of a clear message here."""
    step("Checking for ffmpeg")
    if has_ffmpeg():
        ok("ffmpeg found.")
        return

    if platform.system() != "Windows":
        warn("ffmpeg not found on PATH — Whisper needs it to decode audio. "
             "Install it with your OS package manager (e.g. 'brew install ffmpeg' on Mac).")
        return

    if not shutil.which("winget"):
        warn("ffmpeg not found, and winget isn't available to install it automatically. "
             "Install it manually: https://ffmpeg.org/download.html")
        return

    warn("ffmpeg not found — installing via winget (one-time)...")
    t0 = time.time()
    result = subprocess.run(
        ["winget", "install", "--id", "Gyan.FFmpeg", "-e",
         "--accept-package-agreements", "--accept-source-agreements"],
    )
    if result.returncode != 0:
        warn("Automatic ffmpeg install failed. Install it manually with "
             "'winget install ffmpeg' (or https://ffmpeg.org/download.html), "
             "then restart this launcher.")
        return

    _refresh_path_from_registry()

    if has_ffmpeg():
        ok(f"ffmpeg installed in {human_time(time.time() - t0)}.")
    else:
        warn(f"ffmpeg installed in {human_time(time.time() - t0)}, but not detected on PATH "
             "yet in this window — restart this launcher once (a fresh terminal will have it).")


def install_requirements() -> None:
    step("Checking Python dependencies")
    if requirements_satisfied():
        ok("Dependencies already installed and up to date.")
        return

    warn("Installing dependencies (first run, or requirements.txt changed)...")
    warn("This includes PyTorch/Whisper and can take several minutes.")
    t0 = time.time()
    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        fail("Dependency installation failed — see the pip output above.")

    marker = VENV_DIR / ".requirements_installed"
    marker.write_text(_requirements_fingerprint())
    ok(f"Dependencies installed in {human_time(time.time() - t0)}.")


def _pids_listening_on(port: int) -> list[int]:
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue).OwningProcess"],
        capture_output=True, text=True,
    )
    pids = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def stop_existing_server(port: int) -> None:
    step("Checking for an already-running server")
    if platform.system() != "Windows":
        ok("Skipping (not Windows) — if a server is already running on the "
           "target port, starting a new one below will fail loudly.")
        return

    # Find whatever's actually bound to our port, rather than assuming it was
    # launched as "uvicorn.exe" — we ourselves launch it as "python.exe -m
    # uvicorn", which a name-based taskkill would never match.
    pids = [p for p in _pids_listening_on(port) if p != os.getpid()]
    if not pids:
        ok("No existing server was running on this port.")
        return

    for pid in pids:
        subprocess.run(["taskkill", "/f", "/pid", str(pid)], capture_output=True)
    ok(f"Stopped {len(pids)} process(es) previously listening on port {port}.")
    time.sleep(0.5)  # give the OS a moment to release the socket


def wait_for_server(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5):
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.3)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up and launch WRLD Sync.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open a browser tab")
    parser.add_argument("--auto-update", action="store_true", help="Pull updates automatically without asking")
    parser.add_argument("--no-update-check", action="store_true", help="Don't check for updates at all")
    args = parser.parse_args()

    print(paint("WRLD Sync launcher", C.BOLD, C.MAGENTA))

    check_for_updates(auto_update=args.auto_update, skip=args.no_update_check)

    step("Checking Python")
    ensure_python_version()

    ensure_venv()
    ensure_gpu_torch()
    ensure_ffmpeg()
    install_requirements()
    stop_existing_server(args.port)

    step("Starting the server")
    url = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [str(venv_python()), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=ROOT,
    )

    if wait_for_server(args.port):
        ok(f"Server is up at {url}")
    else:
        warn("Server didn't respond within 60s — it may still be starting. Check the logs above.")

    if not args.no_browser:
        webbrowser.open(url)
        ok("Opened in your browser.")

    print(f"\n{paint('WRLD Sync is running.', C.GREEN, C.BOLD)} Press Ctrl+C here to stop it.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print()
        step("Shutting down")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        ok("Stopped.")


if __name__ == "__main__":
    main()
