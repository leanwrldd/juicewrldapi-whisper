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


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1"


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


def check_for_updates(auto_update: bool, skip: bool) -> None:
    step("Checking for updates")
    if skip:
        ok("Skipped (--no-update-check).")
        return

    if not (ROOT / ".git").exists():
        ok("Not a git checkout, skipping update check.")
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
    build = torch_cuda_build()
    if build:
        ok(f"GPU-accelerated PyTorch already installed (CUDA {build}).")
        return

    warn("NVIDIA GPU detected — installing GPU-accelerated PyTorch instead of "
         "the CPU-only default (one-time, larger download, several minutes)...")
    t0 = time.time()
    result = subprocess.run(
        [str(venv_python()), "-m", "pip", "install", "torch", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cu128"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        warn("GPU PyTorch install failed — continuing with CPU-only PyTorch.")
        return
    ok(f"GPU-accelerated PyTorch installed in {human_time(time.time() - t0)}.")


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
