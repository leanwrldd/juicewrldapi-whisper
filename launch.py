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


def stop_existing_server() -> None:
    step("Checking for an already-running server")
    if platform.system() == "Windows":
        result = subprocess.run(
            ["taskkill", "/f", "/im", "uvicorn.exe"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            ok("Stopped a previously running server.")
        else:
            ok("No existing server was running.")
    else:
        ok("Skipping (not Windows) — if a server is already running on the "
           "target port, starting a new one below will fail loudly.")


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
    args = parser.parse_args()

    print(paint("WRLD Sync launcher", C.BOLD, C.MAGENTA))

    step("Checking Python")
    ensure_python_version()

    ensure_venv()
    install_requirements()
    stop_existing_server()

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
