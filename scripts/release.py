#!/usr/bin/env python3
"""
Interactive release helper for WRLD Sync.

Walks through: commit -> push -> build the Electron installer -> publish a
GitHub release with the installer attached.

Usage:
    python scripts/release.py                  # full interactive flow
    python scripts/release.py -m "fix: thing"   # skip the commit-message prompt
    python scripts/release.py --version 1.2.0   # set an explicit release version
    python scripts/release.py --yes             # don't ask for confirmations
    python scripts/release.py --skip-commit --skip-push   # only build + release
    python scripts/release.py --dry-run         # show what would happen, do nothing
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELECTRON_DIR = ROOT / "electron"
PACKAGE_JSON = ELECTRON_DIR / "package.json"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE per line, no external dependency.
    Never overrides a variable already set in the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# tiny console helpers
# ---------------------------------------------------------------------------
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


def step(n: int, total: int, text: str) -> None:
    print(f"\n{paint(f'[{n}/{total}]', C.BOLD, C.CYAN)} {paint(text, C.BOLD)}")


def info(text: str) -> None:
    print(f"  {paint('->', C.BLUE)} {text}")


def ok(text: str) -> None:
    print(f"  {paint('OK', C.GREEN, C.BOLD)} {text}")


def warn(text: str) -> None:
    print(f"  {paint('!', C.YELLOW, C.BOLD)} {paint(text, C.YELLOW)}")


def err(text: str) -> None:
    print(f"  {paint('X', C.RED, C.BOLD)} {paint(text, C.RED)}")


def fail(text: str) -> None:
    err(text)
    sys.exit(1)


def confirm(prompt: str, auto_yes: bool, default_yes: bool = True) -> bool:
    if auto_yes:
        return True
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        reply = input(f"  {paint('?', C.MAGENTA, C.BOLD)} {prompt} {suffix} ").strip().lower()
        if not reply:
            return default_yes
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
# git / process helpers
# ---------------------------------------------------------------------------
def run(cmd: list[str], cwd: Path | None = None, check: bool = True, capture: bool = False,
        env: dict | None = None) -> subprocess.CompletedProcess:
    # On Windows, things like npm/npx are .cmd shims that CreateProcess can't
    # launch directly without going through a shell.
    use_shell = sys.platform == "win32"
    full_env = {**os.environ, **env} if env else None
    if capture:
        result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, shell=use_shell, env=full_env)
    else:
        result = subprocess.run(cmd, cwd=cwd, text=True, shell=use_shell, env=full_env)
    if check and result.returncode != 0:
        if capture:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
        fail(f"Command failed: {' '.join(cmd)}")
    return result


def git_status_porcelain() -> str:
    return run(["git", "status", "--porcelain"], cwd=ROOT, capture=True).stdout


def current_branch() -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture=True).stdout.strip()


def remote_owner_repo() -> tuple[str, str]:
    url = run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture=True).stdout.strip()
    match = re.search(r"github\.com[/:]([^/]+)/([^/.]+?)(?:\.git)?$", url)
    if not match:
        fail(f"Couldn't parse a GitHub owner/repo from remote URL: {url}")
    return match.group(1), match.group(2)


def github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    # Fall back to a token embedded in the remote URL, if present
    # (https://<token>@github.com/owner/repo.git).
    url = run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture=True).stdout.strip()
    match = re.search(r"https://([^@/]+)@github\.com", url)
    if match:
        return match.group(1)
    import getpass
    warn("No GITHUB_TOKEN/GH_TOKEN env var found and none embedded in the git remote.")
    return getpass.getpass("  Paste a GitHub personal access token (repo scope), input hidden: ").strip()


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------
def step_commit(args, total: int, idx: int) -> bool:
    """Returns True if a commit was made (or one already existed to push)."""
    step(idx, total, "Commit changes")
    status = git_status_porcelain()
    if not status.strip():
        info("Working tree is clean, nothing to commit.")
        return False

    print(paint(status.rstrip(), C.DIM))
    if not confirm("Stage and commit all of the above?", args.yes):
        warn("Skipping commit.")
        return False

    message = args.message
    if not message:
        message = input(f"  {paint('?', C.MAGENTA, C.BOLD)} Commit message: ").strip()
    while not message:
        message = input(f"  {paint('?', C.MAGENTA, C.BOLD)} Commit message (required): ").strip()

    if args.dry_run:
        info(f"[dry-run] git add -A && git commit -m {message!r}")
        return True

    run(["git", "add", "-A"], cwd=ROOT)
    run(["git", "commit", "-m", message], cwd=ROOT)
    ok("Committed.")
    return True


def step_push(args, total: int, idx: int) -> None:
    step(idx, total, "Push to origin")
    branch = current_branch()
    info(f"Branch: {branch}")
    if not confirm(f"Push '{branch}' to origin?", args.yes):
        warn("Skipping push.")
        return
    if args.dry_run:
        info(f"[dry-run] git push origin {branch}")
        return
    run(["git", "push", "origin", branch], cwd=ROOT)
    ok("Pushed.")


def read_version() -> str:
    data = json.loads(PACKAGE_JSON.read_text())
    return data["version"]


def bump_patch(version: str) -> str:
    parts = version.split(".")
    while len(parts) < 3:
        parts.append("0")
    parts[2] = str(int(parts[2]) + 1)
    return ".".join(parts[:3])


def write_version(new_version: str) -> None:
    data = json.loads(PACKAGE_JSON.read_text())
    data["version"] = new_version
    PACKAGE_JSON.write_text(json.dumps(data, indent=2) + "\n")


def step_build(args, total: int, idx: int) -> tuple[str, Path]:
    step(idx, total, "Build the Electron installer")

    current = read_version()
    new_version = args.version or bump_patch(current)
    info(f"Version: {current} -> {new_version}")
    if not confirm(f"Use version {new_version}?", args.yes):
        new_version = input(f"  {paint('?', C.MAGENTA, C.BOLD)} Version to use: ").strip() or new_version

    if args.dry_run:
        info(f"[dry-run] write version {new_version} to electron/package.json")
        info("[dry-run] npm install (if needed) && npm run dist")
        return new_version, ELECTRON_DIR / "dist" / f"WRLD Sync Setup {new_version}.exe"

    write_version(new_version)
    ok(f"electron/package.json set to {new_version}")

    if not (ELECTRON_DIR / "node_modules").exists():
        info("node_modules not found, running npm install...")
        run(["npm", "install"], cwd=ELECTRON_DIR)

    info("Running npm run dist (this can take a few minutes)...")
    # We don't code-sign the installer, so skip electron-builder's automatic
    # signing-certificate discovery: on Windows it otherwise downloads a
    # cross-signing toolkit containing macOS symlinks that fail to extract
    # without Developer Mode/admin (SeCreateSymbolicLinkPrivilege).
    run(["npm", "run", "dist"], cwd=ELECTRON_DIR, env={"CSC_IDENTITY_AUTO_DISCOVERY": "false"})

    dist_dir = ELECTRON_DIR / "dist"
    installers = sorted(dist_dir.glob("*.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not installers:
        fail(f"Build finished but no .exe found in {dist_dir}")
    installer = installers[0]
    ok(f"Built {installer.name} ({installer.stat().st_size / 1_000_000:.1f} MB)")
    return new_version, installer


def create_github_release(owner: str, repo: str, token: str, tag: str, name: str, body: str) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    payload = json.dumps({
        "tag_name": tag,
        "name": name,
        "body": body,
        "draft": False,
        "prerelease": False,
    }).encode()
    req = urllib.request.Request(url, data=payload, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        fail(f"GitHub release creation failed ({e.code}): {e.read().decode(errors='replace')}")


def upload_release_asset(upload_url_template: str, token: str, file_path: Path) -> dict:
    upload_url = upload_url_template.split("{")[0] + f"?name={file_path.name}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    data = file_path.read_bytes()
    req = urllib.request.Request(upload_url, data=data, method="POST", headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": content_type,
        "Content-Length": str(len(data)),
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        fail(f"Asset upload failed ({e.code}): {e.read().decode(errors='replace')}")


def step_release(args, total: int, idx: int, version: str, installer: Path) -> None:
    step(idx, total, "Publish GitHub release")

    tag = f"v{version}"
    owner, repo = remote_owner_repo()
    info(f"Repo: {owner}/{repo}")
    info(f"Tag: {tag}")

    if not confirm(f"Create and push tag {tag}, then publish a release with {installer.name} attached?", args.yes):
        warn("Skipping release.")
        return

    if args.dry_run:
        info(f"[dry-run] git tag {tag} && git push origin {tag}")
        info(f"[dry-run] create GitHub release {tag}, upload {installer}")
        return

    if not installer.exists():
        fail(f"Installer not found at {installer} (did the build step run?)")

    run(["git", "tag", tag], cwd=ROOT)
    run(["git", "push", "origin", tag], cwd=ROOT)
    ok(f"Tag {tag} pushed.")

    token = github_token()
    body = args.message or f"WRLD Sync {version}"
    release = create_github_release(owner, repo, token, tag, f"WRLD Sync {version}", body)
    ok(f"Release created: {release['html_url']}")

    info(f"Uploading {installer.name} ({installer.stat().st_size / 1_000_000:.1f} MB)...")
    upload_release_asset(release["upload_url"], token, installer)
    ok("Asset uploaded.")

    print(f"\n{paint('Release published:', C.GREEN, C.BOLD)} {release['html_url']}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Commit, push, build, and release WRLD Sync.")
    parser.add_argument("-m", "--message", help="Commit message / release notes")
    parser.add_argument("--version", help="Explicit version to release (default: bump patch)")
    parser.add_argument("-y", "--yes", action="store_true", help="Don't ask for confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without doing it")
    parser.add_argument("--skip-commit", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-release", action="store_true")
    args = parser.parse_args()

    print(paint("WRLD Sync release helper", C.BOLD, C.MAGENTA))
    if args.dry_run:
        warn("Dry run: no changes will actually be made.")

    steps = [s for s, skip in [
        ("commit", args.skip_commit),
        ("push", args.skip_push),
        ("build", args.skip_build),
        ("release", args.skip_release),
    ] if not skip]
    total = len(steps)
    idx = 0
    version = None
    installer = None

    try:
        if "commit" in steps:
            idx += 1
            step_commit(args, total, idx)
        if "push" in steps:
            idx += 1
            step_push(args, total, idx)
        if "build" in steps:
            idx += 1
            version, installer = step_build(args, total, idx)
        if "release" in steps:
            idx += 1
            if version is None:
                version = args.version or read_version()
            if installer is None:
                dist_dir = ELECTRON_DIR / "dist"
                candidates = sorted(dist_dir.glob("*.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
                installer = candidates[0] if candidates else dist_dir / f"WRLD Sync Setup {version}.exe"
            step_release(args, total, idx, version, installer)
    except KeyboardInterrupt:
        print()
        fail("Cancelled.")

    print(f"\n{paint('Done.', C.GREEN, C.BOLD)}")


if __name__ == "__main__":
    main()
