#!/usr/bin/env python3
"""
Git credential helper that reads a GitHub token from the GITHUB_TOKEN (or
GH_TOKEN) environment variable instead of storing it anywhere on disk.

Wired up for this repo only via:
    git config --local credential.helper "!python <path-to-this-file>"

If the env var isn't set, this helper prints nothing and git falls back to
its next configured credential source (e.g. a normal login prompt).
"""
import os
import sys


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action != "get":
        return  # nothing to do for store/erase; we don't persist anything

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        return

    print("username=x-access-token")
    print(f"password={token}")


if __name__ == "__main__":
    main()
