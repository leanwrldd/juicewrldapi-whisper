# WRLD Sync

A music player that pulls songs from [juicewrldapi.com](https://juicewrldapi.com), streams audio, and generates karaoke-style synced lyrics using local OpenAI Whisper.

## Setup

**Windows:** double-click `start.bat`. First run sets up a virtual environment and installs everything (including PyTorch/Whisper); later runs are fast.

**Manual (any OS):**

```bash
# 1. Install dependencies (requires Python 3.10+)
pip install -r requirements.txt

# Whisper also needs ffmpeg on your PATH:
#   Windows:  winget install ffmpeg
#   Mac:      brew install ffmpeg

# 2. Run the server
uvicorn app:app --reload

# 3. Open http://localhost:8000
```

Or just run `python launch.py`, which does the setup + launch + browser-opening for you (same thing `start.bat` runs under the hood). Flags: `--port <n>`, `--no-browser`, `--auto-update`, `--no-update-check`.

If an NVIDIA GPU is detected, `launch.py` automatically installs the CUDA build of PyTorch instead of the CPU-only default — otherwise Whisper alignment falls back to (much slower) CPU.

## Usage

1. Type a song name in the search box (or click the `#` button next to it to switch to loading a song directly by its ID, then press Enter)
2. Click a result to load it — audio streams immediately
3. Hit **Sync with Whisper** — the backend downloads the audio and runs local Whisper
4. Watch the lyrics highlight word-by-word as the song plays

## DB Manager

A separate tools page (linked from the main header) for maintaining the song database:

- **Autogroup** — groups songs by title with version tags stripped (e.g. `Can't Die (v3)` → `Can't Die`), so different versions of the same song sit together. Supports manual renaming, merging groups, and moving songs between groups, with changes saved via the juicewrldapi.com `/versions/` route.

## Config

Whisper uses two model slots — an **align** model (does the actual sync) and a **verify** model (double-checks it) — plus a **device** preference (auto / CPU / CUDA). All three are changeable from the Settings controls in the app itself; changes persist locally and take effect on the next sync (no restart needed).

`WHISPER_MODEL` env var sets the initial align model on first run only, before any in-app preference exists:

```bash
WHISPER_MODEL=small uvicorn app:app --reload
```

| Model  | Speed  | Accuracy |
|--------|--------|----------|
| tiny   | fastest | lowest  |
| base   | fast   | good     |
| small  | medium | better   |
| medium | slow   | best     |

## Notes

- First sync call downloads the Whisper model (~150 MB for `base`) if not cached
- The "Synced Lyrics" tab also auto-loads LRC data from the API when available (no Whisper needed)
- Audio proxied through `/api/stream` to handle range requests for seeking
