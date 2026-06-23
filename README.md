# WRLD Sync

A music player that pulls songs from [juicewrldapi.com](https://juicewrldapi.com), streams audio, and generates karaoke-style synced lyrics using local OpenAI Whisper.

## Setup

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

## Usage

1. Type a song name in the search box
2. Click a result to load it — audio streams immediately
3. Hit **Sync with Whisper** — the backend downloads the audio and runs local Whisper
4. Watch the lyrics highlight word-by-word as the song plays

## Config

Set `WHISPER_MODEL` env var to change the model size (default: `base`):

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
