import asyncio
import json
import os
import pathlib
import re
import tempfile
from contextlib import asynccontextmanager

import httpx
import stable_whisper
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = "https://juicewrldapi.com/juicewrld"

# ---------------------------------------------------------------------------
# Audio cache — last downloaded file, reused by /api/sync and /api/verify
# ---------------------------------------------------------------------------
_audio_cache: dict | None = None   # {"path": str, "file": str}
_audio_cache_lock = asyncio.Lock()


async def ensure_audio(song_path: str):
    """Async generator: yields SSE dicts during download, then {"_path": str} as last item."""
    global _audio_cache

    async with _audio_cache_lock:
        if _audio_cache and _audio_cache["path"] == song_path:
            yield {"stage": "downloading", "pct": 100, "msg": "Audio already cached ✓"}
            yield {"_path": _audio_cache["file"]}
            return

        # New song — evict old cached file
        if _audio_cache:
            try:
                os.unlink(_audio_cache["file"])
            except OSError:
                pass
            _audio_cache = None

        yield {"stage": "downloading", "pct": 0, "msg": "Downloading audio…"}
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        downloaded, last_pct = 0, -1
        async with httpx.AsyncClient(timeout=180) as client:
            async with client.stream(
                "GET", BASE + "/files/download/", params={"path": song_path}
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with open(tmp_path, "wb") as f:
                    async for chunk in r.aiter_bytes(65_536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = min(99, int(downloaded / total * 100))
                            if pct >= last_pct + 5:
                                last_pct = pct
                                yield {"stage": "downloading", "pct": pct,
                                       "msg": f"Downloading… {pct}%"}

        _audio_cache = {"path": song_path, "file": tmp_path}
        yield {"stage": "downloading", "pct": 100, "msg": "Download complete"}
        yield {"_path": tmp_path}


# ---------------------------------------------------------------------------
# Model loading — lazy, loaded once on first /api/sync call
# ---------------------------------------------------------------------------
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
_MODEL_PREF_FILE = pathlib.Path(__file__).parent / ".model_pref"

def _load_model_pref() -> str:
    try:
        name = _MODEL_PREF_FILE.read_text().strip()
        if name in WHISPER_MODELS:
            return name
    except OSError:
        pass
    return os.getenv("WHISPER_MODEL", "small")

_model = None
_model_lock = asyncio.Lock()
MODEL_SIZE = _load_model_pref()  # persisted in .model_pref; falls back to WHISPER_MODEL env or "small"


async def get_model():
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[whisper] Loading '{MODEL_SIZE}' model on {device.upper()} …")
                loop = asyncio.get_event_loop()
                _model = await loop.run_in_executor(
                    None, lambda: stable_whisper.load_model(MODEL_SIZE, device=device)
                )
                print(f"[whisper] Model ready on {device.upper()}.")
    return _model


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # model loads lazily on first request


app = FastAPI(title="WRLD Sync", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def jw_get(path: str, params: dict | None = None) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(BASE + path, params=params)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/search")
async def search(q: str, page_size: int = 20):
    return await jw_get("/songs/", {"search": q, "page_size": page_size})


@app.get("/api/song/{song_id}")
async def get_song(song_id: int):
    return await jw_get(f"/songs/{song_id}/")


@app.get("/api/model")
async def get_model_info():
    import torch
    return {
        "model": MODEL_SIZE,
        "loaded": _model is not None,
        "device": "cuda" if (torch.cuda.is_available()) else "cpu",
        "models": WHISPER_MODELS,
    }


@app.post("/api/model")
async def set_model(body: dict):
    global MODEL_SIZE, _model
    name = body.get("model", "").strip()
    if name not in WHISPER_MODELS:
        raise HTTPException(400, f"Unknown model '{name}'. Choose from: {WHISPER_MODELS}")
    async with _model_lock:
        MODEL_SIZE = name
        _model = None   # will reload on next /api/sync call
    try:
        _MODEL_PREF_FILE.write_text(name)
    except OSError:
        pass
    return {"model": MODEL_SIZE, "loaded": False}


@app.get("/api/radio/random")
async def radio_random(
    no_lyrics: bool = False,
    no_synced: bool = False,
    missing_either: bool = False,
    category: str = "",
):
    """Returns a random song, retrying until filter conditions are met (up to 50 attempts).
    no_lyrics:      only songs with no lyrics
    no_synced:      only songs with no synced_lyrics
    missing_either: only songs missing at least one of lyrics / synced_lyrics
    category:       filter by category string (released|unreleased|unsurfaced|recording_session)
    Combining no_lyrics+no_synced = missing both (AND logic).
    """
    for _ in range(50):
        data = await jw_get("/radio/random/")
        song = data.get("song") or {}
        if no_lyrics and song.get("lyrics"):
            continue
        if no_synced and song.get("synced_lyrics"):
            continue
        if missing_either and song.get("lyrics") and song.get("synced_lyrics"):
            continue  # skip songs that have BOTH; keep songs missing at least one
        if category and song.get("category") != category:
            continue
        return data
    raise HTTPException(404, "No matching song found after 50 attempts — try less restrictive filters.")


@app.get("/api/stream")
async def stream_audio(path: str, request: Request):
    # Use httpx params so the path value is properly URL-encoded
    # (paths can contain '&', spaces, etc. that would break the query string)
    req_headers = {}
    if "range" in request.headers:
        req_headers["Range"] = request.headers["range"]

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream_req = client.build_request(
            "GET", BASE + "/files/download/",
            params={"path": path},
            headers=req_headers,
        )
        upstream = await client.send(upstream_req, stream=True)

        resp_headers = {"Accept-Ranges": "bytes"}
        for key in ("content-type", "content-length", "content-range"):
            if key in upstream.headers:
                resp_headers[key] = upstream.headers[key]
        if "content-type" not in resp_headers:
            resp_headers["content-type"] = "audio/mpeg"

        async def gen():
            try:
                async for chunk in upstream.aiter_bytes(32_768):
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(gen(), status_code=upstream.status_code, headers=resp_headers)
    except Exception as e:
        await client.aclose()
        raise HTTPException(502, f"Upstream error: {e}")


class SyncRequest(BaseModel):
    song_id: int
    lyrics: str = ""   # optional override; if set, used instead of song.lyrics


class VerifyRequest(BaseModel):
    song_id: int
    lyrics: str = ""


def _normalize_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9'\s]", "", text.lower()).split()


def _verify_lines(lyric_lines: list[str], transcription: str) -> list[dict]:
    """Score each lyric line against the free transcription by word overlap."""
    trans_words = set(_normalize_words(transcription))
    results = []
    for line in lyric_lines:
        stripped = line.strip()
        if not stripped:
            continue
        words = _normalize_words(stripped)
        if not words:
            continue
        score = sum(1 for w in words if w in trans_words) / len(words)
        status = "present" if score >= 0.55 else ("uncertain" if score >= 0.25 else "absent")
        results.append({"line": stripped, "status": status, "score": round(score, 2)})
    return results


class ProposeRequest(BaseModel):
    song_id: int
    lines: list[dict]
    token: str
    plain_lyrics: str = ""   # optional; if set, included alongside synced_lyrics


@app.post("/api/propose")
async def propose_lyrics(req: ProposeRequest):
    if not req.token:
        raise HTTPException(401, "No auth token provided.")

    # Build LRC string from lines
    lrc_parts = []
    for line in req.lines:
        start = line.get("start", 0)
        m = int(start // 60)
        s = start % 60
        lrc_parts.append(f"[{m}:{s:05.2f}] {line.get('line', '')}")
    lrc = "\n".join(lrc_parts)

    # Fetch song name for the proposal title
    song = await jw_get(f"/songs/{req.song_id}/")

    # follow_redirects=True so any 301/302 on the URL is followed, but also
    # preserve the method (httpx sends a new POST on 307/308 redirects).
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.post(
            BASE + "/accounts/editor/proposals/",
            headers={
                "Authorization": f"Token {req.token}",
                "Content-Type": "application/json",
            },
            json={
                "change_type": "update",
                "song": req.song_id,
                "title": song.get("name", str(req.song_id)),
                "editor_notes": "Synced lyrics generated with WRLD Sync",
                "proposed_data": {
                    "synced_lyrics": lrc,
                    **({"lyrics": req.plain_lyrics} if req.plain_lyrics.strip() else {}),
                },
            },
        )
        body = r.text
        if r.status_code == 403:
            raise HTTPException(403, "Token rejected — editor role required.")
        if r.status_code == 405:
            raise HTTPException(405, f"API returned 405 Method Not Allowed. Your account may not have editor permissions, or the endpoint changed. Response: {body}")
        if not r.is_success:
            raise HTTPException(r.status_code, f"API error {r.status_code}: {body}")
        return r.json()


GENIUS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0 Safari/537.36"
    )
}

def _parse_genius_page(html: str) -> str:
    """Extract and clean lyrics from a Genius page HTML string."""
    soup = BeautifulSoup(html, "html.parser")
    containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
    if not containers:
        raise ValueError("Couldn't find a lyrics container — Genius may have changed their page structure.")

    lines = []
    for container in containers:
        for br in container.find_all("br"):
            br.replace_with("\n")
        lines.append(container.get_text())

    raw = "\n".join(lines)

    # Remove any line that contains a bracketed annotation e.g. [Chorus], [Verse 1]
    bracket_re = re.compile(r"\[.*?\]")
    cleaned = "\n".join(line for line in raw.splitlines() if not bracket_re.search(line))
    return cleaned.strip()


@app.get("/api/genius")
async def genius_lyrics(title: str = "", artist: str = "Juice WRLD", url: str = ""):
    """Fetch lyrics from Genius. Pass `url` to use a specific page; otherwise searches by title+artist."""
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            if url:
                # Direct URL fetch — skip search
                if "genius.com" not in url:
                    raise HTTPException(400, "URL must be a genius.com link.")
                song_url = url
                song_title = url.split("/")[-1].replace("-lyrics", "").replace("-", " ").title()
            else:
                # Search by title + artist
                query = f"{artist} {title}".strip()
                r = await client.get(
                    "https://genius.com/api/search/song",
                    params={"q": query},
                    headers=GENIUS_HEADERS,
                )
                r.raise_for_status()
                sections = r.json().get("response", {}).get("sections", [])
                hits = sections[0].get("hits", []) if sections else []
                if not hits:
                    raise HTTPException(404, f"No Genius results for '{query}'")
                song_url   = hits[0]["result"]["url"]
                song_title = hits[0]["result"]["full_title"]

            r2 = await client.get(song_url, headers=GENIUS_HEADERS)
            r2.raise_for_status()

            lyrics = _parse_genius_page(r2.text)
            return {"lyrics": lyrics, "title": song_title, "url": song_url}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(502, f"Genius fetch failed: {e}")


@app.post("/api/verify")
async def verify_lyrics_audio(req: VerifyRequest):
    """SSE: free-transcribe audio, compare each lyric line by word overlap."""

    async def event_stream():
        tmp_path = None

        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        try:
            yield sse({"stage": "fetching", "msg": "Fetching song info…"})
            song = await jw_get(f"/songs/{req.song_id}/")
            if not song.get("path"):
                yield sse({"stage": "error", "msg": "No audio file for this song."})
                return

            lyrics = (req.lyrics or song.get("lyrics") or "").strip()
            if not lyrics:
                yield sse({"stage": "error", "msg": "No lyrics to verify against. Add lyrics first."})
                return

            tmp_path = None
            async for ev in ensure_audio(song["path"]):
                if "_path" in ev:
                    tmp_path = ev["_path"]
                else:
                    yield sse(ev)

            yield sse({"stage": "loading", "msg": "Loading Whisper model…"})
            model = await get_model()

            yield sse({"stage": "transcribing", "pct": 0, "msg": "Transcribing audio (free pass)…"})
            loop = asyncio.get_event_loop()
            fut = loop.run_in_executor(
                None, lambda: model.transcribe(tmp_path, verbose=False)
            )
            elapsed = 0
            while not fut.done():
                pct = min(95, int((elapsed / 180) ** 0.5 * 95))
                yield sse({"stage": "transcribing", "pct": pct,
                           "msg": f"Transcribing… {elapsed}s"})
                await asyncio.sleep(1)
                elapsed += 1
            result = await fut

            transcription = result.text or ""
            lyric_lines = [l for l in lyrics.split("\n") if l.strip()]
            line_results = _verify_lines(lyric_lines, transcription)

            counts = {"present": 0, "uncertain": 0, "absent": 0}
            for r2 in line_results:
                counts[r2["status"]] += 1

            yield sse({"stage": "done", "results": line_results,
                       "counts": counts, "transcription": transcription})

        except Exception as e:
            yield sse({"stage": "error", "msg": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/sync")
async def sync_lyrics(req: SyncRequest):
    """SSE endpoint — streams progress events then a final 'done' event with lines."""

    async def event_stream():
        tmp_path = None

        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        try:
            # Stage 1: fetch metadata
            yield sse({"stage": "fetching", "msg": "Fetching song info…"})
            song = await jw_get(f"/songs/{req.song_id}/")

            if not song.get("path"):
                yield sse({"stage": "error", "msg": "No audio file for this song."})
                return

            # Stage 2: download (cached — skipped if same song was already downloaded)
            tmp_path = None
            async for ev in ensure_audio(song["path"]):
                if "_path" in ev:
                    tmp_path = ev["_path"]
                else:
                    yield sse(ev)

            # Stage 3: load model (no-op if already cached)
            yield sse({"stage": "loading", "msg": "Loading Whisper model…"})
            model = await get_model()

            # Stage 4: align / transcribe
            # Run in thread pool and stream tick events every second while waiting,
            # since the executor blocks the generator and tqdm only prints to terminal.
            # Custom lyrics (from Genius / manual paste) override song.lyrics
            lyrics = (req.lyrics or song.get("lyrics") or "").strip()
            label = "Aligning lyrics to audio" if lyrics else "Transcribing audio"
            loop = asyncio.get_event_loop()

            if lyrics:
                fut = loop.run_in_executor(
                    None, lambda: model.align(tmp_path, lyrics, language="en")
                )
            else:
                fut = loop.run_in_executor(
                    None, lambda: model.transcribe(tmp_path, word_timestamps=True, verbose=False)
                )

            elapsed = 0
            while not fut.done():
                # Ramp from 0→95% over ~3 min using a sqrt curve so it feels alive
                pct = min(95, int((elapsed / 180) ** 0.5 * 95))
                yield sse({"stage": "aligning", "pct": pct,
                           "msg": f"{label}… {elapsed}s"})
                await asyncio.sleep(1)
                elapsed += 1

            result = await fut

            # Extract word timestamps
            words = []
            for seg in result.segments:
                for w in (seg.words or []):
                    wt = w.word.strip()
                    if wt:
                        words.append({"word": wt, "start": round(w.start, 3), "end": round(w.end, 3)})
            if not words:
                for seg in result.segments:
                    words.append({"word": seg.text.strip(),
                                  "start": round(seg.start, 3), "end": round(seg.end, 3)})

            # Group into lines
            lines = []
            if lyrics and words:
                lyric_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
                ptr = 0
                for line_text in lyric_lines:
                    n = len(line_text.split())
                    chunk = words[ptr:ptr + n]
                    if chunk:
                        lines.append({"line": line_text,
                                      "start": chunk[0]["start"], "end": chunk[-1]["end"]})
                    ptr += n
                    if ptr >= len(words):
                        break
            else:
                for seg in result.segments:
                    lines.append({"line": seg.text.strip(),
                                  "start": round(seg.start, 3), "end": round(seg.end, 3)})

            yield sse({"stage": "done", "lines": lines})

        except Exception as e:
            yield sse({"stage": "error", "msg": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Serve frontend — must be last so API routes take priority
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
