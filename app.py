import asyncio
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field as dc_field

import httpx
import stable_whisper
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = "https://juicewrldapi.com/juicewrld"

# ---------------------------------------------------------------------------
# tqdm progress spy — captures stable_whisper alignment/transcription progress
# ---------------------------------------------------------------------------
_TQDM_RE = re.compile(
    r'([A-Za-z][\w ]*):\s*(\d+)%'          # label + percent
    r'.*?([\d.]+)/([\d.]+)'                  # done / total (seconds)
    r'.*?\[(\d+:\d+)<(\d+:\d+),\s*([\d.]+)s/sec\]'  # [elapsed<eta, speed]
)


class _ProgressSpy:
    """Thread-safe stderr capturer that parses tqdm progress lines."""
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: dict | None = None

    def write(self, s: str) -> int:
        for chunk in re.split(r'[\r\n]', s):
            chunk = chunk.strip()
            if not chunk:
                continue
            m = _TQDM_RE.search(chunk)
            if m:
                with self._lock:
                    self._latest = {
                        'label': m.group(1).strip(),
                        'pct':   int(m.group(2)),
                        'done':  float(m.group(3)),
                        'total': float(m.group(4)),
                        'elapsed': m.group(5),
                        'eta':   m.group(6),
                        'speed': float(m.group(7)),
                    }
        return len(s)

    def flush(self): pass
    def isatty(self) -> bool: return False
    def fileno(self): raise io.UnsupportedOperation("fileno")

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest


class _TeeStderr:
    """Tees stderr to a _ProgressSpy and the original stderr."""
    def __init__(self, spy: _ProgressSpy, orig):
        self._spy = spy
        self._orig = orig

    def write(self, s: str) -> int:
        self._spy.write(s)
        try:
            self._orig.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self) -> bool: return False
    def fileno(self): raise io.UnsupportedOperation("fileno")


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
# Task Queue
# ---------------------------------------------------------------------------

@dataclass
class QueueTask:
    id: str
    type: str           # "sync" | "verify" | "auto"
    song_id: int
    song_name: str
    lyrics: str         # may be empty
    status: str = "pending"   # pending|running|done|error|cancelled
    progress: dict = dc_field(default_factory=dict)
    error: str = ""
    created_at: float = dc_field(default_factory=time.time)
    cancel_requested: bool = False
    result: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.type,
            "song_id": self.song_id,
            "song_name": self.song_name,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
        }
        if self.result and self.status in ("done", "error"):
            d["result"] = self.result
        return d


_task_queue: asyncio.Queue = asyncio.Queue()
_tasks: dict[str, QueueTask] = {}          # id → task (all states)
_active_task: QueueTask | None = None
_q_subscribers: list[asyncio.Queue] = []
_inference_lock = asyncio.Lock()           # one whisper inference at a time


async def _q_broadcast() -> None:
    active  = _active_task.to_dict() if _active_task else None
    pending = [t.to_dict() for t in _tasks.values() if t.status == "pending"]
    done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
    history = [t.to_dict() for t in sorted(done_list, key=lambda t: t.created_at)][-20:]
    data = f"data: {json.dumps({'active': active, 'pending': pending, 'history': history})}\n\n"
    dead = []
    for q in _q_subscribers:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        try:
            _q_subscribers.remove(q)
        except ValueError:
            pass


# ── Shared whisper helpers ────────────────────────────────────────────────

async def _whisper_sync_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    """Run align/transcribe in executor (inference lock held). Returns lines."""
    label = "Aligning" if lyrics else "Transcribing"
    spy   = _ProgressSpy()
    loop  = asyncio.get_event_loop()

    model_obj = await get_model()

    def _run():
        orig = sys.stderr
        sys.stderr = _TeeStderr(spy, orig)
        try:
            if lyrics:
                return model_obj.align(tmp_path, lyrics, language="en")
            else:
                return model_obj.transcribe(tmp_path, word_timestamps=True, verbose=False)
        finally:
            sys.stderr = orig

    async with _inference_lock:
        fut = loop.run_in_executor(None, _run)
        elapsed = 0
        while not fut.done():
            if task.cancel_requested:
                task.progress = {"stage": "aligning", "msg": "Cancelling…", "pct": 0, "step": "aligning"}
                await _q_broadcast()
                await asyncio.shield(fut)   # wait for thread before releasing lock
                raise asyncio.CancelledError()
            prog = spy.latest()
            if prog:
                pct = 55 + prog["pct"] * 0.44
                msg = (f"{label}: {prog['pct']}%  "
                       f"{prog['done']:.1f}/{prog['total']:.1f}s  "
                       f"[{prog['elapsed']}<{prog['eta']}, {prog['speed']:.2f}s/sec]")
            else:
                pct = min(54, int((elapsed / 180) ** 0.5 * 54))
                msg = f"{label}… {elapsed}s"
            task.progress = {"stage": "aligning", "pct": pct, "msg": msg, "step": "aligning",
                             **({"progress": prog} if prog else {})}
            await _q_broadcast()
            await asyncio.sleep(1)
            elapsed += 1
        result = await fut

    # Extract words
    words: list[dict] = []
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
    lines: list[dict] = []
    if lyrics and words:
        lyric_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
        ptr = 0
        for line_text in lyric_lines:
            n = len(line_text.split())
            chunk = words[ptr:ptr + n]
            if chunk:
                lines.append({"line": line_text, "start": chunk[0]["start"], "end": chunk[-1]["end"]})
            ptr += n
            if ptr >= len(words):
                break
    else:
        for seg in result.segments:
            lines.append({"line": seg.text.strip(),
                          "start": round(seg.start, 3), "end": round(seg.end, 3)})
    return lines


async def _whisper_verify_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    """Free-transcribe + compare. Returns verify_results list."""
    spy      = _ProgressSpy()
    loop     = asyncio.get_event_loop()
    model_v  = await get_model()

    def _run():
        orig = sys.stderr
        sys.stderr = _TeeStderr(spy, orig)
        try:
            return model_v.transcribe(tmp_path, verbose=False)
        finally:
            sys.stderr = orig

    async with _inference_lock:
        fut = loop.run_in_executor(None, _run)
        elapsed = 0
        while not fut.done():
            if task.cancel_requested:
                await asyncio.shield(fut)
                raise asyncio.CancelledError()
            prog = spy.latest()
            if prog:
                pct = 42 + prog["pct"] * 0.53
                msg = (f"Transcribing: {prog['pct']}%  "
                       f"{prog['done']:.1f}/{prog['total']:.1f}s  "
                       f"[{prog['elapsed']}<{prog['eta']}, {prog['speed']:.2f}s/sec]")
            else:
                pct = min(41, int((elapsed / 180) ** 0.5 * 41))
                msg = f"Transcribing… {elapsed}s"
            task.progress = {"stage": "transcribing", "pct": pct, "msg": msg, "step": "verifying",
                             **({"progress": prog} if prog else {})}
            await _q_broadcast()
            await asyncio.sleep(1)
            elapsed += 1
        result = await fut

    transcription = result.text or ""
    lyric_lines   = [l for l in lyrics.split("\n") if l.strip()]
    return _verify_lines(lyric_lines, transcription)


async def _download_audio(task: QueueTask, song: dict) -> str:
    """Stream audio via ensure_audio, broadcasting progress. Returns tmp_path."""
    tmp_path = None
    async for ev in ensure_audio(song["path"]):
        if task.cancel_requested:
            raise asyncio.CancelledError()
        if "_path" in ev:
            tmp_path = ev["_path"]
        else:
            task.progress = {**ev, "step": "downloading"}
            await _q_broadcast()
    if tmp_path is None:
        raise ValueError("Audio download failed.")
    return tmp_path


# ── Task runners ──────────────────────────────────────────────────────────

async def _run_sync_task(task: QueueTask) -> None:
    task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
    await _q_broadcast()
    song = await jw_get(f"/songs/{task.song_id}/")
    if not song.get("path"):
        raise ValueError("No audio file for this song.")
    lyrics   = task.lyrics or song.get("lyrics", "") or ""
    tmp_path = await _download_audio(task, song)
    if task.cancel_requested:
        raise asyncio.CancelledError()
    task.progress = {"stage": "loading", "msg": "Loading Whisper model…", "step": "loading", "pct": 52}
    await _q_broadcast()
    lines = await _whisper_sync_worker(task, tmp_path, lyrics)
    task.result   = {"lines": lines}
    task.progress = {"stage": "done", "msg": f"Done — {len(lines)} lines synced", "step": "done", "pct": 100}


async def _run_verify_task(task: QueueTask) -> None:
    task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
    await _q_broadcast()
    song = await jw_get(f"/songs/{task.song_id}/")
    if not song.get("path"):
        raise ValueError("No audio file for this song.")
    lyrics = task.lyrics or song.get("lyrics", "") or ""
    if not lyrics:
        raise ValueError("No lyrics to verify against.")
    tmp_path = await _download_audio(task, song)
    if task.cancel_requested:
        raise asyncio.CancelledError()
    task.progress = {"stage": "loading", "msg": "Loading Whisper model…", "step": "loading", "pct": 38}
    await _q_broadcast()
    verify_results = await _whisper_verify_worker(task, tmp_path, lyrics)
    counts = {"present": 0, "uncertain": 0, "absent": 0}
    for r in verify_results:
        counts[r["status"]] += 1
    task.result   = {"verify": verify_results, "counts": counts}
    task.progress = {"stage": "done",
                     "msg": f"Done — ✓{counts['present']} ?{counts['uncertain']} ✗{counts['absent']}",
                     "step": "done", "pct": 100}


async def _run_auto_task(task: QueueTask) -> None:
    """Genius → verify → sync pipeline."""
    task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
    await _q_broadcast()
    song   = await jw_get(f"/songs/{task.song_id}/")
    lyrics = task.lyrics or song.get("lyrics", "") or ""

    # Step 1: Genius if no lyrics
    if not lyrics:
        task.progress = {"stage": "genius", "msg": "Searching Genius for lyrics…", "step": "genius", "pct": 5}
        await _q_broadcast()
        title = re.sub(r'\s*[\[({].*?[\])}]\s*', '', task.song_name).strip()
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                r = await client.get(
                    "https://genius.com/api/search/song",
                    params={"q": f"Juice WRLD {title}"},
                    headers=GENIUS_HEADERS,
                )
                r.raise_for_status()
                sections = r.json().get("response", {}).get("sections", [])
                hits = sections[0].get("hits", []) if sections else []
                if hits:
                    r2 = await client.get(hits[0]["result"]["url"], headers=GENIUS_HEADERS)
                    r2.raise_for_status()
                    lyrics = _parse_genius_page(r2.text)
        except Exception as exc:
            print(f"[queue/auto] Genius error: {exc}")
        if not lyrics:
            raise ValueError("No lyrics found on Genius — add lyrics manually first.")

    if not song.get("path"):
        raise ValueError("No audio file for this song.")
    if task.cancel_requested:
        raise asyncio.CancelledError()

    # Step 2: Download + Verify
    task.progress = {"stage": "verifying", "msg": "Downloading audio for verify…", "step": "verifying", "pct": 10}
    await _q_broadcast()
    tmp_path = await _download_audio(task, song)
    if task.cancel_requested:
        raise asyncio.CancelledError()
    task.progress = {"stage": "loading", "msg": "Loading Whisper model…", "step": "loading", "pct": 25}
    await _q_broadcast()
    verify_results = await _whisper_verify_worker(task, tmp_path, lyrics)

    counts      = {"present": 0, "uncertain": 0, "absent": 0}
    for r in verify_results:
        counts[r["status"]] += 1
    total        = len(verify_results)
    absent_ratio = counts["absent"] / total if total > 0 else 0

    if absent_ratio > 0.2:
        task.result = {"verify": verify_results, "counts": counts}
        raise ValueError(
            f"{counts['absent']} absent lines ({round(absent_ratio * 100)}%) "
            f"— too many to auto-sync. Review verify results manually."
        )

    if task.cancel_requested:
        raise asyncio.CancelledError()

    # Step 3: Clean + Sync
    cleaned = "\n".join(r["line"] for r in verify_results if r["status"] != "absent")
    task.progress = {"stage": "syncing", "msg": "Syncing cleaned lyrics…", "step": "syncing", "pct": 50}
    await _q_broadcast()
    lines = await _whisper_sync_worker(task, tmp_path, cleaned)
    task.result   = {"lines": lines, "verify": verify_results, "counts": counts}
    task.progress = {"stage": "done", "msg": f"Auto done — {len(lines)} lines", "step": "done", "pct": 100}


# ── Background processor ──────────────────────────────────────────────────

async def _queue_processor() -> None:
    global _active_task
    while True:
        task = await _task_queue.get()

        if task.status == "cancelled":
            _task_queue.task_done()
            await _q_broadcast()
            continue

        _active_task = task
        task.status  = "running"
        await _q_broadcast()

        try:
            if task.type == "sync":
                await _run_sync_task(task)
            elif task.type == "verify":
                await _run_verify_task(task)
            elif task.type == "auto":
                await _run_auto_task(task)
            if task.status == "running":        # runner didn't set error
                task.status = "done"
        except asyncio.CancelledError:
            task.status = "cancelled"
        except Exception as exc:
            task.status = "error"
            task.error  = str(exc)
            print(f"[queue] Task {task.id} failed: {exc}")
        finally:
            _active_task = None
            _task_queue.task_done()
            await _q_broadcast()
            # Prune old history (keep last 30)
            done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
            if len(done_list) > 30:
                for old in sorted(done_list, key=lambda t: t.created_at)[:-30]:
                    _tasks.pop(old.id, None)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    proc = asyncio.create_task(_queue_processor())
    yield
    proc.cancel()
    try:
        await proc
    except asyncio.CancelledError:
        pass


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
            spy = _ProgressSpy()

            def _run_verify():
                orig = sys.stderr
                sys.stderr = _TeeStderr(spy, orig)
                try:
                    return model.transcribe(tmp_path, verbose=False)
                finally:
                    sys.stderr = orig

            fut = loop.run_in_executor(None, _run_verify)
            elapsed = 0
            while not fut.done():
                prog = spy.latest()
                if prog:
                    pct = 42 + prog['pct'] * 0.53
                    msg = (f"{prog['label']}: {prog['pct']}%  "
                           f"{prog['done']:.1f}/{prog['total']:.1f}s  "
                           f"[{prog['elapsed']}<{prog['eta']}, {prog['speed']:.2f}s/sec]")
                else:
                    pct = min(41, int((elapsed / 180) ** 0.5 * 41))
                    msg = f"Transcribing… {elapsed}s"
                yield sse({"stage": "transcribing", "pct": pct, "msg": msg,
                           **({"progress": prog} if prog else {})})
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
            spy = _ProgressSpy()

            def _run_sync():
                orig = sys.stderr
                sys.stderr = _TeeStderr(spy, orig)
                try:
                    if lyrics:
                        return model.align(tmp_path, lyrics, language="en")
                    else:
                        return model.transcribe(tmp_path, word_timestamps=True, verbose=False)
                finally:
                    sys.stderr = orig

            fut = loop.run_in_executor(None, _run_sync)

            elapsed = 0
            while not fut.done():
                prog = spy.latest()
                if prog:
                    pct = 55 + prog['pct'] * 0.44
                    msg = (f"{prog['label']}: {prog['pct']}%  "
                           f"{prog['done']:.1f}/{prog['total']:.1f}s  "
                           f"[{prog['elapsed']}<{prog['eta']}, {prog['speed']:.2f}s/sec]")
                else:
                    pct = min(54, int((elapsed / 180) ** 0.5 * 54))
                    msg = f"{label}… {elapsed}s"
                yield sse({"stage": "aligning", "pct": pct, "msg": msg,
                           **({"progress": prog} if prog else {})})
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
# Queue API
# ---------------------------------------------------------------------------

class QueueAddRequest(BaseModel):
    type: str        # "sync" | "verify" | "auto"
    song_id: int
    song_name: str
    lyrics: str = ""


@app.post("/api/queue")
async def queue_add(req: QueueAddRequest):
    if req.type not in ("sync", "verify", "auto"):
        raise HTTPException(400, f"Unknown task type '{req.type}'")
    task = QueueTask(
        id=str(uuid.uuid4())[:8],
        type=req.type,
        song_id=req.song_id,
        song_name=req.song_name,
        lyrics=req.lyrics,
    )
    _tasks[task.id] = task
    await _task_queue.put(task)
    await _q_broadcast()
    return {"task_id": task.id}


@app.get("/api/queue")
async def queue_state():
    active  = _active_task.to_dict() if _active_task else None
    pending = [t.to_dict() for t in _tasks.values() if t.status == "pending"]
    done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
    history = [t.to_dict() for t in sorted(done_list, key=lambda t: t.created_at)][-20:]
    return {"active": active, "pending": pending, "history": history}


@app.post("/api/queue/{task_id}/cancel")
async def queue_cancel(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task.cancel_requested = True
    if task.status == "pending":
        task.status = "cancelled"
    await _q_broadcast()
    return {"cancelled": True}


@app.get("/api/queue/stream")
async def queue_stream():
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _q_subscribers.append(q)

    # Send initial state immediately
    active  = _active_task.to_dict() if _active_task else None
    pending = [t.to_dict() for t in _tasks.values() if t.status == "pending"]
    done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
    history = [t.to_dict() for t in sorted(done_list, key=lambda t: t.created_at)][-20:]
    initial = f"data: {json.dumps({'active': active, 'pending': pending, 'history': history})}\n\n"

    async def gen():
        yield initial
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=25)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            try:
                _q_subscribers.remove(q)
            except ValueError:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Serve frontend — must be last so API routes take priority
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
