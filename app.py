import asyncio
import io
import json
import os
import pathlib
import re
import signal
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field as dc_field

# ---------------------------------------------------------------------------
# Windows: shut down cleanly when the console window is closed
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes

    _CTRL_CLOSE_EVENT = 2

    @ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)
    def _win_ctrl_handler(ctrl_type):
        if ctrl_type == _CTRL_CLOSE_EVENT:
            # Send SIGINT to ourselves — uvicorn catches it and shuts down gracefully
            os.kill(os.getpid(), signal.SIGINT)
            # Block briefly so uvicorn has time to start shutdown before Windows
            # forcibly terminates the process (~5 s window)
            threading.Event().wait(4)
        return False  # pass event to next handler

    ctypes.windll.kernel32.SetConsoleCtrlHandler(_win_ctrl_handler, True)

import httpx
import stable_whisper
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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
        self.silent = False   # set True to stop forwarding tqdm to terminal

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
        if not self._spy.silent:
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
# Upload directory — local files submitted for whisper processing
# ---------------------------------------------------------------------------
UPLOAD_DIR = pathlib.Path(tempfile.gettempdir()) / "jw_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

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
# Model loading — separate align (sync) and verify models, lazy-loaded
# ---------------------------------------------------------------------------
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
_PREF_DIR = pathlib.Path(__file__).parent

def _read_pref(filename: str, choices: list, default: str) -> str:
    try:
        val = (_PREF_DIR / filename).read_text().strip()
        if val in choices:
            return val
    except OSError:
        pass
    return default

def _write_pref(filename: str, value: str) -> None:
    try:
        (_PREF_DIR / filename).write_text(value)
    except OSError:
        pass

# Device preference: "auto" | "cpu" | "cuda"
DEVICE_PREF: str = _read_pref(".device_pref", ["auto", "cpu", "cuda"], "auto")

def _cuda_usable() -> bool:
    """torch.cuda.is_available() only checks that a driver + device are present —
    it stays True even when the installed PyTorch build has no compiled kernels
    for that GPU's compute capability (e.g. a cu128 wheel on an old Pascal card
    like the GT 1030 / sm_61), which fails loudly mid-computation instead of at
    startup. Cross-check the device's capability against what this build ships
    kernels for.

    Note this is a floor check, not exact membership: CUDA's minor-version
    compatibility means e.g. sm_86-compiled kernels already run fine on sm_89
    (Ada/RTX 40-series) hardware, so a capability that's simply absent from
    get_arch_list() but *higher* than everything in it is still fine. Only a
    capability *lower* than the build's floor is a real incompatibility."""
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
        cap = major + minor / 10
        arch_caps = []
        for arch in torch.cuda.get_arch_list():
            digits = "".join(ch for ch in arch if ch.isdigit())
            if len(digits) >= 2:
                arch_caps.append(int(digits[:-1]) + int(digits[-1]) / 10)
        if arch_caps and cap < min(arch_caps):
            print(f"[whisper] GPU compute capability {major}.{minor} is older than the "
                  f"installed PyTorch build's minimum ({min(arch_caps):.1f}) — falling back to CPU.")
            return False
    except Exception:
        pass
    return True


def _get_device() -> str:
    if DEVICE_PREF == "cpu":
        return "cpu"
    avail = "cuda" if _cuda_usable() else "cpu"
    if DEVICE_PREF == "cuda" and avail != "cuda":
        print("[whisper] CUDA requested but not usable — falling back to CPU.")
        return "cpu"
    return avail

# Align model (used for sync / alignment tasks)
ALIGN_MODEL_SIZE: str = _read_pref(".model_pref_align", WHISPER_MODELS,
                                    _read_pref(".model_pref", WHISPER_MODELS,
                                               os.getenv("WHISPER_MODEL", "small")))
_align_model = None

# Verify model (used for free-transcription / verify tasks)
VERIFY_MODEL_SIZE: str = _read_pref(".model_pref_verify", WHISPER_MODELS, "base")
_verify_model = None

_model_lock     = asyncio.Lock()   # serialises model loads (one at a time)
_inference_lock = asyncio.Lock()   # one whisper inference at a time (model is not thread-safe for concurrent calls)


async def get_align_model():
    global _align_model
    if _align_model is None:
        async with _model_lock:
            if _align_model is None:
                device = _get_device()
                print(f"[whisper] Loading align model '{ALIGN_MODEL_SIZE}' on {device.upper()} …")
                loop = asyncio.get_running_loop()
                _align_model = await loop.run_in_executor(
                    None, lambda: stable_whisper.load_model(ALIGN_MODEL_SIZE, device=device))
                print(f"[whisper] Align model ready.")
    return _align_model


async def get_verify_model():
    global _verify_model
    if _verify_model is None:
        async with _model_lock:
            if _verify_model is None:
                device = _get_device()
                print(f"[whisper] Loading verify model '{VERIFY_MODEL_SIZE}' on {device.upper()} …")
                loop = asyncio.get_running_loop()
                _verify_model = await loop.run_in_executor(
                    None, lambda: stable_whisper.load_model(VERIFY_MODEL_SIZE, device=device))
                print(f"[whisper] Verify model ready.")
    return _verify_model


# Backward-compat alias used by legacy SSE endpoints
async def get_model():
    return await get_align_model()


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
    local_path: str = ""  # non-empty → use this file instead of fetching from API
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

# SSE broadcast via asyncio.Condition — all stream generators wait on this
_q_cond: asyncio.Condition | None = None   # initialised in lifespan
_q_state_json: str = '{"active":null,"pending":[],"history":[]}'


async def _q_broadcast() -> None:
    global _q_state_json
    active   = _active_task.to_dict() if _active_task else None
    pending  = [t.to_dict() for t in _tasks.values() if t.status == "pending"]
    done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
    history  = [t.to_dict() for t in sorted(done_list, key=lambda t: t.created_at)][-20:]
    _q_state_json = json.dumps({"active": active, "pending": pending, "history": history})
    if _q_cond is not None:
        async with _q_cond:
            _q_cond.notify_all()


# ── Shared whisper helpers ────────────────────────────────────────────────

def _align(model_obj, tmp_path: str, lyrics: str):
    # original_split=True keeps one output segment per input lyric line
    # (rather than re-splitting by punctuation lyrics usually lack), which
    # lets stable-ts's per-segment gap_padding actually line up with our
    # real line boundaries and reduces its tendency to predict a line's
    # start too early. nonspeech_skip lower than the 5s default catches the
    # shorter pauses/breaths between song lines too, so word timestamps
    # don't get stretched across them.
    return model_obj.align(
        tmp_path, lyrics, language="en",
        original_split=True,
        nonspeech_skip=1.0,
    )


def _lines_from_alignment(result, lyrics: str) -> list[dict]:
    """Turn a stable-ts alignment/transcription result into {line, start, end} dicts."""
    words: list[dict] = []
    for seg in result.segments:
        for w in (seg.words or []):
            wt = w.word.strip()
            if wt:
                words.append({"word": wt, "start": round(w.start, 3), "end": round(w.end, 3)})
    if not words:
        for seg in result.segments:
            words.append({"word": seg.text.strip(), "start": round(seg.start, 3), "end": round(seg.end, 3)})

    lines: list[dict] = []
    if lyrics and words:
        lyric_lines = [l.strip() for l in lyrics.split("\n") if l.strip()]
        if len(result.segments) == len(lyric_lines):
            # original_split=True guarantees this 1:1 correspondence -- use the
            # segment's own timing directly instead of the word-count slicing
            # below, which can drift if naive whitespace-splitting doesn't
            # match Whisper's own word boundaries (contractions, punctuation).
            for line_text, seg in zip(lyric_lines, result.segments):
                lines.append({"line": line_text, "start": round(seg.start, 3), "end": round(seg.end, 3)})
        else:
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
            lines.append({"line": seg.text.strip(), "start": round(seg.start, 3), "end": round(seg.end, 3)})
    return lines


async def _whisper_sync_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    """Run align/transcribe in executor. Returns lines."""
    label     = "Aligning" if lyrics else "Transcribing"
    spy       = _ProgressSpy()
    loop      = asyncio.get_running_loop()
    model_obj = await get_align_model()

    def _run():
        orig = sys.stderr
        sys.stderr = _TeeStderr(spy, orig)
        try:
            if lyrics:
                return _align(model_obj, tmp_path, lyrics)
            else:
                return model_obj.transcribe(tmp_path, word_timestamps=True, verbose=False)
        finally:
            sys.stderr = orig

    async with _inference_lock:
        fut = loop.run_in_executor(None, _run)
        cancelled = False
        elapsed = 0
        while not fut.done():
            if task.cancel_requested:
                cancelled = True
                spy.silent = True          # silence tqdm in terminal immediately
                task.progress = {"stage": "aligning", "msg": "Cancelling…", "pct": 0, "step": "aligning"}
                await _q_broadcast()
                await asyncio.shield(fut)  # wait for thread (lock held — prevents concurrent model use)
                break
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
            await asyncio.sleep(0.5)
            elapsed += 1
        if not cancelled:
            result = await fut

    if cancelled:
        raise asyncio.CancelledError()

    return _lines_from_alignment(result, lyrics)


async def _whisper_verify_worker(task: QueueTask, tmp_path: str, lyrics: str) -> list[dict]:
    """Free-transcribe + compare. Returns verify_results list."""
    spy      = _ProgressSpy()
    loop     = asyncio.get_running_loop()
    model_v  = await get_verify_model()

    def _run():
        orig = sys.stderr
        sys.stderr = _TeeStderr(spy, orig)
        try:
            return model_v.transcribe(tmp_path, verbose=False)
        finally:
            sys.stderr = orig

    async with _inference_lock:
        fut = loop.run_in_executor(None, _run)
        cancelled = False
        elapsed = 0
        while not fut.done():
            if task.cancel_requested:
                cancelled = True
                spy.silent = True          # silence tqdm in terminal immediately
                task.progress = {"stage": "transcribing", "msg": "Cancelling…", "pct": 0, "step": "verifying"}
                await _q_broadcast()
                await asyncio.shield(fut)  # wait for thread (lock held)
                break
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
            await asyncio.sleep(0.5)
            elapsed += 1
        if not cancelled:
            result = await fut

    if cancelled:
        raise asyncio.CancelledError()

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
    if task.local_path:
        tmp_path = task.local_path
        lyrics   = task.lyrics
    else:
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
    if task.local_path:
        tmp_path = task.local_path
        lyrics   = task.lyrics
    else:
        task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
        await _q_broadcast()
        song = await jw_get(f"/songs/{task.song_id}/")
        if not song.get("path"):
            raise ValueError("No audio file for this song.")
        lyrics = task.lyrics or song.get("lyrics", "") or ""
        tmp_path = await _download_audio(task, song)
    if not lyrics:
        raise ValueError("No lyrics to verify against.")
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


async def _run_transcribe_task(task: QueueTask) -> None:
    """Transcribe audio to plain text, ignoring any existing lyrics."""
    if task.local_path:
        tmp_path = task.local_path
    else:
        task.progress = {"stage": "fetching", "msg": "Fetching song info…", "step": "fetching", "pct": 2}
        await _q_broadcast()
        song = await jw_get(f"/songs/{task.song_id}/")
        if not song.get("path"):
            raise ValueError("No audio file for this song.")
        tmp_path = await _download_audio(task, song)
    if task.cancel_requested:
        raise asyncio.CancelledError()
    task.progress = {"stage": "loading", "msg": "Loading Whisper model…", "step": "loading", "pct": 30}
    await _q_broadcast()
    # Run with empty lyrics → transcription mode
    lines = await _whisper_sync_worker(task, tmp_path, "")
    plain_text = "\n".join(l["line"] for l in lines)
    task.result   = {"lines": lines, "text": plain_text}
    task.progress = {"stage": "done", "msg": f"Transcribed — {len(lines)} lines", "step": "done", "pct": 100}


async def _run_auto_task(task: QueueTask) -> None:
    """Genius → verify → sync pipeline."""
    if task.local_path:
        tmp_path = task.local_path
        lyrics   = task.lyrics
        song     = {"path": task.local_path}  # minimal stub so later code stays happy
    else:
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

    # Step 2: Download (skipped for local files) + Verify
    if not task.local_path:
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

        if task.status in ("cancelled", "cancelling"):
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
            elif task.type == "transcribe":
                await _run_transcribe_task(task)
            if task.status in ("running", "cancelling"):   # runner didn't set error/cancelled
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
def _log_startup_diagnostics() -> None:
    print(f"[startup] Python: {sys.executable}")
    print(f"[startup] Device preference: {DEVICE_PREF}")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"[startup] torch {torch.__version__} (CUDA build: {torch.version.cuda or 'none — CPU-only wheel'})")
        print(f"[startup] torch.cuda.is_available(): {cuda_available}")
        if cuda_available:
            print(f"[startup] GPU: {torch.cuda.get_device_name(0)}")
            if not _cuda_usable():
                print("[startup] This GPU's compute capability isn't supported by the installed "
                      "PyTorch build, so Whisper will run on CPU instead. This usually means the "
                      "GPU is too old for the CUDA build launch.py installed — no fix available "
                      "besides using CPU or a newer GPU.")
        elif DEVICE_PREF == "cuda":
            print("[startup] Device preference is 'cuda' but CUDA isn't available from this "
                  "interpreter — Whisper will fall back to CPU. If you expected GPU here, "
                  "make sure the server was started via start.bat / launch.py so it's using "
                  "the project's .venv (not some other Python on PATH).")
    except Exception as e:
        print(f"[startup] Could not inspect torch/CUDA: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log_startup_diagnostics()
    global _q_cond
    _q_cond = asyncio.Condition()
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
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(BASE + path, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "juicewrldapi.com took too long to respond. Try again in a moment.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"juicewrldapi.com returned an error ({e.response.status_code}).")
    except httpx.HTTPError:
        raise HTTPException(502, "Couldn't reach juicewrldapi.com. Check your connection and try again.")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/search")
async def search(q: str, page_size: int = 20):
    return await jw_get("/songs/", {"search": q, "page_size": page_size})


@app.get("/api/song/{song_id}")
async def get_song(song_id: int):
    return await jw_get(f"/songs/{song_id}/")


@app.get("/api/songs/all")
async def get_all_songs():
    """Every song in the catalog, via the upstream `?all=true` bulk variant
    (bypasses pagination entirely instead of walking pages)."""
    data = await jw_get("/songs/", {"all": "true"})
    songs = data if isinstance(data, list) else data.get("results", [])
    return {"songs": songs}


@app.get("/api/versions/all")
async def get_all_versions():
    """Every saved version/grouping row across the whole catalog, via the
    upstream `?all=true` bulk variant (bypasses pagination entirely)."""
    rows = await jw_get("/versions/", {"all": "true"})
    if not isinstance(rows, list):
        rows = rows.get("results", [])
    for row in rows:
        if "title" in row:
            row["version_title"] = row.pop("title")
    return {"versions": rows}


@app.get("/api/versions/{song_id}")
async def get_versions(song_id: int):
    """Version/grouping rows for a song (and its group-mates), if any.

    Upstream calls the group's display name "title"; our own contract with
    the frontend uses "version_title" (matching the old Supabase column
    name), so translate it here rather than leaking the upstream naming.
    """
    data = await jw_get(f"/versions/{song_id}/")
    for row in data.get("results", []):
        if "title" in row:
            row["version_title"] = row.pop("title")
    return data


class VersionSaveRequest(BaseModel):
    token: str
    group_id: int
    version: str | None = None
    version_title: str | None = None


async def _write_version(song_id: int, req: VersionSaveRequest, method: str, pk: int | None = None):
    if not req.token:
        raise HTTPException(401, "No auth token provided.")
    # Create (POST) targets the list route; update (PATCH) targets the row's
    # own detail route — Django's `versions/<int:song_id>/<int:pk>/`.
    path = f"/versions/{song_id}/{pk}/" if pk is not None else f"/versions/{song_id}/"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.request(
            method,
            BASE + path,
            headers={
                "Authorization": f"Token {req.token}",
                "Content-Type": "application/json",
            },
            json={
                # Upstream requires both fields present and non-blank —
                # "title" (not "version_title") and "version" as a string.
                # Fall back to "Null" when a song has nothing more specific
                # (e.g. it's alone with no other versions, or a "reset to
                # automatic" that has no real label to give).
                "group_id": req.group_id,
                "version": req.version or "Null",
                "title": req.version_title or "unknown",
            },
        )
        body = r.text
        if r.status_code in (401, 403):
            raise HTTPException(403, "Token rejected — editor role required.")
        if not r.is_success:
            raise HTTPException(r.status_code, f"API error {r.status_code}: {body}")
        return r.json()


@app.post("/api/versions/{song_id}")
async def create_version(song_id: int, req: VersionSaveRequest):
    """First-time save for a song that has no version row yet."""
    return await _write_version(song_id, req, "POST")


@app.patch("/api/versions/{song_id}/{pk}")
async def update_version(song_id: int, pk: int, req: VersionSaveRequest):
    """Update of a song's existing version row (pk = that row's own id)."""
    return await _write_version(song_id, req, "PATCH", pk=pk)


@app.get("/api/model")
async def get_model_info():
    avail_device = "cuda" if _cuda_usable() else "cpu"
    return {
        # legacy field kept for backward-compat
        "model": ALIGN_MODEL_SIZE,
        "loaded": _align_model is not None,
        # new fields
        "align_model":   ALIGN_MODEL_SIZE,
        "verify_model":  VERIFY_MODEL_SIZE,
        "device_pref":   DEVICE_PREF,
        "device":        avail_device,
        "align_loaded":  _align_model is not None,
        "verify_loaded": _verify_model is not None,
        "models":        WHISPER_MODELS,
    }


@app.post("/api/model")
async def set_model(body: dict):
    global ALIGN_MODEL_SIZE, VERIFY_MODEL_SIZE, DEVICE_PREF
    global _align_model, _verify_model

    changed_device = False

    if "device_pref" in body:
        pref = body["device_pref"].strip()
        if pref not in ("auto", "cpu", "cuda"):
            raise HTTPException(400, f"device_pref must be auto | cpu | cuda")
        DEVICE_PREF = pref
        _write_pref(".device_pref", pref)
        changed_device = True

    if "align_model" in body or "model" in body:     # "model" = legacy key
        name = (body.get("align_model") or body.get("model", "")).strip()
        if name not in WHISPER_MODELS:
            raise HTTPException(400, f"Unknown model '{name}'")
        ALIGN_MODEL_SIZE = name
        _align_model = None
        _write_pref(".model_pref_align", name)
        _write_pref(".model_pref", name)              # keep legacy file in sync

    if "verify_model" in body:
        name = body["verify_model"].strip()
        if name not in WHISPER_MODELS:
            raise HTTPException(400, f"Unknown model '{name}'")
        VERIFY_MODEL_SIZE = name
        _verify_model = None
        _write_pref(".model_pref_verify", name)

    if changed_device:
        _align_model  = None   # force reload on new device
        _verify_model = None

    return {
        "align_model":   ALIGN_MODEL_SIZE,
        "verify_model":  VERIFY_MODEL_SIZE,
        "device_pref":   DEVICE_PREF,
        "device":        "cuda" if _cuda_usable() else "cpu",
        "align_loaded":  _align_model is not None,
        "verify_loaded": _verify_model is not None,
    }


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
                        return _align(model, tmp_path, lyrics)
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
            lines = _lines_from_alignment(result, lyrics)

            yield sse({"stage": "done", "lines": lines})

        except Exception as e:
            yield sse({"stage": "error", "msg": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Local file upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Accept a local audio file, save it to UPLOAD_DIR, return the server path."""
    suffix = pathlib.Path(file.filename or "audio").suffix or ".mp3"
    uid    = str(uuid.uuid4())[:8]
    dest   = UPLOAD_DIR / f"{uid}{suffix}"
    contents = await file.read()
    dest.write_bytes(contents)
    return {"path": str(dest), "name": file.filename or "local file"}


# ---------------------------------------------------------------------------
# Queue API
# ---------------------------------------------------------------------------

class QueueAddRequest(BaseModel):
    type: str        # "sync" | "verify" | "auto"
    song_id: int = 0
    song_name: str
    lyrics: str = ""
    local_path: str = ""  # set when syncing a local file instead of an API song


@app.post("/api/queue")
async def queue_add(req: QueueAddRequest):
    if req.type not in ("sync", "verify", "auto", "transcribe"):
        raise HTTPException(400, f"Unknown task type '{req.type}'")
    task = QueueTask(
        id=str(uuid.uuid4())[:8],
        type=req.type,
        song_id=req.song_id,
        song_name=req.song_name,
        lyrics=req.lyrics,
        local_path=req.local_path,
    )
    _tasks[task.id] = task
    await _task_queue.put(task)
    await _q_broadcast()
    return {"task_id": task.id}


@app.get("/api/queue")
async def queue_state():
    active   = _active_task.to_dict() if _active_task else None
    pending  = [t.to_dict() for t in _tasks.values() if t.status == "pending"]
    done_list = [t for t in _tasks.values() if t.status in ("done", "error", "cancelled")]
    history  = [t.to_dict() for t in sorted(done_list, key=lambda t: t.created_at)][-20:]
    return {"active": active, "pending": pending, "history": history}


@app.delete("/api/queue/history")
async def clear_queue_history():
    """Remove all completed/error/cancelled tasks from the queue."""
    to_remove = [tid for tid, t in _tasks.items()
                 if t.status in ("done", "error", "cancelled")]
    for tid in to_remove:
        del _tasks[tid]
    await _q_broadcast()
    return {"cleared": len(to_remove)}


@app.post("/api/queue/{task_id}/cancel")
async def queue_cancel(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task.cancel_requested = True
    if task.status == "pending":
        task.status = "cancelled"
    elif task.status == "running":
        task.progress = {**task.progress, "msg": "Cancelling…"}
        task.status = "cancelling"
    await _q_broadcast()
    return {"cancelled": True}


@app.get("/api/queue/stream")
async def queue_stream():
    if _q_cond is None:
        raise HTTPException(503, "Server not ready yet")

    async def gen():
        yield f"data: {_q_state_json}\n\n"
        while True:
            async with _q_cond:
                try:
                    await asyncio.wait_for(_q_cond.wait(), timeout=25)
                    data = _q_state_json
                except asyncio.TimeoutError:
                    data = None
            if data is not None:
                yield f"data: {data}\n\n"
            else:
                yield ": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    # Chrome heuristically caches HTML served with only Last-Modified, which
    # keeps stale copies of the UI alive across edits — force revalidation.
    response = await call_next(request)
    if request.url.path.endswith(".html") or request.url.path in ("", "/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/", StaticFiles(directory="static", html=True), name="static")
