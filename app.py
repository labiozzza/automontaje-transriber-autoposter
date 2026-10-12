import os, json, uuid, asyncio, subprocess, time, re, tempfile, hashlib, urllib.request, importlib, threading, gc, math, io, sys
import shutil
import traceback
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import whisper
from pyannote.audio import Pipeline as DiarizationPipeline
import torch
import soundfile as sf
import numpy as np
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cpu"

app = FastAPI(title="Transcriber API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# In-memory job storage
jobs = {}
job_queues = {}

# Per-job publish locks (single-process uvicorn limited). All job state lives
# in memory, so this app is designed for exactly one server process; running
# uvicorn with --workers > 1 (or any multi-process deployment) would require a
# cross-process lock (e.g. a DB row / Redis / file lock) instead of these.
publish_locks: dict[str, asyncio.Lock] = {}


def _publish_lock(job_id: str) -> asyncio.Lock:
    lock = publish_locks.get(job_id)
    if lock is None:
        lock = asyncio.Lock()
        publish_locks[job_id] = lock
    return lock
whisper_models = {}
WHISPER_PROGRESS_LOCK = threading.Lock()
WHISPER_JOB_LOCK = threading.RLock()
DIARIZATION_JOB_LOCK = threading.RLock()
MODEL_IDLE_SECONDS = 300
whisper_last_used = 0.0
diarization_last_used = 0.0
whisper_idle_timer = None
diarization_idle_timer = None
diarization_pipeline = None
PROJECTS = {}
PROJECTS_LOCK = asyncio.Lock()
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".3gp", ".ts"}
TRANSCRIPTION_STATE_DIR = BASE_DIR / "transcription_jobs"
TRANSCRIPTION_STATE_DIR.mkdir(exist_ok=True)
PROJECTS_PATH = BASE_DIR / "projects.json"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _save_transcription_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if not isinstance(job, dict) or job.get("kind") == "montage":
        return
    _write_json_atomic(TRANSCRIPTION_STATE_DIR / f"{job_id}.json", job)


def _save_projects() -> None:
    _write_json_atomic(PROJECTS_PATH, {"projects": list(PROJECTS.values())})


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_whisper_model(root: str, name: str, progress_cb=None) -> Optional[str]:
    """Докачиваемое (resume) скачивание модели whisper с прогрессом.
    Возвращает путь к локальному файлу модели либо None."""
    url = whisper._MODELS.get(name)
    if url is None:
        return None
    expected = url.split("/")[-2]
    target = os.path.join(root, name + ".pt")
    part = target + ".part"

    if os.path.isfile(target):
        if _file_sha256(target) == expected:
            return target
        os.replace(target, part)

    start = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "Mozilla/5.0 (whisper-mirror-downloader)"}
    if start:
        headers["Range"] = f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as out:
        total = start + int(resp.headers.get("Content-Length") or 0)
        downloaded = start
        last_pct = -1
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
            else:
                pct = int(downloaded * 100 / (start or 1))
            if pct != last_pct and progress_cb:
                last_pct = pct
                progress_cb(pct, downloaded, total)
        if progress_cb:
            progress_cb(100, downloaded, total)
    if _file_sha256(part) != expected:
        raise RuntimeError(f"Скачивание модели {name}: контрольная сумма не совпала")
    os.replace(part, target)
    return target


def load_whisper_model(name: str, progress_cb=None, load_cb=None):
    if name not in whisper_models:
        if whisper_models:
            loaded = ", ".join(whisper_models)
            print(f"[transcriber] Unloading Whisper model(s): {loaded}")
            whisper_models.clear()
            gc.collect()
        print(f"[transcriber] Checking whisper model '{name}'...")
        root = os.path.expanduser("~/.cache/whisper")
        path = ensure_whisper_model(root, name, progress_cb)
        if load_cb:
            load_cb()
        print(f"[transcriber] Loading whisper model '{name}'...")
        whisper_models[name] = whisper.load_model(path or name, device=DEVICE)
        print(f"[transcriber] Whisper '{name}' loaded on {DEVICE}")
    return whisper_models[name]


def run_whisper_job(name, audio_path, model_progress, model_load_start, transcription_start, transcription_progress, **kwargs):
    global whisper_last_used
    with WHISPER_JOB_LOCK:
        _cancel_idle_timer("whisper")
        model = load_whisper_model(name, model_progress, model_load_start)
        transcription_start()
        try:
            return transcribe_with_progress(model, audio_path, transcription_progress, **kwargs)
        finally:
            whisper_last_used = time.monotonic()
            _schedule_idle_unload("whisper")


def unload_whisper_models() -> None:
    global whisper_last_used
    with WHISPER_JOB_LOCK:
        _cancel_idle_timer("whisper")
        if whisper_models:
            loaded = ", ".join(whisper_models)
            print(f"[transcriber] Unloading Whisper model(s): {loaded}")
            whisper_models.clear()
            gc.collect()
        whisper_last_used = 0.0


def unload_diarization_pipeline() -> None:
    global diarization_pipeline, diarization_last_used
    with DIARIZATION_JOB_LOCK:
        _cancel_idle_timer("diarization")
        if diarization_pipeline is not None:
            print("[transcriber] Unloading pyannote diarization pipeline")
            diarization_pipeline = None
            gc.collect()
        diarization_last_used = 0.0


def _cancel_idle_timer(kind: str) -> None:
    global whisper_idle_timer, diarization_idle_timer
    timer = whisper_idle_timer if kind == "whisper" else diarization_idle_timer
    if timer is not None:
        timer.cancel()
    if kind == "whisper":
        whisper_idle_timer = None
    else:
        diarization_idle_timer = None


def _schedule_idle_unload(kind: str, delay: float = MODEL_IDLE_SECONDS) -> None:
    global whisper_idle_timer, diarization_idle_timer
    _cancel_idle_timer(kind)
    timer = threading.Timer(delay, _unload_if_idle, args=(kind,))
    timer.daemon = True
    if kind == "whisper":
        whisper_idle_timer = timer
    else:
        diarization_idle_timer = timer
    timer.start()


def _unload_if_idle(kind: str) -> None:
    lock = WHISPER_JOB_LOCK if kind == "whisper" else DIARIZATION_JOB_LOCK
    with lock:
        last_used = whisper_last_used if kind == "whisper" else diarization_last_used
        remaining = MODEL_IDLE_SECONDS - (time.monotonic() - last_used)
        if last_used and remaining > 0:
            _schedule_idle_unload(kind, remaining)
            return
        if kind == "whisper":
            unload_whisper_models()
        else:
            unload_diarization_pipeline()


def run_diarization_job(audio_path: str, kwargs: dict, start_cb):
    global diarization_last_used
    with DIARIZATION_JOB_LOCK:
        _cancel_idle_timer("diarization")
        pipeline = load_diarization_pipeline()
        start_cb()
        try:
            return pipeline(audio_path, **kwargs)
        finally:
            diarization_last_used = time.monotonic()
            _schedule_idle_unload("diarization")


def transcribe_with_progress(model, audio_path: str, progress_cb, **kwargs):
    """Expose Whisper's internal audio-frame progress without changing Whisper itself."""
    transcribe_module = importlib.import_module("whisper.transcribe")
    original_tqdm = transcribe_module.tqdm.tqdm

    class ProgressTqdm(original_tqdm):
        def update(self, amount=1):
            result = super().update(amount)
            total = float(self.total or 0)
            if total:
                progress_cb(min(1.0, float(self.n) / total))
            return result

    with WHISPER_PROGRESS_LOCK:
        transcribe_module.tqdm.tqdm = ProgressTqdm
        try:
            return model.transcribe(audio_path, verbose=False, **kwargs)
        finally:
            transcribe_module.tqdm.tqdm = original_tqdm


def load_diarization_pipeline():
    global diarization_pipeline
    if diarization_pipeline is None:
        load_secrets_env()
        print("[transcriber] Loading pyannote/speaker-diarization-3.1...")
        diarization_pipeline = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=os.environ.get("HF_TOKEN")
        )
        diarization_pipeline.to(torch.device(DEVICE))
        print(f"[transcriber] Diarization pipeline loaded on {DEVICE}")
    return diarization_pipeline


def convert_to_wav(input_path: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0


def fmt_srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


async def broadcast_progress(job_id: str, progress: float, stage: str, detail: str = ""):
    job = jobs.get(job_id)
    if job:
        job["progress"] = progress
        job["stage"] = stage
        job["detail"] = detail
        q = job_queues.get(job_id)
        if q:
            try:
                await q.put({
                    "type": "progress",
                    "progress": progress,
                    "stage": stage,
                    "detail": detail,
                })
            except Exception:
                pass


async def broadcast_operation(job_id: str, snapshot: dict):
    """Push a normalised operation snapshot (see montage/progress_tracker.py)."""
    job = jobs.get(job_id)
    if job:
        job["operation"] = snapshot
        # Persist only meaningful transitions (status / active stage / stage
        # status changes) so a server restart can restore the last known state
        # without writing to disk on every progress tick (spec 4.3).
        signature = _operation_signature(snapshot)
        if _OPERATION_SAVE_SIGNATURES.get(job_id) != signature:
            _OPERATION_SAVE_SIGNATURES[job_id] = signature
            try:
                _save_montage_job(job_id)
            except Exception:
                pass
        q = job_queues.get(job_id)
        if q:
            try:
                await q.put({"type": "operation", "operation": snapshot})
            except Exception:
                pass


def _operation_signature(snapshot: dict) -> tuple:
    stages = snapshot.get("stages") or []
    return (
        snapshot.get("status"),
        snapshot.get("active_stage"),
        tuple((s.get("key"), s.get("status")) for s in stages),
    )


_OPERATION_SAVE_SIGNATURES: dict[str, tuple] = {}


def mark_operation_failed(job_id: str, error: str) -> Optional[dict]:
    """Record a coherent failed operation state for *job_id* (spec 1.2)."""
    job = jobs.get(job_id)
    if not job:
        return None
    snap = failed_snapshot(job.get("operation"), error)
    job["operation"] = snap
    _OPERATION_SAVE_SIGNATURES[job_id] = _operation_signature(snap)
    return snap


async def process_job(job_id: str, params: dict):
    job = jobs[job_id]
    try:
        job["status"] = "processing"
        wav_path = job["wav_path"]
        original_name = job["original_name"]

        # Stage 1: Conversion
        await broadcast_progress(job_id, 5, "conversion", "Конвертация в WAV...")
        wav_output = str(UPLOAD_DIR / f"{job_id}.wav")
        is_wav = wav_path.lower().endswith(".wav")
        if is_wav:
            wav_output = wav_path
            await broadcast_progress(job_id, 15, "conversion", "Файл уже в формате WAV")
        else:
            success = await asyncio.to_thread(convert_to_wav, wav_path, wav_output)
            if not success:
                job["status"] = "error"
                job["error"] = "Ошибка конвертации файла"
                await broadcast_progress(job_id, 0, "error", "Ошибка конвертации")
                _save_transcription_job(job_id)
                return

        await broadcast_progress(job_id, 15, "conversion", "Конвертация завершена")

        # Stage 2: Transcription
        model_name = params.get("model", "small")
        language = params.get("language", "ru")
        mode = params.get("mode", "words")

        loop = asyncio.get_running_loop()

        def model_progress(pct, downloaded, total):
            mb = downloaded / (1 << 20)
            total_mb = total / (1 << 20)
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(
                    job_id, 20 + pct * 0.28, "model",
                    f"Скачивание модели {model_name}: {mb:.0f} / {total_mb:.0f} MB ({pct}%)"
                ),
                loop
            )

        def model_load_start():
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(
                    job_id, 48, "model",
                    f"Загрузка модели {model_name} в память..."
                ),
                loop
            )

        await broadcast_progress(job_id, 20, "model", f"Проверка модели {model_name}...")
        transcription_span = 12 if params.get("diarization", False) else 38

        def transcription_start():
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(job_id, 50, "transcription", "Транскрибация аудио..."),
                loop,
            )

        def transcription_progress(ratio: float):
            pct = 50 + ratio * transcription_span
            asyncio.run_coroutine_threadsafe(
                broadcast_progress(job_id, pct, "transcription", f"Транскрибация: {ratio * 100:.1f}%"),
                loop,
            )

        result = await asyncio.to_thread(
            run_whisper_job, model_name, wav_output, model_progress, model_load_start,
            transcription_start, transcription_progress,
            language=language, word_timestamps=True
        )
        segments = result["segments"]

        transcription_done = 63 if params.get("diarization", False) else 89
        await broadcast_progress(job_id, transcription_done, "transcription", f"Транскрибация: {len(segments)} сегментов")

        # Stage 3: Diarization
        do_diarization = params.get("diarization", False)
        num_speakers = params.get("num_speakers", None)
        diarization = None

        if do_diarization:
            await broadcast_progress(job_id, 65, "diarization", "Загрузка модели диаризации...")
            diarization_kwargs = {}
            if num_speakers:
                diarization_kwargs["num_speakers"] = int(num_speakers)

            def diarization_start():
                asyncio.run_coroutine_threadsafe(
                    broadcast_progress(job_id, 70, "diarization", "Диаризация..."),
                    loop,
                )

            diarization = await asyncio.to_thread(
                run_diarization_job, wav_output, diarization_kwargs, diarization_start
            )
            await broadcast_progress(job_id, 85, "diarization", "Диаризация завершена")

        # Stage 4: Merge
        await broadcast_progress(job_id, 90, "merge", "Объединение результатов...")

        merged = await asyncio.to_thread(merge_results, segments, diarization, mode)
        job["result"] = merged
        job["status"] = "done"
        job["progress"] = 100

        await broadcast_progress(job_id, 100, "done", "Готово!")
        _save_transcription_job(job_id)
        if await register_video_project(job_id):
            await broadcast_progress(job_id, 100, "done",
                                     "Готово! Ролик добавлен в проекты (Автомонтаж).")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
        await broadcast_progress(job_id, 0, "error", str(e))
        _save_transcription_job(job_id)


def merge_results(segments, diarization, mode):
    result_segments = []

    if diarization is not None:
        for seg in segments:
            start, end = seg["start"], seg["end"]

            if mode == "words" and "words" in seg:
                word_groups = []
                current_group = {"words": [], "word_timestamps": [], "speaker": None, "start": start, "end": start}

                for word_info in seg["words"]:
                    w_start = word_info["start"]
                    w_end = word_info["end"]
                    w_text = word_info["word"]

                    speaker = find_speaker(w_start, w_end, diarization)

                    if speaker != current_group["speaker"] and current_group["words"]:
                        current_group["end"] = w_start
                        current_group["text"] = " ".join(current_group["words"])
                        word_groups.append(current_group)
                        current_group = {"words": [], "word_timestamps": [], "speaker": speaker, "start": w_start, "end": w_start}

                    current_group["words"].append(w_text.strip())
                    current_group["word_timestamps"].append(word_info)
                    current_group["end"] = w_end

                if current_group["words"]:
                    current_group["text"] = " ".join(current_group["words"])
                    word_groups.append(current_group)

                result_segments.extend(word_groups)
            else:
                speaker = find_speaker(start, end, diarization)
                result_segments.append({
                    "start": start,
                    "end": end,
                    "text": seg["text"].strip(),
                    "speaker": speaker,
                    "words": seg.get("words") or [],
                })
    else:
        for seg in segments:
            if mode == "words" and "words" in seg:
                for wi in seg["words"]:
                    result_segments.append({
                        "start": wi["start"],
                        "end": wi["end"],
                        "text": wi["word"].strip(),
                        "speaker": None,
                    })
            else:
                result_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                    "speaker": None,
                    "words": seg.get("words") or [],
                })

    return result_segments


def find_speaker(start, end, diarization):
    best_speaker = "SPEAKER_00"
    best_overlap = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        overlap_start = max(start, turn.start)
        overlap_end = min(end, turn.end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


def export_srt(segments, show_speakers=True, show_timecodes=True):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        if show_timecodes:
            lines.append(f"{fmt_srt_time(seg['start'])} --> {fmt_srt_time(seg['end'])}")
        speaker = f"[{seg['speaker']}] " if show_speakers and seg.get('speaker') else ""
        lines.append(f"{speaker}{seg['text']}")
        lines.append("")
    return "\n".join(lines)


def export_vtt(segments, show_speakers=True, show_timecodes=True):
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        if show_timecodes:
            lines.append(f"{fmt_vtt_time(seg['start'])} --> {fmt_vtt_time(seg['end'])}")
        speaker = f"[{seg['speaker']}] " if show_speakers and seg.get('speaker') else ""
        lines.append(f"{speaker}{seg['text']}")
        lines.append("")
    return "\n".join(lines)


def export_txt(segments, show_speakers=True):
    lines = []
    for seg in segments:
        speaker = f"[{seg['speaker']}] " if show_speakers and seg.get('speaker') else ""
        lines.append(f"{speaker}{seg['text']}")
    return "\n".join(lines)


def export_dialogue(segments, show_timecodes=True):
    lines = []
    current_speaker = None
    for seg in segments:
        sp = seg.get('speaker')
        if sp and sp != current_speaker:
            current_speaker = sp
            lines.append("")
            lines.append(f"═══ {sp} ═══")
        tc = f"[{fmt_srt_time(seg['start'])}] " if show_timecodes else ""
        lines.append(f"{tc}{seg['text']}")
    return "\n".join(lines).strip()


def export_json(segments, show_timecodes=True):
    data = []
    for seg in segments:
        entry = {"text": seg["text"]}
        if seg.get("speaker"):
            entry["speaker"] = seg["speaker"]
        if show_timecodes:
            entry["start"] = seg["start"]
            entry["end"] = seg["end"]
        data.append(entry)
    return json.dumps(data, ensure_ascii=False, indent=2)


def export_tsv(segments, show_speakers=True, show_timecodes=True):
    lines = []
    headers = []
    if show_timecodes:
        headers.extend(["start", "end"])
    if show_speakers:
        headers.append("speaker")
    headers.append("text")
    lines.append("\t".join(headers))

    for seg in segments:
        row = []
        if show_timecodes:
            row.extend([str(seg['start']), str(seg['end'])])
        if show_speakers:
            row.append(seg.get('speaker') or '')
        row.append(seg['text'])
        lines.append("\t".join(row))
    return "\n".join(lines)


# ========== ROUTES ==========

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/manifest.json")
async def manifest():
    return FileResponse(path=str(BASE_DIR / "static" / "manifest.json"), media_type="application/manifest+json")


@app.get("/service-worker.js")
async def service_worker():
    return FileResponse(
        path=str(BASE_DIR / "static" / "sw.js"), media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "file").suffix or ".wav"
    file_path = UPLOAD_DIR / f"{job_id}{ext}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {
        "id": job_id,
        "original_name": file.filename,
        "wav_path": str(file_path),
        "upload_path": str(file_path),
        "status": "uploaded",
        "progress": 0,
        "stage": "",
        "detail": "",
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
    }
    _save_transcription_job(job_id)

    return {"job_id": job_id, "filename": file.filename}


async def register_video_project(job_id: str) -> Optional[str]:
    job = jobs.get(job_id)
    if not job:
        return None
    name = str(job.get("original_name") or "")
    ext = Path(name).suffix.lower()
    src = job.get("upload_path") or job.get("wav_path") or ""
    if ext not in VIDEO_EXTS or not src or not Path(src).exists():
        return None
    if job.get("status") != "done" or not job.get("result"):
        return None
    pid = "p_" + job_id
    async with PROJECTS_LOCK:
        PROJECTS[pid] = {
            "id": pid,
            "name": name,
            "video_path": str(Path(src).resolve()),
            "transcript_job_id": job_id,
            "status": "ready",
            "created_at": job.get("created_at"),
        }
        _save_projects()
    return pid


# ---------------- Thumbs & Video preview ----------------

def _video_response(path: Path, request: Request, media_type: str):
    total = path.stat().st_size
    range_header = request.headers.get("range", "")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600"}
    if not range_header:
        return FileResponse(str(path), media_type=media_type, filename=path.name, headers=headers)

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match or (not match.group(1) and not match.group(2)):
        raise HTTPException(416, "Invalid byte range", headers={"Content-Range": f"bytes */{total}"})
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else total - 1
    else:
        suffix_size = int(match.group(2))
        start = max(0, total - suffix_size)
        end = total - 1
    if start >= total or start > end:
        raise HTTPException(416, "Invalid byte range", headers={"Content-Range": f"bytes */{total}"})
    end = min(end, total - 1)
    length = end - start + 1

    def stream():
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining > 0:
                chunk = source.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers.update({
        "Content-Range": f"bytes {start}-{end}/{total}",
        "Content-Length": str(length),
        "Content-Disposition": f'inline; filename="{path.name}"',
    })
    return StreamingResponse(stream(), status_code=206, media_type=media_type, headers=headers)

def _thumb_path(job_id: str) -> Path:
    return UPLOAD_DIR / f"{job_id}.thumb.jpg"


def _generate_thumb(video_path: Path, out_path: Path) -> bool:
    try:
        if not video_path.exists() or out_path.exists():
            return out_path.exists()
        subprocess.run(
            [shutil.which("ffmpeg") or "ffmpeg", "-y", "-i", str(video_path),
             "-vframes", "1", "-vf", "scale=405:-1", "-q:v", "3",
             str(out_path)],
            capture_output=True, timeout=30,
        )
        return out_path.exists() and out_path.stat().st_size > 100
    except Exception:
        return False


@app.get("/api/thumb/{job_id}")
async def serve_thumb(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    video_src = job.get("upload_path") or job.get("video_path") or ""
    if not video_src or not Path(video_src).exists():
        raise HTTPException(404, "Video not available")
    thumb = _thumb_path(job_id)
    if not thumb.exists():
        ok = await asyncio.to_thread(_generate_thumb, Path(video_src), thumb)
        if not ok:
            raise HTTPException(500, "Thumb generation failed")
    return FileResponse(str(thumb), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


@app.get("/api/video/{job_id}")
async def serve_video(job_id: str, request: Request):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    video_src = job.get("upload_path") or job.get("video_path") or ""
    video = Path(video_src)
    if not video.exists():
        raise HTTPException(404, "Video not available")
    media = {
        ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
        ".webm": "video/webm", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    }.get(video.suffix.lower(), "video/mp4")
    return _video_response(video, request, media)


@app.post("/api/projects/link")
async def link_transcript_to_project(
    transcript_job_id: str = Form(...),
):
    job = jobs.get(transcript_job_id)
    if not job:
        raise HTTPException(404, "Задание транскрибации не найдено")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(400, "Транскрибация ещё не завершена")
    async with PROJECTS_LOCK:
        existing = next(
            (p for p in PROJECTS.values()
             if p.get("transcript_job_id") == transcript_job_id), None)
    if existing:
        return {"project_id": existing["id"], "status": "linked"}
    pid = await register_video_project(transcript_job_id)
    if not pid:
        raise HTTPException(400, "Видео недоступно — проект не создан")
    return {"project_id": pid, "status": "created"}


@app.get("/api/projects")
async def list_projects():
    projects = []
    for pid, pr in PROJECTS.items():
        entry = dict(pr)
        trj = jobs.get(pr.get("transcript_job_id")) if pr.get("transcript_job_id") else None
        entry["transcript_status"] = trj.get("status") if trj else "missing"
        result = trj.get("result") if trj else None
        entry["segments"] = len(result) if isinstance(result, list) else 0
        entry["video_ok"] = bool(pr.get("video_path")) and Path(pr["video_path"]).exists()
        projects.append(entry)
    projects.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    return {"projects": projects}


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    async with PROJECTS_LOCK:
        if project_id in PROJECTS:
            del PROJECTS[project_id]
            _save_projects()
    return {"ok": True}


@app.post("/api/transcribe/{job_id}")
async def start_transcription(
    job_id: str,
    model: str = Form("small"),
    language: str = Form("ru"),
    mode: str = Form("words"),
    diarization: bool = Form(False),
    num_speakers: Optional[int] = Form(None),
):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    params = {
        "model": model,
        "language": language,
        "mode": mode,
        "diarization": diarization,
        "num_speakers": num_speakers,
    }
    jobs[job_id]["params"] = params
    jobs[job_id]["status"] = "processing"
    _save_transcription_job(job_id)

    asyncio.create_task(process_job(job_id, params))
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "progress": job["progress"],
        "stage": job["stage"],
        "detail": job["detail"],
        "error": job.get("error"),
    }


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Job not completed")
    return {"segments": job["result"], "original_name": job["original_name"]}


@app.patch("/api/result/{job_id}")
async def save_result(job_id: str, request: Request):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done" or not isinstance(job.get("result"), list):
        raise HTTPException(400, "Job not completed")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Некорректный JSON")
    texts = payload.get("texts") if isinstance(payload, dict) else None
    if not isinstance(texts, list) or len(texts) != len(job["result"]):
        raise HTTPException(400, "Количество сегментов не совпадает с транскрипцией")
    if any(not isinstance(text, str) for text in texts):
        raise HTTPException(400, "Текст каждого сегмента должен быть строкой")

    normalized = [text.replace("\r\n", "\n").replace("\r", "\n").strip() for text in texts]
    if any("\x00" in text or len(text) > 20_000 for text in normalized):
        raise HTTPException(400, "Один из сегментов содержит недопустимый текст")
    if sum(len(text) for text in normalized) > 1_000_000:
        raise HTTPException(400, "Транскрипция слишком большая")
    if not any(normalized):
        raise HTTPException(400, "Транскрипция не может быть пустой")

    job["result"] = [
        {**segment, "text": text}
        for segment, text in zip(job["result"], normalized)
    ]
    job["transcript_updated_at"] = datetime.now().isoformat()
    _save_transcription_job(job_id)
    return {"segments": job["result"], "saved": True}


@app.get("/api/export/{job_id}/{format_name}")
async def export_file(
    job_id: str,
    format_name: str,
    speakers: bool = True,
    timecodes: bool = True,
):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Job not completed")

    segments = job["result"]
    original = Path(job["original_name"]).stem

    exporters = {
        "srt": lambda: export_srt(segments, speakers, timecodes),
        "vtt": lambda: export_vtt(segments, speakers, timecodes),
        "txt": lambda: export_txt(segments, speakers),
        "json": lambda: export_json(segments, timecodes),
        "tsv": lambda: export_tsv(segments, speakers, timecodes),
        "dialogue": lambda: export_dialogue(segments, timecodes),
    }

    if format_name not in exporters:
        raise HTTPException(400, f"Unknown format: {format_name}")

    content = exporters[format_name]()
    ext = format_name if format_name != "dialogue" else "txt"
    media_type = "text/plain"
    if format_name == "json":
        media_type = "application/json"

    tmp_path = OUTPUT_DIR / f"{job_id}.{ext}"
    tmp_path.write_text(content, encoding="utf-8")

    return FileResponse(
        path=str(tmp_path),
        filename=f"{original}.{ext}",
        media_type=media_type,
    )


# ========== MONTAGE (Автомонтаж) ==========

MONTAGE_WORK = BASE_DIR / "montage_work"
MONTAGE_WORK.mkdir(exist_ok=True)

from montage.secrets_env import load_secrets_env
load_secrets_env()
from montage import engine as montage_engine
from montage import cover_generator
from montage.progress_tracker import failed_snapshot, ProgressTracker, Stage


def _uploaded_video(job_id: str) -> Optional[Path]:
    candidates = [
        path for path in UPLOAD_DIR.glob(f"{job_id}.*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _recover_transcription_projects() -> None:
    """Загружает постоянный каталог и один раз импортирует старые файлы."""
    for path in sorted(TRANSCRIPTION_STATE_DIR.glob("*.json")):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(job, dict) or not job.get("id"):
            continue
        if job.get("status") == "processing":
            job["status"] = "error"
            job["error"] = "Транскрибация была прервана перезапуском сервера"
            job["stage"] = "error"
            job["detail"] = job["error"]
            _write_json_atomic(path, job)
        jobs[str(job["id"])] = job

    montage_sources: dict[str, dict[str, Any]] = {}
    for path in sorted(MONTAGE_WORK.glob("*/job_state.json"), key=lambda item: item.stat().st_mtime):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        transcript_id = str(state.get("transcript_job_id") or "")
        if transcript_id and state.get("srt_text"):
            montage_sources[transcript_id] = state

    recovered_projects: dict[str, dict[str, Any]] = {}
    video_paths = sorted(
        (path for path in UPLOAD_DIR.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTS),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for video_path in video_paths:
        job_id = video_path.stem
        state = montage_sources.get(job_id, {})
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            segments: list[dict[str, Any]] = []
            srt_text = str(state.get("srt_text") or "")
            output_srt = OUTPUT_DIR / f"{job_id}.srt"
            output_json = OUTPUT_DIR / f"{job_id}.json"
            if not srt_text and output_srt.is_file():
                srt_text = output_srt.read_text(encoding="utf-8", errors="replace")
            if output_json.is_file():
                try:
                    value = json.loads(output_json.read_text(encoding="utf-8"))
                    if isinstance(value, list):
                        segments = value
                except Exception:
                    pass
            created_at = datetime.fromtimestamp(video_path.stat().st_mtime).isoformat()
            original_name = str(state.get("original_name") or video_path.name)
            wav_path = UPLOAD_DIR / f"{job_id}.wav"
            job = {
                "id": job_id,
                "original_name": original_name,
                "wav_path": str(wav_path if wav_path.is_file() else video_path),
                "upload_path": str(video_path.resolve()),
                "status": "done" if segments else "uploaded",
                "progress": 100 if segments else 0,
                "stage": "done" if segments else "",
                "detail": "Восстановлено с диска" if segments else "Транскрипция не найдена",
                "result": segments or None,
                "error": None,
                "created_at": created_at,
                "recovered": True,
            }
            jobs[job_id] = job
            _save_transcription_job(job_id)
        recovered_projects[f"p_{job_id}"] = {
            "id": f"p_{job_id}",
            "name": str(job.get("original_name") or video_path.name),
            "video_path": str(video_path.resolve()),
            "transcript_job_id": job_id,
            "status": "ready" if job.get("status") == "done" else "transcript_missing",
            "created_at": job.get("created_at") or datetime.fromtimestamp(video_path.stat().st_mtime).isoformat(),
            "recovered": True,
        }

    catalog_exists = PROJECTS_PATH.is_file()
    if catalog_exists:
        try:
            payload = json.loads(PROJECTS_PATH.read_text(encoding="utf-8"))
            stored = payload.get("projects") if isinstance(payload, dict) else None
            if isinstance(stored, list):
                for project in stored:
                    if isinstance(project, dict) and project.get("id"):
                        PROJECTS[str(project["id"])] = project
        except Exception:
            catalog_exists = False
    if not catalog_exists:
        PROJECTS.update(recovered_projects)
        _save_projects()

app.mount("/animation-assets", StaticFiles(directory=str(montage_engine.ANIMATIONS_DIR)), name="animation-assets")
app.mount("/font-assets", StaticFiles(directory=str(montage_engine.FONTS_DIR)), name="font-assets")


_RENDER_CLEANUP_FILES = (
    "normalized_source.mp4",
    "mirrored_source.mp4",
    "zoom.mp4",
    "sub_sentences.mp4",
    "sub_words.mp4",
    "sub_phrases.mp4",
    "with_visual_layers.mp4",
    "with_animation.mp4",
    "censored_audio.mp4",
    "face_tracking.json",
    "zoom_timeline_config.json",
    "subtitle_timeline_config.json",
    "subtitle_timeline_config_words.json",
    "subtitle_timeline_config_phrases.json",
    "overlay_timeline_config.json",
)


def _safe_montage_workdir(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(job_id)):
        raise ValueError("Недопустимый идентификатор задания")
    root = MONTAGE_WORK.resolve()
    workdir = MONTAGE_WORK / str(job_id)
    if workdir.is_symlink() or workdir.resolve(strict=False).parent != root:
        raise ValueError("Каталог задания находится вне montage_work")
    return workdir


def _protected_montage_paths(job: dict, workdir: Path) -> set[Path]:
    protected = set()
    result = job.get("result") or {}
    for key in ("result_video", "preview_video"):
        raw = result.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = workdir / path
        protected.add(Path(os.path.abspath(str(path))))
    return protected


def _cleanup_path_is_protected(path: Path, protected: set[Path], recursive: bool) -> bool:
    normalized = Path(os.path.abspath(str(path)))
    for item in protected:
        if item == normalized:
            return True
        if recursive:
            try:
                item.relative_to(normalized)
                return True
            except ValueError:
                pass
    return False


def _remove_cleanup_path(path: Path, recursive: bool) -> None:
    if path.is_symlink():
        path.unlink()
    elif recursive and path.is_dir():
        shutil.rmtree(path)
    elif not recursive and path.is_file():
        path.unlink()


def _write_cleanup_log(workdir: Path, summary: dict) -> None:
    line = json.dumps(
        {"timestamp": datetime.now().isoformat(), **summary},
        ensure_ascii=False,
        default=str,
    )
    try:
        with (workdir / "cleanup.log").open("a", encoding="utf-8") as log:
            log.write(line + "\n")
    except Exception as exc:
        print(f"[montage-cleanup] could not write log for {workdir.name}: {exc}", file=sys.stderr)
    try:
        print(f"[montage-cleanup] {workdir.name}: {line}")
    except Exception:
        pass


def _cleanup_allowed_paths(job_id: str, job: dict, phase: str, candidates: list[tuple[Path, bool]]) -> dict:
    try:
        workdir = _safe_montage_workdir(job_id)
    except Exception as exc:
        summary = {"phase": phase, "removed": [], "protected": [], "errors": [str(exc)]}
        print(f"[montage-cleanup] {job_id}: {json.dumps(summary, ensure_ascii=False)}", file=sys.stderr)
        return summary

    summary = {"phase": phase, "removed": [], "protected": [], "errors": []}
    try:
        protected = _protected_montage_paths(job, workdir)
    except Exception as exc:
        summary["errors"].append(f"метаданные результата: {exc}")
        _write_cleanup_log(workdir, summary)
        return summary
    for path, recursive in candidates:
        try:
            if path.parent != workdir:
                summary["errors"].append(f"{path.name}: путь вне каталога задания")
                continue
            if _cleanup_path_is_protected(path, protected, recursive):
                summary["protected"].append(path.name)
                continue
            if not path.exists() and not path.is_symlink():
                continue
            _remove_cleanup_path(path, recursive)
            if not path.exists() and not path.is_symlink():
                summary["removed"].append(path.name)
        except Exception as exc:
            summary["errors"].append(f"{path.name}: {exc}")
    _write_cleanup_log(workdir, summary)
    return summary


def _cleanup_render_artifacts(job_id: str, job: dict) -> dict:
    # The unified pipeline keeps ``status`` at ``processing`` while the optional
    # mirror / preparation phases run, so render artefacts are cleaned up as soon
    # as the main result exists (status ``processing`` or ``done``).
    if job.get("status") not in ("processing", "done") or not job.get("result"):
        return {"phase": "render", "removed": [], "protected": [], "errors": []}
    try:
        workdir = _safe_montage_workdir(job_id)
    except Exception as exc:
        summary = {"phase": "render", "removed": [], "protected": [], "errors": [str(exc)]}
        print(f"[montage-cleanup] {job_id}: {json.dumps(summary, ensure_ascii=False)}", file=sys.stderr)
        return summary
    try:
        candidates = [(workdir / name, False) for name in _RENDER_CLEANUP_FILES]
        return _cleanup_allowed_paths(job_id, job, "render", candidates)
    except Exception as exc:
        summary = {"phase": "render", "removed": [], "protected": [], "errors": [str(exc)]}
        _write_cleanup_log(workdir, summary)
        return summary


def _cleanup_publish_artifacts(job_id: str, job: dict) -> dict:
    if job.get("status") != "done" or job.get("stage") != "published" or job.get("publish_errors"):
        return {"phase": "publish", "removed": [], "protected": [], "errors": []}
    try:
        workdir = _safe_montage_workdir(job_id)
    except Exception as exc:
        summary = {"phase": "publish", "removed": [], "protected": [], "errors": [str(exc)]}
        print(f"[montage-cleanup] {job_id}: {json.dumps(summary, ensure_ascii=False)}", file=sys.stderr)
        return summary
    try:
        candidates = [
            (workdir / "publish_mirrored", True),
            (workdir / "staging", True),
            (workdir / "youtube_staging", True),
            (workdir / f"{job_id}_tg.mp4", False),
        ]
        return _cleanup_allowed_paths(job_id, job, "publish", candidates)
    except Exception as exc:
        summary = {"phase": "publish", "removed": [], "protected": [], "errors": [str(exc)]}
        _write_cleanup_log(workdir, summary)
        return summary


def _run_cleanup_safely(cleanup, job_id: str, job: dict) -> dict:
    try:
        return cleanup(job_id, job)
    except Exception as exc:
        summary = {
            "phase": "publish" if cleanup is _cleanup_publish_artifacts else "render",
            "removed": [],
            "protected": [],
            "errors": [str(exc)],
        }
        try:
            workdir = _safe_montage_workdir(job_id)
            _write_cleanup_log(workdir, summary)
        except Exception:
            pass
        return summary


def _save_montage_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job or job.get("kind") != "montage":
        return
    workdir = _safe_montage_workdir(job_id)
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "job_state.json"
    temporary = workdir / ".job_state.json.tmp"
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, target)


# --------------------------------------------------------------------------
# Mirrored render variant (task spec section 9).
#
# The "mirrored" publishing variant used to be rendered inside
# ``process_publish``.  Publishing must never render/encode (spec 9.1), so the
# variant is now prepared by the montage pipeline: either together with the
# main render (checkbox "prepare_mirrored") or as a separate montage operation
# triggered from the UI.  The ready file is stored on the job together with a
# signature so a stale variant (produced with different montage parameters) is
# never silently reused (spec 9.5).
# --------------------------------------------------------------------------

_MIRROR_VARIANT_DIR = "variants_mirrored"

_MONTAGE_SIGNATURE_KEYS = (
    "mirror_horizontal", "face_tracking", "autozoom", "subtitles",
    "subtitle_mode", "subtitle_position", "subtitle_x", "subtitle_y",
    "subtitle_scale", "subtitle_box", "subtitle_font",
    "blur_banned_words", "edge_mode", "overlays", "animation",
    "zoom_timeline_override",
)


def _montage_signature(params: dict) -> str:
    relevant = {key: params.get(key) for key in _MONTAGE_SIGNATURE_KEYS}
    blob = json.dumps(relevant, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _mirrored_variant_signature(job: dict) -> str:
    options = dict(job.get("params") or {})
    options["mirror_horizontal"] = not bool(options.get("mirror_horizontal"))
    return _montage_signature(options)


def _mirror_variant_entry(job: dict) -> Optional[dict]:
    entry = (job.get("variants") or {}).get("mirrored")
    return entry if isinstance(entry, dict) else None


def mirror_variant_ready(job: dict) -> bool:
    """True only when a ready, non-stale mirrored file exists on disk."""
    entry = _mirror_variant_entry(job)
    if not entry or not entry.get("ready"):
        return False
    if entry.get("signature") != _mirrored_variant_signature(job):
        return False
    path = entry.get("path")
    return bool(path) and Path(str(path)).is_file()


def _public_mirror_variant(job: dict) -> Optional[dict]:
    state = job.get("mirror_variant")
    if not isinstance(state, dict):
        if mirror_variant_ready(job):
            return {"status": "done", "progress": 100, "detail": "Отражённая версия готова",
                    "error": "", "path": _mirror_variant_entry(job).get("path"), "operation": None}
        return None
    public = dict(state)
    public["ready"] = mirror_variant_ready(job)
    return public


async def _render_mirrored_variant(
    job_id: str, *, report_cb=None, progress_cb=None,
) -> tuple:
    """Render the alternate (mirrored) video for *job_id*.

    Reuses the montage pipeline with the mirror flag toggled; overlays are
    re-staged next to the alternate workdir so the render stays self-contained.
    Returns ``(path, alternate_options)``.
    """
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    workdir = MONTAGE_WORK / job_id
    alternate_options = dict(job.get("params") or {})
    alternate_options.pop("prepare_mirrored", None)
    alternate_options["mirror_horizontal"] = not bool(alternate_options.get("mirror_horizontal"))
    if job.get("audio_censor_srt"):
        alternate_options["audio_censor_srt"] = job["audio_censor_srt"]
    alternate_workdir = workdir / _MIRROR_VARIANT_DIR
    alternate_workdir.mkdir(parents=True, exist_ok=True)
    alternate_overlays = []
    overlay_root = (workdir / "overlays").resolve()
    for index, overlay in enumerate(alternate_options.get("overlays") or []):
        staged_overlay = dict(overlay)
        if str(staged_overlay.get("kind") or "") == "image":
            source_image = Path(str(staged_overlay.get("image_path") or "")).resolve()
            try:
                source_image.relative_to(overlay_root)
            except ValueError:
                raise RuntimeError("Файл наложения находится вне каталога задания")
            if not source_image.is_file():
                raise RuntimeError(f"Изображение наложения не найдено: {source_image.name}")
            staged_dir = alternate_workdir / "overlays"
            staged_dir.mkdir(parents=True, exist_ok=True)
            staged_image = staged_dir / f"overlay_{index + 1:02d}{source_image.suffix.lower()}"
            shutil.copy2(source_image, staged_image)
            staged_overlay["image_path"] = str(staged_image)
        alternate_overlays.append(staged_overlay)
    alternate_options["overlays"] = alternate_overlays

    def alt_progress(percent: float, stage: str, detail: str) -> None:
        if progress_cb is not None:
            progress_cb(percent, stage, detail)

    meta = await asyncio.to_thread(
        montage_engine.render_montage,
        job_id=f"{job_id}_mirror",
        workdir=alternate_workdir,
        source_video=Path(job["video_path"]),
        srt_text=job["srt_text"],
        options=alternate_options,
        progress_cb=alt_progress,
        report_cb=report_cb,
        log_cb=lambda message: asyncio.run_coroutine_threadsafe(
            _montage_announce(job_id, f"Отражённая версия: {message}"), loop
        ),
    )
    return Path(meta["result_video"]), alternate_options


async def process_mirror_variant(job_id: str) -> None:
    """Prepare the mirrored publish variant as its own montage operation."""
    job = jobs.get(job_id)
    if not job or job.get("kind") != "montage":
        return
    loop = asyncio.get_running_loop()
    signature = _mirrored_variant_signature(job)
    if mirror_variant_ready(job):
        previous = dict(job.get("mirror_variant") or {})
        previous.update(
            status="done", progress=100, detail="Отражённая версия готова",
            error="", path=_mirror_variant_entry(job).get("path"), signature=signature,
        )
        job["mirror_variant"] = previous
        _save_montage_job(job_id)
        return

    job["mirror_variant"] = {
        "status": "generating", "progress": 0,
        "detail": "Подготовка отражённой версии", "error": "",
        "path": None, "signature": signature, "operation": None,
    }
    _save_montage_job(job_id)
    await _montage_announce(job_id, "Подготовка отражённой версии...")

    def report(snapshot: dict) -> None:
        def store() -> None:
            state = dict(job.get("mirror_variant") or {})
            state["operation"] = snapshot
            state["progress"] = snapshot.get("progress", state.get("progress", 0))
            state["detail"] = snapshot.get("detail") or state.get("detail") or "Подготовка отражённой версии"
            if snapshot.get("status") in ("failed", "error"):
                state["status"] = "error"
            job["mirror_variant"] = state
        loop.call_soon_threadsafe(store)

    def progress(percent: float, stage: str, detail: str) -> None:
        def store() -> None:
            state = dict(job.get("mirror_variant") or {})
            try:
                value = max(0.0, min(100.0, float(percent)))
            except (TypeError, ValueError):
                return
            state["progress"] = max(float(state.get("progress") or 0), value)
            if detail:
                state["detail"] = detail
            job["mirror_variant"] = state
        loop.call_soon_threadsafe(store)

    try:
        path, options = await _render_mirrored_variant(job_id, report_cb=report, progress_cb=progress)
    except BaseException as exc:
        state = dict(job.get("mirror_variant") or {})
        state.update(status="error", error=str(exc), detail=str(exc))
        state["operation"] = failed_snapshot(state.get("operation"), str(exc))
        job["mirror_variant"] = state
        _save_montage_job(job_id)
        await _montage_announce(job_id, f"Отражённая версия не подготовлена: {exc}")
        return
    variants = dict(job.get("variants") or {})
    variants["mirrored"] = {
        "path": str(path),
        "mirror_horizontal": bool(options.get("mirror_horizontal")),
        "signature": signature,
        "ready": True,
        "created_at": time.time(),
    }
    job["variants"] = variants
    state = dict(job.get("mirror_variant") or {})
    state.update(status="done", progress=100, detail="Отражённая версия готова",
                 error="", path=str(path))
    job["mirror_variant"] = state
    _save_montage_job(job_id)
    await _montage_announce(job_id, f"Отражённая версия готова: {path.name}")


# --------------------------------------------------------------------------
# Pre-publication preparation (task: "publish only sends ready files").
#
# Publishing must never run FFmpeg/PIL — no transcoding, FPS normalisation,
# mirroring or cover conversion.  Every local video/image transform happens
# here: during the montage stage or in the dedicated "prepare for publishing"
# operation.  The ready artefacts are stored on the job
# (``job["publish_prep"]``) together with their purpose, readiness, size/params,
# source signature and validation result.  A prepared file is reused while its
# signature matches the current source and montage parameters, so a single file
# serves several platforms (Instagram trial + feed share one ``instagram`` file).
# --------------------------------------------------------------------------

_PUBLISH_PREP_DIR = "publish_prep"

_TARGET_PURPOSE = {
    "instagram": "instagram",
    "trial": "instagram",
    "threads": "instagram",
    "youtube": "youtube",
    "telegram": "telegram",
}

_PURPOSE_LABELS = {
    "instagram": "Instagram",
    "youtube": "YouTube",
    "telegram": "Telegram",
}

_PREP_VIDEO_LABELS = {
    "instagram": "Подготовка видео для Instagram",
    "youtube": "Подготовка видео для YouTube",
    "telegram": "Подготовка видео для Telegram",
}

_PREP_COVER_PURPOSES = ("instagram", "youtube")

_PUBLISH_TARGET_SPECS = {
    "instagram_trial": ("trial", False, "Instagram пробное"),
    "instagram_trial_mirrored": ("trial", True, "Instagram пробное перевёрнутое"),
    "instagram_feed": ("instagram", False, "Instagram основная лента"),
    "instagram_feed_mirrored": ("instagram", True, "Instagram основная лента перевёрнутое"),
    "youtube": ("youtube", False, "YouTube"),
    "youtube_mirrored": ("youtube", True, "YouTube перевёрнутое"),
}


def _default_prepare_targets(job: dict) -> list:
    """Standard publish variants prepared right after the montage render."""
    targets = [
        {"id": "instagram_feed", "kind": "instagram", "mirrored": False},
        {"id": "youtube", "kind": "youtube", "mirrored": False},
    ]
    if mirror_variant_ready(job):
        targets.append({"id": "instagram_feed_mirrored", "kind": "instagram", "mirrored": True})
        targets.append({"id": "youtube_mirrored", "kind": "youtube", "mirrored": True})
    return targets


def _public_publish_ready(job: dict) -> dict:
    ready = {}
    for target_id, (kind, mirrored, _label) in _PUBLISH_TARGET_SPECS.items():
        purpose = _TARGET_PURPOSE.get(kind)
        ready[target_id] = bool(purpose) and _prepared_video_entry(job, purpose, mirrored) is not None
    return ready


def _prep_key(purpose: str, mirrored: bool) -> str:
    return f"{purpose}:{'mirror' if mirrored else 'main'}"


def _file_identity(path) -> str:
    try:
        stat = Path(path).stat()
        return f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        return str(path)


def _publish_prep_signature(job: dict, key: str, source_path) -> str:
    parts = [key, _file_identity(source_path)]
    if key.endswith(":mirror"):
        parts.append(_mirrored_variant_signature(job))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _publish_prep_entries(job: dict) -> dict:
    entries = job.get("publish_prep")
    if not isinstance(entries, dict):
        entries = {}
        job["publish_prep"] = entries
    return entries


def _prep_source_video(job: dict, mirrored: bool) -> Optional[Path]:
    """The current source video for a variant (main result or mirrored file)."""
    if mirrored:
        if not mirror_variant_ready(job):
            return None
        path = Path(str(_mirror_variant_entry(job).get("path")))
        return path if path.is_file() else None
    result = job.get("result") or {}
    path = result.get("result_video")
    if not path:
        return None
    path = Path(str(path))
    return path if path.is_file() else None


def _prepared_video_entry(job: dict, purpose: str, mirrored: bool) -> Optional[dict]:
    """Return a ready, non-stale prepared video entry for purpose/mirrored."""
    key = _prep_key(purpose, mirrored)
    entry = _publish_prep_entries(job).get(key)
    if not isinstance(entry, dict) or not entry.get("ready"):
        return None
    source = _prep_source_video(job, mirrored)
    if source is None:
        return None
    if entry.get("signature") != _publish_prep_signature(job, key, source):
        return None
    path = entry.get("path")
    if not path or not Path(str(path)).is_file():
        return None
    return entry


def _prepared_cover_entry(job: dict, purpose: str, cover_path) -> Optional[dict]:
    """Return a ready prepared cover for *cover_path* (Instagram/YouTube)."""
    if not cover_path:
        return None
    cover_path = Path(str(cover_path))
    if not cover_path.is_file():
        return None
    key = f"cover:{purpose}"
    entry = _publish_prep_entries(job).get(key)
    if not isinstance(entry, dict) or not entry.get("ready"):
        return None
    if entry.get("signature") != _publish_prep_signature(job, key, cover_path):
        return None
    path = entry.get("path")
    if not path or not Path(str(path)).is_file():
        return None
    return entry


def _render_prepared_video(purpose: str, source, target_dir, stem: str,
                           mirrored: bool, progress) -> Path:
    """Produce the platform-ready video (local transform; run off the event loop)."""
    source = Path(source)
    target_dir = Path(target_dir)
    if purpose == "youtube":
        from montage.youtube_publisher import YouTubePublisher
        publisher = YouTubePublisher(
            montage_engine.INSTAPOSTER_DIR, Path(sys.executable),
            target_dir / "youtube.log",
        )
        output = target_dir / f"{stem}_youtube.mp4"
        publisher.prepare_video(source, output, progress)
        return output
    if purpose == "telegram":
        try:
            duration = float(montage_engine.ffprobe(source).get("format", {}).get("duration") or 0) or 30.0
        except Exception:
            duration = 30.0
        output = target_dir / f"{stem}_tg.mp4"
        montage_engine._transcode_to_fit(source, output, 45 * 1024 * 1024, duration, progress)
        return output
    from montage.instagram_publisher import InstagramPublisher
    publisher = InstagramPublisher(
        montage_engine.INSTAPOSTER_DIR, shutil.which("ffmpeg") or "ffmpeg",
        target_dir / "publish.log", lambda message: None,
    )
    return publisher.prepare_variant(source, target_dir, stem, "normal", progress)


def _render_prepared_cover(purpose: str, cover_path, output) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if purpose == "youtube":
        from montage.youtube_publisher import YouTubePublisher
        publisher = YouTubePublisher(
            montage_engine.INSTAPOSTER_DIR, Path(sys.executable),
            output.parent / "youtube.log",
        )
        publisher.prepare_thumbnail(Path(cover_path), output)
        return output
    from PIL import Image
    with Image.open(cover_path) as image:
        image.convert("RGB").save(output, "JPEG", quality=94, optimize=True)
    return output


def _validate_prepared_video(path) -> dict:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Подготовленный файл не создан или пуст")
    try:
        probe = montage_engine.ffprobe(path)
    except Exception as exc:
        raise RuntimeError(f"Подготовленный файл повреждён: {exc}") from exc
    duration = (probe.get("format") or {}).get("duration")
    return {"size": path.stat().st_size, "duration": duration}


def _preparation_error_message(target: dict) -> str:
    """Human-readable reason a publish target is not locally ready (spec 7)."""
    label = target.get("label") or target.get("id") or ""
    mirrored = bool(target.get("mirrored"))
    suffix = " (отражённое)" if mirrored else ""
    reason = str(target.get("reason") or "")
    if reason == "mirror_not_ready":
        return (
            f"Отражённая версия для «{label}»{suffix} не готова. "
            "Включите «Создать отражённую версию» при монтаже и повторите."
        )
    if reason == "cover_not_ready":
        return f"Обложка для «{label}»{suffix} не подготовлена."
    if reason == "preparation_failed":
        return f"Не удалось подготовить видео для «{label}»{suffix}."
    return f"Видео для «{label}»{suffix} не подготовлено."


def _publish_readiness(job: dict, targets: list) -> tuple[dict, list]:
    """Resolve prepared artefacts for resolved publish targets.

    Returns ``(prepared, missing)`` where *prepared* maps a target id to its
    ready paths (video + optional cover/thumbnail) and *missing* lists targets
    that still need local preparation.  Publishing uses only ready files.
    """
    prepared: dict[str, dict] = {}
    missing: list[dict] = []
    for target in targets:
        purpose = _TARGET_PURPOSE.get(str(target.get("kind") or ""))
        target_id = str(target.get("id") or target.get("kind") or "")
        mirrored = bool(target.get("mirrored"))
        entry = _prepared_video_entry(job, purpose, mirrored) if purpose else None
        if entry is None:
            if mirrored and not mirror_variant_ready(job):
                reason = "mirror_not_ready"
            else:
                reason = "video_not_ready"
            missing.append({
                "target": target_id, "label": target.get("label"),
                "kind": target.get("kind"), "mirrored": mirrored, "reason": reason,
            })
            continue
        item: dict = {"video_path": entry.get("path")}
        cover = target.get("cover_path")
        if cover:
            cover_entry = _prepared_cover_entry(job, purpose, cover)
            if cover_entry is None:
                missing.append({
                    "target": target_id, "label": target.get("label"),
                    "kind": target.get("kind"), "mirrored": mirrored, "reason": "cover_not_ready",
                })
                continue
            if purpose == "youtube":
                item["thumbnail_path"] = cover_entry.get("path")
            else:
                item["cover_path"] = cover_entry.get("path")
        prepared[target_id] = item
    return prepared, missing


async def process_prepare_publish(job_id: str, targets: list, cover_path: Optional[str] = None,
                                  cover_id: str = ""):
    """Prepare (locally) every artefact the given publish targets need.

    Runs the platform-specific video transforms and cover conversions *before*
    publishing, stores the results on the job and records a stage snapshot that
    the three-level progress bar renders.  Reuses already prepared artefacts.
    """
    job = jobs.get(job_id)
    if not job or job.get("kind") != "montage":
        return
    workdir = MONTAGE_WORK / job_id
    prep_dir = workdir / _PUBLISH_PREP_DIR
    prep_dir.mkdir(parents=True, exist_ok=True)

    needs: list[dict] = []
    seen: set = set()
    for target in targets:
        purpose = _TARGET_PURPOSE.get(str(target.get("kind") or ""))
        if not purpose:
            continue
        mirrored = bool(target.get("mirrored"))
        key = _prep_key(purpose, mirrored)
        if key in seen:
            continue
        seen.add(key)
        needs.append({"kind": "video", "key": key, "purpose": purpose, "mirrored": mirrored})
    requested_purposes = {item["purpose"] for item in needs}
    if cover_path:
        for purpose in _PREP_COVER_PURPOSES:
            if purpose not in requested_purposes:
                continue
            key = f"cover:{purpose}"
            if key in seen:
                continue
            seen.add(key)
            needs.append({"kind": "cover", "key": key, "purpose": purpose, "mirrored": False})

    # A mirrored artefact can only be produced from an already-rendered mirror
    # variant.  Its absence is never a reason to render here (publishing and
    # preparation never mirror) — drop those needs so the remaining targets are
    # still prepared; publish reports the missing mirrored targets per target.
    unresolved = [
        item for item in needs
        if item["kind"] == "video" and _prep_source_video(job, item["mirrored"]) is None
    ]
    if unresolved:
        needs = [item for item in needs if item not in unresolved]

    if not needs:
        job["prepare_operation"] = None
        job["prepare_status"] = "done"
        _save_montage_job(job_id)
        return

    stages = []
    for item in needs:
        if item["kind"] == "video":
            label = _PREP_VIDEO_LABELS[item["purpose"]]
            if item["mirrored"]:
                label += " (отражённое)"
            stages.append(Stage(item["key"], label, 3.0))
        else:
            stages.append(Stage(item["key"], f"Подготовка обложки для {_PURPOSE_LABELS[item['purpose']]}", 1.0))

    def on_change(tracker: ProgressTracker) -> None:
        job["prepare_operation"] = tracker.snapshot()

    tracker = ProgressTracker(f"{job_id}:prepare", "publish_prepare", stages,
                              label="Подготовка к публикации", on_change=on_change)
    job["prepare_status"] = "running"
    job["prepare_error"] = None
    job["prepare_operation"] = tracker.snapshot()
    _save_montage_job(job_id)
    await _montage_announce(job_id, "Подготовка файлов к публикации...")

    try:
        for item in needs:
            key = item["key"]
            tracker.start(key, "Подготовка")
            if item["kind"] == "video":
                source = _prep_source_video(job, item["mirrored"])
                if source is None:
                    tracker.skip(key, "Источник ещё не готов")
                    continue
                existing = _prepared_video_entry(job, item["purpose"], item["mirrored"])
                if existing is not None:
                    tracker.complete(key, "Уже подготовлено")
                    continue
                stem = f"{job_id}_{key.replace(':', '_')}"

                def video_progress(percent, detail, _key=key):
                    tracker.update(_key, percent, detail)

                path = await asyncio.to_thread(
                    _render_prepared_video, item["purpose"], source, prep_dir, stem,
                    item["mirrored"], video_progress,
                )
                info = _validate_prepared_video(path)
                _publish_prep_entries(job)[key] = {
                    "path": str(path),
                    "purpose": item["purpose"],
                    "kind": "video",
                    "mirrored": item["mirrored"],
                    "ready": True,
                    "signature": _publish_prep_signature(job, key, source),
                    "size": info["size"],
                    "duration": info.get("duration"),
                    "validation": {"ok": True, "detail": f"{info['size']} байт"},
                    "created_at": time.time(),
                }
                tracker.complete(key, f"Готово: {Path(path).name}")
            else:
                existing = _prepared_cover_entry(job, item["purpose"], cover_path)
                if existing is not None:
                    tracker.complete(key, "Уже подготовлено")
                    continue
                output = prep_dir / f"{job_id}_cover_{item['purpose']}.jpg"
                await asyncio.to_thread(_render_prepared_cover, item["purpose"], cover_path, output)
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError("Не удалось подготовить обложку")
                _publish_prep_entries(job)[key] = {
                    "path": str(output),
                    "purpose": item["purpose"],
                    "kind": "cover",
                    "cover_id": cover_id,
                    "ready": True,
                    "signature": _publish_prep_signature(job, key, cover_path),
                    "size": output.stat().st_size,
                    "created_at": time.time(),
                }
                tracker.complete(key, f"Готово: {output.name}")
        tracker.finish("Файлы для публикации готовы")
        job["prepare_status"] = "done"
        _save_montage_job(job_id)
        await _montage_announce(job_id, "Файлы для публикации готовы")
    except BaseException as exc:
        job["prepare_operation"] = failed_snapshot(tracker.snapshot(), str(exc))
        job["prepare_status"] = "error"
        job["prepare_error"] = str(exc)
        _save_montage_job(job_id)
        await _montage_announce(job_id, f"Подготовка к публикации не удалась: {exc}")


def _recover_publish_outcomes(job: dict) -> None:
    """Classify per-platform publish outcomes after an interrupted publish.

    Serves two purposes (task 2):

    * preserved confirmed successes: a platform already reported as published
      is kept published with its ids/links; it is never silently republished;
    * distinguishing the three outcome kinds used after a restart:
      published / failed / unknown.  A target whose outcome could not be
      determined (started but no terminal event was persisted, or a legacy
      job that predates per-target tracking) stays in an explicit *unknown*
      state that the UI shows as "requires manual check" and that is NOT
      retried without an explicit user decision.
    """
    progress = dict(job.get("publish_progress") or {})
    statuses = dict(job.get("publish_status") or {})
    errors = dict(job.get("publish_errors") or {})
    unknown = dict(job.get("publish_unknown") or {})
    started = set((job.get("publish_started") or {}).keys())
    legacy = not bool(job.get("publish_tracking"))
    confirmed = dict(job.get("publish") or {})
    for target in job.get("last_publish_targets") or []:
        tid = str(target.get("id") or target.get("kind") or "")
        if not tid:
            continue
        item = progress.get(tid) or {}
        state = statuses.get(tid)
        if state == "published" or item.get("status") == "done" or tid in confirmed:
            statuses[tid] = "published"
        elif state == "failed" or item.get("status") == "error":
            statuses[tid] = "failed"
            errors.setdefault(tid, item.get("detail") or "Публикация не выполнена")
        elif state == "unknown" or item.get("status") == "unknown":
            statuses[tid] = "unknown"
            unknown.setdefault(tid, item.get("detail") or "Результат публикации неизвестен")
        elif legacy or tid in started:
            statuses[tid] = "unknown"
            unknown[tid] = (
                "Результат неизвестен: сервер перезапущен во время публикации; "
                "проверьте публикацию вручную"
            )
        else:
            statuses[tid] = "failed"
            errors[tid] = "Публикация не начата до перезапуска сервера — можно повторить"
    for tid, state in statuses.items():
        item = progress.get(tid)
        if item is not None:
            item["status"] = {
                "published": "done", "failed": "error", "unknown": "unknown",
            }.get(state, str(item.get("status") or "unknown"))
            if state == "published":
                item["progress"] = 100
                item["detail"] = item.get("detail") or "Опубликовано"
            elif state == "unknown":
                item["progress"] = None
                item["detail"] = unknown.get(tid) or item.get("detail") or "Результат неизвестен"
            else:
                item["progress"] = 100
                item["detail"] = errors.get(tid) or item.get("detail") or "Публикация не выполнена"
    job["publish_status"] = statuses
    job["publish_errors"] = errors
    job["publish_unknown"] = unknown
    job["publish_progress"] = progress


def _recover_montage_jobs() -> None:
    root = MONTAGE_WORK.resolve()
    workdirs = sorted(
        (
            path for path in MONTAGE_WORK.iterdir()
            if not path.is_symlink() and path.is_dir() and path.resolve().parent == root
        ),
        key=lambda path: path.stat().st_mtime,
    )[-20:]
    for workdir in workdirs:
        state_path = workdir / "job_state.json"
        if state_path.is_file():
            try:
                job = json.loads(state_path.read_text(encoding="utf-8"))
                if job.get("kind") != "montage":
                    continue
                if str(job.get("id") or "") != workdir.name:
                    continue
                recovered_after_interruption = False
                in_flight = job.get("status") in {"processing", "publishing"}
                if in_flight:
                    recovered_after_interruption = True
                    recovered_id = str(job.get("id") or "")
                    if not job.get("selected_cover_id"):
                        prepared_cover = next((workdir / "covers").glob(".*-instagram.jpg"), None)
                        if prepared_cover:
                            job["selected_cover_id"] = prepared_cover.name[1:].split("-instagram.", 1)[0]
                    message = "Монтаж прерван перезапуском сервера"
                    result_path = (job.get("result") or {}).get("result_video")
                    has_result = bool(result_path) and Path(str(result_path)).is_file()
                    if has_result:
                        # The render finished; publication (if any) was interrupted.
                        job["status"] = "done"
                        job["stage"] = "done"
                        job["detail"] = "Публикация прервана перезапуском сервера"
                        operation = job.get("operation")
                        if isinstance(operation, dict) and operation.get("status") in {"running", "pending"}:
                            job["operation"] = failed_snapshot(operation, message)
                        _recover_publish_outcomes(job)
                        _run_cleanup_safely(_cleanup_render_artifacts, recovered_id, job)
                        if not job.get("publish_errors") and not job.get("publish_unknown"):
                            _run_cleanup_safely(_cleanup_publish_artifacts, recovered_id, job)
                    else:
                        # Render was interrupted before a finished video existed: never
                        # advertise it as completed without a real result (task 1).
                        job["status"] = "failed"
                        job["stage"] = "interrupted"
                        job["detail"] = message
                        job["error"] = message
                        try:
                            job["progress"] = min(float(job.get("progress") or 0), 99.0)
                        except (TypeError, ValueError):
                            job["progress"] = 0
                        job["result"] = None
                        operation = job.get("operation")
                        if isinstance(operation, dict) and operation.get("status") != "failed":
                            job["operation"] = failed_snapshot(operation, message)
                if job.get("variants") is None:
                    job["variants"] = {}
                mirror_state = job.get("mirror_variant")
                if isinstance(mirror_state, dict) and mirror_state.get("status") == "generating":
                    mirror_msg = "Подготовка отражённой версии прервана перезапуском сервера"
                    mirror_state = dict(mirror_state)
                    mirror_state.update(status="error", error=mirror_msg, detail=mirror_msg)
                    mirror_state["operation"] = failed_snapshot(mirror_state.get("operation"), mirror_msg)
                    job["mirror_variant"] = mirror_state
                    recovered_after_interruption = True
                recovered_id = str(job["id"])
                jobs[recovered_id] = job
                if recovered_after_interruption:
                    _save_montage_job(recovered_id)
                if not in_flight:
                    _run_cleanup_safely(_cleanup_render_artifacts, recovered_id, job)
                    if job.get("stage") == "published" and not job.get("publish_errors") and not job.get("publish_unknown"):
                        _run_cleanup_safely(_cleanup_publish_artifacts, recovered_id, job)
                continue
            except Exception:
                pass

        # Older jobs predate on-disk state; recover enough metadata to download or republish them.
        result_video = workdir / "final.mp4"
        if not result_video.is_file():
            continue
        sources = list(workdir.glob("project_video.*")) + list(workdir.glob("source.*"))
        transcripts = list(workdir.glob("*_words.srt"))
        if not sources or not transcripts:
            continue
        subtitle_mode = "phrases" if (workdir / "sub_phrases.mp4").is_file() else "words"
        timeline = {"subtitle_mode": subtitle_mode, "subtitle_position": "custom", "zoom": []}
        try:
            zoom = json.loads((workdir / "zoom_timeline_config.json").read_text(encoding="utf-8"))
            timeline["zoom"] = (zoom.get("render") or {}).get("zoom", {}).get("timeline") or []
        except Exception:
            pass
        try:
            duration = float(montage_engine.ffprobe(result_video).get("format", {}).get("duration") or 0)
        except Exception:
            duration = 0
        covers = []
        try:
            covers = json.loads((workdir / "covers" / "manifest.json").read_text(encoding="utf-8")).get("covers") or []
        except Exception:
            pass
        instagram_attempted = any((workdir / "staging").glob("*_normal_*.mp4"))
        prepared_cover = next((workdir / "covers").glob(".*-instagram.jpg"), None)
        selected_cover_id = prepared_cover.name[1:].split("-instagram.", 1)[0] if prepared_cover else ""
        job_id = workdir.name
        jobs[job_id] = {
            "id": job_id,
            "original_name": sources[0].name,
            "video_path": str(sources[0]),
            "srt_text": transcripts[0].read_text(encoding="utf-8", errors="replace"),
            "transcript_job_id": None,
            "project_id": None,
            "kind": "montage",
            "status": "done",
            "progress": 100,
            "stage": "done",
            "detail": "Готовый ролик восстановлен после перезапуска",
            "result": {
                "job_id": job_id,
                "result_video": str(result_video.resolve()),
                "preview_video": str((workdir / "preview.mp4").resolve()),
                "duration": duration,
                "subtitle_mode": subtitle_mode,
                "timeline": timeline,
            },
            "params": {"subtitle_mode": subtitle_mode, "mirror_horizontal": False},
            "publish": {} if instagram_attempted else None,
            "publish_errors": ({"instagram_feed": "Предыдущая попытка Instagram не завершилась; можно повторить"}
                               if instagram_attempted else {}),
            "publish_progress": None,
            "error": None,
            "created_at": datetime.fromtimestamp(workdir.stat().st_mtime).isoformat(),
            "covers": covers,
            "selected_cover_id": selected_cover_id,
            "cover_status": "done" if covers else "idle",
            "cover_progress": 100 if covers else 0,
            "cover_detail": "",
            "cover_error": None,
            "operation": None,
            "variants": {},
            "mirror_variant": None,
            "publish_prep": {},
            "prepare_operation": None,
            "prepare_status": "idle",
            "prepare_error": None,
        }
        _save_montage_job(job_id)
        _run_cleanup_safely(_cleanup_render_artifacts, job_id, jobs[job_id])


_recover_montage_jobs()
_recover_transcription_projects()


def _threaded_progress(loop, job_id):
    def progress(pct, stage, detail=""):
        asyncio.run_coroutine_threadsafe(
            broadcast_progress(job_id, pct, stage, detail), loop
        )
    return progress


async def _montage_announce(job_id: str, text: str):
    q = job_queues.get(job_id)
    if q:
        try:
            await q.put({"type": "log", "message": text})
        except Exception:
            pass


async def _finalize_montage(job_id: str) -> None:
    """Finish the unified montage process (``ready`` / ``ready_with_warnings``).

    The main MP4 and the platform artefacts are already produced by the
    preceding phases.  A failure of an optional phase (mirrored variant or
    platform preparation) never hides the main result: it becomes a warning and
    the job still reaches ``done`` so the main video stays usable (spec 4).
    """
    job = jobs.get(job_id)
    if not job or job.get("kind") != "montage":
        return
    if job.get("status") == "error":
        return
    params = job.get("params") or {}
    warnings = []
    if params.get("prepare_mirrored") and not mirror_variant_ready(job):
        reason = str((job.get("mirror_variant") or {}).get("error") or "").strip()
        warnings.append("Отражённая версия не подготовлена" + (f": {reason}" if reason else ""))
    if params.get("prepare_publish", True) and job.get("prepare_status") == "error":
        reason = str(job.get("prepare_error") or "").strip()
        warnings.append(
            "Подготовка файлов к публикации завершилась с ошибкой" + (f": {reason}" if reason else "")
        )
    job["warnings"] = warnings
    job["phase"] = "ready_with_warnings" if warnings else "ready"
    job["status"] = "done"
    job["progress"] = 100
    job["stage"] = "done"
    job["detail"] = "Готово с предупреждениями" if warnings else "Готово"
    _save_montage_job(job_id)
    await broadcast_progress(job_id, 100, "done", job["detail"])
    if warnings:
        await _montage_announce(job_id, "Готово с предупреждениями: " + "; ".join(warnings))
    else:
        await _montage_announce(job_id, "Монтаж полностью готов")


async def process_montage(job_id: str, params: dict):
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    try:
        job["status"] = "processing"
        progress = _threaded_progress(loop, job_id)

        await _montage_announce(job_id, "Проверка установленных компонентов...")
        status = montage_engine.montage_status()

        video = Path(job["video_path"])
        if not video.exists():
            raise RuntimeError("Видеофайл не найден")

        srt_text: str = job["srt_text"] or ""
        transcript_job = job.get("transcript_job_id")
        if not srt_text and transcript_job and transcript_job in jobs:
            source = jobs[transcript_job]
            if source.get("status") == "done" and source.get("result"):
                srt_text = montage_engine.build_words_srt(source["result"])

        if not srt_text.strip():
            raise RuntimeError("Нет текста: загрузите SRT или укажите готовую транскрибацию")

        censor_srt_text = srt_text
        if params.get("blur_banned_words"):
            source_job = jobs.get(transcript_job) or {}
            exact_word_srt = montage_engine.build_word_timing_srt(source_job.get("result") or [])
            if exact_word_srt.strip():
                censor_srt_text = exact_word_srt

        if params.get("blur_banned_words") and montage_engine.needs_word_alignment(censor_srt_text):
            await _montage_announce(job_id, "Уточнение таймкодов слов для цензуры...")
            source_job = jobs.get(transcript_job) or {}
            source_params = source_job.get("params") or {}
            model_name = str(source_params.get("model") or "small")
            language = str(source_params.get("language") or "ru")

            def alignment_progress(ratio: float) -> None:
                progress(2 + ratio * 4, "audio_censor", f"Таймкоды слов: {ratio * 100:.0f}%")

            aligned = await asyncio.to_thread(
                run_whisper_job,
                model_name,
                str(video),
                lambda *_: None,
                lambda: None,
                lambda: None,
                alignment_progress,
                language=language,
                word_timestamps=True,
            )
            aligned_srt = montage_engine.build_word_timing_srt(aligned.get("segments") or [])
            if not aligned_srt.strip() or montage_engine.needs_word_alignment(aligned_srt):
                raise RuntimeError("Не удалось определить точные таймкоды нежелательных слов")
            censor_srt_text = aligned_srt

        render_params = dict(params)
        if params.get("blur_banned_words"):
            render_params["audio_censor_srt"] = censor_srt_text
            job["audio_censor_srt"] = censor_srt_text
            _save_montage_job(job_id)

        if params.get("face_tracking", True) and not status["face_model"]:
            await broadcast_progress(job_id, 3, "model", "Загрузка модели face_landmarker.task...")
            download_face_model(job_id, progress)
            status = montage_engine.montage_status()

        workdir = MONTAGE_WORK / job_id
        workdir.mkdir(parents=True, exist_ok=True)

        def report(snapshot: dict) -> None:
            asyncio.run_coroutine_threadsafe(broadcast_operation(job_id, snapshot), loop)

        await _montage_announce(job_id, "Запуск рендера...")
        meta = await asyncio.to_thread(
            montage_engine.render_montage,
            job_id=job_id,
            workdir=workdir,
            source_video=video,
            srt_text=srt_text,
            options=render_params,
            progress_cb=progress,
            report_cb=report,
            log_cb=lambda message: asyncio.run_coroutine_threadsafe(
                _montage_announce(job_id, message), loop
            ),
        )

        # The montage operation is not finished here: the mirrored variant and
        # the platform-ready artefacts are produced by the same unified process
        # (see ``_run_montage_then_variant``).  ``status`` stays ``processing``
        # until every automatic phase is done, so the UI never reports "done"
        # prematurely (spec section 4).  The main MP4 is already available.
        job["status"] = "processing"
        job["phase"] = "main_ready"
        job["progress"] = 100
        job["stage"] = "main_ready"
        job["detail"] = "Основной ролик готов"
        job["result"] = meta
        _save_montage_job(job_id)
        _run_cleanup_safely(_cleanup_render_artifacts, job_id, job)
        await broadcast_progress(job_id, 100, "main_ready", "Основной ролик готов")
        await _montage_announce(job_id, f"Основной ролик готов: {Path(meta['result_video']).name} ({(os.path.getsize(meta['result_video']) >> 20)} МБ)")
    except Exception as e:
        job["status"] = "error"
        job["phase"] = "failed"
        job["error"] = str(e)
        job["stage"] = "error"
        job["detail"] = str(e)
        operation = mark_operation_failed(job_id, str(e))
        _save_montage_job(job_id)
        print(f"[montage] job {job_id} failed: {traceback.format_exc()}", file=sys.stderr)
        if operation is not None:
            await broadcast_progress(job_id, operation.get("progress", 0), "error", str(e))
            await broadcast_operation(job_id, operation)
        else:
            await broadcast_progress(job_id, 0, "error", str(e))


async def _store_project_covers(job_id: str) -> None:
    job = jobs[job_id]
    project_ids = [job.get("project_id"), f"p_{job_id}"]
    async with PROJECTS_LOCK:
        for project_id in project_ids:
            if project_id and project_id in PROJECTS:
                PROJECTS[project_id]["covers"] = list(job.get("covers") or [])
                PROJECTS[project_id]["covers_dir"] = str(MONTAGE_WORK / job_id / "covers")
        _save_projects()


async def process_cover_suggestions(job_id: str) -> None:
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    job["cover_status"] = "generating"
    job["cover_progress"] = 0
    job["cover_detail"] = "Подготовка стоп-кадров"
    job["cover_error"] = None

    def progress(percent: float, detail: str) -> None:
        def update() -> None:
            job["cover_progress"] = round(max(0, min(100, percent)), 1)
            job["cover_detail"] = detail
        loop.call_soon_threadsafe(update)

    try:
        items = await asyncio.to_thread(
            cover_generator.generate_suggested_covers,
            job_id,
            Path(job["video_path"]),
            str(job.get("srt_text") or ""),
            MONTAGE_WORK / job_id / "covers",
            progress,
            str(job.get("cover_title") or ""),
        )
        job["covers"] = items
        job["cover_status"] = "done"
        job["cover_progress"] = 100
        job["cover_detail"] = f"Обложки готовы: {len(items)}"
        await _store_project_covers(job_id)
        _save_montage_job(job_id)
    except Exception as exc:
        job["cover_status"] = "error"
        job["cover_error"] = str(exc)
        job["cover_detail"] = str(exc)
        _save_montage_job(job_id)
        print(f"[covers] job {job_id} failed: {traceback.format_exc()}", file=sys.stderr)


async def process_custom_cover(job_id: str, timestamp: float, title: str) -> None:
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    job["cover_status"] = "generating"
    job["cover_progress"] = 0
    job["cover_detail"] = "Создание обложки из стоп-кадра"
    job["cover_error"] = None

    def progress(percent: float, detail: str) -> None:
        def update() -> None:
            job.update(cover_progress=round(percent, 1), cover_detail=detail)
        loop.call_soon_threadsafe(update)

    try:
        video = Path((job.get("result") or {}).get("result_video") or job["video_path"])
        items = await asyncio.to_thread(
            cover_generator.generate_custom_cover,
            video, timestamp, title, MONTAGE_WORK / job_id / "covers", progress,
        )
        job["covers"] = items
        job["cover_status"] = "done"
        job["cover_progress"] = 100
        job["cover_detail"] = "Новая обложка готова"
        await _store_project_covers(job_id)
        _save_montage_job(job_id)
    except Exception as exc:
        job["cover_status"] = "error"
        job["cover_error"] = str(exc)
        job["cover_detail"] = str(exc)
        _save_montage_job(job_id)


async def process_publish(job_id: str, targets: list, publish_lock: Optional[asyncio.Lock] = None):
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    try:
        previous_publish = dict(job.get("publish") or {})
        job["status"] = "publishing"
        job["error"] = None
        job["publish"] = dict(previous_publish)
        job["publish_status"] = dict(job.get("publish_status") or {})
        job["publish_started"] = dict(job.get("publish_started") or {})
        job["publish_unknown"] = dict(job.get("publish_unknown") or {})
        job["publish_errors"] = dict(job.get("publish_errors") or {})
        job["publish_tracking"] = True
        for target in targets:
            target_id = str(target.get("id") or target.get("kind") or "")
            if target_id:
                job["publish_unknown"].pop(target_id, None)
                job["publish_errors"].pop(target_id, None)
        job["publish_progress"] = {
            str(target["id"]): {
                "label": str(target.get("label") or target["id"]),
                "progress": 0,
                "status": "queued",
                "detail": "В очереди",
                "steps": [],
            }
            for target in targets
        }
        job["last_publish_targets"] = targets
        _save_montage_job(job_id)
        progress = _threaded_progress(loop, job_id)
        workdir = MONTAGE_WORK / job_id
        source_video = Path(job["result"]["result_video"])
        mirrored_source_video = None

        # Publishing never renders/encodes (spec 9.1).  The mirrored variant must
        # already exist on the job (prepared by the montage pipeline).  If it is
        # missing/stale we fail only the mirrored targets with a clear reason
        # instead of re-rendering here.
        mirrored_targets = [target for target in targets if target.get("mirrored")]
        if mirrored_targets:
            if mirror_variant_ready(job):
                mirrored_source_video = Path(_mirror_variant_entry(job)["path"])
                for target in mirrored_targets:
                    job["publish_progress"][target["id"]].update(
                        status="preparing", detail="Отражённая версия готова"
                    )
            else:
                message = (
                    "Отражённая версия не готова. Включите «Создать отражённую версию» "
                    "при монтаже и повторите."
                )
                for target in mirrored_targets:
                    target["preparation_error"] = message
                await _montage_announce(job_id, message)

        # Publishing only sends ready files: resolve the artefacts produced by
        # the preparation stage.  Anything missing is reported per target
        # instead of transcoding here (spec: publish must not run FFmpeg).
        prepared_targets, missing_targets = _publish_readiness(job, targets)
        for item in missing_targets:
            target_id = item.get("target")
            if target_id and target_id in job["publish_progress"]:
                job["publish_progress"][target_id].update(
                    status="error", progress=0,
                    detail=_preparation_error_message(item),
                )
        for target in targets:
            target_id = str(target.get("id") or target.get("kind") or "")
            if target_id and target_id not in prepared_targets:
                target["preparation_error"] = _preparation_error_message(target)
        if missing_targets:
            labels = ", ".join(str(item.get("label") or item.get("target")) for item in missing_targets)
            await _montage_announce(job_id, f"Не подготовлены к публикации: {labels}")

        await _montage_announce(job_id, "Публикация...")

        def publish_progress(percent: float, stage: str, detail: str) -> None:
            progress(percent, stage, detail)

        def target_progress(target_id: str, percent: float, detail: str, status: str) -> None:
            def update() -> None:
                item = job["publish_progress"].get(target_id)
                if item is None:
                    return
                previous = item.get("status")
                item.update(
                    progress=None if status == "unknown" else round(percent, 1),
                    detail=detail, status=status,
                )
                if status == "done":
                    job["publish_status"][target_id] = "published"
                    job["publish_errors"].pop(target_id, None)
                    job["publish_unknown"].pop(target_id, None)
                elif status == "error":
                    job["publish_status"][target_id] = "failed"
                    job["publish_errors"][target_id] = detail
                    job["publish_unknown"].pop(target_id, None)
                elif status == "unknown":
                    job["publish_status"][target_id] = "unknown"
                    job["publish_unknown"][target_id] = detail
                    job["publish_errors"].pop(target_id, None)
                started = (status == "publishing" and previous == "queued")
                if started:
                    job["publish_started"][target_id] = True
                if status in {"done", "error", "unknown"} or started:
                    _save_montage_job(job_id)
            loop.call_soon_threadsafe(update)

        def confirmed_publish(target_id: str, result_data: dict) -> None:
            def update() -> None:
                job["publish_status"][target_id] = "published"
                job["publish"][target_id] = dict(result_data)
                job["publish_errors"].pop(target_id, None)
                job["publish_unknown"].pop(target_id, None)
                _save_montage_job(job_id)
            loop.call_soon_threadsafe(update)

        def target_step(target_id: str, step_key: str, step_label: str, status: str, detail: str) -> None:
            def update() -> None:
                item = job["publish_progress"].get(target_id)
                if item is None:
                    return
                steps = list(item.get("steps") or [])
                entry = next((s for s in steps if s.get("key") == step_key), None)
                if entry is None:
                    entry = {"key": step_key, "label": step_label, "status": status, "detail": detail or ""}
                    steps.append(entry)
                else:
                    entry["status"] = status
                    entry["label"] = step_label or entry.get("label")
                    if detail:
                        entry["detail"] = detail
                item["steps"] = steps
                _save_montage_job(job_id)
            loop.call_soon_threadsafe(update)

        result = await asyncio.to_thread(
            montage_engine.publish_job,
            job_id=job_id,
            workdir=workdir,
            source_video=source_video,
            mirrored_source_video=mirrored_source_video,
            original_video=Path(job["video_path"]),
            transcript_text=str(job.get("srt_text") or ""),
            targets=targets,
            prepared_targets=prepared_targets,
            progress_cb=publish_progress,
            target_progress_cb=target_progress,
            confirmed_cb=confirmed_publish,
            target_step_cb=target_step,
            log_cb=lambda message: asyncio.run_coroutine_threadsafe(
                _montage_announce(job_id, message), loop
            ),
        )
        publish_results = dict(result.get("results") or {})
        engine_errors = dict(result.get("errors") or {})
        expected_targets = {str(target.get("id") or target.get("kind") or "") for target in targets}
        for missing in expected_targets - set(publish_results) - set(engine_errors):
            if missing:
                engine_errors[missing] = "Публикация не вернула результат"
        for target_id, error_text in engine_errors.items():
            if str(error_text).startswith("AMBIGUOUS_MEDIA_PUBLISH"):
                job["publish_status"][target_id] = "unknown"
                job["publish_unknown"][target_id] = error_text
                job["publish_errors"].pop(target_id, None)
            else:
                job["publish_status"][target_id] = "failed"
                job["publish_errors"][target_id] = error_text
        for target_id in publish_results:
            job["publish_status"][target_id] = "published"
            job["publish_errors"].pop(target_id, None)
            job["publish_unknown"].pop(target_id, None)
        for target_id in expected_targets:
            if job["publish_status"].get(target_id) in {None, "queued", "preparing", "publishing"}:
                job["publish_status"][target_id] = "failed"
                job["publish_errors"].setdefault(target_id, "Публикация не вернула результат")
        terminal_status = {"published": "done", "failed": "error", "unknown": "unknown"}
        for target_id in expected_targets:
            item = job["publish_progress"].get(target_id)
            if item is None:
                continue
            state = job["publish_status"].get(target_id) or "failed"
            item["status"] = terminal_status.get(state, "error")
            if state == "published":
                item.update(progress=100, detail="Опубликовано")
            elif state == "unknown":
                item.update(progress=None, detail=job["publish_unknown"].get(target_id) or "Результат публикации неизвестен")
            else:
                item.update(progress=item.get("progress"), detail=item.get("detail")
                            or job["publish_errors"].get(target_id) or "Публикация не выполнена")
        current_errors = {k: v for k, v in job["publish_errors"].items() if k in expected_targets}
        current_unknown = {k: v for k, v in job["publish_unknown"].items() if k in expected_targets}
        job["status"] = "done"
        job["progress"] = 100
        job["stage"] = "published"
        job["publish"] = {**previous_publish, **publish_results}
        if current_errors and current_unknown:
            detail = "Публикация завершена с ошибками; часть результатов требует ручной проверки"
        elif current_unknown:
            detail = "Публикация завершена; часть результатов требует ручной проверки"
        elif current_errors:
            detail = "Публикация завершена с ошибками"
        else:
            detail = "Публикация завершена"
        job["detail"] = detail
        _save_montage_job(job_id)
        if not current_errors and not current_unknown:
            _run_cleanup_safely(_cleanup_publish_artifacts, job_id, job)
        await broadcast_progress(job_id, 100, "published", detail)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        if job.get("last_publish_targets"):
            _recover_publish_outcomes(job)
        _save_montage_job(job_id)
        await broadcast_progress(job_id, 0, "error", str(e))
    finally:
        # Release the per-job lock on success, on error and on cancellation so
        # a retry is never blocked by a stale lock.
        if publish_lock is not None and publish_lock.locked():
            publish_lock.release()


async def _prepare_then_publish(job_id: str, targets: list, publish_lock,
                                cover_path: Optional[str] = None, cover_id: str = "",
                                prepare_ids: Optional[set] = None) -> None:
    """Auto-prepare missing local artefacts, then publish (spec 6).

    Runs in the background so the HTTP request returns immediately.  Preparation
    failures never stop publishing: ``process_publish`` reports the still-missing
    targets per target with a clear reason.  The lock is released here if the
    preparation step is cancelled before ``process_publish`` takes over.
    """
    try:
        if prepare_ids:
            prep_targets = [t for t in targets if t.get("id") in prepare_ids]
            await _montage_announce(job_id, "Подготовка файлов к публикации перед отправкой...")
            try:
                await process_prepare_publish(
                    job_id, prep_targets, cover_path=cover_path, cover_id=cover_id
                )
            except Exception as exc:
                job = jobs.get(job_id)
                if job is not None:
                    job["prepare_status"] = "error"
                    job["prepare_error"] = str(exc)
                    _save_montage_job(job_id)
                await _montage_announce(job_id, f"Не удалось подготовить файлы: {exc}")
    except BaseException:
        if publish_lock is not None and publish_lock.locked():
            publish_lock.release()
        raise
    await process_publish(job_id, targets, publish_lock=publish_lock)


def download_face_model(job_id: str, progress):
    model = montage_engine.MODELS_DIR / "face_landmarker.task"
    if model.exists():
        return
    montage_engine.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    part = str(model) + ".part"
    start = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "ab") as out:
        total = start + int(resp.headers.get("Content-Length") or 0)
        downloaded = start
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
                progress(pct * 0.03, "model", f"Скачивание модели: {pct}%")
    os.replace(part, str(model))


@app.get("/api/montage/status")
async def montage_status():
    load_secrets_env()
    status = montage_engine.montage_status()
    entries = [
        {"id": j["id"], "name": j["original_name"], "status": j["status"],
         "phase": j.get("phase"), "warnings": list(j.get("warnings") or []),
         "created_at": j.get("created_at", "")}
        for j in jobs.values() if j.get("kind") == "montage"
    ]
    entries.sort(key=lambda item: item["created_at"])
    return {**status, "jobs": entries}


@app.post("/api/montage/fonts/download")
async def download_comfortaa_font():
    target = montage_engine.FONTS_DIR / "Comfortaa.ttf"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return {"font": {"id": target.name, "name": "Comfortaa", "path": str(target)}}
    try:
        await asyncio.to_thread(
            urllib.request.urlretrieve,
            "https://github.com/google/fonts/raw/main/ofl/comfortaa/Comfortaa%5Bwght%5D.ttf",
            target,
        )
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"Не удалось скачать Comfortaa: {exc}")
    return {"font": {"id": target.name, "name": "Comfortaa", "path": str(target)}}


@app.post("/api/montage/render")
async def start_montage(
    video: Optional[UploadFile] = File(None),
    srt: Optional[UploadFile] = File(None),
    overlay_files: list[UploadFile] = File(default=[]),
    transcript_job_id: Optional[str] = Form(None),
    project_id: Optional[str] = Form(None),
    subtitle_mode: str = Form("words"),
    subtitle_position: str = Form("custom"),
    subtitle_x: Optional[float] = Form(None),
    subtitle_y: Optional[float] = Form(None),
    subtitle_scale: Optional[float] = Form(None),
    subtitle_box: Optional[bool] = Form(False),
    subtitle_font: Optional[str] = Form(None),
    face_tracking: bool = Form(False),
    autozoom: bool = Form(False),
    blur_banned_words: bool = Form(False),
    subtitles: bool = Form(True),
    bigpickle: bool = Form(True),
    edge_mode: str = Form("scale"),
    animation_id: str = Form(""),
    animation_start: float = Form(0.0),
    animation_offset_x: int = Form(0),
    animation_item_size: int = Form(72),
    animation_bar: bool = Form(False),
    overlays_json: Optional[str] = Form(None),
    propose_cover: bool = Form(False),
    cover_title: str = Form(""),
    zoom_timeline_json: Optional[str] = Form(None),
    prepare_mirrored: bool = Form(False),
    prepare_publish: bool = Form(True),
):
    job_id = str(uuid.uuid4())[:8]
    work_dir = MONTAGE_WORK / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    video_path: Optional[Path] = None
    srt_text = ""
    src_job_id: Optional[str] = None
    existing_covers: list[dict] = []

    if project_id:
        async with PROJECTS_LOCK:
            project = PROJECTS.get(project_id)
        if not project:
            raise HTTPException(404, "Проект не найден")
        if not project.get("video_path") or not Path(project["video_path"]).exists():
            raise HTTPException(400, "Видео проекта недоступно")
        video_path = work_dir / f"project_video{Path(project['video_path']).suffix or '.mp4'}"
        shutil.copy2(project["video_path"], video_path)
        project_covers_dir = Path(str(project.get("covers_dir") or ""))
        if project_covers_dir.is_dir() and project.get("covers"):
            shutil.copytree(project_covers_dir, work_dir / "covers", dirs_exist_ok=True)
            existing_covers = list(project.get("covers") or [])
        src_job_id = project.get("transcript_job_id")
        if src_job_id and jobs.get(src_job_id) and jobs[src_job_id].get("status") == "done" \
                and isinstance(jobs[src_job_id].get("result"), list):
            srt_text = montage_engine.build_words_srt(jobs[src_job_id]["result"])
    else:
        if video is None:
            raise HTTPException(400, "Загрузите видео или выберите проект")
        ext = Path(video.filename or "video.mp4").suffix or ".mp4"
        video_path = work_dir / f"source{ext}"
        content = await video.read()
        video_path.write_bytes(content)

    if video is not None and video.filename:
        original_name = video.filename
    elif project_id:
        original_name = PROJECTS.get(project_id, {}).get("name", "Проект")
    else:
        original_name = "video.mp4"

    if srt is not None and srt.filename:
        srt_text = (await srt.read()).decode("utf-8", errors="replace")

    if not srt_text and transcript_job_id:
        source_job = jobs.get(transcript_job_id)
        if not source_job or source_job.get("status") != "done" \
                or not isinstance(source_job.get("result"), list):
            raise HTTPException(400, "Сохранённая транскрибация недоступна")
        src_job_id = transcript_job_id
        srt_text = montage_engine.build_words_srt(source_job["result"])

    if not srt_text:
        raise HTTPException(400, "Нет текста для субтитров (загрузите SRT или выберите проект с транскрибацией)")
    cover_title = cover_title.strip()
    if len(cover_title) > 100:
        raise HTTPException(400, "Название обложки должно быть не длиннее 100 символов")

    zoom_override = None
    if zoom_timeline_json and zoom_timeline_json.strip():
        try:
            zoom_override = json.loads(zoom_timeline_json)
        except json.JSONDecodeError:
            raise HTTPException(400, "zoom_timeline_json должен быть валидным JSON")

    overlays: list[dict] = []
    if overlays_json and overlays_json.strip():
        try:
            raw_overlays = json.loads(overlays_json)
        except json.JSONDecodeError:
            raise HTTPException(400, "overlays_json должен быть валидным JSON")
        if not isinstance(raw_overlays, list):
            raise HTTPException(400, "overlays_json должен быть списком")
        if len(raw_overlays) > 20:
            raise HTTPException(400, "Можно добавить не более 20 наложений")
        saved_overlay_paths: dict[int, Path] = {}
        overlay_uploads = overlay_files or []
        total_overlay_bytes = 0
        for index, raw in enumerate(raw_overlays):
            if not isinstance(raw, dict):
                raise HTTPException(400, f"Наложение {index + 1} имеет неверный формат")
            try:
                start = float(raw.get("start") or 0.0)
                end = float(raw.get("end"))
                x = float(raw.get("x", 0.5))
                y = float(raw.get("y", 0.5))
                size = max(20, min(1200, int(raw.get("size") or 180)))
            except (TypeError, ValueError):
                raise HTTPException(400, f"Параметры наложения {index + 1} некорректны")
            if not all(math.isfinite(value) for value in (start, end, x, y)):
                raise HTTPException(400, f"Параметры наложения {index + 1} должны быть конечными числами")
            start = max(0.0, min(86399.95, start))
            end = max(0.0, min(86400.0, end))
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            if end - start < 0.05:
                raise HTTPException(400, f"Длительность наложения {index + 1} должна быть не меньше 0.05 сек")
            overlay = {
                "uid": str(raw.get("uid") or f"overlay-{index + 1}")[:80],
                "start": round(start, 3),
                "end": round(end, 3),
                "x": round(x, 4),
                "y": round(y, 4),
                "size": size,
                "z_index": index,
            }
            overlay_kind = str(raw.get("kind") or "")
            if overlay_kind == "image":
                try:
                    file_index = int(raw.get("file_index"))
                    upload = overlay_uploads[file_index]
                except (TypeError, ValueError, IndexError):
                    raise HTTPException(400, f"Файл наложения {index + 1} не найден")
                if file_index not in saved_overlay_paths:
                    content = await upload.read()
                    total_overlay_bytes += len(content)
                    if not content or len(content) > 20 * 1024 * 1024 or total_overlay_bytes > 100 * 1024 * 1024:
                        raise HTTPException(400, "Изображения наложений слишком большие")
                    try:
                        from PIL import Image
                        with Image.open(io.BytesIO(content)) as image:
                            image.verify()
                        with Image.open(io.BytesIO(content)) as image:
                            width, height = image.size
                            image_format = str(image.format or "").upper()
                    except Exception:
                        raise HTTPException(400, f"Наложение {index + 1} не является корректным изображением")
                    if image_format not in {"PNG", "JPEG", "WEBP"} or width < 1 or height < 1 or width * height > 50_000_000:
                        raise HTTPException(400, f"Формат или размер наложения {index + 1} не поддерживается")
                    extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[image_format]
                    overlay_dir = work_dir / "overlays"
                    overlay_dir.mkdir(exist_ok=True)
                    overlay_path = overlay_dir / f"overlay_{file_index + 1:02d}{extension}"
                    overlay_path.write_bytes(content)
                    saved_overlay_paths[file_index] = overlay_path
                overlay.update({
                    "kind": "image",
                    "image_path": str(saved_overlay_paths[file_index].resolve()),
                    "name": Path(upload.filename or f"Наложение {index + 1}").name[:120],
                })
            else:
                overlay_animation_id = str(raw.get("animation_id") or "")
                if Path(overlay_animation_id).name != overlay_animation_id or not (
                    montage_engine.ANIMATIONS_DIR / overlay_animation_id / "manifest.json"
                ).is_file():
                    raise HTTPException(400, f"Источник наложения {index + 1} не найден")
                overlay.update({"kind": "animation", "animation_id": overlay_animation_id})
            overlays.append(overlay)
    if animation_id:
        if Path(animation_id).name != animation_id or not (
            montage_engine.ANIMATIONS_DIR / animation_id / "manifest.json"
        ).is_file():
            raise HTTPException(400, "Выбранная анимация не найдена")

    jobs[job_id] = {
        "id": job_id,
        "original_name": original_name,
        "video_path": str(video_path),
        "srt_text": srt_text,
        "transcript_job_id": src_job_id,
        "project_id": project_id,
        "kind": "montage",
        "status": "uploaded",
        "phase": "queued",
        "warnings": [],
        "progress": 0,
        "stage": "",
        "detail": "",
        "result": None,
        "publish": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "covers": existing_covers,
        "cover_status": "done" if existing_covers else "idle",
        "cover_progress": 0,
        "cover_detail": "",
        "cover_error": None,
        "cover_title": cover_title,
        "operation": None,
        "variants": {},
        "mirror_variant": None,
        "publish_prep": {},
        "prepare_operation": None,
        "prepare_status": "idle",
        "prepare_error": None,
    }
    params = {
        "subtitle_mode": subtitle_mode,
        "subtitle_position": subtitle_position,
        "subtitle_x": subtitle_x,
        "subtitle_y": subtitle_y,
        "subtitle_scale": subtitle_scale,
        "subtitle_box": subtitle_box,
        "subtitle_font": subtitle_font,
        "face_tracking": face_tracking,
        "autozoom": autozoom,
        "mirror_horizontal": False,
        "blur_banned_words": blur_banned_words,
        "subtitles": subtitles,
        "bigpickle": bigpickle,
        "edge_mode": edge_mode,
        "overlays": overlays,
        "animation": {
            "enabled": bool(animation_id),
            "id": animation_id,
            "start": animation_start,
            "offset_x": animation_offset_x,
            "item_size": max(20, animation_item_size),
            "bar": animation_bar,
        },
        "zoom_timeline_override": zoom_override,
        "prepare_mirrored": bool(prepare_mirrored),
        "prepare_publish": bool(prepare_publish),
    }
    jobs[job_id]["params"] = params
    _save_montage_job(job_id)
    if not project_id and video and video.filename:
        name = video.filename
        ext = Path(name).suffix.lower()
        if ext in VIDEO_EXTS and srt_text:
            async with PROJECTS_LOCK:
                PROJECTS[f"p_{job_id}"] = {
                    "id": f"p_{job_id}",
                    "name": name,
                    "video_path": str(video_path),
                    "transcript_job_id": src_job_id,
                    "status": "ready",
                    "created_at": datetime.now().isoformat(),
                    "covers": list(jobs[job_id].get("covers") or []),
                    "covers_dir": str(MONTAGE_WORK / job_id / "covers"),
                }
                _save_projects()
    async def _run_montage_then_variant() -> None:
        await process_montage(job_id, params)
        job = jobs.get(job_id) or {}
        if job.get("status") == "error":
            return
        # ---- Phase B: mirrored variant (optional, one toggle) ----
        if params.get("prepare_mirrored"):
            job["phase"] = "rendering_mirror"
            _save_montage_job(job_id)
            if not mirror_variant_ready(job):
                await process_mirror_variant(job_id)
        # ---- Phase C: platform-ready artefacts (automatic) ----
        if params.get("prepare_publish", True):
            job["phase"] = "preparing_platform_variants"
            _save_montage_job(job_id)
            try:
                await process_prepare_publish(
                    job_id, _default_prepare_targets(job)
                )
            except Exception as exc:
                job["prepare_status"] = "error"
                job["prepare_error"] = str(exc)
        # ---- Phase D: verify and finalize (no premature done) ----
        job["phase"] = "verifying_outputs"
        _save_montage_job(job_id)
        await _finalize_montage(job_id)

    asyncio.create_task(_run_montage_then_variant())
    if propose_cover:
        asyncio.create_task(process_cover_suggestions(job_id))
    return {"job_id": job_id, "filename": original_name}


@app.post("/api/montage/{job_id}/publish")
async def publish_montage(
    job_id: str,
    targets_json: str = Form(...),
    title: str = Form(""),
    text: str = Form(""),
    cover_id: str = Form(""),
    new_publication: bool = Form(False),
    confirm_unknown: bool = Form(False),
):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    job = jobs[job_id]
    if job["status"] == "publishing":
        raise HTTPException(409, "Публикация уже выполняется для этого ролика")
    # The unified pipeline may still be finishing optional phases (mirroring /
    # platform preparation) after the main render.  The main video is already
    # available, so publishing is allowed as soon as a result exists.
    if not job.get("result") or job["status"] not in {"done", "completed", "processing"}:
        raise HTTPException(400, "Сначала завершите рендер")
    try:
        requested_targets = json.loads(targets_json)
    except json.JSONDecodeError:
        raise HTTPException(400, "targets_json must be valid JSON")
    if not isinstance(requested_targets, list) or not requested_targets:
        raise HTTPException(400, "Список целей пуст")
    if len(title) > 119:
        raise HTTPException(400, "Заголовок должен быть не длиннее 119 символов")
    if len(text) > 2000:
        raise HTTPException(400, "Текст должен быть не длиннее 2000 символов")
    target_specs = _PUBLISH_TARGET_SPECS
    target_ids = list(dict.fromkeys(str(item) for item in requested_targets))
    unknown = [target_id for target_id in target_ids if target_id not in target_specs]
    if unknown:
        raise HTTPException(400, f"Неизвестные цели публикации: {', '.join(unknown)}")
    if not new_publication:
        already_published = set((job.get("publish") or {}).keys())
        already_published |= {
            target_id for target_id, state in (job.get("publish_status") or {}).items()
            if state == "published"
        }
        target_ids = [target_id for target_id in target_ids if target_id not in already_published]
        if not target_ids:
            raise HTTPException(400, "Все выбранные цели уже успешно опубликованы")
    # A platform whose outcome is unknown must not be re-published without an
    # explicit per-request confirmation.  new_publication does NOT bypass this.
    publish_status = job.get("publish_status") or {}
    publish_unknown = job.get("publish_unknown") or {}
    unknown_needing_confirm = [
        target_id for target_id in target_ids
        if publish_status.get(target_id) == "unknown" or target_id in publish_unknown
    ]
    if unknown_needing_confirm and not confirm_unknown:
        return JSONResponse(
            status_code=409,
            content={
                "detail": "Требуется подтверждение повторной публикации: "
                          + ", ".join(unknown_needing_confirm),
                "confirm_targets": unknown_needing_confirm,
            },
        )
    if any(target_id.startswith("youtube") for target_id in target_ids) and not title.strip():
        raise HTTPException(400, "Для YouTube укажите заголовок")
    targets = []
    selected_cover = None
    if cover_id:
        selected_cover = next((item for item in job.get("covers") or [] if item.get("id") == cover_id), None)
        if selected_cover is None:
            raise HTTPException(400, "Выбранная обложка не найдена")
        cover_path = MONTAGE_WORK / job_id / "covers" / Path(str(selected_cover.get("file") or "")).name
        if not cover_path.exists():
            raise HTTPException(400, "Файл выбранной обложки недоступен")
    for target_id in target_ids:
        kind, mirrored, label = target_specs[target_id]
        target = {"id": target_id, "kind": kind, "mirrored": mirrored, "label": label}
        if selected_cover:
            target["cover_path"] = str(cover_path)
        if kind == "youtube":
            target["title"] = title.strip()
        else:
            target["text"] = text.strip()
            target["caption"] = text.strip()
        targets.append(target)
    # Publishing automatically prepares every missing local artefact first (spec
    # 6) — the user never presses a separate "prepare" button.  Mirrored targets
    # whose mirror variant was never rendered cannot be produced here (publish
    # and preparation never mirror); they fail per target with a clear reason.
    _, missing_targets = _publish_readiness(job, targets)
    preparable_ids = {
        item.get("target") for item in missing_targets
        if item.get("reason") in {"video_not_ready", "cover_not_ready"}
    }
    auto_prepare = [t.get("id") for t in targets if t.get("id") in preparable_ids]
    mirror_blocked = [item for item in missing_targets if item.get("reason") == "mirror_not_ready"]
    if mirror_blocked:
        blocked_labels = ", ".join(
            str(item.get("label") or item.get("target")) for item in mirror_blocked
        )
        await _montage_announce(
            job_id,
            f"Отражённая версия не готова — цели ({blocked_labels}) будут пропущены. "
            "Включите «Создать отражённую версию» и повторите монтаж.",
        )
    # Atomic per-job reservation: acquire the lock before creating the
    # background task so two near-simultaneous requests can never start two
    # publications for the same job_id.
    lock = _publish_lock(job_id)
    if lock.locked():
        raise HTTPException(409, "Публикация уже выполняется для этого ролика")
    await lock.acquire()
    try:
        job["selected_cover_id"] = cover_id
        job["last_publish_targets"] = targets
        _save_montage_job(job_id)
        asyncio.create_task(
            _prepare_then_publish(
                job_id, targets, lock,
                cover_path=str(cover_path) if cover_id else None,
                cover_id=cover_id,
                prepare_ids=set(preparable_ids),
            )
        )
    except BaseException:
        if lock.locked():
            lock.release()
        raise
    return {
        "job_id": job_id,
        "status": "publishing",
        "targets": [t.get("kind") for t in targets],
        "auto_prepare": auto_prepare,
    }


_youtube_reauth_state: dict[str, str] = {"status": "idle", "detail": ""}


@app.post("/api/montage/youtube-reauth")
async def youtube_reauth():
    if _youtube_reauth_state["status"] in {"running"}:
        raise HTTPException(409, "Повторная авторизация уже выполняется")
    _youtube_reauth_state.update(status="running", detail="Открытие браузера...")

    def run() -> None:
        try:
            from montage.youtube_publisher import YouTubePublisher
            from montage.secrets_env import load_secrets_env
            load_secrets_env()
            workdir = MONTAGE_WORK / "youtube_reauth"
            workdir.mkdir(parents=True, exist_ok=True)
            youtube = YouTubePublisher(
                montage_engine.INSTAPOSTER_DIR,
                Path(sys.executable),
                workdir / "youtube.log",
            )
            youtube.reauth()
            _youtube_reauth_state.update(status="ok", detail="Авторизация YouTube обновлена")
        except Exception as exc:
            _youtube_reauth_state.update(status="error", detail=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return {"status": "running"}


@app.get("/api/montage/youtube-reauth")
async def youtube_reauth_status():
    return dict(_youtube_reauth_state)


@app.get("/api/montage/result/{job_id}")
async def montage_result(job_id: str):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    job = jobs[job_id]
    return {
        "id": job_id,
        "status": job["status"],
        "phase": job.get("phase"),
        "warnings": list(job.get("warnings") or []),
        "progress": job["progress"],
        "stage": job["stage"],
        "detail": job["detail"],
        "error": job.get("error"),
        "result": job.get("result"),
        "publish": job.get("publish"),
        "publish_errors": job.get("publish_errors"),
        "publish_unknown": job.get("publish_unknown"),
        "publish_status": job.get("publish_status"),
        "publish_progress": job.get("publish_progress"),
        "covers": job.get("covers") or [],
        "cover_status": job.get("cover_status") or "idle",
        "cover_progress": job.get("cover_progress") or 0,
        "cover_detail": job.get("cover_detail") or "",
        "cover_error": job.get("cover_error"),
        "selected_cover_id": job.get("selected_cover_id") or "",
        "last_publish_targets": job.get("last_publish_targets") or [],
        "operation": job.get("operation"),
        "prepare_operation": job.get("prepare_operation"),
        "prepare_status": job.get("prepare_status") or "idle",
        "prepare_error": job.get("prepare_error"),
        "publish_ready": _public_publish_ready(job),
        "mirror_variant": _public_mirror_variant(job),
        "variants": {"mirrored_ready": mirror_variant_ready(job)},
    }


@app.get("/api/montage/{job_id}/covers/{filename}")
async def montage_cover_asset(job_id: str, filename: str, thumbnail: bool = False):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    safe_name = Path(filename).name
    allowed = {Path(str(item.get("file") or "")).name for item in jobs[job_id].get("covers") or []}
    if safe_name not in allowed:
        raise HTTPException(404, "Обложка не найдена")
    path = MONTAGE_WORK / job_id / "covers" / safe_name
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")
    if thumbnail:
        from PIL import Image
        thumbnail_dir = path.parent / ".thumbnails"
        thumbnail_dir.mkdir(exist_ok=True)
        thumbnail_path = thumbnail_dir / f"{path.stem}.jpg"
        if not thumbnail_path.is_file() or thumbnail_path.stat().st_mtime < path.stat().st_mtime:
            temporary = thumbnail_dir / f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp.jpg"
            with Image.open(path) as image:
                image.convert("RGB").resize((360, 640), Image.Resampling.LANCZOS).save(
                    temporary, "JPEG", quality=82, optimize=True
                )
            os.replace(temporary, thumbnail_path)
        path = thumbnail_path
        media_type = "image/jpeg"
    return FileResponse(
        str(path), media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.post("/api/montage/{job_id}/covers/upload")
async def upload_montage_cover(job_id: str, cover: UploadFile = File(...)):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    suffix = Path(cover.filename or "cover.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "Поддерживаются PNG, JPG и WebP")
    covers_dir = MONTAGE_WORK / job_id / "covers"
    covers_dir.mkdir(parents=True, exist_ok=True)
    temporary = covers_dir / f"upload-source-{uuid.uuid4().hex[:8]}{suffix}"
    temporary.write_bytes(await cover.read())
    try:
        items = await asyncio.to_thread(cover_generator.save_uploaded_cover, temporary, covers_dir)
    except Exception as exc:
        raise HTTPException(400, f"Не удалось обработать обложку: {exc}")
    finally:
        temporary.unlink(missing_ok=True)
    jobs[job_id]["covers"] = items
    jobs[job_id]["cover_status"] = "done"
    await _store_project_covers(job_id)
    _save_montage_job(job_id)
    return {"covers": items}


@app.post("/api/montage/{job_id}/covers/generate")
async def generate_montage_cover(job_id: str, timestamp: float = Form(...), title: str = Form(...)):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    title = title.strip()
    if not title or len(title) > 100:
        raise HTTPException(400, "Название обложки должно содержать 1–100 символов")
    if jobs[job_id].get("cover_status") == "generating":
        raise HTTPException(409, "Генерация обложки уже выполняется")
    jobs[job_id].update(
        cover_status="generating",
        cover_progress=0,
        cover_detail="Создание обложки из стоп-кадра",
        cover_error=None,
    )
    _save_montage_job(job_id)
    asyncio.create_task(process_custom_cover(job_id, max(0, timestamp), title))
    return {"status": "generating"}


@app.get("/api/montage/download/{job_id}")
async def montage_download(job_id: str):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    result = jobs[job_id].get("result")
    if not result:
        raise HTTPException(400, "Рендер не завершён")
    video = Path(result["result_video"])
    if not video.exists():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(path=str(video), media_type="video/mp4", filename=video.name, content_disposition_type="inline")


@app.get("/api/montage/preview/{job_id}")
async def montage_preview(job_id: str, request: Request):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    result = jobs[job_id].get("result")
    if not result:
        raise HTTPException(400, "Рендер не завершён")
    video = Path(result.get("preview_video") or result["result_video"])
    if not video.exists():
        raise HTTPException(404, "Файл не найден")
    return _video_response(video, request, "video/mp4")


@app.get("/api/montage/animations")
async def montage_animations():
    return {"animations": montage_engine.animation_ids()}


@app.post("/api/montage/animations")
async def upload_animation(zipfile: UploadFile = File(...)):
    import zipfile as _zipfile
    if not zipfile.filename or not zipfile.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Ожидается ZIP-архив")
    tmp = MONTAGE_WORK / f"anim_{uuid.uuid4().hex[:8]}.zip"
    tmp.write_bytes(await zipfile.read())
    target = montage_engine.ANIMATIONS_DIR / (zipfile.filename.rsplit(".", 1)[0])
    target.mkdir(parents=True, exist_ok=True)
    with _zipfile.ZipFile(tmp) as z:
        for name in z.namelist():
            safe = Path(name).name
            if safe:
                z.extract(name, str(target))
    tmp.unlink(missing_ok=True)
    return {"animations": montage_engine.animation_ids()}


@app.post("/api/montage/animations/create")
async def create_animation(
    name: str = Form(...),
    fps: float = Form(10.0),
    frames: list[UploadFile] = File(...),
):
    from PIL import Image

    clean_name = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip() or "Animation"
    slug = re.sub(r"\s+", "_", clean_name)
    animation_id = f"{slug}_{uuid.uuid4().hex[:8]}"
    target = montage_engine.ANIMATIONS_DIR / animation_id
    target.mkdir(parents=True, exist_ok=False)
    manifest_frames = []
    try:
        for index, upload in enumerate(frames):
            raw = await upload.read()
            if not raw:
                continue
            source = target / f"upload_{index:04d}"
            source.write_bytes(raw)
            output = target / f"frame_{len(manifest_frames):04d}.png"
            try:
                with Image.open(source) as image:
                    image.convert("RGBA").save(output, "PNG")
            finally:
                source.unlink(missing_ok=True)
            manifest_frames.append({"file": output.name})
        if not manifest_frames:
            raise HTTPException(400, "Добавьте хотя бы один корректный кадр")
        manifest = {
            "id": animation_id,
            "name": clean_name,
            "fps": max(1.0, min(60.0, float(fps))),
            "frames": manifest_frames,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {"animation": {"id": animation_id, "name": clean_name}}


@app.get("/api/jobs")
async def list_jobs():
    return [{"id": j["id"], "name": j["original_name"], "status": j["status"]} for j in jobs.values()]


@app.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    q = asyncio.Queue()
    job_queues[job_id] = q
    try:
        job = jobs.get(job_id)
        if job:
            await websocket.send_json({
                "type": "progress",
                "progress": job["progress"],
                "stage": job["stage"],
                "detail": job["detail"],
            })
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=10)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                continue
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        job_queues.pop(job_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
