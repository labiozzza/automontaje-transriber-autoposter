import os, json, uuid, asyncio, subprocess, time, re, tempfile, hashlib, urllib.request, importlib, threading, gc, math, io, sys
import shutil
import traceback
from pathlib import Path
from typing import Optional
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
                return

        await broadcast_progress(job_id, 15, "conversion", "Конвертация завершена")

        # Stage 2: Transcription
        model_name = params.get("model", "small")
        language = params.get("language", "ru")
        mode = params.get("mode", "sentences")

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
        if await register_video_project(job_id):
            await broadcast_progress(job_id, 100, "done",
                                     "Готово! Ролик добавлен в проекты (Автомонтаж).")

    except Exception as e:
        import traceback
        job["status"] = "error"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
        await broadcast_progress(job_id, 0, "error", str(e))


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
    return {"ok": True}


@app.post("/api/transcribe/{job_id}")
async def start_transcription(
    job_id: str,
    model: str = Form("small"),
    language: str = Form("ru"),
    mode: str = Form("sentences"),
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
from montage import drawing_generator

_drawing_generation_lock = asyncio.Lock()

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
    if job.get("status") != "done" or not job.get("result"):
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
                if job.get("kind") == "montage" and (job.get("result") or {}).get("result_video"):
                    if str(job.get("id") or "") != workdir.name:
                        continue
                    recovered_after_interruption = False
                    if not job.get("selected_cover_id"):
                        prepared_cover = next((workdir / "covers").glob(".*-instagram.jpg"), None)
                        if prepared_cover:
                            job["selected_cover_id"] = prepared_cover.name[1:].split("-instagram.", 1)[0]
                    if job.get("status") in {"processing", "publishing"}:
                        job["status"] = "done"
                        job["stage"] = "done"
                        job["detail"] = "Задание восстановлено после перезапуска"
                        targets = job.get("last_publish_targets") or []
                        job["publish_errors"] = {
                            str(item.get("id")): "Попытка была прервана перезапуском; повторите публикацию"
                            for item in targets if item.get("id")
                        }
                        recovered_after_interruption = True
                    recovered_id = str(job["id"])
                    jobs[recovered_id] = job
                    if recovered_after_interruption:
                        _save_montage_job(recovered_id)
                    _run_cleanup_safely(_cleanup_render_artifacts, recovered_id, job)
                    if job.get("stage") == "published" and not job.get("publish_errors"):
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
        }
        _save_montage_job(job_id)
        _run_cleanup_safely(_cleanup_render_artifacts, job_id, jobs[job_id])


_recover_montage_jobs()


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

        await _montage_announce(job_id, "Запуск рендера...")
        meta = await asyncio.to_thread(
            montage_engine.render_montage,
            job_id=job_id,
            workdir=workdir,
            source_video=video,
            srt_text=srt_text,
            options=render_params,
            progress_cb=progress,
            log_cb=lambda message: asyncio.run_coroutine_threadsafe(
                _montage_announce(job_id, message), loop
            ),
        )

        job["status"] = "done"
        job["progress"] = 100
        job["stage"] = "done"
        job["detail"] = "Готово"
        job["result"] = meta
        _save_montage_job(job_id)
        _run_cleanup_safely(_cleanup_render_artifacts, job_id, job)
        await broadcast_progress(job_id, 100, "done", "Рендер завершён")
        await _montage_announce(job_id, f"Готово: {Path(meta['result_video']).name} ({(os.path.getsize(meta['result_video']) >> 20)} МБ)")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        _save_montage_job(job_id)
        print(f"[montage] job {job_id} failed: {traceback.format_exc()}", file=sys.stderr)
        await broadcast_progress(job_id, 0, "error", str(e))


async def _store_project_covers(job_id: str) -> None:
    job = jobs[job_id]
    project_ids = [job.get("project_id"), f"p_{job_id}"]
    async with PROJECTS_LOCK:
        for project_id in project_ids:
            if project_id and project_id in PROJECTS:
                PROJECTS[project_id]["covers"] = list(job.get("covers") or [])
                PROJECTS[project_id]["covers_dir"] = str(MONTAGE_WORK / job_id / "covers")


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


async def process_publish(job_id: str, targets: list):
    job = jobs[job_id]
    loop = asyncio.get_running_loop()
    try:
        previous_publish = dict(job.get("publish") or {})
        job["status"] = "publishing"
        job["error"] = None
        job["publish"] = previous_publish or None
        job["publish_errors"] = None
        job["publish_progress"] = {
            str(target["id"]): {
                "label": str(target.get("label") or target["id"]),
                "progress": 0,
                "status": "queued",
                "detail": "В очереди",
            }
            for target in targets
        }
        job["last_publish_targets"] = targets
        _save_montage_job(job_id)
        progress = _threaded_progress(loop, job_id)
        workdir = MONTAGE_WORK / job_id
        source_video = Path(job["result"]["result_video"])
        mirrored_source_video = None

        if any(bool(target.get("mirrored")) for target in targets):
            try:
                for target in targets:
                    if target.get("mirrored"):
                        job["publish_progress"][target["id"]].update(
                            status="preparing", detail="Подготовка отражённой версии"
                        )
                await _montage_announce(job_id, "Подготовка отражённой версии...")
                alternate_options = dict(job.get("params") or {})
                alternate_options["mirror_horizontal"] = not bool(alternate_options.get("mirror_horizontal"))
                if job.get("audio_censor_srt"):
                    alternate_options["audio_censor_srt"] = job["audio_censor_srt"]
                alternate_workdir = workdir / "publish_mirrored"
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

                def alternate_progress(percent: float, stage: str, detail: str) -> None:
                    progress(min(20, percent * 0.2), f"mirror_{stage}", detail)

                alternate = await asyncio.to_thread(
                    montage_engine.render_montage,
                    job_id=f"{job_id}_mirror",
                    workdir=alternate_workdir,
                    source_video=Path(job["video_path"]),
                    srt_text=job["srt_text"],
                    options=alternate_options,
                    progress_cb=alternate_progress,
                    log_cb=lambda message: asyncio.run_coroutine_threadsafe(
                        _montage_announce(job_id, f"mirror: {message}"), loop
                    ),
                )
                mirrored_source_video = Path(alternate["result_video"])
                for target in targets:
                    if target.get("mirrored"):
                        job["publish_progress"][target["id"]].update(
                            status="queued", detail="Отражённая версия готова"
                        )
            except Exception as exc:
                for target in targets:
                    if target.get("mirrored"):
                        target["preparation_error"] = str(exc)
                await _montage_announce(job_id, f"Отражённая версия не подготовлена: {exc}")

        await _montage_announce(job_id, "Публикация...")

        def publish_progress(percent: float, stage: str, detail: str) -> None:
            offset = 20 if mirrored_source_video else 0
            progress(offset + percent * (100 - offset) / 100, stage, detail)

        def target_progress(target_id: str, percent: float, detail: str, status: str) -> None:
            def update() -> None:
                item = job["publish_progress"].get(target_id)
                if item is not None:
                    item.update(progress=round(percent, 1), detail=detail, status=status)
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
            progress_cb=publish_progress,
            target_progress_cb=target_progress,
            log_cb=lambda message: asyncio.run_coroutine_threadsafe(
                _montage_announce(job_id, message), loop
            ),
        )
        publish_results = dict(result.get("results") or {})
        publish_errors = dict(result.get("errors") or {})
        expected_targets = {str(target.get("id") or target.get("kind") or "") for target in targets}
        for missing in expected_targets - publish_results.keys() - publish_errors.keys():
            if missing:
                publish_errors[missing] = "Публикация не вернула результат"
        job["status"] = "done"
        job["progress"] = 100
        job["stage"] = "published"
        job["publish"] = {**previous_publish, **publish_results}
        job["publish_errors"] = publish_errors
        detail = "Публикация завершена" if not job["publish_errors"] else "Публикация завершена с ошибками"
        job["detail"] = detail
        _save_montage_job(job_id)
        if not job["publish_errors"]:
            _run_cleanup_safely(_cleanup_publish_artifacts, job_id, job)
        await broadcast_progress(job_id, 100, "published", detail)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        _save_montage_job(job_id)
        await broadcast_progress(job_id, 0, "error", str(e))


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


@app.post("/api/montage/drawings/generate")
async def generate_montage_drawings(request: Request):
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(403, "Генерация рисунков доступна только локально")
    origin = request.headers.get("origin", "")
    if origin and origin not in {
        "http://127.0.0.1:8000", "http://localhost:8000",
        "https://127.0.0.1:8000", "https://localhost:8000",
    }:
        raise HTTPException(403, "Недопустимый источник запроса")
    if _drawing_generation_lock.locked():
        raise HTTPException(409, "GPT уже создаёт рисунки")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "Некорректный JSON")
    if not isinstance(payload, dict):
        raise HTTPException(400, "Некорректный запрос")
    transcript = str(payload.get("srt_text") or "")
    transcript_job_id = str(payload.get("transcript_job_id") or "")
    project_id = str(payload.get("project_id") or "")
    if transcript_job_id:
        source_job = jobs.get(transcript_job_id) or {}
        if source_job.get("status") == "done" and isinstance(source_job.get("result"), list):
            transcript = montage_engine.build_words_srt(source_job["result"])
    elif project_id:
        async with PROJECTS_LOCK:
            project = dict(PROJECTS.get(project_id) or {})
        source_job = jobs.get(str(project.get("transcript_job_id") or "")) or {}
        if source_job.get("status") == "done" and isinstance(source_job.get("result"), list):
            transcript = montage_engine.build_words_srt(source_job["result"])
    if not transcript.strip():
        raise HTTPException(400, "Сначала выберите транскрибацию или загрузите SRT")
    if len(transcript) > 200_000:
        raise HTTPException(400, "Транскрипция слишком большая")
    try:
        async with _drawing_generation_lock:
            drawings = await asyncio.to_thread(drawing_generator.generate_drawings, transcript)
    except Exception as exc:
        raise HTTPException(502, f"Не удалось получить рисунки от GPT: {exc}")
    return {"drawings": drawings}


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
    mirror_horizontal: bool = Form(False),
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
            elif overlay_kind == "drawing":
                try:
                    validated = drawing_generator.validate_drawing_overlay(raw, index)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise HTTPException(400, f"Рисунок {index + 1} некорректен: {exc}")
                overlay.update({
                    "kind": "drawing",
                    "name": validated["name"],
                    "trigger_text": validated["trigger_text"],
                    "draw_speed": validated["draw_speed"],
                    "drawing": validated["drawing"],
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
        "mirror_horizontal": mirror_horizontal,
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
    asyncio.create_task(process_montage(job_id, params))
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
):
    if job_id not in jobs or jobs[job_id].get("kind") != "montage":
        raise HTTPException(404, "Montage job not found")
    job = jobs[job_id]
    if job["status"] != "done" or not job.get("result"):
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
    target_specs = {
        "instagram_trial": ("trial", False, "Instagram пробное"),
        "instagram_trial_mirrored": ("trial", True, "Instagram пробное перевёрнутое"),
        "instagram_feed": ("instagram", False, "Instagram основная лента"),
        "instagram_feed_mirrored": ("instagram", True, "Instagram основная лента перевёрнутое"),
        "youtube": ("youtube", False, "YouTube"),
        "youtube_mirrored": ("youtube", True, "YouTube перевёрнутое"),
    }
    target_ids = list(dict.fromkeys(str(item) for item in requested_targets))
    unknown = [target_id for target_id in target_ids if target_id not in target_specs]
    if unknown:
        raise HTTPException(400, f"Неизвестные цели публикации: {', '.join(unknown)}")
    if not new_publication:
        already_published = set((job.get("publish") or {}).keys())
        target_ids = [target_id for target_id in target_ids if target_id not in already_published]
        if not target_ids:
            raise HTTPException(400, "Все выбранные цели уже успешно опубликованы")
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
    job["selected_cover_id"] = cover_id
    job["last_publish_targets"] = targets
    _save_montage_job(job_id)
    asyncio.create_task(process_publish(job_id, targets))
    return {"job_id": job_id, "status": "publishing", "targets": [t.get("kind") for t in targets]}


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
        "progress": job["progress"],
        "stage": job["stage"],
        "detail": job["detail"],
        "error": job.get("error"),
        "result": job.get("result"),
        "publish": job.get("publish"),
        "publish_errors": job.get("publish_errors"),
        "publish_progress": job.get("publish_progress"),
        "covers": job.get("covers") or [],
        "cover_status": job.get("cover_status") or "idle",
        "cover_progress": job.get("cover_progress") or 0,
        "cover_detail": job.get("cover_detail") or "",
        "cover_error": job.get("cover_error"),
        "selected_cover_id": job.get("selected_cover_id") or "",
        "last_publish_targets": job.get("last_publish_targets") or [],
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
