"""Web app: upload a guitar clip -> rig estimate, picture, Logic kit, emulation.

Run locally:  uvicorn server:app --reload
Railway:      uses the Dockerfile; listens on $PORT.
"""
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from gre.pipeline import run

JOBS_DIR = os.environ.get("JOBS_DIR", "/tmp/gre_jobs")
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "40"))
JOB_TTL_H = float(os.environ.get("JOB_TTL_HOURS", "12"))
WORKERS = int(os.environ.get("WORKERS", "2"))
ALLOWED_EXT = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".aac", ".caf"}
os.makedirs(JOBS_DIR, exist_ok=True)

app = FastAPI(title="Guitar Rig Reverse Engineer")
pool = ThreadPoolExecutor(max_workers=WORKERS)
jobs = {}
lock = threading.Lock()
HERE = os.path.dirname(os.path.abspath(__file__))


def _cleanup():
    now = time.time()
    for d in os.listdir(JOBS_DIR):
        p = os.path.join(JOBS_DIR, d)
        try:
            if os.path.isdir(p) and now - os.path.getmtime(p) > JOB_TTL_H * 3600:
                shutil.rmtree(p, ignore_errors=True)
                with lock:
                    jobs.pop(d, None)
        except OSError:
            pass


def _set(jid, **kw):
    with lock:
        jobs.setdefault(jid, {}).update(kw)


def _work(jid, audio, di, start):
    out = os.path.join(JOBS_DIR, jid, "out")
    try:
        _set(jid, status="running")
        res = run(audio, out, di_path=di, start=start,
                  progress=lambda m, f: _set(jid, message=m, progress=round(f, 3)))
        _set(jid, status="done", progress=1.0, message="Done", result=res)
    except Exception as e:  # report to the UI, keep the server alive
        traceback.print_exc()
        _set(jid, status="error", message=str(e) or e.__class__.__name__)


async def _save_upload(up: UploadFile, dest_dir: str, stem: str):
    ext = os.path.splitext(up.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Use WAV, MP3, FLAC, AIFF, M4A or OGG.")
    path = os.path.join(dest_dir, stem + ext)
    size = 0
    with open(path, "wb") as fh:
        while True:
            chunk = await up.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1e6:
                raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB:.0f} MB).")
            fh.write(chunk)
    return path


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as fh:
        return fh.read()


@app.post("/api/jobs")
async def create_job(audio: UploadFile = File(...), di: UploadFile | None = File(None), start: float = Form(0.0)):
    _cleanup()
    jid = uuid.uuid4().hex[:16]
    jdir = os.path.join(JOBS_DIR, jid)
    os.makedirs(jdir, exist_ok=True)
    try:
        a = await _save_upload(audio, jdir, "input")
        d = await _save_upload(di, jdir, "di") if di is not None and di.filename else None
    except HTTPException:
        shutil.rmtree(jdir, ignore_errors=True)
        raise
    _set(jid, status="queued", progress=0.0, message="Queued", name=audio.filename, created=time.time())
    pool.submit(_work, jid, a, d, max(0.0, float(start)))
    return {"id": jid}


@app.post("/api/demo")
def demo():
    """Render a synthetic clip through a known rig and analyse it (no upload needed)."""
    _cleanup()
    from gre.audio_io import save_wav
    from gre.rig import default_rig, render
    from gre.synth import render_di
    from tests.validate import make_events
    jid = uuid.uuid4().hex[:16]
    jdir = os.path.join(JOBS_DIR, jid)
    os.makedirs(jdir, exist_ok=True)
    ev, dur = make_events(reps=2, seed=5)
    rig = default_rig()
    rig["drive"].update(type="overdrive", gain=6.0)
    rig["amp"].update(archetype="plexi", gain=4.5)
    rig["delay"].update(on=True, time_ms=360, feedback=0.3, mix=0.25)
    path = os.path.join(jdir, "demo.wav")
    save_wav(path, render(render_di(ev, dur, seed=9), rig))
    _set(jid, status="queued", progress=0.0, message="Queued", name="Demo riff (Plexi + overdrive + 360 ms delay)",
         created=time.time())
    pool.submit(_work, jid, path, None, 0.0)
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    with lock:
        j = jobs.get(jid)
        if j is None:
            raise HTTPException(404, "Unknown job (it may have expired).")
        return JSONResponse({k: v for k, v in j.items()})


@app.get("/api/jobs/{jid}/files/{name}")
def job_file(jid: str, name: str, download: int = 0):
    if not re.fullmatch(r"[0-9a-f]{16}", jid) or not re.fullmatch(r"[\w.\-]+", name):
        raise HTTPException(400, "Bad path")
    p = os.path.join(JOBS_DIR, jid, "out", name)
    if not os.path.isfile(p):
        raise HTTPException(404, "Not found")
    media = {".wav": "audio/wav", ".svg": "image/svg+xml", ".zip": "application/zip", ".json": "application/json",
             ".md": "text/markdown", ".mid": "audio/midi"}.get(os.path.splitext(name)[1], "application/octet-stream")
    return FileResponse(p, media_type=media, filename=name if download else None)
