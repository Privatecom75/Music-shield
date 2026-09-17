"""Music Shield web server: upload -> protect -> download."""

from __future__ import annotations

import gc
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import soundfile as sf

from music_shield import __version__
from music_shield.audio_io import (
    AudioTooLargeError,
    AudioTooLongError,
    MAX_DURATION_S,
    UnsupportedFormatError,
    load_audio,
    output_extension_for,
    save_audio,
    supported_input_extensions,
)
from music_shield.perturb import DEFAULT_PRESET, PRESETS, protect
from music_shield.synth import DEMO_DURATION_S, DEMO_SAMPLE_RATE, generate_demo_clip

MAX_UPLOAD_BYTES = int(os.environ.get("MUSIC_SHIELD_MAX_UPLOAD_MB", "80")) * 1024 * 1024
JOB_TTL_S = int(os.environ.get("MUSIC_SHIELD_JOB_TTL_S", str(60 * 60)))
DATA_DIR = Path(os.environ.get("MUSIC_SHIELD_DATA_DIR", Path(tempfile.gettempdir()) / "music_shield_jobs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "static"
DEMO_FILENAME = "music-shield-demo.wav"

app = FastAPI(title="Music Shield", version=__version__, docs_url=None, redoc_url=None)

_demo_clip_cache: bytes | None = None

# Public hosts: set MUSIC_SHIELD_BASIC_PASSWORD (and optional MUSIC_SHIELD_BASIC_USER).
# Unset password => auth off (local dev). When set, every request needs HTTP Basic.
_BASIC_USER = (os.environ.get("MUSIC_SHIELD_BASIC_USER") or "musicshield").strip() or "musicshield"
_BASIC_PASSWORD = (os.environ.get("MUSIC_SHIELD_BASIC_PASSWORD") or "").strip()


@app.middleware("http")
async def basic_auth_middleware(request, call_next):
    if not _BASIC_PASSWORD:
        return await call_next(request)
    header = request.headers.get("Authorization") or ""
    if header.lower().startswith("basic "):
        import base64

        try:
            raw = base64.b64decode(header.split(" ", 1)[1].strip()).decode("utf-8")
            username, _, password = raw.partition(":")
        except Exception:
            username = password = ""
        if secrets.compare_digest(username.encode("utf-8"), _BASIC_USER.encode("utf-8")) and secrets.compare_digest(
            password.encode("utf-8"), _BASIC_PASSWORD.encode("utf-8")
        ):
            return await call_next(request)
    return Response(
        content="Authentication required.",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Music Shield"'},
        media_type="text/plain",
    )




def demo_clip_bytes() -> bytes:
    """Render the synthetic demo clip once and keep it in memory (~1.7 MB)."""
    global _demo_clip_cache
    if _demo_clip_cache is None:
        buf = io.BytesIO()
        sf.write(buf, generate_demo_clip(), DEMO_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        _demo_clip_cache = buf.getvalue()
    return _demo_clip_cache


def _safe_stem(filename: str | None) -> str:
    stem = Path(filename or "track").stem
    cleaned = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in stem).strip(" .")
    return cleaned[:80] or "track"


def _cleanup_expired() -> None:
    now = time.time()
    for job_dir in DATA_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        try:
            if now - job_dir.stat().st_mtime > JOB_TTL_S:
                shutil.rmtree(job_dir, ignore_errors=True)
        except FileNotFoundError:
            pass


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


@app.get("/api/info")
def info() -> dict:
    return {
        "version": __version__,
        "presets": [
            {"name": p.name, "label": p.label, "description": p.description, "default": p.name == DEFAULT_PRESET}
            for p in PRESETS.values()
        ],
        "accepted_extensions": supported_input_extensions(),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "max_duration_minutes": MAX_DURATION_S // 60,
        "job_ttl_minutes": JOB_TTL_S // 60,
        "demo": {
            "url": "/api/demo-clip",
            "filename": DEMO_FILENAME,
            "duration_s": DEMO_DURATION_S,
            "sample_rate": DEMO_SAMPLE_RATE,
        },
    }


@app.get("/api/demo-clip")
def demo_clip() -> Response:
    """A short synthetic stereo clip, generated in code, for trying the tool."""
    return Response(
        content=demo_clip_bytes(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{DEMO_FILENAME}"',
            "Cache-Control": "public, max-age=86400",
        },
    )


@app.post("/api/protect")
async def protect_endpoint(file: UploadFile = File(...), strength: str = Form(DEFAULT_PRESET)) -> JSONResponse:
    _cleanup_expired()

    if strength not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown strength '{strength}'.")

    ext = Path(file.filename or "").suffix.lower()
    accepted = supported_input_extensions()
    if ext not in accepted:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext or 'unknown'}'. Upload one of: {', '.join(accepted)}.",
        )

    job_id = uuid.uuid4().hex
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(parents=True)
    input_path = job_dir / f"input{ext}"

    written = 0
    with input_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            out.write(chunk)

    if written == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        loaded = load_audio(input_path)
        source_ext = loaded.source_extension
        sample_rate = loaded.sample_rate
        subtype = loaded.subtype
        protected, stats = protect(loaded.samples, sample_rate, strength, seed=secrets.randbits(63))
        out_ext = output_extension_for(source_ext)
        output_name = f"{_safe_stem(file.filename)}-protected{out_ext}"
        output_path = job_dir / f"output{out_ext}"
        save_audio(output_path, protected, sample_rate, subtype)
        stats_dict = stats.as_dict()
        del protected, loaded, stats
        gc.collect()
    except AudioTooLargeError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (UnsupportedFormatError, AudioTooLongError, ValueError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Processing failed unexpectedly. Try a different file.")
    finally:
        # The original is never kept once processing is done.
        input_path.unlink(missing_ok=True)

    meta = {
        "job_id": job_id,
        "output_name": output_name,
        "output_ext": out_ext,
        "source_ext": source_ext,
        "stats": stats_dict,
        "created_at": time.time(),
    }
    (job_dir / "meta.json").write_text(json.dumps(meta))
    return JSONResponse(
        {
            "job_id": job_id,
            "download_url": f"/api/download/{job_id}",
            "output_name": output_name,
            "output_ext": out_ext,
            "source_ext": source_ext,
            "converted_to_lossless": out_ext != source_ext,
            "stats": stats_dict,
            "expires_in_minutes": JOB_TTL_S // 60,
        }
    )


@app.get("/api/download/{job_id}")
def download(job_id: str) -> FileResponse:
    if not (len(job_id) == 32 and all(c in "0123456789abcdef" for c in job_id)):
        raise HTTPException(status_code=404, detail="Not found.")
    job_dir = DATA_DIR / job_id
    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="This download has expired or never existed.")
    meta = json.loads(meta_path.read_text())
    output_path = job_dir / f"output{meta['output_ext']}"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="This download has expired.")
    media = "audio/flac" if meta["output_ext"] == ".flac" else "audio/wav"
    return FileResponse(output_path, media_type=media, filename=meta["output_name"])


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
