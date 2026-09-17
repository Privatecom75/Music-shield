"""Load and save audio using free tooling only.

WAV and FLAC go through libsndfile (soundfile). MP3 is decoded with ffmpeg
if it is installed; output is always lossless WAV or FLAC because re-encoding
to a lossy codec would partly smooth away the perturbation we just added.

Samples are float32 to keep peak RAM workable on small free-tier hosts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import subprocess

import numpy as np
import soundfile as sf

SUPPORTED_INPUT_EXTENSIONS = {".wav", ".flac", ".mp3"}
LOSSLESS_OUTPUT = {".wav": "WAV", ".flac": "FLAC"}
MAX_DURATION_S = 15 * 60

# Memory model for one protect job, used to reject a file *before* decoding it
# rather than letting the host OOM-kill the service (measured on the streaming
# engine, see LIMITS.md "Operational limits"):
#
#   peak ~= fixed overhead + BYTES_PER_FRAME_CHANNEL * frames * channels
#
# The float32 input and the float32 output are alive together (2 * 4 bytes per
# sample); the MP3 decode path has the same 2x moment (ffmpeg's pipe buffer and
# its numpy copy). Everything else -- interpreter, numpy/scipy/fastapi, the
# ~40 MB chunked-STFT working set, ffmpeg's own process, page cache for the
# output write -- measured at ~155 MiB on a warm server; 256 MB leaves ~100 MB
# of headroom under a 512 MB host at the largest admitted file.
MEMORY_BUDGET_MB = int(os.environ.get("MUSIC_SHIELD_MEMORY_BUDGET_MB", "512"))
FIXED_OVERHEAD_MB = 256
BYTES_PER_FRAME_CHANNEL = 8
MAX_SAMPLE_FRAMES = max(1, MEMORY_BUDGET_MB - FIXED_OVERHEAD_MB) * 1024 * 1024 // BYTES_PER_FRAME_CHANNEL
# Frames per block when writing the output (~1 MB per channel of float32).
WRITE_BLOCK_FRAMES = 1 << 18


class UnsupportedFormatError(ValueError):
    pass


class AudioTooLongError(ValueError):
    pass


class AudioTooLargeError(ValueError):
    pass


@dataclass
class LoadedAudio:
    samples: np.ndarray  # (n, channels) float32 in roughly [-1, 1]
    sample_rate: int
    source_extension: str
    # libsndfile subtype for lossless sources (e.g. "PCM_16"); None for MP3.
    subtype: str | None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def supported_input_extensions() -> list[str]:
    exts = [".wav", ".flac"]
    if ffmpeg_available():
        exts.append(".mp3")
    return exts


def _check_duration(frames: int, sample_rate: int) -> None:
    if sample_rate <= 0:
        raise UnsupportedFormatError("Could not read a valid sample rate from the file.")
    if frames / sample_rate > MAX_DURATION_S:
        raise AudioTooLongError(
            f"Track is longer than {MAX_DURATION_S // 60} minutes. Split it or trim it before protecting."
        )


def max_duration_for(sample_rate: int, channels: int) -> float:
    """Longest track (seconds) the memory budget admits at this rate / channel count."""
    return MAX_SAMPLE_FRAMES / (max(sample_rate, 1) * max(channels, 1))


def _check_size(frames: int, channels: int, sample_rate: int) -> None:
    # Every channel costs the same as a frame of mono: stereo counts 2x.
    channels = max(channels, 1)
    if frames * channels > MAX_SAMPLE_FRAMES:
        limit_s = max_duration_for(sample_rate, channels)
        limit = f"{limit_s / 60:.1f} minutes" if limit_s >= 60 else f"{int(limit_s)} second{'s' if int(limit_s) != 1 else ''}"
        layout = {1: "mono", 2: "stereo"}.get(channels, f"{channels}-channel")
        raise AudioTooLargeError(
            f"Track is too large for this server’s memory budget: about {limit} of "
            f"{layout} audio at {sample_rate} Hz is the most it can protect in one go. "
            "Trim it, split it, or upload a mono version."
        )


def _ffprobe_stream(path: Path) -> tuple[int, int, int]:
    """Return (sample_rate, channels, approx_frames) via ffprobe. frames may be 0 if unknown.

    Do NOT use stream duration_ts alone as frames: on MP3 it is in the stream
    time_base (often 1/14112000), not sample ticks — that falsely rejects short clips.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,duration",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise UnsupportedFormatError("ffprobe could not read this MP3. The file may be corrupt.")
    info = json.loads(result.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise UnsupportedFormatError("No audio stream found in this MP3.")
    s = streams[0]
    sr = int(float(s.get("sample_rate") or 0))
    ch = int(s.get("channels") or 0)
    dur = s.get("duration") or (info.get("format") or {}).get("duration")
    frames = int(float(dur) * sr) if dur and sr else 0
    if sr <= 0 or ch <= 0:
        raise UnsupportedFormatError("Could not read sample rate/channels from this MP3.")
    return sr, ch, frames


def _decode_mp3_f32(path: Path) -> tuple[np.ndarray, int]:
    """Decode MP3 to float32 via ffmpeg pipe (no giant temp WAV in memory)."""
    sr, ch, frames = _ffprobe_stream(path)
    if frames:
        _check_duration(frames, sr)
        _check_size(frames, ch, sr)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(ch), "-ar", str(sr),
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    if result.returncode != 0 or not result.stdout:
        raise UnsupportedFormatError("ffmpeg could not decode this MP3. The file may be corrupt or not really an MP3.")
    data = np.frombuffer(result.stdout, dtype=np.float32)
    if data.size % ch != 0:
        raise UnsupportedFormatError("Decoded MP3 size did not match channel layout.")
    data = data.reshape(-1, ch).copy()  # detach from readonly buffer
    del result
    _check_duration(data.shape[0], sr)
    _check_size(data.shape[0], ch, sr)
    return data, sr


def load_audio(path: str | Path) -> LoadedAudio:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_INPUT_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{ext or 'unknown'}'. Upload {', '.join(supported_input_extensions())}."
        )

    if ext == ".mp3":
        if not ffmpeg_available():
            raise UnsupportedFormatError("MP3 input needs ffmpeg installed on the server. Upload WAV or FLAC instead.")
        data, sr = _decode_mp3_f32(path)
        return LoadedAudio(samples=data, sample_rate=int(sr), source_extension=ext, subtype=None)

    try:
        info = sf.info(str(path))
    except sf.LibsndfileError as exc:
        raise UnsupportedFormatError(f"Could not read this file as {ext}: {exc}") from exc
    _check_duration(info.frames, info.samplerate)
    _check_size(info.frames, info.channels, info.samplerate)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    subtype = info.subtype if info.subtype in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"} else None
    return LoadedAudio(samples=data, sample_rate=int(sr), source_extension=ext, subtype=subtype)


def output_extension_for(source_extension: str) -> str:
    """Lossless sources keep their container; lossy sources become WAV."""
    return source_extension if source_extension in LOSSLESS_OUTPUT else ".wav"


def save_audio(path: str | Path, samples: np.ndarray, sample_rate: int, subtype: str | None) -> None:
    path = Path(path)
    ext = path.suffix.lower()
    fmt = LOSSLESS_OUTPUT.get(ext)
    if fmt is None:
        raise UnsupportedFormatError(f"Refusing to write lossy or unknown output format '{ext}'.")
    chosen = subtype or "PCM_16"
    if fmt == "FLAC" and chosen not in {"PCM_16", "PCM_24"}:
        chosen = "PCM_24"
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    # Clip and write block by block instead of materialising a clipped copy of
    # the whole track next to the original.
    with sf.SoundFile(str(path), mode="w", samplerate=sample_rate, channels=samples.shape[1], format=fmt, subtype=chosen) as out:
        for start in range(0, samples.shape[0], WRITE_BLOCK_FRAMES):
            block = samples[start : start + WRITE_BLOCK_FRAMES].astype(np.float32, copy=True)
            np.clip(block, -1.0, 1.0, out=block)
            out.write(block)
