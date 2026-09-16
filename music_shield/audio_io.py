"""Load and save audio using free tooling only.

WAV and FLAC go through libsndfile (soundfile). MP3 is decoded with ffmpeg
if it is installed; output is always lossless WAV or FLAC because re-encoding
to a lossy codec would partly smooth away the perturbation we just added.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SUPPORTED_INPUT_EXTENSIONS = {".wav", ".flac", ".mp3"}
LOSSLESS_OUTPUT = {".wav": "WAV", ".flac": "FLAC"}
MAX_DURATION_S = 15 * 60


class UnsupportedFormatError(ValueError):
    pass


class AudioTooLongError(ValueError):
    pass


@dataclass
class LoadedAudio:
    samples: np.ndarray  # (n, channels) float64 in roughly [-1, 1]
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
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "decoded.wav"
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-f", "wav", "-acodec", "pcm_f32le", str(wav_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0 or not wav_path.exists():
                raise UnsupportedFormatError("ffmpeg could not decode this MP3. The file may be corrupt or not really an MP3.")
            try:
                data, sr = sf.read(str(wav_path), dtype="float64", always_2d=True)
            except sf.LibsndfileError as exc:
                raise UnsupportedFormatError(f"Decoded MP3 could not be read: {exc}") from exc
        _check_duration(data.shape[0], sr)
        return LoadedAudio(samples=data, sample_rate=int(sr), source_extension=ext, subtype=None)

    try:
        info = sf.info(str(path))
    except sf.LibsndfileError as exc:
        raise UnsupportedFormatError(f"Could not read this file as {ext}: {exc}") from exc
    _check_duration(info.frames, info.samplerate)
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
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
    clipped = np.clip(samples, -1.0, 1.0)
    sf.write(str(path), clipped, sample_rate, format=fmt, subtype=chosen)
