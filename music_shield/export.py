"""Optional lossy exports of an already-protected lossless file.

The protect pipeline always writes lossless WAV/FLAC (see audio_io). This
module turns that stored file into a convenience copy with ffmpeg:

    mp3   libmp3lame, MP3_BITRATE_KBPS
    mp4   H.264 video (a generated still cover: gradient + waveform bars +
          optional title text) with AAC audio at AAC_BITRATE_KBPS

Both are *lossy*. Re-encoding removes the >15 kHz component, buries the masked
noise under codec noise, and adds codec error larger than the whole
perturbation; the gain/phase wobble mostly persists (METRICS.md) but the net
result is less friction than the lossless file. The lossless download stays
the default; these exist for distribution convenience.

Memory: ffmpeg reads the WAV/FLAC from disk itself, so the Python process
never holds the track. The cover image is rendered from the file in small
blocks (a few MB at most). Measured on a 200 s stereo 44.1 kHz file: ffmpeg
peaks at ~55 MiB for MP3 and ~100 MiB for MP4 (720p still, 2 fps, x264
ultrafast, one thread).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

import numpy as np
import soundfile as sf

log = logging.getLogger("music_shield.export")

MP3_BITRATE_KBPS = 192
AAC_BITRATE_KBPS = 192

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 2
# Still image: every frame after the first is a skip frame, so the preset
# mostly decides RAM (ultrafast ~95 MiB vs veryfast ~150 MiB at 720p) and the
# CRF only affects the keyframes.
VIDEO_X264_PRESET = "ultrafast"
VIDEO_CRF = 18
VIDEO_THREADS = 1
COVER_BAR_COUNT = 96
COVER_TITLE_MAX_CHARS = 60
EXPORT_TIMEOUT_S = 600

# Fonts tried for the title overlay, first hit wins. MUSIC_SHIELD_COVER_FONT
# overrides. Without a font (or without drawtext) the video is produced with
# no text, never refused.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


class ExportError(RuntimeError):
    pass


class ExportUnavailableError(ExportError):
    """ffmpeg or a needed encoder is missing on this server."""


class ExportFailedError(ExportError):
    """ffmpeg ran and failed."""


@dataclass(frozen=True)
class ExportFormat:
    name: str
    label: str
    extension: str
    media_type: str
    lossy: bool
    encoders: tuple[str, ...]
    bitrate_kbps: int | None
    note: str


FORMATS: dict[str, ExportFormat] = {
    "wav": ExportFormat("wav", "WAV", ".wav", "audio/wav", False, (), None, "Lossless. Default; keeps the whole perturbation."),
    "flac": ExportFormat("flac", "FLAC", ".flac", "audio/flac", False, (), None, "Lossless. Default when the upload was FLAC."),
    "mp3": ExportFormat(
        "mp3",
        f"MP3 · {MP3_BITRATE_KBPS} kbps",
        ".mp3",
        "audio/mpeg",
        True,
        ("libmp3lame",),
        MP3_BITRATE_KBPS,
        "Lossy. Re-encoding weakens the perturbation; for convenience and distribution only.",
    ),
    "mp4": ExportFormat(
        "mp4",
        f"MP4 video · H.264 + AAC {AAC_BITRATE_KBPS} kbps",
        ".mp4",
        "video/mp4",
        True,
        ("libx264", "aac"),
        AAC_BITRATE_KBPS,
        "Lossy. Protected audio under a simple generated cover (gradient, waveform bars, title). "
        "Re-encoding weakens the perturbation; for convenience and distribution only.",
    ),
}
LOSSLESS_NAMES = tuple(f.name for f in FORMATS.values() if not f.lossy)
LOSSY_NAMES = tuple(f.name for f in FORMATS.values() if f.lossy)


# --- capability probes -----------------------------------------------------


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@lru_cache(maxsize=4)
def _ffmpeg_list(flag: str) -> str:
    if not ffmpeg_available():
        return ""
    try:
        result = subprocess.run(["ffmpeg", "-hide_banner", flag], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def ffmpeg_encoders() -> set[str]:
    names: set[str] = set()
    for line in _ffmpeg_list("-encoders").splitlines():
        parts = line.split()
        # " A....D libmp3lame  libmp3lame MP3 ..." -> flags, name, description;
        # the legend lines look like " V..... = Video" and are skipped.
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS" and parts[1] != "=":
            names.add(parts[1])
    return names


def ffmpeg_has_filter(name: str) -> bool:
    for line in _ffmpeg_list("-filters").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False


def unavailable_reason(name: str) -> str | None:
    """None if `name` can be produced here, else a one-line reason for the client."""
    spec = FORMATS[name]
    if not spec.lossy:
        return None
    if not ffmpeg_available():
        return f"{spec.label.split(' ·')[0]} export needs ffmpeg installed on the server."
    missing = [e for e in spec.encoders if e not in ffmpeg_encoders()]
    if missing:
        return f"{spec.label.split(' ·')[0]} export needs ffmpeg built with {', '.join(missing)} on the server."
    return None


def title_font() -> str | None:
    override = os.environ.get("MUSIC_SHIELD_COVER_FONT")
    candidates = ([override] if override else []) + list(FONT_CANDIDATES)
    for path in candidates:
        if path and Path(path).is_file():
            return path
    return None


def title_overlay_available() -> bool:
    return ffmpeg_available() and ffmpeg_has_filter("drawtext") and title_font() is not None


def describe_formats() -> list[dict]:
    """For /api/info and the UI: what can be downloaded here and what it costs."""
    out = []
    for spec in FORMATS.values():
        reason = unavailable_reason(spec.name)
        out.append(
            {
                "name": spec.name,
                "label": spec.label,
                "extension": spec.extension,
                "lossy": spec.lossy,
                "bitrate_kbps": spec.bitrate_kbps,
                "available": reason is None,
                "unavailable_reason": reason,
                "note": spec.note,
            }
        )
    return out


# --- cover image -------------------------------------------------------------


def waveform_bars(audio_path: Path, count: int = COVER_BAR_COUNT) -> np.ndarray:
    """Per-segment RMS of the mono downmix, normalised to [0, 1], read block by block."""
    info = sf.info(str(audio_path))
    frames = max(int(info.frames), 1)
    seg = max(1, math.ceil(frames / count))
    bars = np.zeros(count, dtype=np.float32)
    i = 0
    with sf.SoundFile(str(audio_path)) as f:
        while i < count:
            block = f.read(seg, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                break
            mono = block.mean(axis=1)
            bars[i] = float(np.sqrt(np.mean(mono * mono)))
            i += 1
    peak = float(bars.max())
    if peak > 0:
        bars /= peak
    # Perceptual-ish curve so quiet passages are still visible.
    return np.sqrt(bars)


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (h, w, 3) uint8 array as an 8-bit RGB PNG (filter 0, no deps)."""
    h, w, _ = rgb.shape
    rows = np.concatenate([np.zeros((h, 1), dtype=np.uint8), rgb.reshape(h, w * 3)], axis=1)
    raw = rows.tobytes()

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def render_cover(audio_path: Path, png_path: Path, width: int = VIDEO_WIDTH, height: int = VIDEO_HEIGHT) -> Path:
    """Soft gradient with mirrored waveform bars in the site's palette. ~10 MB of numpy, briefly."""
    bars = waveform_bars(audio_path)

    top = np.array([23, 53, 43], dtype=np.float32)  # #17352b
    bottom = np.array([14, 17, 22], dtype=np.float32)  # #0e1116
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    img = top * (1.0 - t) + bottom * t
    img = np.broadcast_to(img, (height, width, 3)).copy()

    accent = np.array([110, 231, 183], dtype=np.float32)  # #6ee7b7
    mid = height * 0.46
    max_half = height * 0.22
    span = width * 0.76
    left = (width - span) / 2.0
    slot = span / len(bars)
    bar_w = max(2, int(slot * 0.62))
    for k, level in enumerate(bars):
        half = max(2, int(max_half * float(level)))
        x0 = int(left + k * slot + (slot - bar_w) / 2.0)
        y0, y1 = int(mid - half), int(mid + half)
        img[y0:y1, x0 : x0 + bar_w] = img[y0:y1, x0 : x0 + bar_w] * 0.25 + accent * 0.75
    img[int(mid) : int(mid) + 1, int(left) : int(left + span)] = accent * 0.6 + img[int(mid) : int(mid) + 1, int(left) : int(left + span)] * 0.4

    rgb = np.clip(img, 0, 255).astype(np.uint8)
    png_path.write_bytes(_png_bytes(rgb))
    return png_path


# --- encoders ----------------------------------------------------------------


def _run_ffmpeg(cmd: list[str], dst: Path) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=EXPORT_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        dst.unlink(missing_ok=True)
        raise ExportFailedError("ffmpeg timed out.") from exc
    except OSError as exc:
        dst.unlink(missing_ok=True)
        raise ExportUnavailableError("ffmpeg could not be started.") from exc
    if result.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)
        tail = result.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
        log.error("ffmpeg export failed (rc=%s): %s", result.returncode, " | ".join(tail))
        raise ExportFailedError("ffmpeg failed to encode the export.")


def encode_mp3(src: Path, dst: Path) -> Path:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-c:a", "libmp3lame", "-b:a", f"{MP3_BITRATE_KBPS}k",
        str(dst),
    ]
    _run_ffmpeg(cmd, dst)
    return dst


def _drawtext_value(value: str) -> str:
    """Quote a drawtext option value for the filtergraph parser."""
    escaped = value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    return f"'{escaped}'"


def _title_filter(title_file: Path, caption_file: Path, font: str) -> str:
    common = f"fontfile={_drawtext_value(font)}"
    title = (
        f"drawtext={common}:textfile={_drawtext_value(str(title_file))}:fontcolor=white:fontsize=52"
        ":x=(w-text_w)/2:y=h*0.76"
    )
    caption = (
        f"drawtext={common}:textfile={_drawtext_value(str(caption_file))}:fontcolor=0x98a4b5:fontsize=26"
        ":x=(w-text_w)/2:y=h*0.76+72"
    )
    return f"{title},{caption}"


def encode_mp4(src: Path, dst: Path, title: str | None, workdir: Path | None = None) -> Path:
    """Still cover + protected audio -> H.264/AAC MP4 with faststart."""
    workdir = workdir or dst.parent
    cover = render_cover(src, workdir / "cover.png")
    duration = float(sf.info(str(src)).duration or 0.0)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-framerate", str(VIDEO_FPS), "-i", str(cover),
        "-i", str(src),
    ]
    if title and title_overlay_available():
        font = title_font()
        title_file = workdir / "cover-title.txt"
        caption_file = workdir / "cover-caption.txt"
        title_file.write_text(title.strip()[:COVER_TITLE_MAX_CHARS] or "Protected track", encoding="utf-8")
        caption_file.write_text("Protected with Music Shield", encoding="utf-8")
        cmd += ["-vf", _title_filter(title_file, caption_file, font)]  # type: ignore[arg-type]
    cmd += [
        "-c:v", "libx264", "-tune", "stillimage", "-preset", VIDEO_X264_PRESET, "-crf", str(VIDEO_CRF),
        "-pix_fmt", "yuv420p", "-threads", str(VIDEO_THREADS),
        "-c:a", "aac", "-b:a", f"{AAC_BITRATE_KBPS}k",
        "-shortest", "-movflags", "+faststart",
    ]
    if duration > 0:
        # -shortest alone can overrun a looped image input by a frame or two.
        cmd += ["-t", f"{duration:.3f}"]
    cmd.append(str(dst))
    _run_ffmpeg(cmd, dst)
    return dst


def export_lossy(src: Path, dst: Path, name: str, title: str | None = None) -> Path:
    """Encode the stored lossless file `src` to lossy format `name` at `dst`."""
    if name not in LOSSY_NAMES:
        raise ValueError(f"'{name}' is not a lossy export format.")
    reason = unavailable_reason(name)
    if reason:
        raise ExportUnavailableError(reason)
    if name == "mp3":
        return encode_mp3(Path(src), Path(dst))
    return encode_mp4(Path(src), Path(dst), title)
