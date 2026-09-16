"""Measure how much of the perturbation is still present after a lossy re-encode.

Everything in this module measures *change remaining*, not protection
efficacy. A large residual means the protected file is still different from
the original after the codec; it does not mean any model is affected by that
difference. See LIMITS.md and METRICS.md.

Pipeline for one preset / bitrate:

    original ─── protect ───▶ protected
        │                        │
        ▼ codec                  ▼ codec
    codec(original)         codec(protected)

Four comparisons are reported:

    A  protected            vs original          the change we added
    B  codec(original)      vs original          codec-only control
    C  codec(protected)     vs original          change + codec error together
    D  codec(protected)     vs codec(original)   change that survived the codec

D is the honest "survives re-encode" number: if the codec threw the
perturbation away, codec(protected) collapses onto codec(original) and D goes
to zero. C is what a listener or downstream tool actually sees relative to
the master, but it mixes our change with the codec's own error, so it is
reported for completeness rather than as the headline.

Metrics per comparison:

    residual_db   RMS of (candidate - reference) relative to the reference RMS,
                  in dB (negative; closer to 0 means a bigger change). This is
                  -SNR, chosen so "bigger number = more change remaining".
    logmel_db     mean |log-mel difference| in dB over 0-15 kHz and 64 mel
                  bands, only on frames/bands where the reference has energy.
                  Restricting to <15 kHz keeps codec low-pass behaviour out
                  of the number so it reflects the mid/low-band structure.
    logmel_{low,mid,high}_db
                  the same split into <500 Hz, 500 Hz-4 kHz, 4-15 kHz.

Encoding uses ffmpeg (libmp3lame for MP3, the native encoder for AAC). Both
encoders and ffmpeg's decoders are gapless-aware, but alignment is verified
by cross-correlation and corrected if needed, so a codec delay can never be
mistaken for perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from music_shield.perturb import DEFAULT_PRESET, PRESETS, protect
from music_shield.synth import generate_busy_clip, generate_demo_clip, generate_tone

MEL_BANDS = 64
MEL_FMAX_HZ = 15_000.0
MEL_N_FFT = 2048
MEL_HOP = 512
# Bins whose reference log-mel energy is more than this far under the frame's
# loudest band are ignored: log differences on near-silent bands are noise.
MEL_DYNAMIC_RANGE_DB = 60.0
REGION_EDGES_HZ = (0.0, 500.0, 4000.0, MEL_FMAX_HZ)
MAX_ALIGN_LAG = 8192


class CodecUnavailableError(RuntimeError):
    pass


def ffmpeg_encoder_available(encoder: str) -> bool:
    """True if ffmpeg exists and lists `encoder` (e.g. 'libmp3lame')."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return any(line.split()[1:2] == [encoder] for line in result.stdout.splitlines() if line.strip())


CODECS: dict[str, tuple[str, str, str]] = {
    # name: (ffmpeg encoder, container extension, human label)
    "mp3": ("libmp3lame", ".mp3", "MP3 (libmp3lame, CBR)"),
    "aac": ("aac", ".m4a", "AAC-LC (ffmpeg native)"),
}


def _align(candidate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, int]:
    """Trim/pad `candidate` to `reference` length, correcting any codec delay.

    Returns (aligned candidate, lag in samples) where a positive lag means the
    decoded audio arrived late and was shifted earlier.
    """
    n = reference.shape[0]
    ref_mono = reference.mean(axis=1)
    cand_mono = candidate.mean(axis=1)
    m = min(len(ref_mono), len(cand_mono))
    size = 1
    while size < 2 * m:
        size *= 2
    corr = np.fft.irfft(np.fft.rfft(cand_mono[:m], size) * np.conj(np.fft.rfft(ref_mono[:m], size)), size)
    lags = np.concatenate([np.arange(0, MAX_ALIGN_LAG + 1), np.arange(-MAX_ALIGN_LAG, 0)])
    window = np.concatenate([corr[: MAX_ALIGN_LAG + 1], corr[-MAX_ALIGN_LAG:]])
    lag = int(lags[np.argmax(window)])
    if lag > 0:
        candidate = candidate[lag:]
    elif lag < 0:
        candidate = np.concatenate([np.zeros((-lag, candidate.shape[1])), candidate])
    if candidate.shape[0] < n:
        candidate = np.concatenate([candidate, np.zeros((n - candidate.shape[0], candidate.shape[1]))])
    return candidate[:n], lag


def codec_roundtrip(audio: np.ndarray, sample_rate: int, codec: str = "mp3", bitrate_kbps: int = 128) -> tuple[np.ndarray, int]:
    """Encode `audio` with ffmpeg at `bitrate_kbps`, decode it, and align it.

    Returns (decoded float64 array with the input's shape, alignment lag).
    """
    if codec not in CODECS:
        raise ValueError(f"Unknown codec '{codec}'. Choose one of: {', '.join(CODECS)}")
    encoder, ext, _ = CODECS[codec]
    if not ffmpeg_encoder_available(encoder):
        raise CodecUnavailableError(f"ffmpeg with the '{encoder}' encoder is required for {codec} round-trips.")
    if audio.ndim == 1:
        audio = audio[:, None]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        enc = Path(tmp) / f"enc{ext}"
        dec = Path(tmp) / "out.wav"
        sf.write(str(src), audio.astype(np.float64), sample_rate, format="WAV", subtype="FLOAT")
        common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        encode = common + ["-i", str(src), "-codec:a", encoder, "-b:a", f"{bitrate_kbps}k", str(enc)]
        subprocess.run(encode, check=True, capture_output=True, timeout=600)
        decode = common + ["-i", str(enc), "-f", "wav", "-acodec", "pcm_f32le", str(dec)]
        subprocess.run(decode, check=True, capture_output=True, timeout=600)
        decoded, sr = sf.read(str(dec), dtype="float64", always_2d=True)
    if sr != sample_rate:
        raise RuntimeError(f"Decoder returned {sr} Hz for a {sample_rate} Hz input.")
    if decoded.shape[1] != audio.shape[1]:
        raise RuntimeError("Decoder returned a different channel count.")
    return _align(decoded, audio)


def mp3_roundtrip(audio: np.ndarray, sample_rate: int, bitrate_kbps: int = 128) -> np.ndarray:
    return codec_roundtrip(audio, sample_rate, "mp3", bitrate_kbps)[0]


def residual_db(candidate: np.ndarray, reference: np.ndarray) -> float:
    """RMS of the difference relative to the reference RMS, in dB (= -SNR)."""
    ref_energy = float(np.sum(reference**2))
    diff_energy = float(np.sum((candidate - reference) ** 2))
    if ref_energy <= 0:
        return float("nan")
    if diff_energy <= 0:
        return float("-inf")
    return 10 * math.log10(diff_energy / ref_energy)


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangular mel filters (n_mels, n_bins) and each filter's centre in Hz."""
    fmax = min(fmax, sample_rate / 2.0)
    n_bins = n_fft // 2 + 1
    bin_hz = np.linspace(0, sample_rate / 2.0, n_bins)
    mel_points = np.linspace(_hz_to_mel(0.0), _hz_to_mel(fmax), n_mels + 2)
    hz_points = np.asarray(_mel_to_hz(mel_points))
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lo, mid, hi = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (bin_hz - lo) / max(mid - lo, 1e-9)
        falling = (hi - bin_hz) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0.0, 1.0)
    return fb, hz_points[1:-1]


def _log_mel(x: np.ndarray, sample_rate: int, fb: np.ndarray) -> np.ndarray:
    """(n_mels, n_frames) log-mel power in dB of a mono signal."""
    win = np.hanning(MEL_N_FFT + 1)[:-1]
    n_frames = 1 + max(0, (len(x) - MEL_N_FFT) // MEL_HOP)
    idx = np.arange(MEL_N_FFT)[None, :] + MEL_HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * win[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = fb @ power.T
    return 10 * np.log10(mel + 1e-12)


@dataclass
class Distance:
    """How far `candidate` is from `reference`. Change remaining, not efficacy."""

    residual_db: float
    logmel_db: float
    logmel_low_db: float
    logmel_mid_db: float
    logmel_high_db: float

    def as_dict(self) -> dict:
        return {k: (round(v, 2) if math.isfinite(v) else v) for k, v in asdict(self).items()}


def distance(candidate: np.ndarray, reference: np.ndarray, sample_rate: int) -> Distance:
    if candidate.ndim == 1:
        candidate = candidate[:, None]
    if reference.ndim == 1:
        reference = reference[:, None]
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have the same shape")
    fb, centres = _mel_filterbank(sample_rate, MEL_N_FFT, MEL_BANDS, MEL_FMAX_HZ)
    diffs = []
    masks = []
    for ch in range(reference.shape[1]):
        ref_mel = _log_mel(reference[:, ch], sample_rate, fb)
        cand_mel = _log_mel(candidate[:, ch], sample_rate, fb)
        frame_max = ref_mel.max(axis=0, keepdims=True)
        active = ref_mel > (frame_max - MEL_DYNAMIC_RANGE_DB)
        diffs.append(np.abs(cand_mel - ref_mel))
        masks.append(active)
    diff = np.concatenate(diffs, axis=1)
    mask = np.concatenate(masks, axis=1)

    def region_mean(lo_hz: float, hi_hz: float) -> float:
        rows = (centres >= lo_hz) & (centres < hi_hz)
        sel = mask[rows]
        if not sel.any():
            return float("nan")
        return float(diff[rows][sel].mean())

    return Distance(
        residual_db=residual_db(candidate, reference),
        logmel_db=float(diff[mask].mean()) if mask.any() else float("nan"),
        logmel_low_db=region_mean(REGION_EDGES_HZ[0], REGION_EDGES_HZ[1]),
        logmel_mid_db=region_mean(REGION_EDGES_HZ[1], REGION_EDGES_HZ[2]),
        logmel_high_db=region_mean(REGION_EDGES_HZ[2], REGION_EDGES_HZ[3]),
    )


@dataclass
class RoundtripReport:
    preset: str
    codec: str
    bitrate_kbps: int
    snr_db: float
    protected_vs_original: Distance
    codec_original_vs_original: Distance
    codec_protected_vs_original: Distance
    codec_protected_vs_codec_original: Distance
    # Fraction (linear amplitude) of the *specific* change we added that is
    # still present after the codec: the least-squares projection of D onto
    # A, <D,A>/<A,A>. 1.0 = the codec passed our change through untouched,
    # 0.0 = the codec removed it. Codec noise is uncorrelated with A, so unlike
    # RMS(D)/RMS(A) this is not inflated by the codec re-randomising its own
    # quantisation error between the two encodes.
    retained_fraction: float
    # How much of the difference between the two decoded files is actually
    # our change (correlation of D with A). Low values mean D is mostly codec
    # noise that happened to land differently.
    retained_correlation: float
    decoded_peak: float
    alignment_lag_samples: int

    def as_dict(self) -> dict:
        d = asdict(self)
        for key in ("protected_vs_original", "codec_original_vs_original", "codec_protected_vs_original", "codec_protected_vs_codec_original"):
            d[key] = getattr(self, key).as_dict()
        d["retained_fraction"] = round(self.retained_fraction, 3)
        d["retained_correlation"] = round(self.retained_correlation, 3)
        d["decoded_peak"] = round(self.decoded_peak, 4)
        return d


def evaluate_roundtrip(
    original: np.ndarray,
    sample_rate: int,
    preset: str = DEFAULT_PRESET,
    codec: str = "mp3",
    bitrate_kbps: int = 128,
    seed: int = 1234,
) -> RoundtripReport:
    if original.ndim == 1:
        original = original[:, None]
    original = original.astype(np.float64)
    protected, stats = protect(original, sample_rate, preset, seed=seed)
    codec_original, _ = codec_roundtrip(original, sample_rate, codec, bitrate_kbps)
    codec_protected, lag = codec_roundtrip(protected, sample_rate, codec, bitrate_kbps)

    a = distance(protected, original, sample_rate)
    d = distance(codec_protected, codec_original, sample_rate)
    change = protected - original
    change_after = codec_protected - codec_original
    energy_a = float(np.sum(change**2))
    energy_d = float(np.sum(change_after**2))
    dot = float(np.sum(change * change_after))
    retained = dot / energy_a if energy_a > 0 else float("nan")
    corr = dot / math.sqrt(energy_a * energy_d) if energy_a > 0 and energy_d > 0 else float("nan")
    return RoundtripReport(
        preset=preset,
        codec=codec,
        bitrate_kbps=bitrate_kbps,
        snr_db=stats.snr_db,
        protected_vs_original=a,
        codec_original_vs_original=distance(codec_original, original, sample_rate),
        codec_protected_vs_original=distance(codec_protected, original, sample_rate),
        codec_protected_vs_codec_original=d,
        retained_fraction=retained,
        retained_correlation=corr,
        decoded_peak=float(np.abs(codec_protected).max()),
        alignment_lag_samples=lag,
    )


BUILTIN_CLIPS = ("busy", "demo", "tone")


def load_clip(name_or_path: str, sample_rate: int = 44100) -> tuple[np.ndarray, int]:
    """Built-in synthetic clips by name, or any WAV/FLAC path."""
    if name_or_path == "busy":
        return generate_busy_clip(sample_rate), sample_rate
    if name_or_path == "demo":
        return generate_demo_clip(sample_rate), sample_rate
    if name_or_path == "tone":
        return generate_tone(5.0, sample_rate), sample_rate
    data, sr = sf.read(name_or_path, dtype="float64", always_2d=True)
    return data, int(sr)


def _fmt(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}" if math.isfinite(v) else str(v)


def markdown_table(reports: list[RoundtripReport], clip_label: str) -> str:
    lines = [
        f"Clip: `{clip_label}`. Every number is *change remaining*, not protection efficacy.",
        "",
        "| preset | codec | SNR of change | A: protected vs orig | B: codec(orig) vs orig | C: codec(prot) vs orig | D: codec(prot) vs codec(orig) | retained | corr | D log-mel low / mid / high |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        a, b, c, d = (
            r.protected_vs_original,
            r.codec_original_vs_original,
            r.codec_protected_vs_original,
            r.codec_protected_vs_codec_original,
        )
        lines.append(
            "| {p} | {codec} {br}k | {snr} dB | {a_res} dB / {a_mel} dB | {b_res} dB / {b_mel} dB | {c_res} dB / {c_mel} dB | {d_res} dB / {d_mel} dB | {ret:.0%} | {corr} | {lo} / {mid} / {hi} dB |".format(
                p=r.preset,
                codec=r.codec,
                br=r.bitrate_kbps,
                snr=_fmt(r.snr_db),
                a_res=_fmt(a.residual_db),
                a_mel=_fmt(a.logmel_db, 2),
                b_res=_fmt(b.residual_db),
                b_mel=_fmt(b.logmel_db, 2),
                c_res=_fmt(c.residual_db),
                c_mel=_fmt(c.logmel_db, 2),
                d_res=_fmt(d.residual_db),
                d_mel=_fmt(d.logmel_db, 2),
                ret=r.retained_fraction,
                corr=_fmt(r.retained_correlation, 2),
                lo=_fmt(d.logmel_low_db, 2),
                mid=_fmt(d.logmel_mid_db, 2),
                hi=_fmt(d.logmel_high_db, 2),
            )
        )
    lines.append("")
    lines.append(
        "Cells are `residual dB / mean |log-mel difference| dB (0-15 kHz)`. Residual dB is "
        "RMS(difference) / RMS(reference); closer to 0 means a bigger difference. "
        "`retained` is the least-squares fraction of the added change (A) still present in D; "
        "`corr` is how much of D is that change rather than re-randomised codec noise."
    )
    return "\n".join(lines)


def run_report(
    clip: str = "busy",
    presets: list[str] | None = None,
    codecs: list[str] | None = None,
    bitrates: list[int] | None = None,
    seed: int = 1234,
) -> tuple[list[RoundtripReport], str]:
    presets = presets or list(PRESETS)
    codecs = codecs or ["mp3"]
    bitrates = bitrates or [192, 128]
    audio, sr = load_clip(clip)
    reports = []
    for preset in presets:
        for codec in codecs:
            for br in bitrates:
                reports.append(evaluate_roundtrip(audio, sr, preset, codec, br, seed=seed))
    return reports, markdown_table(reports, clip)
