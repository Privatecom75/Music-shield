"""Psychoacoustically-shaped perturbation engine.

The goal is to add a small, deterministic-per-job perturbation to a track that
is hard to hear but changes the spectrogram-level features that audio ML
models consume. This is *friction*, not protection: see LIMITS.md.

Three components are combined:

1. Masked noise — pseudo-random noise whose per-bin amplitude sits under a
   simplified masking curve derived from the track's own spectrum. Loud
   regions can hide more noise than quiet ones, so the noise "follows" the
   music instead of being a flat hiss.
2. Spectral jitter — a slowly drifting, band-wise gain wobble (a time-varying
   micro-EQ). This is multiplicative rather than additive, so it survives
   simple noise-subtraction better than component 1 alone.
3. High-band perturbation — extra energy in the mostly-inaudible top of the
   spectrum (above ~15 kHz) when the sample rate allows. Cheap friction that
   many feature extractors still see; disappears if the file is downsampled.

The masking model here is deliberately simple (band energies, a fixed offset,
neighbour spreading). It is not an MPEG psychoacoustic model and it makes no
guarantee about inaudibility on every track.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

N_FFT = 2048
HOP = N_FFT // 4
PEAK_CEILING = 0.995
ABSOLUTE_FLOOR_DB = -85.0
NUM_BANDS = 32
HF_BAND_START_HZ = 15_000.0


@dataclass(frozen=True)
class Preset:
    """Tuning knobs for one strength level."""

    name: str
    label: str
    description: str
    # Noise sits this many dB *below* the estimated masking curve.
    noise_offset_db: float
    # Peak-to-peak size of the band gain wobble, in dB.
    jitter_db: float
    # Level of the high-band component relative to full scale, in dB.
    hf_level_db: float
    # Smoothing window for the jitter random walk, in seconds.
    jitter_smooth_s: float


PRESETS: dict[str, Preset] = {
    "light": Preset(
        name="light",
        label="Light (default)",
        description="Conservative. Aims to stay below what most listeners notice on normal playback.",
        noise_offset_db=-12.0,
        jitter_db=0.4,
        hf_level_db=-48.0,
        jitter_smooth_s=0.8,
    ),
    "medium": Preset(
        name="medium",
        label="Medium",
        description="More change to the spectrogram. May be faintly audible on quiet, sparse passages.",
        noise_offset_db=-7.0,
        jitter_db=0.9,
        hf_level_db=-40.0,
        jitter_smooth_s=0.5,
    ),
    "strong": Preset(
        name="strong",
        label="Strong",
        description="Trades listening quality for more disruption. Expect audible texture on headphones.",
        noise_offset_db=-3.0,
        jitter_db=1.5,
        hf_level_db=-32.0,
        jitter_smooth_s=0.3,
    ),
}

DEFAULT_PRESET = "light"


@dataclass
class ProtectStats:
    """Measurements about *how much* the file changed. Not efficacy metrics."""

    preset: str
    sample_rate: int
    channels: int
    duration_s: float
    # Signal-to-perturbation ratio in dB: higher means a smaller change.
    snr_db: float
    original_peak: float
    protected_peak: float
    # True when the output had to be scaled down to avoid clipping.
    peak_limited: bool
    hf_component_applied: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _hann(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:-1].astype(np.float32)


def _stft(x: np.ndarray) -> np.ndarray:
    """Return an (n_bins, n_frames) complex STFT with a periodic Hann window."""
    win = _hann(N_FFT)
    pad = N_FFT
    xp = np.pad(x, (pad, pad + N_FFT))
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = xp[idx] * win[None, :]
    return np.fft.rfft(frames, axis=1).T


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    win = _hann(N_FFT)
    frames = np.fft.irfft(spec.T, n=N_FFT, axis=1) * win[None, :]
    n_frames = frames.shape[0]
    out_len = N_FFT + HOP * (n_frames - 1)
    out = np.zeros(out_len)
    norm = np.zeros(out_len)
    win_sq = win**2
    for i in range(n_frames):
        start = i * HOP
        out[start : start + N_FFT] += frames[i]
        norm[start : start + N_FFT] += win_sq
    norm[norm < 1e-8] = 1.0
    out /= norm
    pad = N_FFT
    return out[pad : pad + length]


def _band_edges(sample_rate: int, n_bins: int) -> np.ndarray:
    """Log-spaced band edges (in bin indices) from ~50 Hz to Nyquist."""
    nyq = sample_rate / 2.0
    freqs = np.geomspace(50.0, nyq, NUM_BANDS + 1)
    edges = np.round(freqs / nyq * (n_bins - 1)).astype(int)
    edges[0] = 1
    edges[-1] = n_bins
    edges = np.maximum.accumulate(edges)
    return edges


def _bin_to_band(edges: np.ndarray, n_bins: int) -> np.ndarray:
    band_of_bin = np.zeros(n_bins, dtype=int)
    for b in range(NUM_BANDS):
        band_of_bin[edges[b] : edges[b + 1]] = b
    return band_of_bin


def _masking_curve(power: np.ndarray, edges: np.ndarray, band_of_bin: np.ndarray) -> np.ndarray:
    """Estimate a per-bin masking power from the signal's own band energies.

    Simplified model: mean band power, spread to neighbours at -10 dB, offset
    by -20 dB, then floored at an absolute threshold. Returned as *power*.
    """
    n_bins, n_frames = power.shape
    band_power = np.zeros((NUM_BANDS, n_frames))
    for b in range(NUM_BANDS):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            band_power[b] = power[lo:hi].mean(axis=0)
    spread = band_power.copy()
    neighbour_gain = 10 ** (-10 / 10)
    spread[1:] = np.maximum(spread[1:], band_power[:-1] * neighbour_gain)
    spread[:-1] = np.maximum(spread[:-1], band_power[1:] * neighbour_gain)
    masked = spread * 10 ** (-20 / 10)
    floor = 10 ** (ABSOLUTE_FLOOR_DB / 10)
    masked = np.maximum(masked, floor)
    return masked[band_of_bin]


def _smooth_walk(rng: np.random.Generator, n_frames: int, smooth_frames: int, n_series: int) -> np.ndarray:
    """Smoothed Gaussian noise per series, normalised to roughly unit std."""
    raw = rng.standard_normal((n_series, n_frames + 2 * smooth_frames))
    kernel = np.ones(max(1, smooth_frames)) / max(1, smooth_frames)
    smoothed = np.stack([np.convolve(r, kernel, mode="same") for r in raw])
    smoothed = smoothed[:, smooth_frames : smooth_frames + n_frames]
    std = smoothed.std(axis=1, keepdims=True)
    std[std < 1e-9] = 1.0
    return smoothed / std


def _perturb_channel(x: np.ndarray, sample_rate: int, preset: Preset, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    n = len(x)
    spec = _stft(x)
    n_bins, n_frames = spec.shape
    power = np.abs(spec) ** 2
    edges = _band_edges(sample_rate, n_bins)
    band_of_bin = _bin_to_band(edges, n_bins)

    # 1. Masked noise, shaped under the estimated masking curve.
    mask_power = _masking_curve(power, edges, band_of_bin)
    noise_power = mask_power * 10 ** (preset.noise_offset_db / 10)
    # Never inject noise into bins where the track itself is essentially silent
    # (digital silence at intro/outro should stay silent).
    silent = power < 10 ** (ABSOLUTE_FLOOR_DB / 10)
    noise_power[silent] = 0.0
    phase = rng.uniform(0, 2 * math.pi, size=spec.shape)
    # Mask and noise are both in STFT units, so no window scaling is needed.
    noise_spec = np.sqrt(noise_power) * np.exp(1j * phase)

    # 2. Spectral jitter: slowly drifting band gains (multiplicative).
    frames_per_s = sample_rate / HOP
    smooth_frames = max(1, int(preset.jitter_smooth_s * frames_per_s))
    walk = _smooth_walk(rng, n_frames, smooth_frames, NUM_BANDS)
    gain_db = np.clip(walk * (preset.jitter_db / 2.0), -preset.jitter_db, preset.jitter_db)
    gain = 10 ** (gain_db / 20)
    spec_jittered = spec * gain[band_of_bin]

    y = _istft(spec_jittered + noise_spec, n)

    # 3. High-band perturbation, only if the sample rate actually has room.
    hf_applied = False
    hf_start_bin = int(HF_BAND_START_HZ / (sample_rate / 2.0) * (n_bins - 1))
    if hf_start_bin < n_bins - 4:
        hf_applied = True
        frame_silent = silent.all(axis=0)
        hf_spec = np.zeros_like(spec)
        hf_phase = rng.uniform(0, 2 * math.pi, size=(n_bins - hf_start_bin, n_frames))
        # Follow the track's own loudness so the component ducks in quiet parts.
        envelope = np.sqrt(power.mean(axis=0))
        envelope = envelope / (envelope.max() + 1e-12)
        envelope[frame_silent] = 0.0
        hf_spec[hf_start_bin:] = envelope[None, :] * np.exp(1j * hf_phase)
        hf = _istft(hf_spec, n)
        # Calibrate against the track's RMS so the preset level means the same
        # thing regardless of sample rate or FFT size.
        track_rms = math.sqrt(float(np.mean(x**2))) + 1e-12
        hf_rms = math.sqrt(float(np.mean(hf**2))) + 1e-12
        target_rms = track_rms * 10 ** (preset.hf_level_db / 20)
        y = y + hf * (target_rms / hf_rms)

    return y, hf_applied


def protect(audio: np.ndarray, sample_rate: int, preset_name: str = DEFAULT_PRESET, seed: int | None = None) -> tuple[np.ndarray, ProtectStats]:
    """Apply the perturbation to a (samples, channels) float array in [-1, 1].

    Returns the protected audio (same shape, float32) and change statistics.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Choose one of: {', '.join(PRESETS)}")
    preset = PRESETS[preset_name]
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError("audio must have shape (samples,) or (samples, channels)")
    if audio.shape[0] < N_FFT:
        raise ValueError(f"Track is too short to process (need at least {N_FFT} samples).")
    if not np.isfinite(audio).all():
        raise ValueError("Input audio contains NaN or infinite samples.")

    x = np.ascontiguousarray(audio, dtype=np.float32)
    base_rng = np.random.default_rng(seed)
    channel_seeds = base_rng.integers(0, 2**63 - 1, size=x.shape[1])

    out = np.empty_like(x)
    hf_any = False
    for ch in range(x.shape[1]):
        rng = np.random.default_rng(int(channel_seeds[ch]))
        out[:, ch], hf = _perturb_channel(x[:, ch], sample_rate, preset, rng)
        hf_any = hf_any or hf

    original_peak = float(np.abs(x).max()) if x.size else 0.0
    peak = float(np.abs(out).max()) if out.size else 0.0
    peak_limited = False
    if peak > PEAK_CEILING:
        # Scale the whole file rather than hard-clipping individual samples.
        out *= PEAK_CEILING / peak
        peak_limited = True
        peak = float(np.abs(out).max())

    diff = out - x
    sig_energy = float(np.sum(x**2))
    diff_energy = float(np.sum(diff**2))
    if diff_energy <= 0:
        snr_db = float("inf")
    elif sig_energy <= 0:
        snr_db = float("-inf")
    else:
        snr_db = 10 * math.log10(sig_energy / diff_energy)

    stats = ProtectStats(
        preset=preset.name,
        sample_rate=int(sample_rate),
        channels=int(x.shape[1]),
        duration_s=round(x.shape[0] / sample_rate, 3),
        snr_db=round(snr_db, 2) if math.isfinite(snr_db) else snr_db,
        original_peak=round(original_peak, 4),
        protected_peak=round(peak, 4),
        peak_limited=peak_limited,
        hf_component_applied=hf_any,
    )
    return out, stats
