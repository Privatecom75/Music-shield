"""Psychoacoustically-shaped perturbation engine.

The goal is to add a small, deterministic-per-job perturbation to a track that
is hard to hear but changes the features that audio ML models consume. This is
*friction*, not protection: see LIMITS.md. METRICS.md has measurements of how
much of the change survives a lossy re-encode.

Four components are combined:

1. Spectral jitter — a slowly drifting gain wobble (a time-varying micro-EQ)
   defined on ~30 knots spread over a warped frequency scale and interpolated
   smoothly across bins, so it has no hard band edges. It is *multiplicative*
   on the track's own content: a codec that reproduces the loud parts of the
   spectrum accurately also reproduces the wobble, which is why this survives
   MP3/AAC far better than additive noise.
2. Phase drift — a slow, per-knot rotation of the STFT phase (a time-varying
   all-pass). Human hearing is nearly insensitive to slow monaural phase
   changes and the drift is shared across channels so the stereo image does
   not move, yet the waveform and the complex spectrum change substantially
   and lossy codecs reproduce the change faithfully. Magnitude spectrograms
   are almost untouched by this component; it targets waveform / complex-
   spectrum features, not mel features.
3. Masked noise — pseudo-random noise whose per-bin amplitude sits under a
   simplified masking curve derived from the track's own spectrum. Loud
   regions can hide more noise than quiet ones, so the noise "follows" the
   music instead of being a flat hiss. Lossy codecs replace much of this with
   their own quantisation noise; it is kept small at `light`.
4. High-band component — extra masked noise above ~15 kHz, at a smaller
   offset under the masking curve, when the sample rate allows. Cheap
   friction that some feature extractors still see, but it is removed by the
   low-pass filter in every common lossy encoder and is not counted on for
   anything after a re-encode. Because it is mask-shaped, a track with no
   content above 15 kHz gets nothing added there.

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
# Masking spread into neighbouring bands. Bands are ~0.27 octave (0.6 Bark at
# 300 Hz, 1-2 Bark in the mids). Real masking slopes are roughly -25 dB/Bark
# upward and steeper downward, so these are still on the generous side. Both
# were -10 dB, which put noise into empty bands next to loud ones at a level
# the audibility proxies flagged on sparse material.
NEIGHBOUR_SPREAD_UP_DB = -18.0
NEIGHBOUR_SPREAD_DOWN_DB = -27.0
# Frames of backward running-minimum applied to the masking curve so noise
# cannot precede an attack (3 hops = the earlier frames overlapping a window).
PRE_ECHO_FRAMES = 3

# Jitter / phase knots: at least this far apart in Hz at the bottom of the
# spectrum (wider than the STFT main lobe, so one low partial is not split
# across several knots) and this many per octave higher up.
KNOT_MIN_SPACING_HZ = 150.0
KNOTS_PER_OCTAVE = 6.0
KNOT_START_HZ = 50.0
# Phase drift is faded in between these frequencies; DC and sub-bass keep
# their phase so the rotation never turns into a DC offset.
PHASE_FADE_LO_HZ = 20.0
PHASE_FADE_HI_HZ = 50.0


@dataclass(frozen=True)
class Preset:
    """Tuning knobs for one strength level."""

    name: str
    label: str
    description: str
    # Noise sits this many dB *below* the estimated masking curve.
    noise_offset_db: float
    # Peak size of the interpolated gain wobble, in dB (reached at 2 sigma).
    jitter_db: float
    # Peak size of the phase drift, in degrees (reached at 2 sigma).
    phase_deg: float
    # The >15 kHz component sits this many dB below the masking curve (a
    # smaller offset than the main noise: hearing is poor up there).
    hf_offset_db: float
    # Smoothing window for the jitter / phase random walks, in seconds.
    jitter_smooth_s: float


PRESETS: dict[str, Preset] = {
    "light": Preset(
        name="light",
        label="Light (default)",
        description="Conservative. Aims to stay below what most listeners notice on normal playback.",
        noise_offset_db=-14.0,
        jitter_db=0.6,
        phase_deg=3.0,
        hf_offset_db=-10.0,
        jitter_smooth_s=0.8,
    ),
    "medium": Preset(
        name="medium",
        label="Medium",
        description="More change to the spectrogram. May be faintly audible as a slow EQ wobble on sustained or sparse passages.",
        noise_offset_db=-12.0,
        jitter_db=1.3,
        phase_deg=8.0,
        hf_offset_db=-8.0,
        jitter_smooth_s=0.5,
    ),
    "strong": Preset(
        name="strong",
        label="Strong",
        description="Trades listening quality for more disruption. Expect audible texture on headphones.",
        noise_offset_db=-8.0,
        jitter_db=2.2,
        phase_deg=14.0,
        hf_offset_db=-4.0,
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


def _n_frames_for(length: int) -> int:
    return 1 + (length + 3 * N_FFT - N_FFT) // HOP


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


def _masking_curve(power: np.ndarray, edges: np.ndarray, band_of_bin: np.ndarray, pre_echo_frames: int = PRE_ECHO_FRAMES) -> np.ndarray:
    """Estimate a per-bin masking power from the signal's own band energies.

    Simplified model: mean band power, spread to neighbours (-18 dB upward, -27 dB downward), offset
    by -20 dB, then floored at an absolute threshold. Returned as *power*.
    `pre_echo_frames=0` gives the instantaneous estimate (used by the
    audibility proxies); the default applies the synthesis-side pre-echo limit.
    """
    n_bins, n_frames = power.shape
    band_power = np.zeros((NUM_BANDS, n_frames))
    for b in range(NUM_BANDS):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            band_power[b] = power[lo:hi].mean(axis=0)
    spread = band_power.copy()
    # Masking spreads upward in frequency more than downward.
    spread[1:] = np.maximum(spread[1:], band_power[:-1] * 10 ** (NEIGHBOUR_SPREAD_UP_DB / 10))
    spread[:-1] = np.maximum(spread[:-1], band_power[1:] * 10 ** (NEIGHBOUR_SPREAD_DOWN_DB / 10))
    masked = spread * 10 ** (-20 / 10)
    # Pre-echo control: a frame's noise is synthesised over its whole window,
    # so an attack near the end of the window would otherwise get noise
    # ~40 ms *before* it, in the quiet, where backward masking cannot hide it.
    # Limit each frame's mask by the earlier frames that overlap its span.
    limited = masked.copy()
    for k in range(1, pre_echo_frames + 1):
        limited[:, k:] = np.minimum(limited[:, k:], masked[:, :-k])
    floor = 10 ** (ABSOLUTE_FLOOR_DB / 10)
    limited = np.maximum(limited, floor)
    return limited[band_of_bin]


def _smooth_walk(rng: np.random.Generator, n_frames: int, smooth_frames: int, n_series: int) -> np.ndarray:
    """Smoothed Gaussian noise per series, normalised to roughly unit std.

    A Hann kernel (rather than a boxcar) keeps the walk's derivative smooth,
    so the modulation has no fast, tremolo-like components.
    """
    width = max(2, 2 * smooth_frames)
    raw = rng.standard_normal((n_series, n_frames + 2 * width))
    kernel = np.hanning(width + 2)[1:-1]
    kernel /= kernel.sum()
    smoothed = np.stack([np.convolve(r, kernel, mode="same") for r in raw])
    smoothed = smoothed[:, width : width + n_frames]
    std = smoothed.std(axis=1, keepdims=True)
    std[std < 1e-9] = 1.0
    return smoothed / std


def knot_frequencies(sample_rate: int) -> np.ndarray:
    """Knot centres in Hz for the jitter / phase components.

    Walks up from KNOT_START_HZ taking steps of at least KNOT_MIN_SPACING_HZ
    (so low partials are not split across knots) and at most 1/KNOTS_PER_OCTAVE
    octave. Ends at Nyquist.
    """
    nyq = sample_rate / 2.0
    ratio = 2 ** (1.0 / KNOTS_PER_OCTAVE)
    knots = [KNOT_START_HZ]
    while True:
        f = knots[-1]
        nxt = max(f + KNOT_MIN_SPACING_HZ, f * ratio)
        if nxt >= nyq:
            break
        knots.append(nxt)
    knots.append(nyq)
    return np.asarray(knots)


def _knot_interpolation(sample_rate: int, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-bin (lower knot index, upper knot index, weight of upper knot)."""
    knots = knot_frequencies(sample_rate)
    bin_hz = np.arange(n_bins) * (sample_rate / 2.0) / (n_bins - 1)
    log_bins = np.log2(np.maximum(bin_hz, 1.0))
    log_knots = np.log2(knots)
    pos = np.interp(log_bins, log_knots, np.arange(len(knots), dtype=float))
    lo = np.floor(pos).astype(int)
    lo = np.clip(lo, 0, len(knots) - 2)
    w = np.clip(pos - lo, 0.0, 1.0)
    return lo, lo + 1, w, len(knots)


def _modulation(rng: np.random.Generator, sample_rate: int, n_bins: int, n_frames: int, preset: Preset) -> np.ndarray:
    """Complex per-bin, per-frame multiplier combining gain jitter and phase drift.

    Shared by all channels of a file so that the stereo image is not modulated.
    """
    lo, hi, w, n_knots = _knot_interpolation(sample_rate, n_bins)
    frames_per_s = sample_rate / HOP
    smooth_frames = max(1, int(preset.jitter_smooth_s * frames_per_s))

    gain_walk = _smooth_walk(rng, n_frames, smooth_frames, n_knots)
    gain_db = np.clip(gain_walk * (preset.jitter_db / 2.0), -preset.jitter_db, preset.jitter_db)
    gain_db_bins = gain_db[lo] * (1.0 - w)[:, None] + gain_db[hi] * w[:, None]

    phase_walk = _smooth_walk(rng, n_frames, smooth_frames, n_knots)
    phase_deg = np.clip(phase_walk * (preset.phase_deg / 2.0), -preset.phase_deg, preset.phase_deg)
    phase_bins = np.deg2rad(phase_deg[lo] * (1.0 - w)[:, None] + phase_deg[hi] * w[:, None])
    bin_hz = np.arange(n_bins) * (sample_rate / 2.0) / (n_bins - 1)
    fade = np.clip((bin_hz - PHASE_FADE_LO_HZ) / (PHASE_FADE_HI_HZ - PHASE_FADE_LO_HZ), 0.0, 1.0)
    fade[0] = 0.0
    fade[-1] = 0.0
    phase_bins *= fade[:, None]

    return 10 ** (gain_db_bins / 20) * np.exp(1j * phase_bins)


def _perturb_channel(
    x: np.ndarray, sample_rate: int, preset: Preset, rng: np.random.Generator, modulation: np.ndarray
) -> tuple[np.ndarray, bool]:
    n = len(x)
    spec = _stft(x)
    n_bins, n_frames = spec.shape
    power = np.abs(spec) ** 2
    edges = _band_edges(sample_rate, n_bins)
    band_of_bin = _bin_to_band(edges, n_bins)

    # 1 + 2. Spectral jitter and phase drift (multiplicative, shared across channels).
    spec_mod = spec * modulation

    # 3. Masked noise, shaped under the estimated masking curve.
    mask_power = _masking_curve(power, edges, band_of_bin)
    noise_power = mask_power * 10 ** (preset.noise_offset_db / 10)
    # Never inject noise into bins where the track itself is essentially silent
    # (digital silence at intro/outro should stay silent).
    silent = power < 10 ** (ABSOLUTE_FLOOR_DB / 10)
    noise_power[silent] = 0.0
    phase = rng.uniform(0, 2 * math.pi, size=spec.shape)
    # Mask and noise are both in STFT units, so no window scaling is needed.
    noise_spec = np.sqrt(noise_power) * np.exp(1j * phase)

    y = _istft(spec_mod + noise_spec, n)

    # 4. High-band component: extra masked noise above 15 kHz, at a smaller
    # offset under the same masking estimate (hearing is poor up there). It
    # used to be RMS-calibrated and ignored the mask, which put energy into
    # empty air bands on sparse material; shaping it under the mask means the
    # whole additive part of the perturbation sits under the mask estimate by
    # construction, and a track with nothing above 15 kHz gets nothing added.
    hf_applied = False
    hf_start_bin = int(HF_BAND_START_HZ / (sample_rate / 2.0) * (n_bins - 1))
    if hf_start_bin < n_bins - 4:
        hf_power = mask_power[hf_start_bin:] * 10 ** (preset.hf_offset_db / 10)
        hf_power[silent[hf_start_bin:]] = 0.0
        # Only bins whose own content is above the absolute floor got noise,
        # so the mask floor alone never creates a high band from nothing.
        if hf_power.any():
            hf_applied = True
            hf_phase = rng.uniform(0, 2 * math.pi, size=hf_power.shape)
            hf_spec = np.zeros_like(spec)
            hf_spec[hf_start_bin:] = np.sqrt(hf_power) * np.exp(1j * hf_phase)
            y = y + _istft(hf_spec, n)

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
    modulation_seed = int(base_rng.integers(0, 2**63 - 1))
    channel_seeds = base_rng.integers(0, 2**63 - 1, size=x.shape[1])

    n_bins = N_FFT // 2 + 1
    n_frames = _n_frames_for(x.shape[0])
    modulation = _modulation(np.random.default_rng(modulation_seed), sample_rate, n_bins, n_frames, preset)

    out = np.empty_like(x)
    hf_any = False
    for ch in range(x.shape[1]):
        rng = np.random.default_rng(int(channel_seeds[ch]))
        out[:, ch], hf = _perturb_channel(x[:, ch], sample_rate, preset, rng, modulation)
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
