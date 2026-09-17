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
import scipy.fft

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


# ---------------------------------------------------------------------------
# Streaming STFT engine.
#
# The track is never turned into one big (frames x bins) matrix. The slow
# random walks that drive the jitter / phase drift are planned for the whole
# track up front (they are tiny: ~35 knots x frames), and each channel is then
# processed CHUNK_FRAMES analysis frames at a time: frame -> rfft -> shape ->
# irfft -> overlap-add straight into the float32 output array. Everything is
# float32 / complex64. The peak working set of a job is therefore the input,
# the output, and a few chunk-sized buffers, whatever the track length. The
# only per-chunk state carried across chunk borders is the last
# PRE_ECHO_FRAMES frames of the masking estimate.
# ---------------------------------------------------------------------------

# Analysis frames per chunk. 512 frames is ~6 s at 44.1 kHz and keeps the
# per-chunk working set (complex64 spectrum, float32 power / mask / noise /
# modulation buffers) around 20-40 MB. Fixed, not adaptive, so a given seed
# always produces the same output.
CHUNK_FRAMES = 512
# Frames per block for the streaming input scan / statistics (~1 MB/channel).
_STATS_BLOCK_FRAMES = 1 << 18
_OVERLAP = N_FFT // HOP
_ABSOLUTE_FLOOR_POWER = 10 ** (ABSOLUTE_FLOOR_DB / 10)


def _hann(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:-1].astype(np.float32)


_WINDOW = _hann(N_FFT)
# Periodic Hann at 75 % overlap: sum of the squared, shifted windows is the
# constant 1.5 everywhere the signal is covered by all _OVERLAP frames, which
# the N_FFT of zero padding on each side guarantees for every real sample.
_OLA_NORM_PERIOD = (_WINDOW.astype(np.float64) ** 2).reshape(_OVERLAP, HOP).sum(axis=0)
assert np.ptp(_OLA_NORM_PERIOD) < 1e-5, "window / hop pair is not constant-overlap-add"
_OLA_NORM = float(_OLA_NORM_PERIOD.mean())
_SYNTH_WINDOW = (_WINDOW / _OLA_NORM).astype(np.float32)


def _n_frames_for(length: int) -> int:
    """Number of analysis frames for a signal padded by N_FFT before and 2*N_FFT after."""
    return 1 + (length + 2 * N_FFT) // HOP


def _frames_for_chunk(x: np.ndarray, f0: int, f1: int) -> np.ndarray:
    """Windowed analysis frames f0..f1-1 of the virtually zero-padded signal.

    Returns an (f1 - f0, N_FFT) float32 array. Only the slice of `x` that the
    chunk touches is copied; the zero padding is materialised per chunk.
    """
    m = f1 - f0
    start = f0 * HOP - N_FFT  # first sample needed, in x coordinates
    span = (m - 1) * HOP + N_FFT
    seg = np.zeros(span, dtype=np.float32)
    a, b = max(start, 0), min(start + span, len(x))
    if b > a:
        seg[a - start : b - start] = x[a:b]
    frames = np.lib.stride_tricks.sliding_window_view(seg, N_FFT)[::HOP]
    return frames * _WINDOW


def _stft_chunk(x: np.ndarray, f0: int, f1: int) -> np.ndarray:
    """(f1 - f0, n_bins) complex64 spectrum of analysis frames f0..f1-1."""
    return scipy.fft.rfft(_frames_for_chunk(x, f0, f1), axis=1)


def _overlap_add_chunk(spec: np.ndarray, f0: int, out: np.ndarray) -> None:
    """Inverse-transform a (m, n_bins) chunk and add it in place into `out` (1-D)."""
    frames = scipy.fft.irfft(spec, n=N_FFT, axis=1)
    frames *= _SYNTH_WINDOW
    m = frames.shape[0]
    span = (m - 1) * HOP + N_FFT
    seg = np.zeros(span, dtype=np.float32)
    # Frame i occupies seg[i*HOP : i*HOP + N_FFT]; split every frame into
    # _OVERLAP hop-sized lanes so the overlap-add is _OVERLAP vector adds
    # instead of a Python loop over frames.
    lanes = frames.reshape(m, _OVERLAP, HOP)
    for k in range(_OVERLAP):
        seg[k * HOP : k * HOP + m * HOP] += lanes[:, k, :].reshape(-1)
    start = f0 * HOP - N_FFT
    a, b = max(start, 0), min(start + span, len(out))
    if b > a:
        out[a:b] += seg[a - start : b - start]


def _stft(x: np.ndarray) -> np.ndarray:
    """Full (n_bins, n_frames) complex64 STFT, assembled chunk by chunk.

    Kept for the offline audibility proxies, which want the whole spectrogram;
    `protect` itself never builds one.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n_frames = _n_frames_for(len(x))
    n_bins = N_FFT // 2 + 1
    spec = np.empty((n_frames, n_bins), dtype=np.complex64)
    for f0 in range(0, n_frames, CHUNK_FRAMES):
        f1 = min(f0 + CHUNK_FRAMES, n_frames)
        spec[f0:f1] = _stft_chunk(x, f0, f1)
    return spec.T


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


def _masked_band_power(power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-frame masking power per band, before the pre-echo limit and floor.

    `power` is (frames, bins). Simplified model: mean band power, spread to
    neighbours (-18 dB upward, -27 dB downward), offset by -20 dB.
    """
    n_frames = power.shape[0]
    band_power = np.zeros((n_frames, NUM_BANDS), dtype=np.float32)
    for b in range(NUM_BANDS):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            band_power[:, b] = power[:, lo:hi].mean(axis=1)
    spread = band_power.copy()
    # Masking spreads upward in frequency more than downward.
    np.maximum(spread[:, 1:], band_power[:, :-1] * np.float32(10 ** (NEIGHBOUR_SPREAD_UP_DB / 10)), out=spread[:, 1:])
    np.maximum(spread[:, :-1], band_power[:, 1:] * np.float32(10 ** (NEIGHBOUR_SPREAD_DOWN_DB / 10)), out=spread[:, :-1])
    spread *= np.float32(10 ** (-20 / 10))
    return spread


def _pre_echo_limit(
    masked: np.ndarray, prev_masked: np.ndarray | None, pre_echo_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Backward running minimum over `pre_echo_frames` frames, then the absolute floor.

    `prev_masked` holds the unlimited mask of the last frames of the previous
    chunk so the running minimum sees across chunk borders exactly as it would
    on the whole track. Returns (limited mask for this chunk, state for the
    next chunk).
    """
    if prev_masked is not None and len(prev_masked):
        stacked = np.concatenate([prev_masked, masked], axis=0)
        skip = len(prev_masked)
    else:
        stacked, skip = masked, 0
    # Pre-echo control: a frame's noise is synthesised over its whole window,
    # so an attack near the end of the window would otherwise get noise
    # ~40 ms *before* it, in the quiet, where backward masking cannot hide it.
    # Limit each frame's mask by the earlier frames that overlap its span.
    limited = stacked.copy()
    for k in range(1, pre_echo_frames + 1):
        np.minimum(limited[k:], stacked[:-k], out=limited[k:])
    limited = limited[skip:]
    np.maximum(limited, np.float32(_ABSOLUTE_FLOOR_POWER), out=limited)
    carry = stacked[len(stacked) - pre_echo_frames :] if pre_echo_frames else stacked[:0]
    return limited, carry.copy()


def _masking_curve(
    power: np.ndarray, edges: np.ndarray, band_of_bin: np.ndarray, pre_echo_frames: int = PRE_ECHO_FRAMES
) -> np.ndarray:
    """Per-bin masking power for a whole (n_bins, n_frames) power spectrogram.

    Whole-track convenience wrapper used by the audibility proxies; `protect`
    runs the same model chunk by chunk. `pre_echo_frames=0` gives the
    instantaneous estimate.
    """
    masked = _masked_band_power(np.ascontiguousarray(power.T, dtype=np.float32), edges)
    limited, _ = _pre_echo_limit(masked, None, pre_echo_frames)
    return limited[:, band_of_bin].T


def _smooth_walk(rng: np.random.Generator, n_frames: int, smooth_frames: int, n_series: int) -> np.ndarray:
    """Smoothed Gaussian noise per series, normalised to roughly unit std.

    A Hann kernel (rather than a boxcar) keeps the walk's derivative smooth,
    so the modulation has no fast, tremolo-like components.
    """
    width = max(2, 2 * smooth_frames)
    # float32 throughout: the walks are (n_series, n_frames) for the whole
    # track, the only whole-track arrays the engine keeps besides the audio.
    raw = rng.standard_normal((n_series, n_frames + 2 * width), dtype=np.float32)
    kernel = np.hanning(width + 2)[1:-1].astype(np.float32)
    kernel /= kernel.sum()
    smoothed = np.stack([np.convolve(r, kernel, mode="same") for r in raw])
    del raw
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


@dataclass
class _ModulationPlan:
    """Whole-track jitter / phase-drift walks at the knots, expanded per chunk.

    Shared by all channels of a file so that the stereo image is not modulated.
    The per-knot walks are (n_knots, n_frames) and tiny; the per-bin complex
    multiplier is only ever built for one chunk of frames at a time.
    """

    gain_db: np.ndarray  # (n_knots, n_frames) float32
    phase_rad: np.ndarray  # (n_knots, n_frames) float32
    lo: np.ndarray  # (n_bins,) lower knot per bin
    hi: np.ndarray  # (n_bins,) upper knot per bin
    w: np.ndarray  # (n_bins,) float32 weight of the upper knot
    fade: np.ndarray  # (n_bins,) float32 phase fade-in below PHASE_FADE_HI_HZ

    @property
    def n_frames(self) -> int:
        return self.gain_db.shape[1]

    def chunk(self, f0: int, f1: int) -> np.ndarray:
        """(f1 - f0, n_bins) complex64 multiplier for frames f0..f1-1."""
        one_minus_w = np.float32(1.0) - self.w
        g = self.gain_db[:, f0:f1].T
        gain_db_bins = g[:, self.lo] * one_minus_w + g[:, self.hi] * self.w
        p = self.phase_rad[:, f0:f1].T
        phase_bins = (p[:, self.lo] * one_minus_w + p[:, self.hi] * self.w) * self.fade
        mag = np.float32(10) ** (gain_db_bins / np.float32(20))
        mod = np.empty(gain_db_bins.shape, dtype=np.complex64)
        mod.real = mag * np.cos(phase_bins)
        mod.imag = mag * np.sin(phase_bins)
        return mod


def _plan_modulation(rng: np.random.Generator, sample_rate: int, n_bins: int, n_frames: int, preset: Preset) -> _ModulationPlan:
    lo, hi, w, n_knots = _knot_interpolation(sample_rate, n_bins)
    frames_per_s = sample_rate / HOP
    smooth_frames = max(1, int(preset.jitter_smooth_s * frames_per_s))

    gain_walk = _smooth_walk(rng, n_frames, smooth_frames, n_knots)
    gain_db = np.clip(gain_walk * (preset.jitter_db / 2.0), -preset.jitter_db, preset.jitter_db)

    phase_walk = _smooth_walk(rng, n_frames, smooth_frames, n_knots)
    phase_deg = np.clip(phase_walk * (preset.phase_deg / 2.0), -preset.phase_deg, preset.phase_deg)
    bin_hz = np.arange(n_bins) * (sample_rate / 2.0) / (n_bins - 1)
    fade = np.clip((bin_hz - PHASE_FADE_LO_HZ) / (PHASE_FADE_HI_HZ - PHASE_FADE_LO_HZ), 0.0, 1.0)
    fade[0] = 0.0
    fade[-1] = 0.0
    return _ModulationPlan(
        gain_db=gain_db.astype(np.float32, copy=False),
        phase_rad=np.deg2rad(phase_deg).astype(np.float32, copy=False),
        lo=lo,
        hi=hi,
        w=w.astype(np.float32),
        fade=fade.astype(np.float32),
    )


def _add_random_phase_noise(spec: np.ndarray, noise_power: np.ndarray, rng: np.random.Generator) -> None:
    """Add noise of the given per-bin power with uniformly random phase, in place."""
    phase = rng.uniform(0, 2 * math.pi, size=noise_power.shape).astype(np.float32, copy=False)
    amp = np.sqrt(noise_power)
    spec.real += amp * np.cos(phase)
    spec.imag += amp * np.sin(phase)


def _perturb_channel(
    x: np.ndarray, out: np.ndarray, sample_rate: int, preset: Preset, rng: np.random.Generator, plan: _ModulationPlan
) -> bool:
    """Protect one channel, chunk by chunk, overlap-adding into `out` (zeroed, same length).

    Returns whether the >15 kHz component was applied anywhere.
    """
    n_bins = N_FFT // 2 + 1
    edges = _band_edges(sample_rate, n_bins)
    band_of_bin = _bin_to_band(edges, n_bins)
    noise_gain = np.float32(10 ** (preset.noise_offset_db / 10))
    hf_gain = np.float32(10 ** (preset.hf_offset_db / 10))
    hf_start_bin = int(HF_BAND_START_HZ / (sample_rate / 2.0) * (n_bins - 1))
    hf_possible = hf_start_bin < n_bins - 4
    hf_applied = False
    prev_masked: np.ndarray | None = None

    for f0 in range(0, plan.n_frames, CHUNK_FRAMES):
        f1 = min(f0 + CHUNK_FRAMES, plan.n_frames)
        spec = _stft_chunk(x, f0, f1)  # (m, n_bins) complex64
        power = spec.real**2 + spec.imag**2
        # Never inject noise into bins where the track itself is essentially
        # silent (digital silence at intro/outro should stay silent).
        silent = power < np.float32(_ABSOLUTE_FLOOR_POWER)
        masked = _masked_band_power(power, edges)
        del power
        limited, prev_masked = _pre_echo_limit(masked, prev_masked, PRE_ECHO_FRAMES)
        mask_power = limited[:, band_of_bin]  # (m, n_bins) float32

        # 1 + 2. Spectral jitter and phase drift (multiplicative, shared across channels).
        spec *= plan.chunk(f0, f1)

        # 3. Masked noise, shaped under the estimated masking curve. Mask and
        # noise are both in STFT units, so no window scaling is needed.
        noise_power = mask_power * noise_gain
        noise_power[silent] = 0.0
        _add_random_phase_noise(spec, noise_power, rng)
        del noise_power

        # 4. High-band component: extra masked noise above 15 kHz, at a smaller
        # offset under the same masking estimate (hearing is poor up there).
        # Shaping it under the mask means the whole additive part of the
        # perturbation sits under the mask estimate by construction, and a
        # track with nothing above 15 kHz gets nothing added. Synthesis is
        # linear, so it is summed into the same chunk spectrum.
        if hf_possible:
            hf_power = mask_power[:, hf_start_bin:] * hf_gain
            hf_power[silent[:, hf_start_bin:]] = 0.0
            # Only bins whose own content is above the absolute floor got noise,
            # so the mask floor alone never creates a high band from nothing.
            if hf_power.any():
                hf_applied = True
                _add_random_phase_noise(spec[:, hf_start_bin:], hf_power, rng)
            del hf_power
        del mask_power, silent

        _overlap_add_chunk(spec, f0, out)
        del spec
    return hf_applied


def _block_ranges(n: int):
    for start in range(0, n, _STATS_BLOCK_FRAMES):
        yield start, min(start + _STATS_BLOCK_FRAMES, n)


def _scan_input(x: np.ndarray) -> tuple[bool, float, float]:
    """(all finite, peak, energy) of a (samples, channels) array, in blocks."""
    peak = 0.0
    energy = 0.0
    for a, b in _block_ranges(x.shape[0]):
        blk = x[a:b]
        if not np.isfinite(blk).all():
            return False, peak, energy
        if blk.size:
            peak = max(peak, float(blk.max()), float(-blk.min()))
            blk64 = blk.astype(np.float64)
            energy += float(np.einsum("ij,ij->", blk64, blk64))
    return True, peak, energy


def _peak(x: np.ndarray) -> float:
    peak = 0.0
    for a, b in _block_ranges(x.shape[0]):
        blk = x[a:b]
        if blk.size:
            peak = max(peak, float(blk.max()), float(-blk.min()))
    return peak


def _diff_energy(x: np.ndarray, y: np.ndarray) -> float:
    energy = 0.0
    for a, b in _block_ranges(x.shape[0]):
        d = y[a:b].astype(np.float64)
        d -= x[a:b]
        energy += float(np.einsum("ij,ij->", d, d))
    return energy


def protect(audio: np.ndarray, sample_rate: int, preset_name: str = DEFAULT_PRESET, seed: int | None = None) -> tuple[np.ndarray, ProtectStats]:
    """Apply the perturbation to a (samples, channels) float array in [-1, 1].

    Returns the protected audio (same shape, float32) and change statistics.
    Peak memory is the input plus an output of the same size plus a few tens
    of MB of chunk buffers, independent of track length (see LIMITS.md).
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

    x = np.ascontiguousarray(audio, dtype=np.float32)
    finite, original_peak, sig_energy = _scan_input(x)
    if not finite:
        raise ValueError("Input audio contains NaN or infinite samples.")

    base_rng = np.random.default_rng(seed)
    modulation_seed = int(base_rng.integers(0, 2**63 - 1))
    channel_seeds = base_rng.integers(0, 2**63 - 1, size=x.shape[1])

    n_bins = N_FFT // 2 + 1
    n_frames = _n_frames_for(x.shape[0])
    plan = _plan_modulation(np.random.default_rng(modulation_seed), sample_rate, n_bins, n_frames, preset)

    out = np.zeros_like(x)
    hf_any = False
    for ch in range(x.shape[1]):
        rng = np.random.default_rng(int(channel_seeds[ch]))
        hf = _perturb_channel(x[:, ch], out[:, ch], sample_rate, preset, rng, plan)
        hf_any = hf_any or hf

    peak = _peak(out)
    peak_limited = False
    if peak > PEAK_CEILING:
        # Scale the whole file rather than hard-clipping individual samples.
        out *= np.float32(PEAK_CEILING / peak)
        peak_limited = True
        peak = _peak(out)

    diff_energy = _diff_energy(x, out)
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
