"""Synthetic test signals used by the CLI demo and the test-suite."""

from __future__ import annotations

import numpy as np


def generate_tone(duration_s: float = 3.0, sample_rate: int = 44100, stereo: bool = True) -> np.ndarray:
    """A small chord with a slow amplitude envelope, peaking around -6 dBFS."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    freqs = (220.0, 277.18, 329.63, 440.0)
    tone = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate(freqs))
    tone /= np.abs(tone).max()
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t - np.pi / 2)
    mono = 0.5 * tone * envelope
    if not stereo:
        return mono[:, None]
    right = 0.5 * np.roll(tone, sample_rate // 100) * envelope
    return np.stack([mono, right], axis=1)


DEMO_SAMPLE_RATE = 44100
DEMO_DURATION_S = 5.0


def generate_demo_clip(sample_rate: int = DEMO_SAMPLE_RATE, duration_s: float = DEMO_DURATION_S) -> np.ndarray:
    """A deterministic, musical-ish stereo clip for the in-app demo.

    Two plucked chords over a soft bass note, plus a quiet band-limited
    noise "pad" so the spectrum is not just a handful of pure tones. This is
    synthetic and free of any third-party material.
    """
    rng = np.random.default_rng(2024)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    out = np.zeros((n, 2))

    # Two chords, each held for half the clip: A minor then F major.
    chords = [(220.0, 261.63, 329.63, 440.0), (174.61, 220.0, 261.63, 349.23)]
    half = n // 2
    for ci, chord in enumerate(chords):
        start = ci * half
        seg_t = t[: n - start] if ci == len(chords) - 1 else t[:half]
        seg = np.zeros((len(seg_t), 2))
        for i, f in enumerate(chord):
            # Pluck: fast attack, exponential decay, gentle harmonics.
            decay = np.exp(-seg_t * 1.4)
            partial = np.sin(2 * np.pi * f * seg_t) + 0.35 * np.sin(2 * np.pi * 2 * f * seg_t) + 0.12 * np.sin(2 * np.pi * 3 * f * seg_t)
            pan = 0.35 + 0.3 * (i / (len(chord) - 1))
            voice = partial * decay / (i + 1.5)
            seg[:, 0] += voice * (1 - pan)
            seg[:, 1] += voice * pan
        out[start : start + len(seg_t)] += seg

    # Sustained bass an octave under each chord root.
    bass = np.where(t < duration_s / 2, np.sin(2 * np.pi * 110.0 * t), np.sin(2 * np.pi * 87.31 * t))
    bass *= 0.25 * (1 - np.exp(-t * 6))
    out += bass[:, None]

    # Quiet band-limited noise pad (roughly 300 Hz to 6 kHz) via FFT masking.
    noise = rng.standard_normal((n, 2))
    spectrum = np.fft.rfft(noise, axis=0)
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)
    band = ((freqs > 300) & (freqs < 6000)).astype(float)[:, None]
    pad = np.fft.irfft(spectrum * band, n=n, axis=0)
    pad /= np.abs(pad).max() + 1e-12
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * t - np.pi / 2)
    out += 0.06 * pad * swell[:, None]

    # Short fades so the clip does not click.
    fade = int(0.02 * sample_rate)
    ramp = np.linspace(0, 1, fade)
    out[:fade] *= ramp[:, None]
    out[-fade:] *= ramp[::-1][:, None]

    out /= np.abs(out).max() + 1e-12
    return 0.6 * out


BUSY_SAMPLE_RATE = 44100
BUSY_DURATION_S = 8.0


def _band_limited_burst(rng: np.random.Generator, length: int, sample_rate: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    noise = rng.standard_normal(length)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(length, 1 / sample_rate)
    spectrum *= (freqs > lo_hz) & (freqs < hi_hz)
    burst = np.fft.irfft(spectrum, length)
    return burst / (np.abs(burst).max() + 1e-12)


def generate_busy_clip(sample_rate: int = BUSY_SAMPLE_RATE, duration_s: float = BUSY_DURATION_S) -> np.ndarray:
    """A dense, broadband, deterministic stereo clip for codec stress tests.

    Saw-like chords with harmonics up to ~16 kHz, a gated bass, and a kick /
    snare / hi-hat pattern at 120 BPM. Unlike the sparse demo clip this keeps
    an MP3 encoder bit-constrained at 128 kbps, so its quantisation noise
    sits near the masking threshold the way it does on real mixes. Entirely
    synthetic; no third-party material.
    """
    rng = np.random.default_rng(7)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    out = np.zeros((n, 2))
    beat = 60.0 / 120.0
    bar = 4 * beat
    max_harmonic_hz = min(16_000.0, sample_rate * 0.45)

    chords = [
        (130.81, 164.81, 196.0, 246.94),
        (146.83, 174.61, 220.0, 261.63),
        (110.0, 130.81, 164.81, 196.0),
        (123.47, 146.83, 185.0, 220.0),
    ]
    bass = np.zeros(n)
    for ci, chord in enumerate(chords):
        start = int(ci * bar * sample_rate)
        stop = min(n, int((ci + 1) * bar * sample_rate))
        if start >= n:
            break
        seg_t = t[: stop - start]
        env = np.minimum(1.0, seg_t / 0.02) * np.exp(-seg_t * 0.35)
        seg = np.zeros((len(seg_t), 2))
        for vi, f in enumerate(chord):
            for side, detune in ((0, 0.998), (1, 1.002)):
                voice = np.zeros(len(seg_t))
                k = 1
                while f * k * detune < max_harmonic_hz:
                    voice += np.sin(2 * np.pi * f * k * detune * seg_t + rng.uniform(0, 2 * np.pi)) / k
                    k += 1
                seg[:, side] += voice * env / (vi + 2)
        out[start:stop] += 0.25 * seg
        root = chord[0] / 2
        for k in (1, 3, 5, 7):
            bass[start:stop] += np.sin(2 * np.pi * root * k * seg_t) / k
    gate = ((t % (beat / 2)) < beat * 0.4).astype(float)
    out += (0.35 * bass * gate)[:, None]

    kick_len = int(0.25 * sample_rate)
    kick_t = t[:kick_len]
    sweep = 45 + 115 * np.exp(-kick_t * 30)
    kick = np.sin(2 * np.pi * np.cumsum(sweep) / sample_rate) * np.exp(-kick_t * 12)
    snare_len = int(0.18 * sample_rate)
    hat_len = int(0.06 * sample_rate)
    for b in range(int(duration_s / beat)):
        s = int(b * beat * sample_rate)
        if s + kick_len <= n:
            out[s : s + kick_len] += 0.7 * kick[:, None]
        if b % 2 == 1 and s + snare_len <= n:
            body = _band_limited_burst(rng, snare_len, sample_rate, 180.0, 7000.0)
            snare = (0.8 * body + 0.5 * np.sin(2 * np.pi * 190 * t[:snare_len])) * np.exp(-t[:snare_len] * 18)
            out[s : s + snare_len] += 0.45 * snare[:, None]
        for sub, pan in ((0.0, 0.35), (0.5, 0.65)):
            hs = int((b + sub) * beat * sample_rate)
            if hs + hat_len > n:
                continue
            hat = _band_limited_burst(rng, hat_len, sample_rate, 5000.0, sample_rate / 2.0) * np.exp(-t[:hat_len] * 60)
            out[hs : hs + hat_len, 0] += 0.25 * hat * (1 - pan)
            out[hs : hs + hat_len, 1] += 0.25 * hat * pan

    fade = int(0.01 * sample_rate)
    ramp = np.linspace(0, 1, fade)
    out[:fade] *= ramp[:, None]
    out[-fade:] *= ramp[::-1][:, None]
    out /= np.abs(out).max() + 1e-12
    return 0.8 * out


SPARSE_DURATION_S = 8.0
SUSTAINED_DURATION_S = 8.0


def generate_sparse_clip(sample_rate: int = 44100, duration_s: float = SPARSE_DURATION_S) -> np.ndarray:
    """Quiet solo plucked notes with gaps between them: the audibility risk case.

    Peaks around -18 dBFS, one note at a time, near-silence between notes, no
    pad or percussion. There is very little energy to hide a perturbation
    under, so this is where masked noise and modulation are most exposed.
    Entirely synthetic.
    """
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    out = np.zeros((n, 2))
    notes = (329.63, 392.0, 440.0, 523.25, 392.0, 293.66, 329.63)
    for i, f in enumerate(notes):
        start = int(i * 1.1 * sample_rate)
        if start >= n:
            break
        seg_t = t[: n - start]
        env = np.minimum(1.0, seg_t / 0.005) * np.exp(-seg_t * 2.2)
        voice = np.sin(2 * np.pi * f * seg_t) + 0.3 * np.sin(2 * np.pi * 2 * f * seg_t) + 0.08 * np.sin(2 * np.pi * 3 * f * seg_t)
        pan = 0.4 + 0.2 * (i % 2)
        out[start:, 0] += voice * env * (1 - pan)
        out[start:, 1] += voice * env * pan
    fade = int(0.01 * sample_rate)
    ramp = np.linspace(0, 1, fade)
    out[:fade] *= ramp[:, None]
    out[-fade:] *= ramp[::-1][:, None]
    out /= np.abs(out).max() + 1e-12
    return 0.125 * out


def generate_sustained_clip(sample_rate: int = 44100, duration_s: float = SUSTAINED_DURATION_S) -> np.ndarray:
    """Steady organ-like chord over an exposed sustained sub-bass, no transients.

    Steady tones are where a slow gain wobble or phase drift has nothing to
    hide behind in time; the 41 Hz sub-bass sits right at the phase fade-in.
    Entirely synthetic.
    """
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    out = np.zeros((n, 2))
    for i, f in enumerate((164.81, 246.94, 329.63, 493.88)):
        voice = sum(np.sin(2 * np.pi * f * k * t) / (k * k) for k in (1, 2, 3, 4, 6))
        pan = 0.3 + 0.4 * (i / 3)
        out[:, 0] += voice * (1 - pan) / (i + 1.5)
        out[:, 1] += voice * pan / (i + 1.5)
    out += (0.5 * np.sin(2 * np.pi * 41.2 * t))[:, None]
    swell = 0.7 + 0.3 * np.sin(2 * np.pi * 0.15 * t - np.pi / 2)
    out *= swell[:, None]
    fade = int(0.05 * sample_rate)
    ramp = np.linspace(0, 1, fade)
    out[:fade] *= ramp[:, None]
    out[-fade:] *= ramp[::-1][:, None]
    out /= np.abs(out).max() + 1e-12
    return 0.5 * out
