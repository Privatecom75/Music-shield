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
