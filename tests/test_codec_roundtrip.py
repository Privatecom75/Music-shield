"""Does the perturbation survive a lossy re-encode?

These tests measure *change remaining* after an ffmpeg MP3 round-trip. They
say nothing about whether any model is affected by that change. The floors
below are documented in METRICS.md; if you tune presets, re-run
`python -m music_shield codec-metrics` and update both together.
"""

from dataclasses import replace

import numpy as np
import pytest

from music_shield import perturb
from music_shield.codec_eval import (
    _align,
    codec_roundtrip,
    distance,
    evaluate_roundtrip,
    ffmpeg_encoder_available,
    residual_db,
)
from music_shield.synth import generate_busy_clip, generate_demo_clip

SR = 44100
needs_mp3 = pytest.mark.skipif(not ffmpeg_encoder_available("libmp3lame"), reason="ffmpeg with libmp3lame not available")

# Documented floors (see METRICS.md). Measured values are well above these on
# the synthetic clips; the margin is there so encoder version differences do
# not make the suite flaky, not because the measured numbers are marginal.
LIGHT_MIN_RETAINED_FRACTION = 0.70
LIGHT_MIN_RESIDUAL_AFTER_MP3_DB = -36.0
# Light must stay conservative: this is the SNR floor for the default preset
# on the synthetic clips (measured ~29-30 dB).
LIGHT_MIN_SNR_DB = 25.0


@pytest.fixture(scope="module")
def busy():
    return generate_busy_clip(SR)


@pytest.fixture(scope="module")
def demo():
    return generate_demo_clip(SR)


def test_residual_metric_basics():
    x = generate_demo_clip(SR)
    assert residual_db(x, x) == float("-inf")
    half = x * 0.5
    assert residual_db(half, x) == pytest.approx(-6.02, abs=0.01)
    d = distance(x, x, SR)
    assert d.logmel_db == 0.0 and d.logmel_mid_db == 0.0


def test_alignment_recovers_a_shift():
    x = generate_demo_clip(SR)
    shifted = np.concatenate([np.zeros((300, 2)), x])[: len(x)]
    aligned, lag = _align(shifted, x)
    assert lag == 300
    assert aligned.shape == x.shape
    assert np.allclose(aligned[: len(x) - 300], x[: len(x) - 300])


@needs_mp3
def test_mp3_roundtrip_is_sample_aligned_and_sane(demo):
    decoded, lag = codec_roundtrip(demo, SR, "mp3", 128)
    assert decoded.shape == demo.shape
    assert abs(lag) <= 2048
    assert np.isfinite(decoded).all()
    # A codec is lossy but not destructive: decoded audio is close to the input.
    assert -40 < residual_db(decoded, demo) < -15


@needs_mp3
@pytest.mark.parametrize("bitrate", [192, 128])
@pytest.mark.parametrize("clip_name", ["busy", "demo"])
def test_light_perturbation_survives_mp3(clip_name, bitrate, busy, demo):
    clip = busy if clip_name == "busy" else demo
    report = evaluate_roundtrip(clip, SR, "light", "mp3", bitrate, seed=1234)

    # Light stays the conservative default.
    assert report.snr_db >= LIGHT_MIN_SNR_DB
    # No NaN / clipping regressions through the codec.
    assert np.isfinite(report.decoded_peak) and report.decoded_peak < 1.0
    # protected != original still holds after the round-trip, by a useful margin.
    d = report.codec_protected_vs_codec_original
    assert d.residual_db > LIGHT_MIN_RESIDUAL_AFTER_MP3_DB
    assert report.retained_fraction >= LIGHT_MIN_RETAINED_FRACTION
    assert np.isfinite(d.logmel_db) and d.logmel_db > 0.0


@needs_mp3
@pytest.mark.parametrize("preset", ["medium", "strong"])
def test_stronger_presets_survive_at_least_as_well(preset, busy):
    light = evaluate_roundtrip(busy, SR, "light", "mp3", 128, seed=1234)
    other = evaluate_roundtrip(busy, SR, preset, "mp3", 128, seed=1234)
    assert other.decoded_peak < 1.0
    assert other.retained_fraction >= LIGHT_MIN_RETAINED_FRACTION
    # More change in -> more change out.
    assert other.codec_protected_vs_codec_original.residual_db > light.codec_protected_vs_codec_original.residual_db


@needs_mp3
def test_high_band_only_perturbation_is_removed_by_mp3(busy, monkeypatch):
    """Negative control: the metric must be able to say 'this did not survive'.

    A perturbation living only above 15 kHz is low-passed away by libmp3lame at
    128 kbps, so its retained fraction should collapse. This is the behaviour
    that motivated moving the presets' weight onto jitter and phase drift.
    """
    hf_only = replace(perturb.PRESETS["light"], name="hf_only", noise_offset_db=-200.0, jitter_db=0.0, phase_deg=0.0, hf_offset_db=0.0)
    monkeypatch.setitem(perturb.PRESETS, "hf_only", hf_only)
    report = evaluate_roundtrip(busy, SR, "hf_only", "mp3", 128, seed=1234)
    assert report.retained_fraction < 0.5
    assert report.retained_correlation < 0.3


def test_modulation_is_shared_across_channels(demo):
    """Jitter/phase drift are identical on L and R so the stereo image is not modulated.

    With a dual-mono input, both channels must receive exactly the same
    multiplicative change; only the per-channel masked noise may differ.
    """
    mono = demo[:, :1]
    dual = np.concatenate([mono, mono], axis=1)
    quiet = replace(perturb.PRESETS["light"], name="mod_only", noise_offset_db=-200.0, hf_offset_db=-200.0)
    perturb.PRESETS["mod_only"] = quiet
    try:
        out, _ = perturb.protect(dual, SR, "mod_only", seed=3)
    finally:
        del perturb.PRESETS["mod_only"]
    assert np.allclose(out[:, 0], out[:, 1], atol=1e-9)
    assert not np.allclose(out[:, 0], mono[:, 0])
