"""Objective audibility proxies. These are not listening tests; see METRICS.md."""

from dataclasses import replace

import numpy as np
import pytest

from music_shield import perturb
from music_shield.audibility import audibility, evaluate_presets
from music_shield.synth import generate_busy_clip, generate_sparse_clip, generate_sustained_clip

SR = 44100

# Documented floors for the default preset on the dense synthetic clip
# (METRICS.md, "Listening notes"). Measured: seg-SNR p05 24.6 dB, additive
# NMR p99 -10.9 dB, log-mel p99 0.55 dB.
LIGHT_BUSY_MIN_SEG_SNR_DB = 20.0
LIGHT_BUSY_MAX_ADDITIVE_NMR_P99_DB = -3.0
LIGHT_BUSY_MAX_LOGMEL_P99_DB = 1.0


@pytest.fixture(scope="module")
def busy():
    return generate_busy_clip(SR)


def test_identical_input_is_silent_change(busy):
    r = audibility(busy, busy, SR, preset="none")
    assert r.snr_db == float("inf")
    assert r.rms_delta_db == 0.0
    assert r.risk.startswith("likely fine")


def test_light_on_dense_material_meets_documented_floors(busy):
    (r,) = evaluate_presets(busy, SR, presets=["light"], seed=1234)
    assert r.nmr_is_additive_only
    assert r.seg_snr_p05_db > LIGHT_BUSY_MIN_SEG_SNR_DB
    assert r.nmr_p99_db < LIGHT_BUSY_MAX_ADDITIVE_NMR_P99_DB
    assert r.logmel_p99_db < LIGHT_BUSY_MAX_LOGMEL_P99_DB
    assert abs(r.rms_delta_db) < 0.2
    assert r.risk.startswith("likely fine")


def test_presets_get_monotonically_more_audible(busy):
    reports = evaluate_presets(busy, SR, seed=1234)
    seg = [r.seg_snr_p05_db for r in reports]
    mel = [r.logmel_p99_db for r in reports]
    assert seg[0] > seg[1] > seg[2]
    assert mel[0] < mel[1] < mel[2]


def test_no_high_band_added_where_track_has_none():
    """The >15 kHz component is mask-shaped: a track with nothing up there gets nothing."""
    x = generate_sustained_clip(SR)
    out, stats = perturb.protect(x, SR, "strong", seed=1)
    assert stats.hf_component_applied is False
    spec = np.abs(np.fft.rfft((out - x)[:, 0])) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / SR)
    hf_energy = spec[freqs >= 15_000].sum()
    total = spec.sum()
    assert hf_energy / total < 1e-6


def test_high_band_is_applied_on_material_that_has_one(busy):
    _, stats = perturb.protect(busy, SR, "light", seed=1)
    assert stats.hf_component_applied is True


def test_noise_does_not_precede_an_attack_from_silence():
    """Pre-echo control: no perturbation energy in the 30 ms before an onset out of digital silence."""
    n = SR * 2
    t = np.arange(n) / SR
    x = np.zeros((n, 1))
    onset = SR  # attack at exactly 1.0 s
    burst = np.sin(2 * np.pi * 440 * t[: n - onset]) * np.exp(-t[: n - onset] * 3)
    x[onset:, 0] = 0.5 * burst
    additive = replace(perturb.PRESETS["strong"], name="additive_strong", jitter_db=0.0, phase_deg=0.0)
    perturb.PRESETS["additive_strong"] = additive
    try:
        noise_only, _ = perturb.protect(x, SR, "additive_strong", seed=2)
        full, _ = perturb.protect(x, SR, "strong", seed=2)
    finally:
        del perturb.PRESETS["additive_strong"]

    def rms(a):
        return float(np.sqrt(np.mean(a**2)))

    pre = slice(onset - int(0.030 * SR), onset - int(0.005 * SR))
    post = slice(onset, onset + int(0.030 * SR))
    diff_noise = (noise_only - x)[:, 0]
    diff_full = (full - x)[:, 0]
    # Added noise before the attack is at least 40 dB under the noise after it.
    assert rms(diff_noise[pre]) < rms(diff_noise[post]) * 10 ** (-40 / 20)
    # The modulation's own pre-ringing (STFT-domain filtering) is confined to
    # the last few ms, inside backward masking: 5-30 ms before the onset the
    # whole residual is at least 30 dB under the post-onset residual.
    assert rms(diff_full[pre]) < rms(diff_full[post]) * 10 ** (-30 / 20)


def test_sparse_material_is_flagged_not_hidden():
    """Honesty check: the proxies must not call the sparse clip 'likely fine' at strong."""
    x = generate_sparse_clip(SR)
    reports = evaluate_presets(x, SR, presets=["strong"], seed=1234)
    assert not reports[0].risk.startswith("likely fine")
