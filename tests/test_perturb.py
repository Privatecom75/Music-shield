import numpy as np
import pytest

from music_shield.perturb import PEAK_CEILING, PRESETS, protect
from music_shield.synth import DEMO_DURATION_S, DEMO_SAMPLE_RATE, generate_demo_clip, generate_tone

SR = 44100


@pytest.fixture(scope="module")
def tone():
    return generate_tone(duration_s=3.0, sample_rate=SR)


def test_default_preset_changes_audio_but_stays_sane(tone):
    protected, stats = protect(tone, SR, seed=123)

    assert protected.shape == tone.shape
    assert np.isfinite(protected).all(), "output must not contain NaN/inf"
    assert not np.allclose(protected, tone), "output must differ from input"
    assert np.abs(protected).max() <= PEAK_CEILING + 1e-9
    # Default is conservative: the change should be small relative to the signal.
    assert stats.snr_db > 20
    assert stats.preset == "light"
    assert stats.peak_limited is False


@pytest.mark.parametrize("preset", list(PRESETS))
def test_all_presets_produce_finite_output(tone, preset):
    protected, stats = protect(tone, SR, preset, seed=7)
    assert np.isfinite(protected).all()
    assert np.abs(protected).max() <= PEAK_CEILING + 1e-9
    assert stats.snr_db > 10


def test_presets_are_ordered_by_strength(tone):
    snrs = [protect(tone, SR, p, seed=7)[1].snr_db for p in ("light", "medium", "strong")]
    assert snrs[0] > snrs[1] > snrs[2]


def test_digital_silence_stays_silent(tone):
    pad = np.zeros((SR, tone.shape[1]))
    padded = np.concatenate([pad, tone, pad])
    protected, _ = protect(padded, SR, seed=5)
    margin = 4096  # allow for STFT frame spill at the boundaries
    assert np.abs(protected[: SR - margin]).max() == 0.0
    assert np.abs(protected[-(SR - margin) :]).max() == 0.0


def test_peak_limiting_kicks_in_for_hot_input():
    hot = generate_tone(duration_s=2.0, sample_rate=SR)
    hot = hot / np.abs(hot).max() * 0.999
    protected, stats = protect(hot, SR, "strong", seed=9)
    assert np.abs(protected).max() <= PEAK_CEILING + 1e-9
    assert stats.peak_limited is True


def test_mono_and_low_sample_rate():
    mono = generate_tone(duration_s=2.0, sample_rate=16000, stereo=False)
    protected, stats = protect(mono, 16000, seed=3)
    assert protected.shape == mono.shape
    assert stats.channels == 1
    assert stats.hf_component_applied is False


def test_seed_makes_output_reproducible(tone):
    a, _ = protect(tone, SR, seed=42)
    b, _ = protect(tone, SR, seed=42)
    c, _ = protect(tone, SR, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_demo_clip_is_deterministic_and_protectable():
    a = generate_demo_clip()
    b = generate_demo_clip()
    assert np.array_equal(a, b)
    assert a.shape == (int(DEMO_DURATION_S * DEMO_SAMPLE_RATE), 2)
    assert np.isfinite(a).all()
    assert np.abs(a).max() <= 0.6 + 1e-9
    # Starts and ends silent-ish thanks to the fades.
    assert np.abs(a[:5]).max() < 0.01 and np.abs(a[-5:]).max() < 0.01

    protected, stats = protect(a, DEMO_SAMPLE_RATE, seed=11)
    assert np.isfinite(protected).all()
    assert not np.allclose(protected, a)
    assert stats.snr_db > 20


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        protect(np.zeros((100, 1)), SR)
    with pytest.raises(ValueError):
        protect(generate_tone(1.0), SR, "ultra")
    bad = generate_tone(1.0)
    bad[10, 0] = np.nan
    with pytest.raises(ValueError):
        protect(bad, SR)
