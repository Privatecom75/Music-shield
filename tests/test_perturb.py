from dataclasses import replace
import tracemalloc

import numpy as np
import pytest

from music_shield import perturb
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
    """A near-full-scale input must never leave the ceiling, and the limiter must engage.

    Whether a given random walk pushes the peak *up* depends on the seed
    (the gain jitter is symmetric), so several seeds are tried: every output
    respects the ceiling and at least one needed the limiter.
    """
    hot = generate_tone(duration_s=2.0, sample_rate=SR)
    hot = hot / np.abs(hot).max() * 0.999
    limited = []
    for seed in range(6):
        protected, stats = protect(hot, SR, "strong", seed=seed)
        assert np.abs(protected).max() <= PEAK_CEILING + 1e-9
        assert stats.protected_peak <= PEAK_CEILING + 1e-4
        limited.append(stats.peak_limited)
    assert any(limited)


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


def _synthetic_float32(seconds: float, channels: int, sr: int = SR) -> np.ndarray:
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    mono = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 3300 * t)
    mono += 0.02 * np.random.default_rng(0).standard_normal(n, dtype=np.float32)
    x = np.repeat(mono[:, None], channels, axis=1).astype(np.float32)
    x[:, 1:] *= 0.9
    return np.ascontiguousarray(x)


def test_protect_of_long_float32_track_stays_within_memory_budget():
    """The engine streams the STFT; its working set must not grow with track length.

    numpy reports its buffers to tracemalloc, so this measures the arrays
    protect() allocates, independent of the C allocator. Budget: the output
    (same size as the input) plus the chunk buffers and the knot walks. With
    the old whole-track STFT a 60 s stereo clip needed well over 1 GB here.
    """
    x = _synthetic_float32(60.0, 2)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before, _ = tracemalloc.get_traced_memory()
        out, stats = protect(x, SR, seed=1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    budget = out.nbytes + 80 * 1024 * 1024
    assert peak - before < budget, f"protect() peaked at {(peak - before) / 2**20:.0f} MiB for a {x.nbytes / 2**20:.0f} MiB input"
    assert out.dtype == np.float32
    assert out.shape == x.shape
    assert np.isfinite(out).all()
    assert stats.snr_db > 20


def test_output_is_float32_and_input_untouched():
    x = _synthetic_float32(3.0, 2)
    copy = x.copy()
    out, _ = protect(x, SR, seed=4)
    assert out.dtype == np.float32
    assert np.array_equal(x, copy)
    # float64 input is fine too; the output is still float32.
    out64, _ = protect(x.astype(np.float64), SR, seed=4)
    assert out64.dtype == np.float32
    assert np.array_equal(out, out64)


def test_chunked_analysis_synthesis_is_an_identity():
    x = _synthetic_float32(2.5, 1)[:, 0]
    n_frames = perturb._n_frames_for(len(x))
    out = np.zeros_like(x)
    for f0 in range(0, n_frames, perturb.CHUNK_FRAMES):
        f1 = min(f0 + perturb.CHUNK_FRAMES, n_frames)
        perturb._overlap_add_chunk(perturb._stft_chunk(x, f0, f1), f0, out)
    assert np.abs(out - x).max() < 1e-5


def test_result_does_not_depend_on_chunk_size(monkeypatch):
    """Chunk borders are invisible: the multiplicative part and the masking
    estimate (which carries PRE_ECHO_FRAMES of state across chunks) must give
    the same answer for any chunk size. Noise is disabled because its random
    draws are consumed per chunk and would legitimately differ."""
    x = generate_demo_clip()[: 4 * DEMO_SAMPLE_RATE]
    quiet = replace(PRESETS["light"], name="mod_only", noise_offset_db=-400.0, hf_offset_db=-400.0)
    monkeypatch.setitem(PRESETS, "mod_only", quiet)
    outs = []
    for chunk in (perturb.CHUNK_FRAMES, 37, 10**9):
        monkeypatch.setattr(perturb, "CHUNK_FRAMES", chunk)
        outs.append(protect(x, DEMO_SAMPLE_RATE, "mod_only", seed=3)[0])
    assert np.abs(outs[0] - outs[1]).max() < 1e-6
    assert np.abs(outs[0] - outs[2]).max() < 1e-6

    # And the masking curve computed chunk by chunk equals the whole-track one.
    spec = perturb._stft(x[:, 0])
    power = np.abs(spec) ** 2
    edges = perturb._band_edges(DEMO_SAMPLE_RATE, spec.shape[0])
    band_of_bin = perturb._bin_to_band(edges, spec.shape[0])
    whole = perturb._masking_curve(power, edges, band_of_bin).T
    power_fb = np.ascontiguousarray(power.T)
    chunked = np.empty_like(whole)
    prev = None
    for f0 in range(0, whole.shape[0], 37):
        f1 = min(f0 + 37, whole.shape[0])
        masked = perturb._masked_band_power(power_fb[f0:f1], edges)
        limited, prev = perturb._pre_echo_limit(masked, prev, perturb.PRE_ECHO_FRAMES)
        chunked[f0:f1] = limited[:, band_of_bin]
    assert np.array_equal(whole, chunked)
