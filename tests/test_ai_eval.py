"""Copy-friction pipelines. Change-remaining numbers, not model-efficacy claims."""

import numpy as np
import pytest

from music_shield import ai_eval
from music_shield.codec_eval import ffmpeg_encoder_available
from music_shield.synth import generate_busy_clip

SR = 44100
needs_mp3 = pytest.mark.skipif(not ffmpeg_encoder_available("libmp3lame"), reason="ffmpeg with libmp3lame not available")
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_encoder_available("pcm_f32le"), reason="ffmpeg not available")

# Documented floors (METRICS.md, "Copy-friction pipelines"). Measured on the
# busy clip at light: mp3-denoise retained 94 %, resample16k 100 %.
LIGHT_MIN_RETAINED = 0.70


@pytest.fixture(scope="module")
def busy():
    return generate_busy_clip(SR)


@needs_mp3
def test_denoise_chain_does_not_remove_the_multiplicative_change(busy):
    r = ai_eval.evaluate_pipeline(busy, SR, "mp3-denoise", "light", seed=1234)
    assert r.retained_fraction >= LIGHT_MIN_RETAINED
    assert np.isfinite(r.after_residual_db)
    # The denoiser itself moves the file more than we do: the protected copy is
    # only slightly further from the master than the control. Documented, not hidden.
    assert r.protected_vs_original.logmel_db >= r.control_vs_original.logmel_db
    assert r.protected_vs_original.logmel_db - r.control_vs_original.logmel_db < 0.5


@needs_ffmpeg
def test_resample_to_16k_keeps_the_change(busy):
    r = ai_eval.evaluate_pipeline(busy, SR, "resample16k", "light", seed=1234)
    assert r.retained_fraction >= 0.9
    assert r.retained_correlation >= 0.9


@pytest.mark.skipif(not ai_eval.encodec_available(), reason="torch/encodec not installed")
@needs_mp3
def test_encodec_pipeline_reports_sane_numbers(busy):
    try:
        r = ai_eval.evaluate_pipeline(busy, SR, "encodec-6k", "light", seed=1234)
    except Exception as exc:  # weights download needs network; do not fail offline
        pytest.skip(f"EnCodec unavailable: {exc}")
    tp, tc = r.tokens_perturbation, r.tokens_mp3_control
    assert tp is not None and tc is not None
    assert 0.0 <= tp.token_agreement_all <= 1.0 and 0.0 <= tp.token_agreement_first_codebook <= 1.0
    assert tp.latent_cosine > 0.99
    assert np.isfinite(r.retained_fraction) and r.retained_fraction > 0.5


def test_unknown_pipeline_is_rejected(busy):
    with pytest.raises(ValueError):
        ai_eval.evaluate_pipeline(busy, SR, "magic", "light")
