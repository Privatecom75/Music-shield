"""Copy-friction evaluation against free, local pipelines.

Everything here measures how a protected file behaves relative to its
original when pushed through pipelines a copier or an ML preprocessor might
use. It does **not** measure whether any model copies, imitates or trains on
the track any worse; nobody can get that number from a CPU in a few seconds
and this module does not pretend to. Read METRICS.md for the numbers and the
interpretation, including the parts where the answer is "barely matters".

Pipelines (all $0, all run on a CPU VM):

    mp3-denoise   MP3 128 kbps -> ffmpeg `afftdn` FFT denoiser (noise-floor
                  tracking on) -> decode. A crude "clean the copy up" chain.
    resample16k   44.1 kHz -> 16 kHz -> 44.1 kHz with ffmpeg's resampler. The
                  first step of many ML feature pipelines; kills everything
                  above 8 kHz.
    encodec-6k    Meta's EnCodec 24 kHz neural codec (MIT licence, ~90 MB
    encodec-24k   weights fetched once from Meta's public bucket) at 6 and
                  24 kbps. A neural re-synthesis, plus two things a codec-
                  token-based generative model would actually see: the
                  fraction of RVQ tokens that are identical between the
                  protected and original files, and the cosine similarity of
                  the continuous encoder latents. Requires `torch` and
                  `encodec` (`pip install torch encodec`, CPU wheel is fine);
                  skipped cleanly when they are missing.

For each pipeline P and preset, with A = protected - original:

    retained     <P(protected) - P(original), A> / <A, A>: least-squares
                 fraction of our change that is still present after P.
    corr         correlation of that difference with A.
    control      distance(P(original), original): what P alone does to a
                 clean file. Any friction we add has to be read against this.
    protected    distance(P(protected), original): how far the processed
                 protected file ends up from the clean master.

Controls for the EnCodec token / latent numbers are computed on an MP3 128k
round-trip of the *original*, so "how much does our perturbation move the
tokens" can be compared with "how much does an ordinary MP3 move them".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from music_shield.codec_eval import (
    Distance,
    _align,
    codec_roundtrip,
    distance,
    ffmpeg_encoder_available,
    load_clip,
    residual_db,
)
from music_shield.perturb import DEFAULT_PRESET, PRESETS, protect

try:  # optional heavy dependency; the module stays importable without it
    import torch
    from encodec import EncodecModel
except ImportError:  # pragma: no cover - exercised only where torch is absent
    torch = None
    EncodecModel = None

ENCODEC_SR = 24_000
PIPELINES = ("mp3-denoise", "resample16k", "encodec-6k", "encodec-24k")
DEFAULT_PIPELINES = ("mp3-denoise", "resample16k", "encodec-6k", "encodec-24k")
CLEANUP_MP3_KBPS = 128
AFFTDN_FILTER = "afftdn=nr=12:nf=-45:tn=1"


class PipelineUnavailableError(RuntimeError):
    pass


def encodec_available() -> bool:
    return torch is not None and EncodecModel is not None


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, capture_output=True, timeout=600)


def _run_ffmpeg_chain(audio: np.ndarray, sample_rate: int, steps: list[list[str]], exts: list[str]) -> np.ndarray:
    """Run `audio` through consecutive ffmpeg invocations and return aligned float64 audio."""
    with tempfile.TemporaryDirectory() as tmp:
        current = Path(tmp) / "in.wav"
        sf.write(str(current), audio.astype(np.float64), sample_rate, format="WAV", subtype="FLOAT")
        for i, (step, ext) in enumerate(zip(steps, exts)):
            nxt = Path(tmp) / f"step{i}{ext}"
            _ffmpeg(["-i", str(current), *step, str(nxt)])
            current = nxt
        out = Path(tmp) / "out.wav"
        _ffmpeg(["-i", str(current), "-ar", str(sample_rate), "-f", "wav", "-acodec", "pcm_f32le", str(out)])
        decoded, sr = sf.read(str(out), dtype="float64", always_2d=True)
    if sr != sample_rate or decoded.shape[1] != audio.shape[1]:
        raise RuntimeError("ffmpeg chain returned a different format")
    return _align(decoded, audio)[0]


def mp3_denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if not ffmpeg_encoder_available("libmp3lame"):
        raise PipelineUnavailableError("ffmpeg with libmp3lame is required for mp3-denoise")
    return _run_ffmpeg_chain(
        audio,
        sample_rate,
        [["-codec:a", "libmp3lame", "-b:a", f"{CLEANUP_MP3_KBPS}k"], ["-af", AFFTDN_FILTER, "-f", "wav", "-acodec", "pcm_f32le"]],
        [".mp3", ".wav"],
    )


def resample16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if not ffmpeg_encoder_available("pcm_f32le"):
        raise PipelineUnavailableError("ffmpeg is required for resample16k")
    return _run_ffmpeg_chain(audio, sample_rate, [["-ar", "16000", "-f", "wav", "-acodec", "pcm_f32le"]], [".wav"])


def _to_encodec_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono, resampled to the EnCodec model rate with ffmpeg (soxr/swr), float64."""
    mono = audio.mean(axis=1, keepdims=True) if audio.ndim == 2 else audio[:, None]
    if sample_rate == ENCODEC_SR:
        return mono[:, 0]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        out = Path(tmp) / "out.wav"
        sf.write(str(src), mono, sample_rate, format="WAV", subtype="FLOAT")
        _ffmpeg(["-i", str(src), "-ar", str(ENCODEC_SR), "-f", "wav", "-acodec", "pcm_f32le", str(out)])
        data, _ = sf.read(str(out), dtype="float64", always_2d=True)
    return data[:, 0]


_MODEL_CACHE: dict[float, object] = {}


def _encodec_model(bandwidth_kbps: float):
    if not encodec_available():
        raise PipelineUnavailableError("`torch` and `encodec` are required for the EnCodec pipelines (pip install torch encodec)")
    model = _MODEL_CACHE.get(bandwidth_kbps)
    if model is None:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bandwidth_kbps)
        model.eval()
        _MODEL_CACHE[bandwidth_kbps] = model
    return model


def encodec_analyse(mono_24k: np.ndarray, bandwidth_kbps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (decoded audio, RVQ codes (n_q, T), continuous encoder latents (128, T))."""
    model = _encodec_model(bandwidth_kbps)
    x = torch.from_numpy(mono_24k.astype(np.float32))[None, None, :]
    with torch.no_grad():
        frames = model.encode(x)
        codes = frames[0][0][0].numpy()
        decoded = model.decode(frames)[0, 0].numpy().astype(np.float64)
        latents = model.encoder(x)[0].numpy()
    n = len(mono_24k)
    if len(decoded) < n:
        decoded = np.concatenate([decoded, np.zeros(n - len(decoded))])
    return decoded[:n], codes, latents


@dataclass
class TokenStats:
    """How much the discrete / continuous EnCodec representation moved."""

    token_agreement_all: float
    token_agreement_first_codebook: float
    latent_cosine: float
    latent_rel_l2: float

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def _token_stats(codes_a: np.ndarray, codes_b: np.ndarray, lat_a: np.ndarray, lat_b: np.ndarray) -> TokenStats:
    t = min(codes_a.shape[1], codes_b.shape[1])
    ca, cb = codes_a[:, :t], codes_b[:, :t]
    la, lb = lat_a[:, :t], lat_b[:, :t]
    cos = np.sum(la * lb, axis=0) / (np.linalg.norm(la, axis=0) * np.linalg.norm(lb, axis=0) + 1e-12)
    return TokenStats(
        token_agreement_all=float((ca == cb).mean()),
        token_agreement_first_codebook=float((ca[0] == cb[0]).mean()),
        latent_cosine=float(cos.mean()),
        latent_rel_l2=float(np.linalg.norm(lb - la) / (np.linalg.norm(la) + 1e-12)),
    )


@dataclass
class PipelineReport:
    pipeline: str
    preset: str
    snr_db: float
    change_db: float
    retained_fraction: float
    retained_correlation: float
    after_residual_db: float
    control_vs_original: Distance
    protected_vs_original: Distance
    # EnCodec only: representation shift caused by our perturbation, and by a
    # plain MP3 128k re-encode of the original (control).
    tokens_perturbation: TokenStats | None = None
    tokens_mp3_control: TokenStats | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["control_vs_original"] = self.control_vs_original.as_dict()
        d["protected_vs_original"] = self.protected_vs_original.as_dict()
        for key in ("snr_db", "change_db", "retained_fraction", "retained_correlation", "after_residual_db"):
            v = d[key]
            d[key] = round(v, 3) if math.isfinite(v) else v
        d["tokens_perturbation"] = self.tokens_perturbation.as_dict() if self.tokens_perturbation else None
        d["tokens_mp3_control"] = self.tokens_mp3_control.as_dict() if self.tokens_mp3_control else None
        return d


def _retained(change: np.ndarray, change_after: np.ndarray) -> tuple[float, float]:
    ea = float(np.sum(change**2))
    ed = float(np.sum(change_after**2))
    dot = float(np.sum(change * change_after))
    retained = dot / ea if ea > 0 else float("nan")
    corr = dot / math.sqrt(ea * ed) if ea > 0 and ed > 0 else float("nan")
    return retained, corr


def evaluate_pipeline(
    original: np.ndarray, sample_rate: int, pipeline: str, preset: str = DEFAULT_PRESET, seed: int = 1234
) -> PipelineReport:
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown pipeline '{pipeline}'. Choose one of: {', '.join(PIPELINES)}")
    if original.ndim == 1:
        original = original[:, None]
    original = original.astype(np.float64)
    protected, stats = protect(original, sample_rate, preset, seed=seed)
    change_db = residual_db(protected, original)
    tokens_p = tokens_c = None

    if pipeline.startswith("encodec"):
        kbps = 6.0 if pipeline == "encodec-6k" else 24.0
        orig24 = _to_encodec_input(original, sample_rate)
        prot24 = _to_encodec_input(protected, sample_rate)
        dec_o, codes_o, lat_o = encodec_analyse(orig24, kbps)
        dec_p, codes_p, lat_p = encodec_analyse(prot24, kbps)
        mp3_o = _to_encodec_input(codec_roundtrip(original, sample_rate, "mp3", CLEANUP_MP3_KBPS)[0], sample_rate)
        _, codes_m, lat_m = encodec_analyse(mp3_o, kbps)
        tokens_p = _token_stats(codes_o, codes_p, lat_o, lat_p)
        tokens_c = _token_stats(codes_o, codes_m, lat_o, lat_m)
        ref, prot, p_o, p_p, sr = orig24[:, None], prot24[:, None], dec_o[:, None], dec_p[:, None], ENCODEC_SR
        p_o = _align(p_o, ref)[0]
        p_p = _align(p_p, ref)[0]
    else:
        fn = mp3_denoise if pipeline == "mp3-denoise" else resample16k
        ref, prot, sr = original, protected, sample_rate
        p_o = fn(original, sample_rate)
        p_p = fn(protected, sample_rate)

    retained, corr = _retained(prot - ref, p_p - p_o)
    return PipelineReport(
        pipeline=pipeline,
        preset=preset,
        snr_db=stats.snr_db,
        change_db=change_db,
        retained_fraction=retained,
        retained_correlation=corr,
        after_residual_db=residual_db(p_p, p_o),
        control_vs_original=distance(p_o, ref, sr),
        protected_vs_original=distance(p_p, ref, sr),
        tokens_perturbation=tokens_p,
        tokens_mp3_control=tokens_c,
    )


def _fmt(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}" if math.isfinite(v) else str(v)


def markdown_table(reports: list[PipelineReport], clip_label: str) -> str:
    lines = [
        f"Clip: `{clip_label}`. Change remaining / representation shift through free local pipelines. Not model-efficacy numbers.",
        "",
        "| pipeline | preset | change added | retained | corr | P(prot) vs P(orig) | control: P(orig) vs orig | P(prot) vs orig | tokens same (all / cb0) | latent cos / rel-L2 | MP3 control: tokens same (all / cb0) | latent cos / rel-L2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        c, p = r.control_vs_original, r.protected_vs_original
        tp, tc = r.tokens_perturbation, r.tokens_mp3_control
        tok = f"{tp.token_agreement_all:.1%} / {tp.token_agreement_first_codebook:.1%}" if tp else "-"
        lat = f"{_fmt(tp.latent_cosine, 4)} / {_fmt(tp.latent_rel_l2, 3)}" if tp else "-"
        tokc = f"{tc.token_agreement_all:.1%} / {tc.token_agreement_first_codebook:.1%}" if tc else "-"
        latc = f"{_fmt(tc.latent_cosine, 4)} / {_fmt(tc.latent_rel_l2, 3)}" if tc else "-"
        lines.append(
            f"| {r.pipeline} | {r.preset} | {_fmt(r.change_db)} dB | {r.retained_fraction:.0%} | {_fmt(r.retained_correlation, 2)} | "
            f"{_fmt(r.after_residual_db)} dB / {p.logmel_db - c.logmel_db:+.2f} dB | "
            f"{_fmt(c.residual_db)} dB / {_fmt(c.logmel_db, 2)} dB | {_fmt(p.residual_db)} dB / {_fmt(p.logmel_db, 2)} dB | {tok} | {lat} | {tokc} | {latc} |"
        )
    lines.append("")
    lines.append(
        "Cells `x dB / y dB` are residual dB / mean |log-mel difference| (0-15 kHz, or to Nyquist for EnCodec at 24 kHz). "
        "`P(prot) vs P(orig)` shows residual dB and the *extra* log-mel distance to the master the protected copy carries over the control. "
        "`retained` is the least-squares fraction of the added change still present after the pipeline; `corr` how much of the post-pipeline difference is that change. "
        "Token agreement is the fraction of EnCodec RVQ codes identical to the original's (all codebooks / first codebook); the MP3 control is the same for a plain 128 kbps re-encode of the original."
    )
    return "\n".join(lines)


def run_report(
    clip: str = "busy",
    presets: list[str] | None = None,
    pipelines: list[str] | None = None,
    seed: int = 1234,
) -> tuple[list[PipelineReport], str]:
    presets = presets or list(PRESETS)
    pipelines = pipelines or list(DEFAULT_PIPELINES)
    audio, sr = load_clip(clip)
    reports = []
    for pipeline in pipelines:
        for preset in presets:
            reports.append(evaluate_pipeline(audio, sr, pipeline, preset, seed=seed))
    return reports, markdown_table(reports, clip)
