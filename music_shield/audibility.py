"""Objective audibility proxies for a protected file versus its original.

No listening panel has been run on this project. This module computes the
things one *can* compute without ears and reports them honestly:

    snr_db                 global signal-to-perturbation ratio
    seg_snr_p05_db         5th percentile of 50 ms segmental SNR over active
                           segments: the worst-case passages, where sparse
                           material exposes the change
    nmr_mean_db / nmr_p99  noise-to-mask ratio: residual power per STFT cell
                           relative to the same simplified masking estimate
                           the engine uses (band energy, -20 dB, spreading).
                           > 0 dB means the residual is above that estimate.
                           Circular for the masked-noise component by
                           construction, informative for jitter/phase whose
                           residual was not shaped against it.
    nmr_over_0_pct         percentage of active cells with NMR > 0 dB
    band_residual_db       residual energy relative to reference energy in
                           <500 Hz, 500 Hz-4 kHz, 4-15 kHz and >15 kHz
    hf_energy_delta_db     change in energy above 10 kHz (hiss / air proxy)
    rms_delta_db           overall level change
    crest_db / crest_delta crest factor (peak/RMS) before and after; a large
                           change means transients were altered
    logmel_p99_db          99th percentile |Δ log-mel| over active cells:
                           the largest short-term band-level deviation, the
                           closest thing here to a 'you would hear the EQ
                           wobble' number
    stereo_residual_db     residual of the side (L-R) signal relative to the
                           original side signal: image stability

The `risk` label is a heuristic derived from these numbers. It is not a
listening result and must not be presented as one.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math

import numpy as np

from music_shield import perturb
from music_shield.codec_eval import MEL_BANDS, MEL_DYNAMIC_RANGE_DB, MEL_FMAX_HZ, _log_mel, _mel_filterbank
from music_shield.perturb import protect

SEGMENT_S = 0.05
ACTIVE_FLOOR_DBFS = -60.0
BAND_EDGES_HZ = ((0.0, 500.0), (500.0, 4000.0), (4000.0, 15000.0), (15000.0, 1e9))
HF_START_HZ = 10_000.0

# Heuristic thresholds for the risk label. Chosen from textbook-ish values
# (segmental SNR near 20 dB, sub-1 dB band-level steps, residual under the
# masking estimate), not from a listening test.
RISK_LIKELY_FINE = {"seg_snr_p05_db": 22.0, "nmr_p99_db": 0.0, "logmel_p99_db": 1.0}
RISK_POSSIBLY_AUDIBLE = {"seg_snr_p05_db": 15.0, "nmr_p99_db": 6.0, "logmel_p99_db": 2.0}


@dataclass
class AudibilityReport:
    preset: str
    snr_db: float
    seg_snr_p05_db: float
    # NMR of the *additive* components (masked noise + high band) when the
    # split is available, else of the whole residual. A multiplicative
    # -30 dB change *of* a partial is not noise beside it, so total-residual
    # NMR over-reads on loud tonal bins.
    nmr_mean_db: float
    nmr_p99_db: float
    nmr_over_0_pct: float
    nmr_is_additive_only: bool
    band_residual_db: tuple[float, float, float, float]
    # Absolute residual level per band in dBFS (RMS). What matters when the
    # band was empty in the original.
    band_residual_dbfs: tuple[float, float, float, float]
    hf_energy_delta_db: float
    rms_delta_db: float
    crest_db: float
    crest_delta_db: float
    logmel_mean_db: float
    logmel_p99_db: float
    stereo_residual_db: float
    risk: str

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 2) if math.isfinite(v) else v
        for key in ("band_residual_db", "band_residual_dbfs"):
            d[key] = [round(v, 1) if math.isfinite(v) else v for v in getattr(self, key)]
        return d


def _db_ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    if num <= 0:
        return float("-inf")
    return 10 * math.log10(num / den)


def _segmental_snr_p05(x: np.ndarray, y: np.ndarray, sample_rate: int) -> float:
    seg = max(1, int(SEGMENT_S * sample_rate))
    n = (len(x) // seg) * seg
    if n == 0:
        return float("nan")
    xs = x[:n].reshape(-1, seg)
    ds = (y[:n] - x[:n]).reshape(-1, seg)
    sig = (xs**2).mean(axis=1)
    err = (ds**2).mean(axis=1)
    active = sig > 10 ** (ACTIVE_FLOOR_DBFS / 10)
    if not active.any():
        return float("nan")
    with np.errstate(divide="ignore"):
        snr = 10 * np.log10(sig[active] / np.maximum(err[active], 1e-30))
    return float(np.percentile(snr, 5))


def _nmr(x: np.ndarray, y: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    spec_x = perturb._stft(x)
    spec_d = perturb._stft(y - x)
    power = np.abs(spec_x) ** 2
    n_bins = spec_x.shape[0]
    edges = perturb._band_edges(sample_rate, n_bins)
    mask = perturb._masking_curve(power, edges, perturb._bin_to_band(edges, n_bins), pre_echo_frames=0)
    active = power > 10 ** ((ACTIVE_FLOOR_DBFS - 20) / 10) * power.max()
    active &= power > 10 ** (perturb.ABSOLUTE_FLOOR_DB / 10)
    if not active.any():
        return float("nan"), float("nan"), float("nan")
    with np.errstate(divide="ignore"):
        nmr = 10 * np.log10(np.maximum(np.abs(spec_d) ** 2, 1e-30) / mask)[active]
    return float(nmr.mean()), float(np.percentile(nmr, 99)), float(100.0 * (nmr > 0).mean())


def _band_energies(x: np.ndarray, sample_rate: int) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sample_rate)
    return np.array([spec[(freqs >= lo) & (freqs < hi)].sum() for lo, hi in BAND_EDGES_HZ]), spec, freqs


def _logmel_stats(x: np.ndarray, y: np.ndarray, sample_rate: int) -> tuple[float, float]:
    fb, _ = _mel_filterbank(sample_rate, 2048, MEL_BANDS, MEL_FMAX_HZ)
    ref = _log_mel(x, sample_rate, fb)
    cand = _log_mel(y, sample_rate, fb)
    active = ref > (ref.max(axis=0, keepdims=True) - MEL_DYNAMIC_RANGE_DB)
    active &= ref > (ref.max() - 80.0)
    if not active.any():
        return float("nan"), float("nan")
    diff = np.abs(cand - ref)[active]
    return float(diff.mean()), float(np.percentile(diff, 99))


def _crest_db(x: np.ndarray) -> float:
    rms = math.sqrt(float(np.mean(x**2))) + 1e-12
    return 20 * math.log10(float(np.abs(x).max()) / rms + 1e-12)


def _classify(seg_snr: float, nmr_p99: float, logmel_p99: float) -> str:
    fine = RISK_LIKELY_FINE
    maybe = RISK_POSSIBLY_AUDIBLE
    if seg_snr >= fine["seg_snr_p05_db"] and nmr_p99 <= fine["nmr_p99_db"] and logmel_p99 <= fine["logmel_p99_db"]:
        return "likely fine (heuristic)"
    if seg_snr >= maybe["seg_snr_p05_db"] and nmr_p99 <= maybe["nmr_p99_db"] and logmel_p99 <= maybe["logmel_p99_db"]:
        return "possibly audible on headphones / sparse material (heuristic)"
    return "likely audible (heuristic)"


def audibility(
    original: np.ndarray,
    protected: np.ndarray,
    sample_rate: int,
    preset: str = "",
    additive_residual: np.ndarray | None = None,
) -> AudibilityReport:
    """Objective proxies for how audible `protected - original` may be. Not a listening test.

    `additive_residual`, when given, is the part of `protected - original` that
    is added noise (masked noise + high band) rather than a modulation of the
    track; NMR is then computed on that part only.
    """
    if original.ndim == 1:
        original = original[:, None]
    if protected.ndim == 1:
        protected = protected[:, None]
    if original.shape != protected.shape:
        raise ValueError("original and protected must have the same shape")
    x = original.astype(np.float64)
    y = protected.astype(np.float64)
    mono_x = x.mean(axis=1)
    mono_y = y.mean(axis=1)

    sig = float(np.sum(x**2))
    err = float(np.sum((y - x) ** 2))
    snr = -_db_ratio(err, sig) if err > 0 else float("inf")

    nmr_target = x + additive_residual if additive_residual is not None else y
    nmr_mean, nmr_p99, nmr_over = zip(*[_nmr(x[:, c], nmr_target[:, c], sample_rate) for c in range(x.shape[1])])
    bands_x, spec_x, freqs = _band_energies(mono_x, sample_rate)
    bands_d, _, _ = _band_energies(mono_y - mono_x, sample_rate)
    spec_y = np.abs(np.fft.rfft(mono_y)) ** 2
    hf = freqs >= HF_START_HZ
    band_res = tuple(_db_ratio(float(d), float(s)) for d, s in zip(bands_d, bands_x))
    # Parseval: band energy / N = mean-square over the whole clip; RMS in dBFS.
    n = len(mono_x)
    band_dbfs = tuple(_db_ratio(float(d) / (n * n / 2.0), 1.0) for d in bands_d)
    mel_mean, mel_p99 = zip(*[_logmel_stats(x[:, c], y[:, c], sample_rate) for c in range(x.shape[1])])

    if x.shape[1] >= 2:
        side_x = x[:, 0] - x[:, 1]
        side_y = y[:, 0] - y[:, 1]
        stereo_res = _db_ratio(float(np.sum((side_y - side_x) ** 2)), float(np.sum(side_x**2)))
    else:
        stereo_res = float("nan")

    seg = _segmental_snr_p05(mono_x, mono_y, sample_rate)
    crest_x = _crest_db(mono_x)
    crest_y = _crest_db(mono_y)
    nmr_p99_v = float(max(nmr_p99))
    mel_p99_v = float(max(mel_p99))
    return AudibilityReport(
        preset=preset,
        snr_db=float(snr),
        seg_snr_p05_db=seg,
        nmr_mean_db=float(np.mean(nmr_mean)),
        nmr_p99_db=nmr_p99_v,
        nmr_over_0_pct=float(np.mean(nmr_over)),
        nmr_is_additive_only=additive_residual is not None,
        band_residual_db=band_res,
        band_residual_dbfs=band_dbfs,
        hf_energy_delta_db=_db_ratio(float(spec_y[hf].sum()), float(spec_x[hf].sum())),
        rms_delta_db=_db_ratio(float(np.mean(y**2)), float(np.mean(x**2))),
        crest_db=crest_x,
        crest_delta_db=crest_y - crest_x,
        logmel_mean_db=float(np.mean(mel_mean)),
        logmel_p99_db=mel_p99_v,
        stereo_residual_db=stereo_res,
        risk=_classify(seg, nmr_p99_v, mel_p99_v),
    )


def evaluate_presets(audio: np.ndarray, sample_rate: int, presets: list[str] | None = None, seed: int = 1234) -> list[AudibilityReport]:
    presets = presets or list(perturb.PRESETS)
    out = []
    for name in presets:
        protected, _ = protect(audio, sample_rate, name, seed=seed)
        # Same seed with jitter/phase switched off reproduces the additive
        # components bit-for-bit (the RNG draw order does not depend on them).
        additive_name = f"__additive_{name}"
        perturb.PRESETS[additive_name] = replace(perturb.PRESETS[name], name=additive_name, jitter_db=0.0, phase_deg=0.0)
        try:
            additive_only, _ = protect(audio, sample_rate, additive_name, seed=seed)
        finally:
            del perturb.PRESETS[additive_name]
        out.append(audibility(audio, protected, sample_rate, preset=name, additive_residual=additive_only - audio))
    return out


def _fmt(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}" if math.isfinite(v) else str(v)


def markdown_table(reports: list[AudibilityReport], clip_label: str) -> str:
    lines = [
        f"Clip: `{clip_label}`. Objective proxies only; no human listening test was run.",
        "",
        "| preset | SNR | seg-SNR p05 | additive NMR mean / p99 | cells > mask | residual rel. low / mid / high / >15k | residual dBFS low / mid / high / >15k | HF Δ | RMS Δ | crest Δ | log-mel mean / p99 | side residual | risk (heuristic) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        b = r.band_residual_db
        f = r.band_residual_dbfs
        lines.append(
            f"| {r.preset} | {_fmt(r.snr_db)} dB | {_fmt(r.seg_snr_p05_db)} dB | {_fmt(r.nmr_mean_db)} / {_fmt(r.nmr_p99_db)} dB | "
            f"{_fmt(r.nmr_over_0_pct)} % | {_fmt(b[0])} / {_fmt(b[1])} / {_fmt(b[2])} / {_fmt(b[3])} dB | "
            f"{_fmt(f[0])} / {_fmt(f[1])} / {_fmt(f[2])} / {_fmt(f[3])} dBFS | {_fmt(r.hf_energy_delta_db, 2)} dB | "
            f"{_fmt(r.rms_delta_db, 2)} dB | {_fmt(r.crest_delta_db, 2)} dB | {_fmt(r.logmel_mean_db, 2)} / {_fmt(r.logmel_p99_db, 2)} dB | "
            f"{_fmt(r.stereo_residual_db)} dB | {r.risk} |"
        )
    additive = all(r.nmr_is_additive_only for r in reports)
    lines.append("")
    lines.append(
        "seg-SNR p05 = 5th percentile of 50 ms segmental SNR on active segments (worst passages). "
        + ("NMR is computed on the additive components only (masked noise + high band), " if additive else "NMR is computed on the whole residual, ")
        + "against the engine's own simplified masking estimate; > 0 dB is above that estimate. "
        "Relative residual columns are residual energy / reference energy per band (large positive values mean the band was empty in the original); "
        "dBFS columns are the absolute RMS level of the residual per band. Risk is a threshold heuristic, not a listening result."
    )
    return "\n".join(lines)
