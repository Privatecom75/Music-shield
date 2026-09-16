"""Command-line entry point: `python -m music_shield ...`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from music_shield.audio_io import load_audio, output_extension_for, save_audio
from music_shield.codec_eval import BUILTIN_CLIPS, CODECS, ffmpeg_encoder_available, run_report
from music_shield.perturb import DEFAULT_PRESET, PRESETS, protect
from music_shield.synth import generate_tone


def _cmd_protect(args: argparse.Namespace) -> int:
    loaded = load_audio(args.input)
    out_path = Path(args.output) if args.output else Path(args.input).with_name(
        Path(args.input).stem + "-protected" + output_extension_for(loaded.source_extension)
    )
    protected, stats = protect(loaded.samples, loaded.sample_rate, args.strength, seed=args.seed)
    save_audio(out_path, protected, loaded.sample_rate, loaded.subtype)
    print(f"Wrote {out_path}")
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tone = generate_tone(duration_s=args.seconds)
    sr = 44100
    original = out_dir / "tone.wav"
    protected_path = out_dir / "tone-protected.wav"
    save_audio(original, tone, sr, "PCM_16")
    protected, stats = protect(tone, sr, args.strength, seed=args.seed)
    save_audio(protected_path, protected, sr, "PCM_16")
    print(f"Wrote {original} and {protected_path}")
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def _cmd_presets(_: argparse.Namespace) -> int:
    for p in PRESETS.values():
        marker = " (default)" if p.name == DEFAULT_PRESET else ""
        print(f"{p.name}{marker}: {p.description}")
    return 0


def _cmd_codec_metrics(args: argparse.Namespace) -> int:
    codecs = [c.strip() for c in args.codecs.split(",") if c.strip()]
    bitrates = [int(b) for b in args.bitrates.split(",") if b.strip()]
    for codec in codecs:
        encoder = CODECS[codec][0] if codec in CODECS else codec
        if not ffmpeg_encoder_available(encoder):
            print(f"error: ffmpeg with the '{encoder}' encoder is required for {codec} round-trips.", file=sys.stderr)
            return 1
    reports, table = run_report(args.clip, presets=args.presets or None, codecs=codecs, bitrates=bitrates, seed=args.seed)
    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2))
    else:
        print(table)
    return 0


def _cmd_audibility(args: argparse.Namespace) -> int:
    from music_shield.audibility import evaluate_presets, markdown_table as audibility_table
    from music_shield.codec_eval import load_clip

    audio, sr = load_clip(args.clip)
    reports = evaluate_presets(audio, sr, presets=args.presets or None, seed=args.seed)
    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2))
    else:
        print(audibility_table(reports, args.clip))
    return 0


def _cmd_ai_metrics(args: argparse.Namespace) -> int:
    from music_shield import ai_eval

    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    for p in pipelines:
        if p not in ai_eval.PIPELINES:
            print(f"error: unknown pipeline '{p}'. Choose from {', '.join(ai_eval.PIPELINES)}", file=sys.stderr)
            return 1
        if p.startswith("encodec") and not ai_eval.encodec_available():
            print("error: the EnCodec pipelines need `pip install torch encodec` (CPU wheel is fine).", file=sys.stderr)
            return 1
    reports, table = ai_eval.run_report(args.clip, presets=args.presets or None, pipelines=pipelines, seed=args.seed)
    if args.json:
        print(json.dumps([r.as_dict() for r in reports], indent=2))
    else:
        print(table)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music_shield",
        description="Add a light psychoacoustic perturbation to your own tracks. Friction, not DRM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("protect", help="Protect one audio file")
    p.add_argument("input", help="Input .wav / .flac / .mp3")
    p.add_argument("output", nargs="?", help="Output path (.wav or .flac). Defaults to <input>-protected.<ext>")
    p.add_argument("--strength", choices=list(PRESETS), default=DEFAULT_PRESET)
    p.add_argument("--seed", type=int, default=None, help="Fix the random seed (for reproducible output)")
    p.set_defaults(func=_cmd_protect)

    d = sub.add_parser("demo", help="Generate a synthetic tone, protect it, write both files")
    d.add_argument("--out-dir", default="demo_out")
    d.add_argument("--seconds", type=float, default=3.0)
    d.add_argument("--strength", choices=list(PRESETS), default=DEFAULT_PRESET)
    d.add_argument("--seed", type=int, default=None)
    d.set_defaults(func=_cmd_demo)

    s = sub.add_parser("presets", help="List strength presets")
    s.set_defaults(func=_cmd_presets)

    m = sub.add_parser(
        "codec-metrics",
        help="Measure how much of the perturbation remains after an MP3/AAC round-trip (needs ffmpeg)",
        description=(
            "Protect a clip, re-encode both the original and the protected file with ffmpeg, decode, "
            "and report how much change remains. These are change-remaining numbers, not efficacy claims; "
            "see METRICS.md."
        ),
    )
    m.add_argument("--clip", default="busy", help=f"One of {', '.join(BUILTIN_CLIPS)} (synthetic) or a path to a WAV/FLAC you own")
    m.add_argument("--presets", nargs="*", choices=list(PRESETS), help="Presets to evaluate (default: all)")
    m.add_argument("--codecs", default="mp3", help="Comma-separated: mp3, aac (default: mp3)")
    m.add_argument("--bitrates", default="192,128", help="Comma-separated kbps (default: 192,128)")
    m.add_argument("--seed", type=int, default=1234)
    m.add_argument("--json", action="store_true", help="Print full JSON instead of a markdown table")
    m.set_defaults(func=_cmd_codec_metrics)

    a = sub.add_parser(
        "audibility",
        help="Objective audibility proxies (segmental SNR, noise-to-mask ratio, band residuals). Not a listening test.",
    )
    a.add_argument("--clip", default="busy", help=f"One of {', '.join(BUILTIN_CLIPS)} (synthetic) or a path to a WAV/FLAC you own")
    a.add_argument("--presets", nargs="*", choices=list(PRESETS))
    a.add_argument("--seed", type=int, default=1234)
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=_cmd_audibility)

    ai = sub.add_parser(
        "ai-metrics",
        help="Change remaining through free local copy/ML pipelines (MP3+denoise, 16 kHz resample, EnCodec). Not efficacy.",
    )
    ai.add_argument("--clip", default="busy", help=f"One of {', '.join(BUILTIN_CLIPS)} (synthetic) or a path to a WAV/FLAC you own")
    ai.add_argument("--presets", nargs="*", choices=list(PRESETS))
    ai.add_argument("--pipelines", default="mp3-denoise,resample16k,encodec-6k,encodec-24k", help="Comma-separated pipeline names")
    ai.add_argument("--seed", type=int, default=1234)
    ai.add_argument("--json", action="store_true")
    ai.set_defaults(func=_cmd_ai_metrics)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
