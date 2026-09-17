# Music Shield

Anti-copy friction for your own music. Upload a track you own → get back a copy
with a light, psychoacoustically-shaped perturbation → download it. The
perturbation is designed to be hard to hear on normal playback while altering
the spectrogram-level features that audio ML models consume.

**Run locally:** `./run.sh` then open http://127.0.0.1:8420. A hosted
instance, if you run one, should sit behind the built-in HTTP Basic auth (see
[DEPLOY.md](DEPLOY.md)). **Honest limits:** read [LIMITS.md](LIMITS.md).

**This is friction, not protection.** It is not DRM, not AI-proof, and not a
proof of ownership. Re-encoding, denoising, remixing, resampling, or an
attacker who knows the method can weaken or remove it. See [LIMITS.md](LIMITS.md)
before relying on it for anything.

Only use it on recordings you own or have permission to modify. This project
does not, and will not, include tools for stripping protections from or
otherwise handling other people's music.

## What it does

The engine (`music_shield/perturb.py`) combines four model-free components and
mixes them into a lossless output:

| Component | What it is | Why |
| --- | --- | --- |
| Spectral jitter | A slowly drifting gain wobble on ~34 knots over a warped frequency scale, interpolated smoothly across bins (a time-varying micro-EQ, ±0.6 dB peak at the default preset) | Multiplicative on the track's own content, so codecs that reproduce the music also reproduce the wobble; the only component that moves magnitude spectrograms |
| Phase drift | A slow per-knot rotation of the STFT phase (a time-varying all-pass, ±3° peak at the default preset), identical on all channels | Hearing is nearly insensitive to slow monaural phase changes, codecs pass it through, and the stereo image does not move. Changes the waveform and complex spectrum; leaves magnitude spectrograms essentially untouched |
| Masked noise | Pseudo-random noise whose per-bin level sits under a simplified masking curve computed from the track's own band energies (asymmetric neighbour spreading, backward running-minimum so noise never precedes an attack) | Hides behind loud content instead of sounding like flat hiss; changes fine spectral detail. Kept small at `light`: codecs replace it with their own noise, neural codecs barely see it, and it is the only hiss risk |
| High-band component | A little extra masked noise above 15 kHz at a smaller offset under the same masking curve, only where the track already has content there and the sample rate allows | Cheap extra change in a region most adults barely hear. Removed by every common lossy encoder's low-pass, so nothing relies on it; a track with no air band gets nothing added |

Digital silence stays silent. If the result would clip, the whole file is
scaled down rather than hard-clipped, and the UI tells you.

Three presets: `light` (default, conservative), `medium`, `strong`. The
returned "change level (SNR)" number describes how much the file changed —
higher means smaller change. It says nothing about how well the file is
protected; nobody can honestly give you that number and this tool does not try.

What we measured, on synthetic clips only, is in [METRICS.md](METRICS.md):

- **Audibility** (objective proxies, no human panel): `light` passes fixed
  thresholds on dense and mixed material; steady tonal material may expose
  the gain wobble; quiet sparse solo material is flagged at every preset.
- **Copy friction** against free local pipelines (MP3+denoise, 16 kHz
  resample, Meta's open EnCodec neural codec): 91–103 % of the change
  survives all of them, but every pipeline's own error is louder than our
  perturbation, and `light` moves EnCodec's tokens and latents *less than an
  ordinary MP3 does*. For codec-token models the honest summary is "barely
  matters".
- **Codec survival**: at MP3 128 kbps about 95 % of the `light` change is
  still in the decoded file, buried under codec error that is louder.

These are *change remaining* and *representation shift* numbers, not
protection efficacy. What it deliberately does **not** do: model-targeted
adversarial optimisation, and audio watermarking / ownership verification.
See [Roadmap](#roadmap-not-promises).

## Run locally

Requirements: Python 3.10+, and `ffmpeg` on your PATH if you want MP3 input
(WAV and FLAC work without it).

```sh
./run.sh
# → http://127.0.0.1:8420
```

`run.sh` creates a `.venv`, installs `requirements.txt`, and starts uvicorn.
Manually:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8420
```

Or with Docker (includes ffmpeg):

```sh
docker build -t music-shield .
docker run --rm -p 8420:8420 music-shield
```

Open the URL, drop a WAV/FLAC/MP3, tick the ownership box, click **Protect
track**, download the result. MP3 input comes back as WAV because re-encoding
to a lossy codec would partly smooth away the perturbation.

No track handy? Click **Try demo clip** to load a 5-second synthetic chord
generated on the server (`GET /api/demo-clip`) and run it through the same
pipeline and presets.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8420` | Listen port (`run.sh` / Docker) |
| `MUSIC_SHIELD_DATA_DIR` | `$TMPDIR/music_shield_jobs` | Where protected outputs are kept until they expire |
| `MUSIC_SHIELD_JOB_TTL_S` | `3600` | Seconds before a protected file is deleted |
| `MUSIC_SHIELD_MAX_UPLOAD_MB` | `80` | Upload size limit |
| `MUSIC_SHIELD_MEMORY_BUDGET_MB` | `512` | RAM the host has; sizes the guard that rejects tracks too big to protect in memory |

Tracks longer than 15 minutes are rejected, and so are tracks that would not
fit the memory budget — about 6.3 minutes of stereo or 12.7 minutes of mono
at 44.1 kHz on the default 512 MB (see LIMITS.md). Originals are deleted as
soon as processing finishes; only the protected output is kept, and only
until it expires.

## CLI

```sh
# protect a file (output defaults to <name>-protected.<wav|flac>)
python -m music_shield protect my-track.wav --strength light

# generate a synthetic tone, protect it, write both files to demo_out/
python -m music_shield demo

# list presets
python -m music_shield presets

# measure how much change remains after an MP3 / AAC round-trip (needs ffmpeg)
python -m music_shield codec-metrics                          # synthetic "busy" clip, all presets, 192k + 128k
python -m music_shield codec-metrics --clip my-track.wav --presets light --codecs mp3,aac --json

# objective audibility proxies (segmental SNR, noise-to-mask ratio, band residuals); not a listening test
python -m music_shield audibility --clip sparse
python -m music_shield audibility --clip my-track.wav --presets light

# change remaining through free copy / ML pipelines; EnCodec rows need `pip install torch encodec`
python -m music_shield ai-metrics --clip busy
python -m music_shield ai-metrics --clip my-track.wav --pipelines mp3-denoise,resample16k
```

Built-in synthetic clips: `busy`, `demo`, `sparse`, `sustained`, `tone`.

## Tests

```sh
pip install -r requirements-dev.txt
pytest
```

The suite generates a synthetic chord, protects it with every preset, and
checks that the output differs from the input, contains no NaN/inf, stays
under the peak ceiling, keeps digital silence silent, is reproducible for a
fixed seed, and that presets are ordered by strength. The API tests exercise
upload → protect → download, unsupported formats, and corrupt input.

`tests/test_codec_roundtrip.py` (skipped without `ffmpeg` + `libmp3lame`)
re-encodes protected and unprotected clips to MP3 at 192k and 128k, decodes
them, and asserts that the difference is still there above a documented floor,
that the `light` SNR stays conservative, that nothing clips or goes NaN, that
jitter/phase are shared across channels, and — as a negative control — that a
>15 kHz-only perturbation is removed by the codec.
`tests/test_audibility.py` asserts the `light` proxy floors on the dense clip,
that presets get monotonically more audible, that no high band is added to a
track that has none, and that no noise precedes an attack from silence.
`tests/test_ai_eval.py` asserts the change survives the denoise and resample
chains (EnCodec test skipped without torch/encodec). Floors are listed in
[METRICS.md](METRICS.md).

## API

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/api/info` | Presets, accepted extensions, limits, demo clip metadata |
| `GET` | `/api/demo-clip` | 5-second synthetic stereo WAV for trying the tool |
| `POST` | `/api/protect` | multipart `file` + optional `strength`; returns JSON with `download_url` and `stats` |
| `GET` | `/api/download/{job_id}` | The protected file (`audio/wav` or `audio/flac`) |
| `GET` | `/api/health` | Liveness |

Authentication: when `MUSIC_SHIELD_BASIC_PASSWORD` is set (optionally with
`MUSIC_SHIELD_BASIC_USER`), every request requires HTTP Basic auth with that
single shared credential. Unset, the app is open, which is only acceptable on
`127.0.0.1`. See [DEPLOY.md](DEPLOY.md). Never commit the password.

## Project layout

```
music_shield/      engine (perturb.py), codec round-trip metrics (codec_eval.py),
                   audibility proxies (audibility.py), copy-friction pipelines
                   (ai_eval.py), audio I/O, synthetic signals, CLI
app/main.py        FastAPI server
app/static/        plain HTML / CSS / JS front end
tests/             pytest suite
LIMITS.md          honest failure modes
METRICS.md         audibility proxies, copy-friction pipelines, codec survival; how measured
DEPLOY.md          $0 / cheap hosting notes (no automation)
```

## Roadmap (not promises)

- Done in this version: objective audibility proxies on four synthetic clips
  (no human panel), evaluation against a free open model (EnCodec) and two
  ffmpeg pipelines with the finding published as-is ("barely matters" for
  codec-token models), and a preset retune from both.
- Still open: **human listening tests** on real, licensed material. The
  proxies flag sparse solo material at every preset and steady tonal material
  at `light`; only ears can settle those.
- Run `codec-metrics`, `audibility` and `ai-metrics` on real, licensed
  material rather than synthetic clips.
- Optional watermark for ownership marking. Not built: given that the
  perturbation itself barely registers on a neural codec, a light verification
  stub would invite exactly the false confidence this project refuses to
  sell, and it would need its own robustness study first. No strip tools,
  ever.
- Model-targeted perturbations (research, not v1).

## License and intent

Use this on your own music. If you are looking for a way to remove
protections from someone else's audio, this is not it and we will not help.
