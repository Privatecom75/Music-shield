# Music Shield

Anti-copy friction for your own music. Upload a track you own → get back a copy
with a light, psychoacoustically-shaped perturbation → download it. The
perturbation is designed to be hard to hear on normal playback while altering
the spectrogram-level features that audio ML models consume.

**Run locally:** `./run.sh` then open http://127.0.0.1:8420. Local only — this
is not deployed anywhere. **Honest limits:** read [LIMITS.md](LIMITS.md).

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
| Masked noise | Pseudo-random noise whose per-bin level sits under a simplified masking curve computed from the track's own band energies | Hides behind loud content instead of sounding like flat hiss; changes fine spectral detail. Kept small at `light` because codecs largely replace it with their own noise |
| High-band component | Very low-level energy above 15 kHz, following the track's loudness envelope, only when the sample rate allows | Cheap extra change in a region most adults barely hear. Removed by every common lossy encoder's low-pass, so it is no longer relied on |

Digital silence stays silent. If the result would clip, the whole file is
scaled down rather than hard-clipped, and the UI tells you.

Three presets: `light` (default, conservative), `medium`, `strong`. The
returned "change level (SNR)" number describes how much the file changed —
higher means smaller change. It says nothing about how well the file is
protected; nobody can honestly give you that number and this tool does not try.

How much of the change survives an MP3/AAC re-encode is measured, on synthetic
clips only, in [METRICS.md](METRICS.md). Short version: at MP3 128 kbps about
94 % of the `light` change is still in the decoded file, but the codec's own
error is louder than our perturbation. Those are *change remaining* numbers,
not protection efficacy.

What it deliberately does **not** do in v1: model-targeted adversarial
optimisation (no gradient attacks against specific models), and audio
watermarking / ownership verification. Both are deferred — see
[Roadmap](#roadmap-not-promises).

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

Tracks longer than 15 minutes are rejected. Originals are deleted as soon as
processing finishes; only the protected output is kept, and only until it
expires.

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
```

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
>15 kHz-only perturbation is removed by the codec. Floors are listed in
[METRICS.md](METRICS.md).

## API

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/api/info` | Presets, accepted extensions, limits, demo clip metadata |
| `GET` | `/api/demo-clip` | 5-second synthetic stereo WAV for trying the tool |
| `POST` | `/api/protect` | multipart `file` + optional `strength`; returns JSON with `download_url` and `stats` |
| `GET` | `/api/download/{job_id}` | The protected file (`audio/wav` or `audio/flac`) |
| `GET` | `/api/health` | Liveness |

No authentication. This is a local MVP; if you host it anywhere shared, put it
behind a login or a reverse proxy with basic auth (see [DEPLOY.md](DEPLOY.md)).

## Project layout

```
music_shield/      engine (perturb.py), codec round-trip metrics (codec_eval.py),
                   audio I/O, synthetic signals, CLI
app/main.py        FastAPI server
app/static/        plain HTML / CSS / JS front end
tests/             pytest suite
LIMITS.md          honest failure modes
METRICS.md         what survives an MP3/AAC re-encode, and how it was measured
DEPLOY.md          $0 / cheap hosting notes (no automation)
```

## Roadmap (not promises)

- Listening tests on real material to tune the presets — the current values
  are based on SNR, construction arguments and a handful of synthetic
  signals, not on human trials.
- Run `codec-metrics` on real, licensed material rather than synthetic clips.
- Evaluate against actual open-source audio models and publish whatever we
  find, including if the answer is "it barely matters".
- Optional watermark for ownership marking (deferred from v1; verification was
  not built).
- Model-targeted perturbations (research, not v1).

## License and intent

Use this on your own music. If you are looking for a way to remove
protections from someone else's audio, this is not it and we will not help.
