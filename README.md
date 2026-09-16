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

The engine (`music_shield/perturb.py`) combines three model-free components and
mixes them into a lossless output:

| Component | What it is | Why |
| --- | --- | --- |
| Masked noise | Pseudo-random noise whose per-bin level sits under a simplified masking curve computed from the track's own band energies | Hides behind loud content instead of sounding like flat hiss; changes fine spectral detail |
| Spectral jitter | A slowly drifting gain wobble across 32 log-spaced bands (a time-varying micro-EQ, ±0.4 dB at the default preset) | Multiplicative rather than additive, so simple noise subtraction does not undo it |
| High-band component | Low-level energy above 15 kHz, following the track's loudness envelope, only when the sample rate allows | Cheap extra change in a region most adults barely hear; disappears if the file is downsampled |

Digital silence stays silent. If the result would clip, the whole file is
scaled down rather than hard-clipped, and the UI tells you.

Three presets: `light` (default, conservative), `medium`, `strong`. The
returned "change level (SNR)" number describes how much the file changed —
higher means smaller change. It says nothing about how well the file is
protected; nobody can honestly give you that number and this tool does not try.

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
music_shield/      engine (perturb.py), audio I/O, synthetic signals, CLI
app/main.py        FastAPI server
app/static/        plain HTML / CSS / JS front end
tests/             pytest suite
LIMITS.md          honest failure modes
DEPLOY.md          $0 / cheap hosting notes (no automation)
```

## Roadmap (not promises)

- Listening tests on real material to tune the presets — the current values
  are based on SNR and a handful of synthetic signals, not on human trials.
- Evaluate against actual open-source audio models and publish whatever we
  find, including if the answer is "it barely matters".
- Optional watermark for ownership marking (deferred from v1; verification was
  not built).
- Model-targeted perturbations (research, not v1).

## License and intent

Use this on your own music. If you are looking for a way to remove
protections from someone else's audio, this is not it and we will not help.
