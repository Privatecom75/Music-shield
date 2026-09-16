# Deployment notes ($0 / cheap)

**Jab approved go-live (via Big Boss) for free-tier only — keep basic auth on.**

These are notes, not automation. Nothing in this repo deploys anywhere on its
own. Before hosting a shared instance, remember the app has **no auth**: put
it behind basic auth, a login page, or keep it private.

## What the app needs

- Python 3.10+ (or the Dockerfile), `libsndfile` (bundled with the
  `soundfile` wheel on most platforms), and optionally `ffmpeg` for MP3 input.
- Writable scratch space for `MUSIC_SHIELD_DATA_DIR` (defaults to the system
  temp dir). Everything in it is disposable and expires after an hour.
- Enough CPU to run an STFT over a full track in one request. It is
  single-threaded numpy; a 5-minute stereo 44.1 kHz file takes ~2–4 s on a
  laptop core.
- Listens on `$PORT` (Docker image) — every platform below sets that.

## Render (free web service)

- Create a **Web Service** from the repo; pick **Docker** as the runtime so
  ffmpeg is included. Free instance type.
- Render injects `PORT`; the Dockerfile already honours it.
- Caveats: free services spin down after 15 minutes idle (first request after
  that takes ~30–60 s), the filesystem is ephemeral (fine — outputs are
  disposable anyway), and free CPU is slow, so long tracks may hit Render's
  request timeout. Set `MUSIC_SHIELD_MAX_UPLOAD_MB` lower (e.g. `30`) if that
  bites.
- Set `MUSIC_SHIELD_BASIC_USER` + `MUSIC_SHIELD_BASIC_PASSWORD` (app enforces HTTP Basic).

## Fly.io

- `fly launch` will detect the Dockerfile. Choose a `shared-cpu-1x` machine
  with 256–512 MB RAM (numpy on a long 48 kHz file can exceed 256 MB — use 512).
- Set `internal_port = 8420` in `fly.toml` or set `PORT` to match.
- Fly's free allowance has been reduced/changed several times; check current
  pricing. Scale to zero (`min_machines_running = 0`) keeps the bill near zero
  but adds cold-start latency.

## Railway

- Deploy from the repo with the Dockerfile. Railway sets `PORT` automatically.
- The free/trial tier is credit-limited and may pause the service when credits
  run out; check the current plan before assuming $0.

## Any VPS / your own machine

```sh
docker build -t music-shield .
docker run -d --restart unless-stopped -p 127.0.0.1:8420:8420 music-shield
```

Then put Caddy or nginx in front with basic auth and TLS. Bind to
`127.0.0.1` as above so the raw port is not exposed.

## Hugging Face Spaces (Docker Space)

- Free CPU Spaces run Dockerfiles. Set the Space port to 8420 (or `PORT`
  env) in the Space settings / `README.md` front-matter (`app_port: 8420`).
- Public by default — make the Space private or add auth.

## Things to change before anyone else uses it

1. Add authentication (even a shared password via basic auth).
2. Lower `MUSIC_SHIELD_MAX_UPLOAD_MB` and consider a shorter
   `MUSIC_SHIELD_JOB_TTL_S`.
3. Consider moving processing to a background worker if you see request
   timeouts; the current design is synchronous on purpose to stay small.
4. Do not change the in-app copy to promise more than it does.
