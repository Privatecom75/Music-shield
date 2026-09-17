# Deployment notes ($0 / cheap)

**Jab approved go-live (via Big Boss) for free-tier only — keep basic auth on.**

These are notes, not automation. Nothing in this repo deploys anywhere on its
own. The app enforces HTTP Basic auth **only when** `MUSIC_SHIELD_BASIC_PASSWORD`
is set (optional `MUSIC_SHIELD_BASIC_USER`, default `musicshield`). Set it on
every shared instance, choose the value out of band, and never commit it or
paste it into an issue, commit message, or this file.

## What the app needs

- Python 3.10+ (or the Dockerfile), `libsndfile` (bundled with the
  `soundfile` wheel on most platforms), and optionally `ffmpeg` for MP3 input
  and the MP3 / MP4-video exports (needs `libmp3lame`, `libx264`, `aac`;
  Debian's `ffmpeg` package has all three). A TTF font (`fonts-dejavu-core`
  in the Dockerfile) puts the title on the MP4 cover; without one the video
  is produced with no text.
- Writable scratch space for `MUSIC_SHIELD_DATA_DIR` (defaults to the system
  temp dir). Everything in it is disposable and expires after an hour.
- Enough CPU to run an STFT over a full track in one request. It is
  single-threaded numpy/scipy; a 5-minute stereo 44.1 kHz file takes ~3 s on
  a laptop core.
- RAM: the STFT is streamed in fixed-size chunks, so a job needs roughly
  8 bytes per sample per channel (float32 in + float32 out) on top of a
  ~150–180 MB warm baseline. `MUSIC_SHIELD_MEMORY_BUDGET_MB` (default `512`)
  tells the size guard how much the host has; files that would not fit get
  a 413 before decoding. Set it to the instance's real RAM. See LIMITS.md
  "Operational limits" for what that means in minutes. MP3/MP4 exports run
  as a separate ffmpeg process (~55 MiB / ~100 MiB peak) behind the same
  one-job-at-a-time lock, so they add nothing to the protect job's peak.
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
- The free instance has 512 MB RAM, which is the default
  `MUSIC_SHIELD_MEMORY_BUDGET_MB`; leave it unless you move to a bigger
  instance. Tracks the budget cannot afford are rejected with a 413 up front
  (about 6.3 min stereo / 12.7 min mono at 44.1 kHz).
- Set `MUSIC_SHIELD_BASIC_USER` + `MUSIC_SHIELD_BASIC_PASSWORD` (app enforces HTTP Basic).

## Fly.io

- `fly launch` will detect the Dockerfile. Choose a `shared-cpu-1x` machine
  with 512 MB RAM. 256 MB is not enough: that is the fixed overhead the size
  guard assumes, so nothing useful would fit in the remaining budget.
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

1. Set `MUSIC_SHIELD_BASIC_PASSWORD` (the built-in basic auth) or put a
   proper login in front. Rotate it if it ever leaks.
2. Lower `MUSIC_SHIELD_MAX_UPLOAD_MB` and consider a shorter
   `MUSIC_SHIELD_JOB_TTL_S`.
3. Consider moving processing to a background worker if you see request
   timeouts; the current design is synchronous on purpose to stay small.
4. Do not change the in-app copy to promise more than it does.
