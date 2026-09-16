FROM python:3.12-slim

# ffmpeg is only needed for MP3 input; libsndfile handles WAV/FLAC.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY music_shield ./music_shield
COPY app ./app

ENV PORT=8420 \
    MUSIC_SHIELD_DATA_DIR=/tmp/music_shield_jobs \
    PYTHONUNBUFFERED=1

EXPOSE 8420
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
