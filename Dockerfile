FROM python:3.12-slim

# ffmpeg: MP3 input and the optional MP3 / MP4-video exports (libmp3lame,
# libx264, aac). fonts-dejavu-core: title text on the MP4 cover; without a
# font the video is still produced, just without text. libsndfile: WAV/FLAC.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 fonts-dejavu-core \
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
