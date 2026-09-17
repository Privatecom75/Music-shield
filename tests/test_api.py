import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from music_shield.synth import generate_tone


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSIC_SHIELD_DATA_DIR", str(tmp_path / "jobs"))
    import importlib

    import app.main as main

    importlib.reload(main)
    return TestClient(main.app)


def _wav_bytes(seconds=2.0, sr=22050):
    buf = io.BytesIO()
    sf.write(buf, generate_tone(seconds, sr), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_info_lists_presets(client):
    data = client.get("/api/info").json()
    names = {p["name"] for p in data["presets"]}
    assert {"light", "medium", "strong"} <= names
    assert sum(p["default"] for p in data["presets"]) == 1
    assert ".wav" in data["accepted_extensions"]


def test_protect_roundtrip(client):
    res = client.post("/api/protect", files={"file": ("demo.wav", _wav_bytes(), "audio/wav")}, data={"strength": "light"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["output_name"] == "demo-protected.wav"
    assert body["stats"]["preset"] == "light"
    assert body["stats"]["snr_db"] > 20

    dl = client.get(body["download_url"])
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("audio/wav")
    protected, sr = sf.read(io.BytesIO(dl.content), always_2d=True)
    original, _ = sf.read(io.BytesIO(_wav_bytes()), always_2d=True)
    assert sr == 22050
    assert protected.shape == original.shape
    assert np.isfinite(protected).all()
    assert not np.allclose(protected, original)


def test_rejects_unsupported_extension(client):
    res = client.post("/api/protect", files={"file": ("song.ogg", b"not audio", "audio/ogg")})
    assert res.status_code == 415
    assert "Unsupported" in res.json()["detail"]


def test_rejects_garbage_wav(client):
    res = client.post("/api/protect", files={"file": ("song.wav", b"definitely not a wav", "audio/wav")})
    assert res.status_code == 422


def test_rejects_unknown_strength(client):
    res = client.post("/api/protect", files={"file": ("demo.wav", _wav_bytes(), "audio/wav")}, data={"strength": "nuclear"})
    assert res.status_code == 400


def test_download_404_for_unknown_job(client):
    assert client.get("/api/download/" + "0" * 32).status_code == 404
    assert client.get("/api/download/not-a-job").status_code == 404


def test_demo_clip_is_valid_wav_and_protects(client):
    info = client.get("/api/info").json()["demo"]
    res = client.get(info["url"])
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("audio/wav")

    clip, sr = sf.read(io.BytesIO(res.content), always_2d=True)
    assert sr == info["sample_rate"] == 44100
    assert clip.shape[1] == 2
    assert abs(clip.shape[0] / sr - info["duration_s"]) < 0.01
    assert np.isfinite(clip).all()
    assert 0.3 < np.abs(clip).max() < 1.0

    # Same bytes every time, so the browser can cache it.
    assert client.get(info["url"]).content == res.content

    protected = client.post(
        "/api/protect", files={"file": (info["filename"], res.content, "audio/wav")}, data={"strength": "light"}
    )
    assert protected.status_code == 200, protected.text
    body = protected.json()
    assert body["output_name"] == "music-shield-demo-protected.wav"
    assert body["stats"]["duration_s"] == info["duration_s"]
    assert body["stats"]["snr_db"] > 20


def test_info_reports_memory_guard(client):
    data = client.get("/api/info").json()
    assert data["max_sample_frames"] > 0
    assert 0 < data["max_stereo_minutes_44k"] <= data["max_duration_minutes"]
    assert data["max_stereo_minutes_44k"] <= data["max_mono_minutes_44k"] <= data["max_duration_minutes"]


def test_track_over_memory_budget_is_rejected_with_413_before_decoding(client, monkeypatch):
    """A file the memory model cannot afford gets a clean 413, not an OOM-killed worker.

    The guard runs on the header (libsndfile info / ffprobe), so a WAV whose
    *declared* frame count is over budget is rejected without reading samples.
    """
    import music_shield.audio_io as audio_io

    # Budget for a 1 s stereo clip at 22.05 kHz: a 2 s clip is over.
    monkeypatch.setattr(audio_io, "MAX_SAMPLE_FRAMES", 22050 * 2)
    ok = client.post("/api/protect", files={"file": ("short.wav", _wav_bytes(1.0), "audio/wav")})
    assert ok.status_code == 200, ok.text
    res = client.post("/api/protect", files={"file": ("long.wav", _wav_bytes(2.0), "audio/wav")})
    assert res.status_code == 413, res.text
    detail = res.json()["detail"]
    assert "memory budget" in detail
    # The limit is quoted for this file's own rate / layout.
    assert "stereo" in detail and "22050 Hz" in detail and "1 second" in detail


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Music Shield" in res.text
    assert "not AI-proof" in res.text
