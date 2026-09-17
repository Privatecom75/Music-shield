"""Optional MP3 / MP4 exports of the stored lossless output.

Contract under test: the lossless file stays the default download, `format`
is validated, lossy exports go through ffmpeg (and say so when they cannot),
and the encoder invocation carries the documented settings.
"""

import io
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from music_shield import export
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


def _flac_bytes(seconds=2.0, sr=22050):
    buf = io.BytesIO()
    sf.write(buf, generate_tone(seconds, sr), sr, format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def _protect(client, payload=None, name="demo.wav", mime="audio/wav"):
    res = client.post("/api/protect", files={"file": (name, payload or _wav_bytes(), mime)}, data={"strength": "light"})
    assert res.status_code == 200, res.text
    return res.json()


def _all_available(monkeypatch):
    monkeypatch.setattr(export, "unavailable_reason", lambda name: None)


def _none_available(monkeypatch):
    monkeypatch.setattr(export, "ffmpeg_available", lambda: False)
    monkeypatch.setattr(export, "unavailable_reason", lambda name: None if not export.FORMATS[name].lossy else "needs ffmpeg")


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,codec_name,width,height,sample_rate,channels:format=duration,format_name",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    return json.loads(out)


real_ffmpeg = pytest.mark.skipif(
    export.unavailable_reason("mp3") is not None or export.unavailable_reason("mp4") is not None,
    reason="needs ffmpeg with libmp3lame, libx264 and aac",
)


# --- /api/info and the protect response -------------------------------------------


def test_info_lists_export_formats(client):
    formats = {f["name"]: f for f in client.get("/api/info").json()["export_formats"]}
    assert set(formats) == {"wav", "flac", "mp3", "mp4"}
    assert not formats["wav"]["lossy"] and not formats["flac"]["lossy"]
    for name in ("mp3", "mp4"):
        assert formats[name]["lossy"] is True
        assert formats[name]["bitrate_kbps"] == 192
        assert isinstance(formats[name]["available"], bool)
        assert "weakens" in formats[name]["note"]
        if not formats[name]["available"]:
            assert "ffmpeg" in formats[name]["unavailable_reason"]


def test_protect_response_lists_only_available_export_urls(client, monkeypatch):
    _all_available(monkeypatch)
    body = _protect(client)
    assert body["export_urls"] == {
        "mp3": f"{body['download_url']}?format=mp3",
        "mp4": f"{body['download_url']}?format=mp4",
    }
    _none_available(monkeypatch)
    assert _protect(client)["export_urls"] == {}


# --- format query validation --------------------------------------------------------


def test_default_download_is_lossless_wav(client):
    body = _protect(client)
    dl = client.get(body["download_url"])
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("audio/wav")
    assert dl.headers["content-disposition"].endswith('filename="demo-protected.wav"')
    explicit = client.get(body["download_url"], params={"format": "wav"})
    assert explicit.status_code == 200
    assert explicit.content == dl.content


def test_flac_source_defaults_to_flac(client):
    body = _protect(client, _flac_bytes(), name="song.flac", mime="audio/flac")
    assert body["output_ext"] == ".flac"
    dl = client.get(body["download_url"])
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("audio/flac")
    assert client.get(body["download_url"], params={"format": "FLAC"}).status_code == 200
    # No lossless-to-lossless transcoding: asking for WAV on a FLAC job is a 400 that names the alternative.
    res = client.get(body["download_url"], params={"format": "wav"})
    assert res.status_code == 400
    assert "FLAC" in res.json()["detail"]


def test_unknown_format_is_400_listing_valid_formats(client):
    body = _protect(client)
    for bad in ("ogg", "m4a", "wav;rm", "../x"):
        res = client.get(body["download_url"], params={"format": bad})
        assert res.status_code == 400, (bad, res.text)
        detail = res.json()["detail"]
        assert "Unknown format" in detail
        for name in ("wav", "flac", "mp3", "mp4"):
            assert name in detail
    # An empty format means "the default".
    assert client.get(body["download_url"], params={"format": ""}).status_code == 200


def test_format_query_is_case_and_space_insensitive(client, monkeypatch):
    calls = []

    def fake_export(src, dst, name, title=None):
        calls.append(name)
        Path(dst).write_bytes(b"x")
        return dst

    _all_available(monkeypatch)
    monkeypatch.setattr(export, "export_lossy", fake_export)
    body = _protect(client)
    assert client.get(body["download_url"], params={"format": " MP3 "}).status_code == 200
    assert calls == ["mp3"]


def test_format_on_unknown_job_is_404(client):
    assert client.get("/api/download/" + "0" * 32, params={"format": "mp3"}).status_code == 404


# --- lossy export needs ffmpeg ---------------------------------------------------------


def test_lossy_export_is_501_without_ffmpeg(client, monkeypatch):
    _none_available(monkeypatch)
    body = _protect(client)
    for name in ("mp3", "mp4"):
        res = client.get(body["download_url"], params={"format": name})
        assert res.status_code == 501, res.text
        assert "ffmpeg" in res.json()["detail"]
    # The lossless default is untouched by a missing ffmpeg.
    assert client.get(body["download_url"]).status_code == 200


def test_unavailable_reason_names_missing_encoder(monkeypatch):
    monkeypatch.setattr(export, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(export, "ffmpeg_encoders", lambda: {"aac"})
    assert export.unavailable_reason("wav") is None
    assert "libmp3lame" in export.unavailable_reason("mp3")
    assert "libx264" in export.unavailable_reason("mp4")
    monkeypatch.setattr(export, "ffmpeg_available", lambda: False)
    assert "ffmpeg" in export.unavailable_reason("mp3")


def test_encoder_list_parser_skips_legend(monkeypatch):
    sample = (
        "Encoders:\n V..... = Video\n A..... = Audio\n ------\n"
        " V....D libx264              libx264 H.264\n A....D libmp3lame           libmp3lame MP3\n A....D aac                  AAC\n"
    )
    monkeypatch.setattr(export, "_ffmpeg_list", lambda flag: sample)
    assert export.ffmpeg_encoders() == {"libx264", "libmp3lame", "aac"}


# --- lossy export uses ffmpeg with the documented settings (mocked ffmpeg) -------------


class _FakeRun:
    """Records ffmpeg invocations and writes the output file ffmpeg would have produced."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"fake-encoded")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")


@pytest.fixture()
def fake_ffmpeg(monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(export.subprocess, "run", fake)
    _all_available(monkeypatch)
    return fake


def test_mp3_export_invokes_libmp3lame_at_192k(client, fake_ffmpeg):
    body = _protect(client)
    res = client.get(body["download_url"], params={"format": "mp3"})
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("audio/mpeg")
    assert res.headers["content-disposition"].endswith('filename="demo-protected.mp3"')
    assert res.content == b"fake-encoded"

    assert len(fake_ffmpeg.calls) == 1
    cmd = fake_ffmpeg.calls[0]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-i") + 1].endswith("output.wav")  # encodes the stored lossless file
    assert cmd[cmd.index("-c:a") + 1] == "libmp3lame"
    assert cmd[cmd.index("-b:a") + 1] == "192k"
    assert cmd[-1].endswith("export.mp3")


def test_mp4_export_is_still_image_video_with_aac(client, fake_ffmpeg, monkeypatch):
    monkeypatch.setattr(export, "title_overlay_available", lambda: True)
    monkeypatch.setattr(export, "title_font", lambda: "/fonts/Fake.ttf")
    body = _protect(client, name="My Song.wav")
    res = client.get(body["download_url"], params={"format": "mp4"})
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("video/mp4")
    # Starlette percent-encodes names with spaces (RFC 5987), so match loosely.
    assert "Song-protected.mp4" in res.headers["content-disposition"]

    cmd = fake_ffmpeg.calls[0]
    inputs = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"]
    assert len(inputs) == 2
    cover, audio = inputs
    assert cover.endswith("cover.png") and Path(cover).exists()
    assert audio.endswith("output.wav")
    assert cmd[cmd.index("-loop") + 1] == "1"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-tune") + 1] == "stillimage"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-b:a") + 1] == "192k"
    assert "-shortest" in cmd
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert float(cmd[cmd.index("-t") + 1]) == pytest.approx(2.0, abs=0.01)
    vf = cmd[cmd.index("-vf") + 1]
    assert "drawtext" in vf and "cover-title.txt" in vf
    title_file = Path(cover).with_name("cover-title.txt")
    assert title_file.read_text() == "My Song"
    assert cmd[-1].endswith("export.mp4")


def test_mp4_export_without_font_has_no_text_filter(client, fake_ffmpeg, monkeypatch):
    monkeypatch.setattr(export, "title_overlay_available", lambda: False)
    body = _protect(client)
    assert client.get(body["download_url"], params={"format": "mp4"}).status_code == 200
    cmd = fake_ffmpeg.calls[0]
    assert "-vf" not in cmd
    assert cmd[cmd.index("-c:v") + 1] == "libx264"


def test_export_is_encoded_once_and_cached(client, fake_ffmpeg):
    body = _protect(client)
    for _ in range(3):
        assert client.get(body["download_url"], params={"format": "mp3"}).status_code == 200
    assert len(fake_ffmpeg.calls) == 1
    assert client.get(body["download_url"], params={"format": "mp4"}).status_code == 200
    assert len(fake_ffmpeg.calls) == 2


def test_ffmpeg_failure_is_500_and_leaves_no_partial_file(client, monkeypatch):
    calls = {"n": 0}

    def failing_run(cmd, **kwargs):
        calls["n"] += 1
        Path(cmd[-1]).write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"boom")

    _all_available(monkeypatch)
    monkeypatch.setattr(export.subprocess, "run", failing_run)
    body = _protect(client)
    res = client.get(body["download_url"], params={"format": "mp3"})
    assert res.status_code == 500
    assert "MP3 export failed" in res.json()["detail"]
    # The lossless download still works, and a retry re-runs ffmpeg rather than serving the partial file.
    assert client.get(body["download_url"]).status_code == 200
    assert client.get(body["download_url"], params={"format": "mp3"}).status_code == 500
    assert calls["n"] == 2


def test_drawtext_values_are_quoted_and_escaped():
    assert export._drawtext_value("plain") == "'plain'"
    assert export._drawtext_value("a:b'c\\d") == "'a\\:b\\'c\\\\d'"


# --- cover image -------------------------------------------------------------------------


def test_cover_is_valid_png_with_expected_size(tmp_path):
    wav = tmp_path / "t.wav"
    sr = 22050
    x = generate_tone(1.0, sr)
    x[: sr // 2] *= 0.1  # quiet first half so bars vary
    sf.write(wav, x, sr, subtype="PCM_16")
    png = export.render_cover(wav, tmp_path / "cover.png", width=320, height=180)
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    w, h, depth, ctype = struct.unpack(">IIBB", data[16:26])
    assert (w, h, depth, ctype) == (320, 180, 8, 2)
    bars = export.waveform_bars(wav, count=8)
    assert bars.shape == (8,)
    assert 0 <= bars.min() and bars.max() == pytest.approx(1.0)
    assert bars[0] < bars[-1]


def test_waveform_bars_of_silence_do_not_divide_by_zero(tmp_path):
    wav = tmp_path / "s.wav"
    sf.write(wav, np.zeros((4000, 2), dtype=np.float32), 8000, subtype="PCM_16")
    bars = export.waveform_bars(wav, count=5)
    assert np.all(bars == 0)


# --- real ffmpeg (skipped when the encoders are missing) -----------------------------


@real_ffmpeg
def test_real_mp3_export_decodes_to_the_same_length(client, tmp_path):
    body = _protect(client)
    res = client.get(body["download_url"], params={"format": "mp3"})
    assert res.status_code == 200, res.text
    out = tmp_path / "x.mp3"
    out.write_bytes(res.content)
    probe = _ffprobe(out)
    assert probe["streams"][0]["codec_name"] == "mp3"
    assert int(probe["streams"][0]["sample_rate"]) == 22050
    assert float(probe["format"]["duration"]) == pytest.approx(2.0, abs=0.15)


@real_ffmpeg
def test_real_mp4_export_is_h264_video_plus_aac(client, tmp_path):
    body = _protect(client, name="Night Drive.wav")
    res = client.get(body["download_url"], params={"format": "mp4"})
    assert res.status_code == 200, res.text
    assert res.headers["content-type"].startswith("video/mp4")
    out = tmp_path / "x.mp4"
    out.write_bytes(res.content)
    probe = _ffprobe(out)
    by_type = {s["codec_type"]: s for s in probe["streams"]}
    assert by_type["video"]["codec_name"] == "h264"
    assert (by_type["video"]["width"], by_type["video"]["height"]) == (export.VIDEO_WIDTH, export.VIDEO_HEIGHT)
    assert by_type["audio"]["codec_name"] == "aac"
    assert int(by_type["audio"]["channels"]) == 2
    assert "mp4" in probe["format"]["format_name"]
    assert float(probe["format"]["duration"]) == pytest.approx(2.0, abs=0.2)
    # faststart: the moov atom precedes mdat so the file streams from the first byte.
    head = res.content[: 1 << 20]
    assert head.index(b"moov") < head.index(b"mdat")
