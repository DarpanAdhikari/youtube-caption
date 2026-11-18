from pathlib import Path

from transcript_pipeline.paths import (
    JobPaths,
    atomic_write_json,
    find_audio_file,
    human_duration,
    parse_video_id,
    safe_title,
)


def test_parse_video_id_variants():
    expected = "dQw4w9WgXcQ"
    assert parse_video_id(expected) == expected
    assert parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == expected
    assert parse_video_id("https://youtu.be/dQw4w9WgXcQ") == expected
    assert parse_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == expected
    assert parse_video_id("https://www.youtube.com/live/dQw4w9WgXcQ") == expected
    assert parse_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == expected


def test_parse_video_id_falls_back_to_hash():
    value = parse_video_id("https://example.com/not-youtube")
    assert value.startswith("url-")
    assert len(value) == 15


def test_safe_title():
    assert safe_title("  a/b:c*?  ", fallback="video") == "a b c"
    assert safe_title("", fallback="video") == "video"
    assert len(safe_title("x" * 200, fallback="video")) == 80


def test_atomic_json_roundtrip(tmp_path: Path):
    path = tmp_path / "nested" / "data.json"
    atomic_write_json(path, {"ok": True, "text": "नेपाल"})
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert '"ok": true' in path.read_text(encoding="utf-8")
    assert not list(path.parent.glob("*.tmp"))


def test_job_paths_and_audio_lookup(tmp_path: Path):
    paths = JobPaths.for_video(tmp_path, "dQw4w9WgXcQ", "Test Video")
    paths.ensure()
    assert paths.root.name.startswith("dQw4w9WgXcQ_Test Video")
    audio = paths.audio_dir / "original.m4a"
    audio.write_bytes(b"audio")
    (paths.audio_dir / "original.info.json").write_text("{}", encoding="utf-8")
    assert find_audio_file(paths.audio_dir) == audio
    assert human_duration(3661) == "1h 01m 01s"
