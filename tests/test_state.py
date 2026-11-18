from pathlib import Path

from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.paths import JobPaths
from transcript_pipeline.state import (
    ChunkRecord,
    load_job,
    mark_chunk,
    new_job,
    record_error,
    reset_failed_chunks,
    save_job,
    set_status,
)


def make_job(tmp_path: Path):
    config = PipelineConfig(output_root=tmp_path)
    paths = JobPaths.for_video(tmp_path, "dQw4w9WgXcQ", "Test Video").ensure()
    job = new_job(
        config,
        video_id="dQw4w9WgXcQ",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Test Video",
        paths=paths,
    )
    job.chunks = [
        ChunkRecord(
            index=1,
            start=0,
            end=600,
            audio_start=0,
            audio_end=600,
            path=str(paths.chunk_audio_file(1)),
        ),
        ChunkRecord(
            index=2,
            start=600,
            end=1200,
            audio_start=599.5,
            audio_end=1200,
            path=str(paths.chunk_audio_file(2)),
        ),
    ]
    save_job(paths, job)
    return config, paths, job


def test_roundtrip_and_counts(tmp_path: Path):
    _, paths, job = make_job(tmp_path)
    mark_chunk(paths, job, job.chunks[0], "done", segments_path="chunk_1.json")
    mark_chunk(paths, job, job.chunks[1], "failed", error="boom")
    loaded = load_job(paths)
    assert loaded is not None
    assert loaded.total_chunks == 2
    assert loaded.completed_chunks == 1
    assert loaded.failed_chunks == 1
    assert loaded.chunks[0].segments_path == "chunk_1.json"
    assert loaded.chunks[1].error == "boom"


def test_reset_failed_chunks(tmp_path: Path):
    _, paths, job = make_job(tmp_path)
    mark_chunk(paths, job, job.chunks[0], "failed", error="boom")
    changed = reset_failed_chunks(paths, job)
    assert changed == 1
    assert job.chunks[0].status == "pending"
    assert job.chunks[0].error is None


def test_record_error_appends_jsonl(tmp_path: Path):
    _, paths, job = make_job(tmp_path)
    record_error(paths, job, "transcribe", RuntimeError("boom"), chunk_index=1)
    lines = paths.errors_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"error_type": "RuntimeError"' in lines[0]
    assert '"chunk_index": 1' in lines[0]


def test_completed_with_errors_sets_completion_time(tmp_path: Path):
    _, paths, job = make_job(tmp_path)
    set_status(paths, job, "completed_with_errors", "completed")
    assert job.completed_at is not None
    assert load_job(paths).completed_at is not None
