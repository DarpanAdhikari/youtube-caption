from pathlib import Path
from types import SimpleNamespace

import pytest

from transcript_pipeline.asr import ASRResult, ASRSegment, write_chunk_result
from transcript_pipeline.chunking import ChunkPlan
from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.merge import (
    TranscriptSegment,
    assess_quality,
    save_raw_transcript,
)
from transcript_pipeline.paths import JobPaths
from transcript_pipeline.pipeline import run_pipeline
from transcript_pipeline.state import ChunkRecord, mark_chunk, new_job, save_job
import transcript_pipeline.pipeline as pipeline_module


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
    job.duration = 10
    return config, paths, job


def test_export_only_from_raw(tmp_path: Path):
    _, paths, job = make_job(tmp_path)
    segments = [
        TranscriptSegment(0, 5, "पहिलो वाक्य।"),
        TranscriptSegment(5, 10, "दोस्रो वाक्य।"),
    ]
    save_raw_transcript(paths, job, segments, assess_quality(segments, 10))
    save_job(paths, job)

    config = PipelineConfig(
        video_url=job.url,
        output_root=tmp_path,
        run_mode="export_only",
        refine_mode="off",
    )
    result = run_pipeline(config)
    assert result.job.status == "completed"
    assert result.exports.files["transcript_txt"].exists()
    assert "पहिलो" in result.exports.files["transcript_txt"].read_text(encoding="utf-8")


def test_refine_only(tmp_path: Path):
    _, paths, job = make_job(tmp_path)
    segments = [TranscriptSegment(0, 5, "यो परिक्षा हो।")]
    save_raw_transcript(paths, job, segments, assess_quality(segments, 10))
    save_job(paths, job)

    config = PipelineConfig(
        video_url=job.url,
        output_root=tmp_path,
        run_mode="refine_only",
        refine_mode="off",
    )
    result = run_pipeline(config)
    assert result.job.status == "completed"
    assert result.refined_segments[0].text == "यो परिक्षा हो।"


def test_export_only_rebuilds_from_chunks(tmp_path: Path):
    _, paths, job = make_job(tmp_path)
    job.chunks = [
        ChunkRecord(
            index=1,
            start=0,
            end=10,
            audio_start=0,
            audio_end=10,
            path=str(paths.chunk_audio_file(1)),
            status="done",
        )
    ]
    result = ASRResult(
        segments=[ASRSegment(0, 5, "पहिलो वाक्य।")],
        model="fake",
    )
    chunk = job.chunks[0]
    write_chunk_result(paths, chunk, result)
    mark_chunk(paths, job, chunk, "done", segments_path=str(paths.chunk_result_file(1)))
    save_job(paths, job)

    config = PipelineConfig(
        video_url=job.url,
        output_root=tmp_path,
        run_mode="export_only",
        refine_mode="off",
    )
    pipeline_result = run_pipeline(config)
    assert pipeline_result.job.status == "completed"
    assert pipeline_result.exports.files["transcript_txt"].exists()


def test_full_run_with_fake_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "input.m4a"
    source.write_bytes(b"fake")

    def fake_download(config, paths, reporter, *, known_metadata=None):
        return pipeline_module.AudioMetadata(
            video_id="dQw4w9WgXcQ",
            title="Fake Video",
            duration=10.0,
            audio_path=str(source),
        )

    def fake_normalize(input_path, output_path, reporter, *, force=False):
        return output_path

    def fake_probe(path):
        return {"duration": 10.0}

    def fake_prepare(self, wav_path, job):
        plan = ChunkPlan(1, 0, 10, 0, 10, "whole")
        chunk = ChunkRecord(
            index=1,
            start=0,
            end=10,
            audio_start=0,
            audio_end=10,
            path=str(self.paths.chunk_audio_file(1)),
        )
        Path(chunk.path).write_bytes(b"fake")
        job.chunks = [chunk]
        return [plan]

    def fake_transcribe(engine, plans, paths, job, reporter):
        chunk = job.chunks[0]
        result = ASRResult(segments=[ASRSegment(0, 10, "पहिलो वाक्य।")], model="fake")
        write_chunk_result(paths, chunk, result)
        mark_chunk(
            paths,
            job,
            chunk,
            "done",
            segments_path=str(paths.chunk_result_file(1)),
        )

    monkeypatch.setattr(pipeline_module, "download_audio", fake_download)
    monkeypatch.setattr(pipeline_module, "normalize_audio", fake_normalize)
    monkeypatch.setattr(pipeline_module, "probe_media", fake_probe)
    monkeypatch.setattr(pipeline_module.TranscriptionPipeline, "_prepare_chunks", fake_prepare)
    monkeypatch.setattr(pipeline_module, "transcribe_chunks", fake_transcribe)
    monkeypatch.setattr(
        pipeline_module.TranscriptionPipeline,
        "_build_engine",
        lambda self, job: (SimpleNamespace(), None),
    )

    config = PipelineConfig(
        input_audio=source,
        output_root=tmp_path,
        run_mode="auto",
        refine_mode="off",
    )
    result = run_pipeline(config)
    assert result.job.status == "completed"
    assert result.job.audio_path == str(source)
    assert result.raw_segments[0].text == "पहिलो वाक्य।"
    assert result.exports.files["transcript_txt"].exists()
    assert result.exports.files["subtitles_srt"].exists()
