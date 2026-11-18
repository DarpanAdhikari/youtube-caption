from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .paths import (
    JobPaths,
    append_jsonl,
    atomic_write_json,
    read_json_or_none,
    utc_now,
)

SCHEMA_VERSION = 1
JOB_STATUSES = {
    "pending",
    "downloading",
    "preparing",
    "chunking",
    "transcribing",
    "merging",
    "refining",
    "exporting",
    "completed",
    "completed_with_errors",
    "interrupted",
    "failed",
}
CHUNK_STATUSES = {"pending", "transcribing", "done", "failed"}


def _now() -> str:
    return utc_now()


@dataclass
class ChunkRecord:
    index: int
    start: float
    end: float
    audio_start: float
    audio_end: float
    path: str
    cut_reason: str = "fixed"
    status: str = "pending"
    attempts: int = 0
    segments_path: str | None = None
    error: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "audio_start": round(self.audio_start, 3),
            "audio_end": round(self.audio_end, 3),
            "path": self.path,
            "cut_reason": self.cut_reason,
            "status": self.status,
            "attempts": self.attempts,
            "segments_path": self.segments_path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        return cls(
            index=int(data["index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            audio_start=float(data.get("audio_start", data["start"])),
            audio_end=float(data.get("audio_end", data["end"])),
            path=str(data["path"]),
            cut_reason=str(data.get("cut_reason", "fixed")),
            status=str(data.get("status", "pending")),
            attempts=int(data.get("attempts", 0)),
            segments_path=data.get("segments_path"),
            error=data.get("error"),
        )


@dataclass
class JobState:
    video_id: str
    url: str
    title: str
    output_dir: str
    schema_version: int = SCHEMA_VERSION
    channel: str = ""
    duration: float = 0.0
    audio_path: str | None = None
    language: str = "ne"
    model: str = ""
    refine_mode: str = "rule"
    chunk_seconds: int = 600
    status: str = "pending"
    stage: str = "created"
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    last_error: str | None = None
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    chunks: list[ChunkRecord] = field(default_factory=list)

    @classmethod
    def new(
        cls,
        config: PipelineConfig,
        *,
        video_id: str,
        url: str,
        title: str,
        output_dir: Path,
    ) -> "JobState":
        return cls(
            video_id=video_id,
            url=url,
            title=title,
            output_dir=str(output_dir),
            language=config.language,
            model=config.model,
            refine_mode=config.refine_mode,
            chunk_seconds=config.chunk_seconds,
            config=config.to_dict(),
            started_at=_now(),
        )

    @property
    def is_completed(self) -> bool:
        return self.status in {"completed", "completed_with_errors"}

    @property
    def pending_chunks(self) -> list[ChunkRecord]:
        return [chunk for chunk in self.chunks if chunk.status != "done"]

    def refresh_counts(self) -> None:
        self.total_chunks = len(self.chunks)
        self.completed_chunks = sum(1 for chunk in self.chunks if chunk.status == "done")
        self.failed_chunks = sum(1 for chunk in self.chunks if chunk.status == "failed")

    def touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        self.refresh_counts()
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "channel": self.channel,
            "duration": self.duration,
            "audio_path": self.audio_path,
            "output_dir": self.output_dir,
            "language": self.language,
            "model": self.model,
            "refine_mode": self.refine_mode,
            "chunk_seconds": self.chunk_seconds,
            "status": self.status,
            "stage": self.stage,
            "total_chunks": self.total_chunks,
            "completed_chunks": self.completed_chunks,
            "failed_chunks": self.failed_chunks,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "last_error": self.last_error,
            "warnings": list(self.warnings),
            "config": dict(self.config),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            video_id=str(data["video_id"]),
            url=str(data.get("url", "")),
            title=str(data.get("title", "")),
            channel=str(data.get("channel", "")),
            duration=float(data.get("duration", 0.0)),
            audio_path=data.get("audio_path"),
            output_dir=str(data["output_dir"]),
            language=str(data.get("language", "ne")),
            model=str(data.get("model", "")),
            refine_mode=str(data.get("refine_mode", "rule")),
            chunk_seconds=int(data.get("chunk_seconds", 600)),
            status=str(data.get("status", "pending")),
            stage=str(data.get("stage", "created")),
            total_chunks=int(data.get("total_chunks", 0)),
            completed_chunks=int(data.get("completed_chunks", 0)),
            failed_chunks=int(data.get("failed_chunks", 0)),
            created_at=str(data.get("created_at", _now())),
            updated_at=str(data.get("updated_at", _now())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            last_error=data.get("last_error"),
            warnings=list(data.get("warnings", [])),
            config=dict(data.get("config", {})),
            chunks=[ChunkRecord.from_dict(item) for item in data.get("chunks", [])],
        )


def new_job(
    config: PipelineConfig,
    *,
    video_id: str,
    url: str,
    title: str,
    paths: JobPaths,
) -> JobState:
    job = JobState.new(
        config,
        video_id=video_id,
        url=url,
        title=title,
        output_dir=paths.root,
    )
    save_job(paths, job)
    return job


def load_job(paths: JobPaths) -> JobState | None:
    data = read_json_or_none(paths.job_file)
    if data is None:
        return None
    return JobState.from_dict(data)


def save_job(paths: JobPaths, job: JobState) -> None:
    job.touch()
    atomic_write_json(paths.job_file, job.to_dict())


def set_status(
    paths: JobPaths,
    job: JobState,
    status: str,
    stage: str | None = None,
    error: str | None = None,
) -> None:
    if status not in JOB_STATUSES:
        raise ValueError(f"unknown job status: {status}")
    job.status = status
    if stage is not None:
        job.stage = stage
    if error is not None:
        job.last_error = error
    if status in {"completed", "completed_with_errors"}:
        job.completed_at = _now()
    save_job(paths, job)


def mark_chunk(
    paths: JobPaths,
    job: JobState,
    chunk: ChunkRecord,
    status: str,
    *,
    segments_path: str | None = None,
    error: str | None = None,
) -> None:
    if status not in CHUNK_STATUSES:
        raise ValueError(f"unknown chunk status: {status}")
    chunk.status = status
    if segments_path is not None:
        chunk.segments_path = segments_path
    if error is not None:
        chunk.error = error
    job.refresh_counts()
    save_job(paths, job)


def reset_failed_chunks(paths: JobPaths, job: JobState) -> int:
    changed = 0
    for chunk in job.chunks:
        if chunk.status == "failed":
            chunk.status = "pending"
            chunk.error = None
            changed += 1
    job.refresh_counts()
    save_job(paths, job)
    return changed


def record_error(
    paths: JobPaths,
    job: JobState,
    stage: str,
    exc: BaseException,
    *,
    chunk_index: int | None = None,
) -> None:
    append_jsonl(
        paths.errors_file,
        {
            "timestamp": _now(),
            "stage": stage,
            "chunk_index": chunk_index,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
        },
    )
    save_job(paths, job)
