from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import PipelineError

REFINE_MODES = ("off", "rule", "neural")
RUN_MODES = ("auto", "fresh", "resume", "retry_failed", "export_only", "refine_only")


def default_model_cache_dir() -> Path:
    if Path("/content").is_dir():
        return Path("/content/models")
    return Path.home() / ".cache" / "transcript_pipeline" / "models"


def default_output_root() -> Path:
    value = os.getenv("TRANSCRIPT_OUTPUT_ROOT", "")
    return Path(value).expanduser() if value else Path.cwd() / "DRP-Transcripts"


@dataclass
class PipelineConfig:
    video_url: str = ""
    input_audio: Path | None = None
    output_root: Path = field(default_factory=default_output_root)
    model: str = "nepali-medium-ct2"
    fallback_model: str | None = None
    language: str = "ne"
    chunk_seconds: int = 600
    min_chunk_seconds: int = 480
    max_chunk_seconds: int = 720
    silence_noise_db: float = -35.0
    silence_min_seconds: float = 0.5
    chunk_overlap_seconds: float = 0.5
    refine_mode: str = "rule"
    refine_min_similarity: float = 0.90
    run_mode: str = "auto"
    force_overwrite: bool = False
    keep_chunks: bool = False
    keep_wav: bool = False
    cookies_path: Path | None = None
    glossary_path: Path | None = None
    corrections_path: Path | None = None
    corrector_url: str | None = None
    device: str = "auto"
    compute_type: str = "float16"
    beam_size: int = 5
    best_of: int = 5
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    word_timestamps: bool = False
    initial_prompt: str | None = None
    retries: int = 3
    retry_delay: float = 2.0
    cpu_threads: int = 4
    num_workers: int = 1
    model_cache_dir: Path = field(default_factory=default_model_cache_dir)
    drive_model_cache_dir: Path | None = None
    max_chars_per_line: int = 42
    max_lines: int = 2
    max_cps: float = 17.0
    min_caption_duration: float = 1.0
    max_caption_duration: float = 7.0
    log_level: str = "INFO"

    def validate(self) -> None:
        self.output_root = Path(self.output_root).expanduser()
        if self.input_audio is not None:
            self.input_audio = Path(self.input_audio).expanduser()
        for name in (
            "cookies_path",
            "glossary_path",
            "corrections_path",
            "model_cache_dir",
            "drive_model_cache_dir",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, Path(value).expanduser())
        if not str(self.output_root).strip():
            raise PipelineError("output_root must not be empty")
        if self.refine_mode not in REFINE_MODES:
            raise PipelineError(f"refine_mode must be one of {REFINE_MODES}")
        if self.run_mode not in RUN_MODES:
            raise PipelineError(f"run_mode must be one of {RUN_MODES}")
        if self.chunk_seconds <= 0:
            raise PipelineError("chunk_seconds must be positive")
        if not 0 < self.min_chunk_seconds <= self.chunk_seconds:
            raise PipelineError("min_chunk_seconds must be > 0 and <= chunk_seconds")
        if self.max_chunk_seconds < self.chunk_seconds:
            raise PipelineError("max_chunk_seconds must be >= chunk_seconds")
        if self.silence_min_seconds <= 0:
            raise PipelineError("silence_min_seconds must be positive")
        if self.chunk_overlap_seconds < 0:
            raise PipelineError("chunk_overlap_seconds must not be negative")
        if self.retries < 0:
            raise PipelineError("retries must not be negative")
        if self.retry_delay < 0:
            raise PipelineError("retry_delay must not be negative")
        if self.max_chars_per_line <= 0 or self.max_lines <= 0:
            raise PipelineError("caption line settings must be positive")
        if self.max_cps <= 0:
            raise PipelineError("max_cps must be positive")
        if not 0 <= self.refine_min_similarity <= 1:
            raise PipelineError("refine_min_similarity must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Path):
                result[name] = str(value)
            else:
                result[name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        known = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: value for key, value in data.items() if key in known}
        for key in (
            "input_audio",
            "output_root",
            "cookies_path",
            "glossary_path",
            "corrections_path",
            "model_cache_dir",
            "drive_model_cache_dir",
        ):
            if values.get(key):
                values[key] = Path(values[key])
        if "input_audio" in values and values["input_audio"] is None:
            values.pop("input_audio")
        return cls(**values)
