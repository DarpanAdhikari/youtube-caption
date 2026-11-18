from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunking import ChunkPlan
from .config import PipelineConfig
from .errors import ASRError, DependencyError
from .paths import JobPaths, atomic_write_json, read_json
from .progress import ProgressReporter
from .state import ChunkRecord, JobState, mark_chunk, record_error, save_job


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "confidence": (
                None if self.confidence is None else round(self.confidence, 4)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ASRSegment":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data.get("text", "")),
            confidence=(
                None if data.get("confidence") is None else float(data["confidence"])
            ),
        )


@dataclass
class ASRResult:
    segments: list[ASRSegment]
    language: str = ""
    language_probability: float | None = None
    model: str = ""
    engine: str = "faster-whisper"

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [segment.to_dict() for segment in self.segments],
            "language": self.language,
            "language_probability": self.language_probability,
            "model": self.model,
            "engine": self.engine,
        }


class FasterWhisperEngine:
    def __init__(self, model_ref: str | Path, config: PipelineConfig):
        self.model_ref = str(model_ref)
        self.config = config
        self._model = None
        self._device = ""
        self._compute_type = ""

    @property
    def device(self) -> str:
        if not self._device:
            self._device = self._resolve_device()
        return self._device

    @property
    def compute_type(self) -> str:
        if not self._compute_type:
            if self.device == "cuda":
                self._compute_type = self.config.compute_type or "float16"
            else:
                self._compute_type = "int8"
        return self._compute_type

    def _resolve_device(self) -> str:
        requested = (self.config.device or "auto").lower()
        cuda_count = 0
        try:
            import ctranslate2

            cuda_count = int(ctranslate2.get_cuda_device_count())
        except Exception:
            cuda_count = 0
        if requested == "auto":
            return "cuda" if cuda_count > 0 else "cpu"
        if requested == "cuda" and cuda_count <= 0:
            raise ASRError(
                "CUDA was requested but no CUDA device is visible to CTranslate2. "
                "Set device='cpu' or enable a GPU runtime."
            )
        return requested

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyError(
                "faster-whisper is required. Install it with "
                "`pip install faster-whisper`."
            ) from exc

        try:
            self._model = WhisperModel(
                self.model_ref,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(self.config.model_cache_dir),
                cpu_threads=self.config.cpu_threads,
                num_workers=max(1, self.config.num_workers),
            )
        except Exception as exc:
            raise ASRError(
                f"could not load faster-whisper model {self.model_ref}: {exc}"
            ) from exc

    def unload(self) -> None:
        self._model = None

    def transcribe(
        self,
        audio_path: Path,
        *,
        offset: float = 0.0,
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> ASRResult:
        self.load()
        assert self._model is not None
        prompt = initial_prompt if initial_prompt is not None else self.config.initial_prompt
        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=(
                    None
                    if (language or self.config.language) in ("", "auto")
                    else (language or self.config.language)
                ),
                task="transcribe",
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=0.0,
                vad_filter=self.config.vad_filter,
                condition_on_previous_text=self.config.condition_on_previous_text,
                word_timestamps=self.config.word_timestamps,
                initial_prompt=prompt or None,
            )
            segments: list[ASRSegment] = []
            for segment in segments_iter:
                text = str(segment.text or "").strip()
                if not text:
                    continue
                start = float(segment.start or 0.0) + offset
                end = float(segment.end or 0.0) + offset
                if end <= start:
                    end = start + 0.01
                confidence = None
                if segment.avg_logprob is not None:
                    confidence = max(0.0, min(1.0, 1.0 + float(segment.avg_logprob)))
                segments.append(
                    ASRSegment(
                        start=start,
                        end=end,
                        text=text,
                        confidence=confidence,
                    )
                )
        except Exception as exc:
            raise ASRError(f"transcription failed for {audio_path}: {exc}") from exc

        return ASRResult(
            segments=segments,
            language=str(getattr(info, "language", "") or ""),
            language_probability=(
                None
                if getattr(info, "language_probability", None) is None
                else float(info.language_probability)
            ),
            model=self.model_ref,
            engine="faster-whisper",
        )


def chunk_result_path(paths: JobPaths, chunk: ChunkRecord) -> Path:
    return paths.chunk_result_file(chunk.index)


def write_chunk_result(
    paths: JobPaths,
    chunk: ChunkRecord,
    result: ASRResult,
) -> Path:
    target = chunk_result_path(paths, chunk)
    atomic_write_json(
        target,
        {
            "chunk": chunk.to_dict(),
            "model": result.model,
            "engine": result.engine,
            "language": result.language,
            "language_probability": result.language_probability,
            "segments": [segment.to_dict() for segment in result.segments],
        },
    )
    return target


def read_chunk_result(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ASRError(f"invalid chunk result: {path}")
    return data


def _all_chunks_failed(job: JobState) -> bool:
    return bool(job.chunks) and all(chunk.status == "failed" for chunk in job.chunks)


def transcribe_chunks(
    engine: FasterWhisperEngine,
    plans: list[ChunkPlan],
    paths: JobPaths,
    job: JobState,
    reporter: ProgressReporter,
) -> None:
    total = len(plans)
    if total == 0:
        raise ASRError("no chunks to transcribe")
    plan_by_index = {plan.index: plan for plan in plans}
    try:
        engine.load()
    except Exception as exc:
        raise ASRError(f"could not load ASR model: {exc}") from exc

    for position, chunk in enumerate(job.chunks, start=1):
        plan = plan_by_index.get(chunk.index)
        if plan is None:
            raise ASRError(f"missing plan for chunk {chunk.index}")
        result_path = chunk_result_path(paths, chunk)
        if chunk.status == "done" and result_path.exists():
            reporter.update(
                "transcribe",
                position,
                total,
                message="transcribing chunks",
                detail=f"skipping completed chunk {chunk.index}",
            )
            continue
        if chunk.status == "done" and not result_path.exists():
            chunk.status = "pending"

        audio_path = paths.chunk_audio_file(chunk.index)
        if not audio_path.exists():
            chunk.status = "failed"
            chunk.error = f"missing chunk audio: {audio_path}"
            mark_chunk(paths, job, chunk, "failed", error=chunk.error)
            record_error(
                paths,
                job,
                "transcribe",
                ASRError(chunk.error),
                chunk_index=chunk.index,
            )
            continue

        max_attempts = max(0, engine.config.retries) + 1
        for attempt in range(1, max_attempts + 1):
            chunk.attempts += 1
            mark_chunk(paths, job, chunk, "transcribing")
            reporter.update(
                "transcribe",
                position - 1,
                total,
                message="transcribing chunks",
                detail=(
                    f"chunk {chunk.index} attempt {attempt}/{max_attempts}"
                ),
            )
            try:
                result = engine.transcribe(
                    audio_path,
                    offset=plan.audio_start,
                )
                write_chunk_result(paths, chunk, result)
                mark_chunk(
                    paths,
                    job,
                    chunk,
                    "done",
                    segments_path=str(result_path),
                )
                reporter.update(
                    "transcribe",
                    position,
                    total,
                    message="transcribing chunks",
                    detail=(
                        f"chunk {chunk.index}: {len(result.segments)} segments"
                    ),
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                chunk.error = str(exc)
                if attempt < max_attempts:
                    reporter.emit(
                        "transcribe",
                        message=f"chunk {chunk.index} failed; retrying",
                        detail=str(exc),
                    )
                    time.sleep(max(0.0, engine.config.retry_delay) * attempt)
                    continue
                mark_chunk(paths, job, chunk, "failed", error=str(exc))
                record_error(
                    paths,
                    job,
                    "transcribe",
                    exc,
                    chunk_index=chunk.index,
                )
                reporter.emit(
                    "transcribe",
                    message=f"chunk {chunk.index} failed permanently",
                    detail=str(exc),
                )

    job.refresh_counts()
    save_job(paths, job)
    if _all_chunks_failed(job):
        raise ASRError("all chunks failed transcription")
