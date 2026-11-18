from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asr import FasterWhisperEngine, transcribe_chunks
from .audio import (
    AudioMetadata,
    download_audio,
    fetch_video_metadata,
    normalize_audio,
    probe_media,
)
from .chunking import (
    ChunkPlan,
    chunks_from_dicts,
    detect_silences,
    plan_chunks,
    slice_chunks,
    validate_chunk_plan,
)
from .config import PipelineConfig
from .errors import ASRError, DownloadError, PipelineError, StateError
from .exports import ExportResult, export_all
from .merge import (
    TranscriptSegment,
    assess_quality,
    load_chunk_segments,
    load_raw_segments,
    load_refined_segments,
    merge_segments,
    save_raw_transcript,
    save_refined_transcript,
)
from .models import ModelSpec, ensure_model, resolve_model
from .paths import (
    JobPaths,
    find_existing_job,
    local_video_id,
    parse_video_id,
    validate_root_is_writable,
)
from .progress import ProgressReporter
from .refine import (
    CorrectionDictionary,
    Glossary,
    build_initial_prompt,
    refine_segments,
    save_refinement_outputs,
)
from .state import (
    ChunkRecord,
    JobState,
    load_job,
    new_job,
    record_error,
    reset_failed_chunks,
    save_job,
    set_status,
)


@dataclass
class PipelineResult:
    job: JobState
    paths: JobPaths
    exports: ExportResult
    raw_segments: list[TranscriptSegment]
    refined_segments: list[TranscriptSegment]


class TranscriptionPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        reporter: ProgressReporter | None = None,
    ):
        self.config = config
        self.reporter = reporter or ProgressReporter()
        self.paths: JobPaths | None = None
        self.job: JobState | None = None
        self.known_metadata: dict[str, Any] | None = None

    def run(self) -> PipelineResult:
        self.config.validate()
        validate_root_is_writable(self.config.output_root)
        try:
            self._prepare()
            assert self.paths is not None and self.job is not None
            mode = self._effective_mode()
            if mode == "export_only":
                return self._export_only()
            if mode == "refine_only":
                return self._refine_only()
            return self._run_full(mode)
        except KeyboardInterrupt:
            self._mark_interrupted()
            raise
        except BaseException as exc:
            self._mark_failed(exc)
            raise

    def _prepare(self) -> None:
        config = self.config
        if config.input_audio is not None:
            video_id = local_video_id(config.input_audio)
            existing = find_existing_job(config.output_root, video_id)
            self._prepare_with_existing(existing, video_id, title=config.input_audio.stem)
            return

        if not config.video_url.strip():
            raise PipelineError("either video_url or input_audio is required")
        video_id = parse_video_id(config.video_url)
        existing = find_existing_job(config.output_root, video_id)
        if existing is not None:
            self._prepare_with_existing(existing, video_id)
            return

        preflight_error: str | None = None
        try:
            self.known_metadata = fetch_video_metadata(config.video_url, config)
            title = str(self.known_metadata.get("title") or video_id)
            preflight_id = str(self.known_metadata.get("video_id") or "").strip()
            if preflight_id:
                video_id = preflight_id
        except DownloadError as exc:
            preflight_error = str(exc)
            title = video_id
        except Exception:
            preflight_error = "metadata preflight failed"
            title = video_id

        existing = find_existing_job(config.output_root, video_id)
        if existing is not None:
            self._prepare_with_existing(existing, video_id)
            return

        paths = JobPaths.for_video(config.output_root, video_id, title).ensure()
        if paths.job_file.exists():
            job = load_job(paths)
            if job is not None and config.run_mode != "fresh":
                self.paths = paths
                self.job = job
                self._apply_config_to_job()
                return
        if config.run_mode in {"resume", "export_only", "refine_only", "retry_failed"}:
            raise StateError(
                f"no existing job found for {video_id} under {config.output_root}"
            )
        job = new_job(
            config,
            video_id=video_id,
            url=config.video_url,
            title=title,
            paths=paths,
        )
        if preflight_error:
            job.warnings.append(preflight_error)
        self.paths = paths
        self.job = job
        save_job(paths, job)

    def _prepare_with_existing(
        self,
        paths: JobPaths | None,
        video_id: str,
        *,
        title: str | None = None,
    ) -> None:
        if paths is None:
            if title is None:
                title = video_id
            paths = JobPaths.for_video(self.config.output_root, video_id, title).ensure()
        paths.ensure()
        job = load_job(paths)
        if job is None:
            if self.config.run_mode in {
                "resume",
                "export_only",
                "refine_only",
                "retry_failed",
            }:
                raise StateError(f"job file is missing under {paths.root}")
            job = new_job(
                self.config,
                video_id=video_id,
                url=self.config.video_url,
                title=title or video_id,
                paths=paths,
            )
        self.paths = paths
        self.job = job
        if self.config.run_mode == "fresh":
            if not self.config.force_overwrite:
                raise StateError(
                    f"job already exists at {paths.root}; use --force-overwrite "
                    "or choose --run-mode resume"
                )
            paths.reset_run_outputs()
            self.job = new_job(
                self.config,
                video_id=video_id,
                url=job.url or self.config.video_url,
                title=job.title or title or video_id,
                paths=paths,
            )
            self.paths = paths
            return
        if self.job.model:
            self.config.model = self.job.model
        self._apply_config_to_job()

    def _apply_config_to_job(self) -> None:
        assert self.job is not None and self.paths is not None
        self.job.config = self.config.to_dict()
        save_job(self.paths, self.job)

    def _effective_mode(self) -> str:
        assert self.job is not None
        if self.config.run_mode in {"export_only", "refine_only"}:
            return self.config.run_mode
        if self.config.run_mode == "auto" and self.job.is_completed:
            return "export_only"
        return self.config.run_mode

    def _check_resume_compatibility(self) -> None:
        assert self.job is not None
        if not self.job.chunks:
            return
        if self.job.model and self.config.model and self.job.model != self.config.model:
            raise StateError(
                f"job was started with model {self.job.model!r} but the config asks "
                f"for {self.config.model!r}. Use --run-mode fresh --force-overwrite "
                "to restart with another model."
            )
        if self.job.language != self.config.language:
            raise StateError(
                "language changed for an existing job; restart with --run-mode fresh"
            )
        if self.job.chunk_seconds != self.config.chunk_seconds:
            raise StateError(
                "chunk_seconds changed for an existing job; restart with --run-mode fresh"
            )
        if (
            self.job.config.get("min_chunk_seconds") != self.config.min_chunk_seconds
            or self.job.config.get("max_chunk_seconds") != self.config.max_chunk_seconds
            or self.job.config.get("silence_noise_db") != self.config.silence_noise_db
            or self.job.config.get("silence_min_seconds")
            != self.config.silence_min_seconds
        ):
            raise StateError(
                "silence/chunk settings changed for an existing job; "
                "restart with --run-mode fresh"
            )

    def _run_full(self, mode: str) -> PipelineResult:
        assert self.paths is not None and self.job is not None
        paths = self.paths
        job = self.job
        if mode in {"resume", "retry_failed", "auto"}:
            self._check_resume_compatibility()
        if mode == "retry_failed":
            changed = reset_failed_chunks(paths, job)
            if changed:
                self.reporter.stage(
                    "resume",
                    f"reset {changed} failed chunk(s) for another attempt",
                )

        set_status(paths, job, "downloading", "download")
        metadata = download_audio(
            self.config,
            paths,
            self.reporter,
            known_metadata=self.known_metadata,
        )
        self._update_job_from_audio(job, metadata)

        set_status(paths, job, "preparing", "prepare")
        audio_path = Path(str(job.audio_path))
        wav_path = normalize_audio(
            audio_path,
            paths.normalized_wav,
            self.reporter,
            force=False,
        )
        try:
            duration = float(probe_media(wav_path)["duration"])
        except Exception:
            duration = float(job.duration or metadata.duration)
        if duration > 0:
            job.duration = duration

        plans = self._prepare_chunks(wav_path, job)
        pending = [chunk for chunk in job.chunks if chunk.status != "done"]
        if pending:
            set_status(paths, job, "transcribing", "transcribe")
            engine, warning = self._build_engine(job)
            if warning:
                job.warnings.append(warning)
            glossary = self._load_glossary()
            prompt = build_initial_prompt(self.config, glossary)
            if prompt:
                self.config.initial_prompt = prompt
                engine.config.initial_prompt = prompt
            try:
                transcribe_chunks(engine, plans, paths, job, self.reporter)
            except ASRError:
                raise
            except Exception as exc:
                raise ASRError(str(exc)) from exc

        set_status(paths, job, "merging", "merge")
        chunk_segments = load_chunk_segments(paths, job)
        if not chunk_segments:
            raise ASRError("no chunk transcripts were produced")
        raw_segments = merge_segments(
            chunk_segments,
            job,
            total_duration=job.duration,
        )
        quality = assess_quality(raw_segments, job.duration)
        save_raw_transcript(paths, job, raw_segments, quality)
        job.warnings.extend(
            warning for warning in quality.get("warnings", []) if warning not in job.warnings
        )

        set_status(paths, job, "refining", "refine")
        refined_segments, refinement_report = self._refine(raw_segments, job)
        save_refined_transcript(
            paths,
            job,
            refined_segments,
            report=refinement_report,
        )

        set_status(paths, job, "exporting", "export")
        export_result = export_all(
            paths,
            job,
            raw_segments,
            refined_segments,
            self.config,
            quality=quality,
            refinement_report=refinement_report,
            warnings=job.warnings,
        )
        job.warnings = list(dict.fromkeys(job.warnings + export_result.warnings))
        final_status = (
            "completed_with_errors"
            if any(chunk.status == "failed" for chunk in job.chunks)
            else "completed"
        )
        set_status(paths, job, final_status, "completed")
        self._cleanup(paths, job)
        self.reporter.stage(
            "done",
            f"{job.title or job.video_id} ({final_status})",
        )
        return PipelineResult(
            job=job,
            paths=paths,
            exports=export_result,
            raw_segments=raw_segments,
            refined_segments=refined_segments,
        )

    def _update_job_from_audio(
        self,
        job: JobState,
        metadata: AudioMetadata,
    ) -> None:
        assert self.paths is not None
        if metadata.video_id:
            job.video_id = metadata.video_id
        if metadata.title:
            job.title = metadata.title
        if metadata.channel:
            job.channel = metadata.channel
        if metadata.url:
            job.url = metadata.url
        if metadata.duration > 0:
            job.duration = metadata.duration
        job.audio_path = metadata.audio_path
        save_job(self.paths, job)

    def _prepare_chunks(self, wav_path: Path, job: JobState) -> list[ChunkPlan]:
        assert self.paths is not None
        paths = self.paths
        if job.chunks:
            plans = [
                ChunkPlan(
                    index=chunk.index,
                    start=chunk.start,
                    end=chunk.end,
                    audio_start=chunk.audio_start,
                    audio_end=chunk.audio_end,
                    cut_reason=chunk.cut_reason,
                )
                for chunk in job.chunks
            ]
            validate_chunk_plan(plans, job.duration)
            self.reporter.stage("chunk", "reusing existing chunk plan")
        else:
            set_status(paths, job, "chunking", "chunk")
            self.reporter.stage("chunk", "detecting silences")
            silences = detect_silences(
                wav_path,
                noise_db=self.config.silence_noise_db,
                min_silence=self.config.silence_min_seconds,
                duration=job.duration,
            )
            plans = plan_chunks(job.duration, silences, self.config)
            validate_chunk_plan(plans, job.duration)
            job.chunks = [
                ChunkRecord(
                    index=plan.index,
                    start=plan.start,
                    end=plan.end,
                    audio_start=plan.audio_start,
                    audio_end=plan.audio_end,
                    path=str(paths.chunk_audio_file(plan.index)),
                    cut_reason=plan.cut_reason,
                )
                for plan in plans
            ]
            save_job(paths, job)

        slice_chunks(
            wav_path,
            plans,
            paths.chunk_audio_dir,
            self.reporter,
            reuse=True,
        )
        return plans

    def _build_engine(self, job: JobState) -> tuple[FasterWhisperEngine, str | None]:
        spec = resolve_model(self.config.model)
        try:
            model_ref = ensure_model(spec, self.config, self.reporter)
            engine = FasterWhisperEngine(model_ref, self.config)
            engine.load()
            return engine, None
        except Exception as exc:
            fallback_name = self.config.fallback_model
            if not fallback_name:
                raise
            self.reporter.stage(
                "model",
                f"primary model failed ({exc}); trying {fallback_name}",
            )
            fallback = resolve_model(fallback_name)
            model_ref = ensure_model(fallback, self.config, self.reporter)
            engine = FasterWhisperEngine(model_ref, self.config)
            engine.load()
            job.model = fallback.key
            save_job(self.paths, job)
            return (
                engine,
                f"primary model failed; used fallback {fallback.key}: {exc}",
            )

    def _load_glossary(self) -> Glossary:
        return Glossary.from_file(self.config.glossary_path)

    def _load_dictionary(self, glossary: Glossary) -> CorrectionDictionary:
        dictionary = CorrectionDictionary.from_file(self.config.corrections_path)
        if glossary.variants:
            merged = CorrectionDictionary({**glossary.variants, **dictionary.entries})
            return merged
        return dictionary

    def _refine(
        self,
        raw_segments: list[TranscriptSegment],
        job: JobState,
    ) -> tuple[list[TranscriptSegment], dict[str, Any]]:
        job.refine_mode = self.config.refine_mode
        glossary = self._load_glossary()
        dictionary = self._load_dictionary(glossary)
        result = refine_segments(
            raw_segments,
            self.config,
            glossary=glossary,
            dictionary=dictionary,
            reporter=self.reporter,
        )
        save_refinement_outputs(
            self.paths,
            result.changes,
            result.report,
        )
        if (
            self.config.refine_mode == "neural"
            and not self.config.corrector_url
            and "neural refinement requested without a corrector URL; used rules"
            not in job.warnings
        ):
            job.warnings.append(
                "neural refinement requested without a corrector URL; used rules"
            )
        return result.segments, result.report

    def _export_only(self) -> PipelineResult:
        assert self.paths is not None and self.job is not None
        paths = self.paths
        job = self.job
        try:
            raw_segments, _ = load_raw_segments(paths)
        except FileNotFoundError:
            chunk_segments = load_chunk_segments(paths, job)
            if not chunk_segments:
                raise StateError(
                    "no raw transcript or completed chunks are available for export"
                )
            raw_segments = merge_segments(
                chunk_segments,
                job,
                total_duration=job.duration,
            )
            quality = assess_quality(raw_segments, job.duration)
            save_raw_transcript(paths, job, raw_segments, quality)
        else:
            quality = assess_quality(raw_segments, job.duration)

        refined = load_refined_segments(paths)
        if refined is not None:
            refined_segments, refinement_report = refined
        else:
            set_status(paths, job, "refining", "refine")
            refined_segments, refinement_report = self._refine(raw_segments, job)
            save_refined_transcript(
                paths,
                job,
                refined_segments,
                report=refinement_report,
            )

        set_status(paths, job, "exporting", "export")
        export_result = export_all(
            paths,
            job,
            raw_segments,
            refined_segments,
            self.config,
            quality=quality,
            refinement_report=refinement_report,
            warnings=job.warnings,
        )
        job.warnings = list(dict.fromkeys(job.warnings + export_result.warnings))
        final_status = (
            "completed_with_errors"
            if any(chunk.status == "failed" for chunk in job.chunks)
            else "completed"
        )
        set_status(paths, job, final_status, "completed")
        self._cleanup(paths, job)
        return PipelineResult(
            job=job,
            paths=paths,
            exports=export_result,
            raw_segments=raw_segments,
            refined_segments=refined_segments,
        )

    def _refine_only(self) -> PipelineResult:
        assert self.paths is not None and self.job is not None
        paths = self.paths
        job = self.job
        raw_segments, _ = load_raw_segments(paths)
        set_status(paths, job, "refining", "refine")
        refined_segments, refinement_report = self._refine(raw_segments, job)
        save_refined_transcript(
            paths,
            job,
            refined_segments,
            report=refinement_report,
        )
        set_status(paths, job, "exporting", "export")
        export_result = export_all(
            paths,
            job,
            raw_segments,
            refined_segments,
            self.config,
            quality=assess_quality(raw_segments, job.duration),
            refinement_report=refinement_report,
            warnings=job.warnings,
        )
        job.warnings = list(dict.fromkeys(job.warnings + export_result.warnings))
        set_status(paths, job, "completed", "completed")
        return PipelineResult(
            job=job,
            paths=paths,
            exports=export_result,
            raw_segments=raw_segments,
            refined_segments=refined_segments,
        )

    def _cleanup(self, paths: JobPaths, job: JobState) -> None:
        if any(chunk.status == "failed" for chunk in job.chunks):
            return
        if not self.config.keep_chunks and paths.chunk_audio_dir.exists():
            shutil.rmtree(paths.chunk_audio_dir, ignore_errors=True)
        if not self.config.keep_wav and paths.normalized_wav.exists():
            paths.normalized_wav.unlink(missing_ok=True)

    def _mark_interrupted(self) -> None:
        if self.paths is not None and self.job is not None:
            self.job.status = "interrupted"
            self.job.stage = "interrupted"
            save_job(self.paths, self.job)

    def _mark_failed(self, exc: BaseException) -> None:
        if self.paths is not None and self.job is not None:
            self.job.status = "failed"
            self.job.last_error = str(exc)
            try:
                record_error(self.paths, self.job, self.job.stage or "pipeline", exc)
            except Exception:
                save_job(self.paths, self.job)


def run_pipeline(
    config: PipelineConfig,
    reporter: ProgressReporter | None = None,
) -> PipelineResult:
    return TranscriptionPipeline(config, reporter).run()
