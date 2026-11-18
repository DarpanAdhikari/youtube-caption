from __future__ import annotations

import importlib.util
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .merge import (
    TranscriptSegment,
    segments_to_text,
    timestamped_paragraphs,
)
from .paths import JobPaths, atomic_write_json, atomic_write_text
from .state import JobState


@dataclass
class CaptionCue:
    index: int
    start: float
    end: float
    lines: list[str]


@dataclass
class ExportResult:
    files: dict[str, Path]
    warnings: list[str]


def _wrap(text: str, width: int, max_lines: int | None) -> list[str]:
    lines = textwrap.wrap(text, width=width) or [""]
    if max_lines is not None and len(lines) > max_lines:
        return lines[:max_lines]
    return lines


def build_caption_cues(
    segments: list[TranscriptSegment],
    config: PipelineConfig,
) -> list[CaptionCue]:
    cues: list[CaptionCue] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        duration = max(0.001, segment.end - segment.start)
        required_read_time = len(text) / config.max_cps
        wrapped = textwrap.wrap(text, width=config.max_chars_per_line) or [""]
        needs_split = (
            duration > config.max_caption_duration
            or required_read_time > duration
            or len(wrapped) > config.max_lines
        )
        if not needs_split or duration <= config.min_caption_duration:
            cues.append(
                CaptionCue(
                    index=len(cues) + 1,
                    start=segment.start,
                    end=segment.end,
                    lines=_wrap(
                        text,
                        config.max_chars_per_line,
                        None if len(wrapped) > config.max_lines else config.max_lines,
                    ),
                )
            )
            continue

        chunk_width = config.max_chars_per_line * config.max_lines
        words = text.split()
        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join(current + [word])
            if len(candidate) > chunk_width and current:
                chunks.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current))

        total_chars = sum(len(chunk) for chunk in chunks) or 1
        cursor = segment.start
        for position, chunk in enumerate(chunks):
            share = len(chunk) / total_chars
            chunk_duration = max(duration * share, 0.01)
            chunk_end = (
                segment.end
                if position == len(chunks) - 1
                else cursor + chunk_duration
            )
            cues.append(
                CaptionCue(
                    index=len(cues) + 1,
                    start=round(cursor, 3),
                    end=round(chunk_end, 3),
                    lines=_wrap(
                        chunk,
                        config.max_chars_per_line,
                        config.max_lines,
                    ),
                )
            )
            cursor = chunk_end
    return cues


def _format_srt_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _format_clock(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_srt(cues: list[CaptionCue]) -> str:
    blocks = []
    for cue in cues:
        blocks.append(
            f"{cue.index}\n"
            f"{_format_srt_timestamp(cue.start)} --> {_format_srt_timestamp(cue.end)}\n"
            f"{chr(10).join(cue.lines)}\n"
        )
    return "\n".join(blocks)


def render_vtt(cues: list[CaptionCue]) -> str:
    blocks = ["WEBVTT\n"]
    for cue in cues:
        blocks.append(
            f"{cue.index}\n"
            f"{_format_vtt_timestamp(cue.start)} --> {_format_vtt_timestamp(cue.end)}\n"
            f"{chr(10).join(cue.lines)}\n"
        )
    return "\n".join(blocks)


def _video_payload(job: JobState) -> dict[str, Any]:
    return {
        "video_id": job.video_id,
        "url": job.url,
        "title": job.title,
        "channel": job.channel,
        "duration": job.duration,
        "language": job.language,
        "model": job.model,
        "refine_mode": job.refine_mode,
    }


def _write_docx(
    path: Path,
    job: JobState,
    segments: list[TranscriptSegment],
    config: PipelineConfig,
) -> bool:
    if importlib.util.find_spec("docx") is None:
        return False
    from docx import Document

    document = Document()
    document.add_heading(job.title or job.video_id, level=0)
    document.add_paragraph(f"Source: {job.url or 'local audio'}")
    document.add_paragraph(
        f"Duration: {_format_clock(job.duration)} | "
        f"Language: {job.language} | "
        f"Model: {job.model} | "
        f"Refinement: {job.refine_mode}"
    )
    document.add_heading("Transcript", level=1)
    for start, text in timestamped_paragraphs(
        segments,
        paragraph_seconds=60.0,
        paragraph_chars=700,
    ):
        paragraph = document.add_paragraph()
        run = paragraph.add_run(f"[{_format_clock(start)}] ")
        run.bold = True
        paragraph.add_run(text)
    document.save(str(path))
    return True


def export_all(
    paths: JobPaths,
    job: JobState,
    raw_segments: list[TranscriptSegment],
    refined_segments: list[TranscriptSegment],
    config: PipelineConfig,
    *,
    quality: dict[str, Any] | None = None,
    refinement_report: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> ExportResult:
    output = paths.output_dir
    output.mkdir(parents=True, exist_ok=True)
    known_warnings = list(warnings or [])
    files: dict[str, Path] = {}

    raw_text = segments_to_text(raw_segments)
    refined_text = segments_to_text(refined_segments)
    raw_path = output / "transcript.raw.txt"
    refined_path = output / "transcript.txt"
    atomic_write_text(raw_path, raw_text + ("\n" if raw_text else ""))
    atomic_write_text(refined_path, refined_text + ("\n" if refined_text else ""))
    files["transcript_raw_txt"] = raw_path
    files["transcript_txt"] = refined_path

    raw_cues = build_caption_cues(raw_segments, config)
    refined_cues = build_caption_cues(refined_segments, config)
    raw_srt = output / "subtitles.raw.srt"
    raw_vtt = output / "subtitles.raw.vtt"
    refined_srt = output / "subtitles.srt"
    refined_vtt = output / "subtitles.vtt"
    atomic_write_text(raw_srt, render_srt(raw_cues))
    atomic_write_text(raw_vtt, render_vtt(raw_cues))
    atomic_write_text(refined_srt, render_srt(refined_cues))
    atomic_write_text(refined_vtt, render_vtt(refined_cues))
    files["subtitles_raw_srt"] = raw_srt
    files["subtitles_raw_vtt"] = raw_vtt
    files["subtitles_srt"] = refined_srt
    files["subtitles_vtt"] = refined_vtt

    combined = {
        "schema_version": 1,
        "video": _video_payload(job),
        "settings": config.to_dict(),
        "raw": {
            "segments": [segment.to_dict() for segment in raw_segments],
            "text": raw_text,
        },
        "refined": {
            "segments": [segment.to_dict() for segment in refined_segments],
            "text": refined_text,
        },
        "quality": quality or {},
        "refinement": refinement_report or {},
    }
    combined_path = output / "transcript.json"
    atomic_write_json(combined_path, combined)
    files["transcript_json"] = combined_path

    docx_path = output / "transcript.docx"
    if _write_docx(docx_path, job, refined_segments, config):
        files["transcript_docx"] = docx_path
    else:
        known_warnings.append(
            "python-docx is not installed; transcript.docx was skipped"
        )

    metadata = {
        "schema_version": 1,
        "video": _video_payload(job),
        "source": "youtube" if job.url else "local",
        "created_at": job.updated_at,
        "counts": {
            "raw_segments": len(raw_segments),
            "refined_segments": len(refined_segments),
            "raw_characters": len(raw_text),
            "refined_characters": len(refined_text),
        },
        "quality": quality or {},
        "refinement": refinement_report or {},
        "warnings": known_warnings,
        "files": sorted(path.name for path in files.values()),
    }
    metadata_path = output / "metadata.json"
    atomic_write_json(metadata_path, metadata)
    files["metadata_json"] = metadata_path

    bundle_path = output / "transcript_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files.values()):
            archive.write(path, arcname=path.name)
    files["bundle_zip"] = bundle_path

    return ExportResult(files=files, warnings=known_warnings)
