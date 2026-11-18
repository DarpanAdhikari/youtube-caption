from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .asr import read_chunk_result
from .paths import JobPaths, atomic_write_json, read_json_or_none
from .state import JobState


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    confidence: float | None = None
    chunk_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "confidence": (
                None if self.confidence is None else round(self.confidence, 4)
            ),
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            text=str(data.get("text", "")),
            confidence=(
                None if data.get("confidence") is None else float(data["confidence"])
            ),
            chunk_index=(
                None if data.get("chunk_index") is None else int(data["chunk_index"])
            ),
        )


def load_chunk_segments(paths: JobPaths, job: JobState) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for chunk in job.chunks:
        if chunk.status != "done":
            continue
        result_path = (
            Path(chunk.segments_path)
            if chunk.segments_path
            else paths.chunk_result_file(chunk.index)
        )
        if not result_path.exists():
            continue
        data = read_chunk_result(result_path)
        for item in data.get("segments", []):
            segments.append(
                TranscriptSegment(
                    start=float(item["start"]),
                    end=float(item["end"]),
                    text=str(item.get("text", "")).strip(),
                    confidence=(
                        None
                        if item.get("confidence") is None
                        else float(item["confidence"])
                    ),
                    chunk_index=chunk.index,
                )
            )
    return segments


def merge_segments(
    segments: list[TranscriptSegment],
    job: JobState,
    *,
    total_duration: float | None = None,
) -> list[TranscriptSegment]:
    duration = float(total_duration or job.duration or 0.0)
    chunk_bounds = {
        chunk.index: (chunk.start, chunk.end)
        for chunk in job.chunks
    }
    filtered: list[TranscriptSegment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        if duration > 0:
            start = min(start, duration)
            end = min(end, duration)
        if end - start <= 0.001:
            continue
        if segment.chunk_index is not None and segment.chunk_index in chunk_bounds:
            chunk_start, chunk_end = chunk_bounds[segment.chunk_index]
            midpoint = (start + end) / 2
            if midpoint < chunk_start - 0.001 or midpoint > chunk_end + 0.001:
                continue
        filtered.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                confidence=segment.confidence,
                chunk_index=segment.chunk_index,
            )
        )

    filtered.sort(key=lambda item: (item.start, item.end))
    merged: list[TranscriptSegment] = []
    for segment in filtered:
        if not merged:
            merged.append(segment)
            continue
        previous = merged[-1]
        if (
            segment.text == previous.text
            and segment.start <= previous.end + 0.25
        ):
            previous.end = max(previous.end, segment.end)
            continue
        if segment.start < previous.end - 0.01:
            segment.start = previous.end
        if segment.end <= segment.start:
            continue
        merged.append(segment)

    monotonic: list[TranscriptSegment] = []
    for segment in merged:
        if monotonic:
            previous = monotonic[-1]
            if segment.start < previous.start:
                segment.start = previous.start
            if segment.start < previous.end:
                segment.start = previous.end
            if segment.end <= segment.start:
                continue
        monotonic.append(segment)
    return monotonic


def segments_to_text(
    segments: list[TranscriptSegment],
    *,
    paragraph_seconds: float = 60.0,
    paragraph_chars: int = 700,
) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    current_start = None
    current_chars = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if current_start is None:
            current_start = segment.start
        current.append(text)
        current_chars += len(text)
        sentence_end = text.endswith(("।", "?", "!", ".", "॥"))
        long_gap = segment.end - current_start >= paragraph_seconds
        if (sentence_end and current_chars >= 120) or long_gap or current_chars >= paragraph_chars:
            paragraphs.append(" ".join(current))
            current = []
            current_start = None
            current_chars = 0
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def timestamped_paragraphs(
    segments: list[TranscriptSegment],
    *,
    paragraph_seconds: float = 60.0,
    paragraph_chars: int = 700,
) -> list[tuple[float, str]]:
    paragraphs: list[tuple[float, str]] = []
    current: list[str] = []
    current_start = None
    current_chars = 0
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if current_start is None:
            current_start = segment.start
        current.append(text)
        current_chars += len(text)
        sentence_end = text.endswith(("।", "?", "!", ".", "॥"))
        long_gap = segment.end - current_start >= paragraph_seconds
        if (sentence_end and current_chars >= 120) or long_gap or current_chars >= paragraph_chars:
            paragraphs.append((current_start, " ".join(current)))
            current = []
            current_start = None
            current_chars = 0
    if current:
        paragraphs.append((current_start or 0.0, " ".join(current)))
    return paragraphs


def _repetition_ratio(text: str, n: int = 8) -> float:
    tokens = text.split()
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def assess_quality(
    segments: list[TranscriptSegment],
    duration: float,
) -> dict[str, Any]:
    text = " ".join(segment.text for segment in segments)
    chars = len(text.replace(" ", ""))
    chars_per_second = chars / duration if duration > 0 else 0.0
    repetition = _repetition_ratio(text)
    warnings: list[str] = []
    if segments and chars_per_second < 1.0:
        warnings.append(
            f"very low text density ({chars_per_second:.2f} chars/second); "
            "check the model/language"
        )
    if chars_per_second > 40.0:
        warnings.append(
            f"very high text density ({chars_per_second:.2f} chars/second); "
            "possible repeated or hallucinated text"
        )
    if repetition >= 0.35:
        warnings.append(
            f"high repeated n-gram ratio ({repetition:.0%}); inspect for loops"
        )
    empty = sum(1 for segment in segments if not segment.text.strip())
    if segments and empty / len(segments) > 0.2:
        warnings.append("many empty segments")
    return {
        "segments": len(segments),
        "characters": chars,
        "chars_per_second": round(chars_per_second, 3),
        "repetition_ratio": round(repetition, 4),
        "warnings": warnings,
    }


def save_raw_transcript(
    paths: JobPaths,
    job: JobState,
    segments: list[TranscriptSegment],
    quality: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": 1,
        "video_id": job.video_id,
        "url": job.url,
        "title": job.title,
        "language": job.language,
        "model": job.model,
        "duration": job.duration,
        "segments": [segment.to_dict() for segment in segments],
        "text": segments_to_text(segments),
        "quality": quality,
    }
    atomic_write_json(paths.raw_transcript_file, payload)
    return paths.raw_transcript_file


def save_refined_transcript(
    paths: JobPaths,
    job: JobState,
    segments: list[TranscriptSegment],
    *,
    report: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "schema_version": 1,
        "video_id": job.video_id,
        "url": job.url,
        "title": job.title,
        "language": job.language,
        "model": job.model,
        "duration": job.duration,
        "refine_mode": job.refine_mode,
        "segments": [segment.to_dict() for segment in segments],
        "text": segments_to_text(segments),
        "refinement": report or {},
    }
    atomic_write_json(paths.refined_transcript_file, payload)
    return paths.refined_transcript_file


def _segments_from_payload(payload: dict[str, Any]) -> list[TranscriptSegment]:
    return [
        TranscriptSegment.from_dict(item)
        for item in payload.get("segments", [])
    ]


def load_raw_segments(paths: JobPaths) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    payload = read_json_or_none(paths.raw_transcript_file)
    if payload is None:
        raise FileNotFoundError(paths.raw_transcript_file)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid raw transcript: {paths.raw_transcript_file}")
    return _segments_from_payload(payload), payload


def load_refined_segments(
    paths: JobPaths,
) -> tuple[list[TranscriptSegment], dict[str, Any]] | None:
    payload = read_json_or_none(paths.refined_transcript_file)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"invalid refined transcript: {paths.refined_transcript_file}")
    return _segments_from_payload(payload), payload
