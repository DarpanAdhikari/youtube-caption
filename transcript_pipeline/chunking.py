from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .errors import ChunkingError
from .progress import ProgressReporter

_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")
_SILENCE_DURATION = re.compile(r"silence_duration:\s*([0-9.]+)")


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class ChunkPlan:
    index: int
    start: float
    end: float
    audio_start: float
    audio_end: float
    cut_reason: str = "fixed"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def audio_duration(self) -> float:
        return max(0.0, self.audio_end - self.audio_start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "audio_start": round(self.audio_start, 3),
            "audio_end": round(self.audio_end, 3),
            "cut_reason": self.cut_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkPlan":
        return cls(
            index=int(data["index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            audio_start=float(data.get("audio_start", data["start"])),
            audio_end=float(data.get("audio_end", data["end"])),
            cut_reason=str(data.get("cut_reason", "fixed")),
        )


def detect_silences(
    wav_path: Path,
    *,
    noise_db: float = -35.0,
    min_silence: float = 0.5,
    duration: float | None = None,
) -> list[Silence]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(wav_path),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ChunkingError(
            f"ffmpeg silencedetect failed: {result.stderr.strip() or 'unknown error'}"
        )
    silences: list[Silence] = []
    current_start: float | None = None
    for line in result.stderr.splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match:
            current_start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END.search(line)
        if end_match and current_start is not None:
            end = float(end_match.group(1))
            if end > current_start:
                silences.append(Silence(current_start, end))
            current_start = None
    if current_start is not None and duration is not None and duration > current_start:
        silences.append(Silence(current_start, duration))
    return silences


def _choose_cut(
    silences: list[Silence],
    *,
    lower: float,
    upper: float,
    target: float,
    min_silence: float,
) -> float | None:
    candidates = [
        silence
        for silence in silences
        if lower <= silence.midpoint <= upper and silence.duration >= min_silence
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda silence: (
            abs(silence.midpoint - target),
            -silence.duration,
        ),
    ).midpoint


def plan_chunks(
    duration: float,
    silences: list[Silence],
    config: PipelineConfig,
) -> list[ChunkPlan]:
    if duration <= 0:
        raise ChunkingError("audio duration must be positive")
    target = float(config.chunk_seconds)
    minimum = float(config.min_chunk_seconds)
    maximum = float(config.max_chunk_seconds)
    overlap = max(0.0, float(config.chunk_overlap_seconds))

    if duration <= maximum:
        return [
            ChunkPlan(
                index=1,
                start=0.0,
                end=duration,
                audio_start=0.0,
                audio_end=duration,
                cut_reason="whole",
            )
        ]

    plans: list[tuple[float, float, str]] = []
    cursor = 0.0
    guard = 0
    while cursor < duration - 0.01:
        guard += 1
        if guard > 10000:
            raise ChunkingError("chunk planning did not converge")
        remaining = duration - cursor
        if remaining <= maximum:
            plans.append((cursor, duration, "tail"))
            break

        lower = min(cursor + minimum, duration - 0.01)
        upper = min(cursor + maximum, duration - 0.01)
        desired = min(cursor + target, upper)
        cut = _choose_cut(
            silences,
            lower=lower,
            upper=upper,
            target=desired,
            min_silence=config.silence_min_seconds,
        )
        reason = "silence" if cut is not None else "fixed"
        if cut is None:
            cut = desired

        if cut <= cursor + 0.01:
            cut = min(cursor + minimum, duration - 0.01)
        if cut >= duration - 0.01:
            plans.append((cursor, duration, "tail"))
            break

        leftover = duration - cut
        if 0 < leftover < minimum:
            if (cut - cursor) + leftover <= maximum:
                plans.append((cursor, duration, "tail"))
                break
            latest = max(cursor + minimum, duration - minimum)
            cut = min(cut, latest)
            reason = "fixed" if reason != "silence" else "silence"

        plans.append((cursor, cut, reason))
        cursor = cut

    if not plans:
        plans.append((0.0, duration, "whole"))

    result: list[ChunkPlan] = []
    previous_reason = ""
    for index, (start, end, reason) in enumerate(plans, start=1):
        audio_start = start
        audio_end = end
        if start > 0 and previous_reason == "fixed":
            audio_start = max(0.0, start - overlap)
        if reason == "fixed":
            audio_end = min(duration, end + overlap)
        result.append(
            ChunkPlan(
                index=index,
                start=start,
                end=end,
                audio_start=audio_start,
                audio_end=audio_end,
                cut_reason=reason,
            )
        )
        previous_reason = reason
    return result


def prune_tiny_chunks(plans: list[ChunkPlan], minimum: float) -> list[ChunkPlan]:
    if len(plans) <= 1:
        return plans
    last = plans[-1]
    if last.duration >= minimum:
        return plans
    previous = plans[-2]
    merged_end = last.end
    if merged_end - previous.start > 0:
        merged = ChunkPlan(
            index=previous.index,
            start=previous.start,
            end=merged_end,
            audio_start=previous.audio_start,
            audio_end=max(previous.audio_end, last.audio_end),
            cut_reason=previous.cut_reason,
        )
        return plans[:-2] + [merged]
    return plans


def chunks_to_dicts(plans: list[ChunkPlan]) -> list[dict[str, Any]]:
    return [plan.to_dict() for plan in plans]


def chunks_from_dicts(data: list[dict[str, Any]]) -> list[ChunkPlan]:
    return [ChunkPlan.from_dict(item) for item in data]


def plan_fingerprint(plans: list[ChunkPlan]) -> tuple[tuple[float, float], ...]:
    return tuple(
        (round(plan.start, 3), round(plan.end, 3))
        for plan in plans
    )


def validate_chunk_plan(plans: list[ChunkPlan], duration: float) -> None:
    if not plans:
        raise ChunkingError("chunk plan is empty")
    previous_end = 0.0
    for plan in plans:
        if plan.start < -0.001 or plan.end > duration + 0.001:
            raise ChunkingError(f"chunk {plan.index} lies outside the audio")
        if plan.end <= plan.start:
            raise ChunkingError(f"chunk {plan.index} has non-positive duration")
        if plan.start < previous_end - 0.01:
            raise ChunkingError(f"chunk {plan.index} overlaps the previous chunk")
        if plan.audio_start > plan.start + 0.001:
            raise ChunkingError(f"chunk {plan.index} has invalid audio_start")
        if plan.audio_end < plan.end - 0.001:
            raise ChunkingError(f"chunk {plan.index} has invalid audio_end")
        previous_end = plan.end


def slice_chunks(
    wav_path: Path,
    plans: list[ChunkPlan],
    chunk_audio_dir: Path,
    reporter: ProgressReporter,
    *,
    reuse: bool = True,
) -> list[Path]:
    chunk_audio_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    total = len(plans)
    for position, plan in enumerate(plans, start=1):
        output = chunk_audio_dir / f"chunk_{plan.index:04d}.wav"
        outputs.append(output)
        if reuse and output.exists() and output.stat().st_size > 0:
            reporter.update(
                "chunk",
                position,
                total,
                message="chunks ready",
                detail=f"reusing chunk {plan.index}",
            )
            continue
        duration = plan.audio_duration
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{plan.audio_start:.3f}",
            "-i",
            str(wav_path),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise ChunkingError(
                f"ffmpeg failed for chunk {plan.index}: "
                f"{result.stderr.strip() or 'unknown error'}"
            )
        if not output.exists() or output.stat().st_size == 0:
            raise ChunkingError(f"chunk {plan.index} is empty")
        reporter.update(
            "chunk",
            position,
            total,
            message="chunks ready",
            detail=f"chunk {plan.index} {plan.start:.1f}-{plan.end:.1f}s",
        )
    return outputs


def estimate_chunk_count(duration: float, target_seconds: int) -> int:
    if duration <= 0 or target_seconds <= 0:
        return 0
    return max(1, math.ceil(duration / target_seconds))
