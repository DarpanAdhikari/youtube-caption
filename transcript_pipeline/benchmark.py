from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .asr import ASRResult, FasterWhisperEngine
from .audio import download_audio, ensure_tools, probe_media
from .config import PipelineConfig
from .errors import PipelineError
from .models import ensure_model, resolve_model
from .paths import (
    JobPaths,
    atomic_write_json,
    ensure_dir,
    find_audio_file,
    parse_video_id,
    safe_title,
)
from .progress import ProgressReporter

_METRIC_PUNCT = re.compile(r"[।॥,;:?!.()\[\]{}\"'“”‘’]+")
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4", ".opus", ".webm", ".flac"}


def normalize_metric_text(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text or "")).lower()
    value = _METRIC_PUNCT.sub(" ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_metric_text(reference).split()
    hyp_tokens = normalize_metric_text(hypothesis).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref_text = normalize_metric_text(reference).replace(" ", "")
    hyp_text = normalize_metric_text(hypothesis).replace(" ", "")
    if not ref_text:
        return 0.0 if not hyp_text else 1.0
    return edit_distance(list(ref_text), list(hyp_text)) / len(ref_text)


def _model_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace("@", "__")


@dataclass
class BenchmarkRow:
    model: str
    repo_id: str
    audio: str
    duration: float
    elapsed: float
    rtf: float
    segments: int
    config: str
    languages: dict[str, float]
    transcript_path: str
    wer: float | None = None
    cer: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "repo_id": self.repo_id,
            "audio": self.audio,
            "duration": round(self.duration, 3),
            "elapsed": round(self.elapsed, 3),
            "rtf": round(self.rtf, 3),
            "segments": self.segments,
            "languages": self.languages,
            "transcript_path": self.transcript_path,
            "wer": None if self.wer is None else round(self.wer, 4),
            "cer": None if self.cer is None else round(self.cer, 4),
            "error": self.error,
        }


def _aggregate_rows(
    rows: list[BenchmarkRow],
    model_info: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for info in model_info:
        model = info["model"]
        model_rows = [row for row in rows if row.model == model]
        scored = [row for row in model_rows if row.wer is not None]
        wers = [row.wer for row in scored if row.wer is not None]
        cers = [row.cer for row in scored if row.cer is not None]
        total_duration = sum(row.duration for row in model_rows)
        total_elapsed = sum(row.elapsed for row in model_rows)
        aggregates.append(
            {
                "model": model,
                "repo_id": info.get("repo_id", ""),
                "clips": len(model_rows),
                "scored_clips": len(scored),
                "failed_clips": sum(1 for row in model_rows if row.error),
                "mean_wer": round(statistics.mean(wers), 4) if wers else None,
                "median_wer": round(statistics.median(wers), 4) if wers else None,
                "mean_cer": round(statistics.mean(cers), 4) if cers else None,
                "median_cer": round(statistics.median(cers), 4) if cers else None,
                "total_duration": round(total_duration, 3),
                "total_elapsed": round(total_elapsed, 3),
                "rtf": (
                    round(total_elapsed / total_duration, 3)
                    if total_duration
                    else None
                ),
                "load_error": info.get("load_error"),
            }
        )
    return aggregates


def benchmark_models(
    model_names: list[str],
    audio_paths: list[Path],
    references: dict[str, str] | None,
    config: PipelineConfig,
    output_dir: Path,
    reporter: ProgressReporter | None = None,
) -> dict[str, Any]:
    reporter = reporter or ProgressReporter()
    ensure_dir(output_dir)
    references = references or {}
    rows: list[BenchmarkRow] = []
    model_info: list[dict[str, Any]] = []

    for model_name in model_names:
        spec = resolve_model(model_name)
        reporter.stage("benchmark", f"preparing {spec.repo_id}")
        model_start = time.perf_counter()
        model_error: str | None = None
        try:
            model_ref = ensure_model(spec, config, reporter)
            engine = FasterWhisperEngine(model_ref, config)
            engine.load()
        except Exception as exc:
            model_error = str(exc)
            model_info.append(
                {
                    "model": spec.key,
                    "repo_id": spec.repo_id,
                    "license": spec.license,
                    "load_error": model_error,
                }
            )
            reporter.stage("benchmark", f"{spec.key} failed to load: {exc}")
            continue
        load_elapsed = time.perf_counter() - model_start
        model_info.append(
            {
                "model": spec.key,
                "repo_id": spec.repo_id,
                "license": spec.license,
                "architecture": spec.architecture,
                "load_seconds": round(load_elapsed, 2),
                "device": engine.device,
                "compute_type": engine.compute_type,
            }
        )
        model_dir = output_dir / _model_dir_name(spec.repo_id)
        ensure_dir(model_dir)

        for audio_path in audio_paths:
            audio_path = audio_path.expanduser().resolve()
            reporter.stage("benchmark", f"{spec.key}: {audio_path.name}")
            try:
                media = probe_media(audio_path)
                started = time.perf_counter()
                result = engine.transcribe(audio_path, offset=0.0)
                elapsed = time.perf_counter() - started
                text = " ".join(segment.text for segment in result.segments).strip()
                transcript_path = model_dir / f"{audio_path.stem}.txt"
                transcript_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
                reference = (
                    references.get(str(audio_path))
                    or references.get(audio_path.stem)
                    or references.get(audio_path.parent.name)
                )
                row = BenchmarkRow(
                    model=spec.key,
                    repo_id=spec.repo_id,
                    audio=str(audio_path),
                    duration=media["duration"],
                    elapsed=elapsed,
                    rtf=(elapsed / media["duration"]) if media["duration"] else 0.0,
                    segments=len(result.segments),
                    config=spec.architecture,
                    languages=_language_info(result),
                    transcript_path=str(transcript_path),
                    wer=(
                        word_error_rate(reference, text)
                        if reference is not None
                        else None
                    ),
                    cer=(
                        char_error_rate(reference, text)
                        if reference is not None
                        else None
                    ),
                )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    BenchmarkRow(
                        model=spec.key,
                        repo_id=spec.repo_id,
                        audio=str(audio_path),
                        duration=0.0,
                        elapsed=0.0,
                        rtf=0.0,
                        segments=0,
                        config=spec.architecture,
                        languages={},
                        transcript_path="",
                        error=str(exc),
                    )
                )
                reporter.stage(
                    "benchmark",
                    f"{spec.key}: {audio_path.name} failed: {exc}",
                )
        engine.unload()

    summary = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": model_info,
        "aggregates": _aggregate_rows(rows, model_info),
        "rows": [row.to_dict() for row in rows],
        "notes": [
            "WER/CER are only reported when a matching reference file is supplied.",
            "Spot-check names, numbers, dates, and institutions manually.",
            "RTF is elapsed transcription time divided by audio duration.",
        ],
    }
    atomic_write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_render_markdown(summary), encoding="utf-8")
    return summary


def _language_info(result: ASRResult) -> dict[str, float]:
    value: dict[str, float] = {}
    if result.language:
        value[result.language] = (
            round(result.language_probability, 4)
            if result.language_probability is not None
            else 0.0
        )
    return value


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Nepali ASR benchmark", ""]
    aggregates = summary.get("aggregates", [])
    if aggregates:
        lines.append("## Aggregate")
        lines.append("")
        lines.append(
            "| Model | Clips | Scored | Failed | Mean WER | Median WER | "
            "Mean CER | Median CER | RTF |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for aggregate in aggregates:
            if aggregate.get("load_error"):
                lines.append(
                    "| {model} | load failed | | | | | | | |".format(
                        model=aggregate["model"]
                    )
                )
                continue
            lines.append(
                "| {model} | {clips} | {scored} | {failed} | {mean_wer} | "
                "{median_wer} | {mean_cer} | {median_cer} | {rtf} |".format(
                    model=aggregate["model"],
                    clips=aggregate["clips"],
                    scored=aggregate["scored_clips"],
                    failed=aggregate["failed_clips"],
                    mean_wer=(
                        ""
                        if aggregate["mean_wer"] is None
                        else f"{aggregate['mean_wer']:.2%}"
                    ),
                    median_wer=(
                        ""
                        if aggregate["median_wer"] is None
                        else f"{aggregate['median_wer']:.2%}"
                    ),
                    mean_cer=(
                        ""
                        if aggregate["mean_cer"] is None
                        else f"{aggregate['mean_cer']:.2%}"
                    ),
                    median_cer=(
                        ""
                        if aggregate["median_cer"] is None
                        else f"{aggregate['median_cer']:.2%}"
                    ),
                    rtf="" if aggregate["rtf"] is None else f"{aggregate['rtf']:.3f}",
                )
            )
        lines.append("")
    lines.append("## Per-clip")
    lines.append("")
    lines.append("| Model | Audio | Duration | RTF | WER | CER | Error |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in summary.get("rows", []):
        lines.append(
            "| {model} | {audio} | {duration:.1f}s | {rtf:.3f} | {wer} | {cer} | {error} |".format(
                model=row["model"],
                audio=Path(row["audio"]).name,
                duration=row["duration"],
                rtf=row["rtf"],
                wer="" if row["wer"] is None else f"{row['wer']:.2%}",
                cer="" if row["cer"] is None else f"{row['cer']:.2%}",
                error=row.get("error") or "",
            )
        )
    lines.append("")
    for info in summary.get("models", []):
        if info.get("load_error"):
            lines.append(f"- **{info['model']}**: load failed: {info['load_error']}")
        else:
            lines.append(
                f"- **{info['model']}**: {info.get('architecture', '')}, "
                f"{info.get('license', '')}, load {info.get('load_seconds', '?')}s, "
                f"{info.get('device', '?')}/{info.get('compute_type', '?')}"
            )
    return "\n".join(lines) + "\n"


def _collect_audio_files(directories: list[str] | None) -> list[Path]:
    result: list[Path] = []
    for value in directories or []:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"audio directory not found: {path}")
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(
                item
                for item in path.iterdir()
                if item.is_file() and item.suffix.lower() in _AUDIO_SUFFIXES
            )
        if not candidates:
            raise FileNotFoundError(f"no supported audio files found in: {path}")
        result.extend(candidates)
    return result


def _load_references(
    values: list[str] | None,
    directories: list[str] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in directories or []:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"reference directory not found: {path}")
        for item in sorted(path.glob("*.txt")):
            result[item.stem] = item.read_text(encoding="utf-8")
    for value in values or []:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"reference file not found: {path}")
        result[path.stem] = path.read_text(encoding="utf-8")
    return result


def _download_url_audio(
    url: str,
    config: PipelineConfig,
    output_dir: Path,
    max_seconds: float,
    reporter: ProgressReporter,
) -> Path:
    video_id = parse_video_id(url)
    downloads = output_dir / "downloads"
    paths = JobPaths.for_video(downloads, video_id, "benchmark").ensure()
    run_config = replace(config, video_url=url)
    metadata = download_audio(run_config, paths, reporter)
    source = Path(metadata.audio_path)
    if not source.exists():
        found = find_audio_file(paths.audio_dir)
        if found is not None:
            source = found
    if not source.exists():
        raise PipelineError(f"download produced no audio file for {url}")

    if max_seconds <= 0:
        return source

    target = (
        output_dir
        / "inputs"
        / safe_title(video_id, fallback="video", limit=60)
        / f"first-{int(max_seconds)}s.wav"
    )
    if target.exists() and target.stat().st_size > 0:
        return target
    ensure_dir(target.parent)
    ensure_tools()
    reporter.stage("benchmark", f"trimming {video_id} to {int(max_seconds)} seconds")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-t",
        f"{max_seconds:.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(
            "ffmpeg benchmark trim failed: "
            + (result.stderr.strip() or "unknown error")
        )
    if not target.exists() or target.stat().st_size == 0:
        raise PipelineError("ffmpeg produced an empty benchmark sample")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark permissive Nepali Whisper models.",
    )
    parser.add_argument(
        "--models",
        default="nepali-medium-ct2,nepali-turbo-ct2,openai-turbo",
        help="comma-separated model keys or Hugging Face repo IDs",
    )
    parser.add_argument(
        "--audio",
        action="append",
        default=None,
        help="local audio file to benchmark; repeat for multiple files",
    )
    parser.add_argument(
        "--audio-dir",
        action="append",
        default=None,
        help="directory of local audio files; repeat for multiple directories",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help="YouTube URL to download and benchmark; repeat as needed",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=600,
        help="trim each YouTube input to this many seconds; 0 uses full audio",
    )
    parser.add_argument(
        "--reference",
        action="append",
        help=(
            "reference transcript; matched by audio stem, parent folder, "
            "or full path; repeat as needed"
        ),
    )
    parser.add_argument(
        "--reference-dir",
        action="append",
        help="directory of reference .txt files; repeat for multiple directories",
    )
    parser.add_argument(
        "--output",
        default="benchmark",
        help="benchmark output directory",
    )
    parser.add_argument("--language", default="ne")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--model-cache-dir", default=None)
    args = parser.parse_args(argv)

    if not args.audio and not args.audio_dir and not args.url:
        parser.error("at least one --audio, --audio-dir, or --url is required")

    config = PipelineConfig(
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
    )
    if args.model_cache_dir:
        config.model_cache_dir = Path(args.model_cache_dir).expanduser()
    output_dir = Path(args.output).expanduser()
    reporter = ProgressReporter()
    audio_paths = [Path(value) for value in (args.audio or [])]
    audio_paths.extend(_collect_audio_files(args.audio_dir))
    for url in args.url or []:
        audio_paths.append(
            _download_url_audio(
                url,
                config,
                output_dir,
                max(0.0, float(args.max_seconds)),
                reporter,
            )
        )
    audio_paths = list(
        dict.fromkeys(path.expanduser().resolve() for path in audio_paths)
    )
    references = _load_references(args.reference, args.reference_dir)
    model_names = [value.strip() for value in args.models.split(",") if value.strip()]
    summary = benchmark_models(
        model_names,
        audio_paths,
        references,
        config,
        output_dir,
        reporter,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
