from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import REFINE_MODES, RUN_MODES, PipelineConfig
from .errors import PipelineError
from .models import DEFAULT_MODEL_KEY, model_choices_text
from .pipeline import run_pipeline
from .progress import ProgressReporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m transcript_pipeline",
        description=(
            "Download a YouTube video, transcribe it with a permissive Nepali "
            "Whisper model, refine conservatively, and export transcripts."
        ),
    )
    parser.add_argument("--url", help="YouTube URL or 11-character video ID")
    parser.add_argument("--audio", help="local audio/video file instead of YouTube")
    parser.add_argument(
        "--output-root",
        default=None,
        help="root directory for job folders (default: ./DRP-Transcripts)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, help="model key or HF repo")
    parser.add_argument("--fallback-model", default=None, help="fallback model key or HF repo")
    parser.add_argument("--language", default="ne", help="Whisper language code")
    parser.add_argument("--chunk-seconds", type=int, default=600)
    parser.add_argument("--min-chunk-seconds", type=int, default=480)
    parser.add_argument("--max-chunk-seconds", type=int, default=720)
    parser.add_argument("--silence-noise-db", type=float, default=-35.0)
    parser.add_argument("--silence-min-seconds", type=float, default=0.5)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=0.5)
    parser.add_argument(
        "--refine-mode",
        choices=REFINE_MODES,
        default="rule",
        help="off, rule, or neural correction",
    )
    parser.add_argument("--corrector-url", default=None, help="nepali-corrector base URL")
    parser.add_argument("--glossary", default=None, help="JSON glossary path")
    parser.add_argument("--corrections", default=None, help="JSON corrections path")
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="auto",
        help="auto, fresh, resume, retry_failed, export_only, refine_only",
    )
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument("--keep-wav", action="store_true")
    parser.add_argument("--cookies", default=None, help="yt-dlp cookies.txt path")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--condition-on-previous-text", action="store_true")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--model-cache-dir", default=None)
    parser.add_argument("--drive-model-cache-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a JSON result summary instead of the human summary",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        video_url=args.url or "",
        input_audio=Path(args.audio).expanduser() if args.audio else None,
        model=args.model,
        fallback_model=args.fallback_model,
        language=args.language,
        chunk_seconds=args.chunk_seconds,
        min_chunk_seconds=args.min_chunk_seconds,
        max_chunk_seconds=args.max_chunk_seconds,
        silence_noise_db=args.silence_noise_db,
        silence_min_seconds=args.silence_min_seconds,
        chunk_overlap_seconds=args.chunk_overlap_seconds,
        refine_mode=args.refine_mode,
        corrector_url=args.corrector_url,
        glossary_path=Path(args.glossary).expanduser() if args.glossary else None,
        corrections_path=Path(args.corrections).expanduser() if args.corrections else None,
        run_mode=args.run_mode,
        force_overwrite=args.force_overwrite,
        keep_chunks=args.keep_chunks,
        keep_wav=args.keep_wav,
        cookies_path=Path(args.cookies).expanduser() if args.cookies else None,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        best_of=args.best_of,
        vad_filter=not args.no_vad,
        condition_on_previous_text=args.condition_on_previous_text,
        word_timestamps=args.word_timestamps,
        initial_prompt=args.initial_prompt,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )
    if args.output_root:
        config.output_root = Path(args.output_root).expanduser()
    if args.model_cache_dir:
        config.model_cache_dir = Path(args.model_cache_dir).expanduser()
    if args.drive_model_cache_dir:
        config.drive_model_cache_dir = Path(args.drive_model_cache_dir).expanduser()
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_models:
        print(model_choices_text())
        return 0
    if not args.url and not args.audio:
        parser.error("one of --url or --audio is required")
    config = config_from_args(args)
    reporter = ProgressReporter(quiet=args.quiet)
    try:
        result = run_pipeline(config, reporter)
    except KeyboardInterrupt:
        print("interrupted; resume with --run-mode resume", file=sys.stderr)
        return 130
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 1

    summary = {
        "status": result.job.status,
        "video_id": result.job.video_id,
        "title": result.job.title,
        "output_dir": str(result.paths.root),
        "completed_chunks": result.job.completed_chunks,
        "failed_chunks": result.job.failed_chunks,
        "warnings": result.job.warnings,
        "files": {
            key: str(path)
            for key, path in result.exports.files.items()
        },
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"{result.job.status}: {result.job.title or result.job.video_id}")
        print(f"output: {result.paths.root}")
        for key, path in result.exports.files.items():
            print(f"  {key}: {path}")
        if result.job.warnings:
            print("warnings:")
            for warning in result.job.warnings:
                print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
