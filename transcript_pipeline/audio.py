from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .errors import DependencyError, DownloadError
from .paths import (
    JobPaths,
    atomic_write_json,
    ensure_dir,
    find_audio_file,
    human_duration,
    read_json_or_none,
)
from .progress import ProgressReporter


@dataclass
class AudioMetadata:
    video_id: str
    title: str
    url: str = ""
    channel: str = ""
    duration: float = 0.0
    upload_date: str = ""
    thumbnail: str = ""
    audio_path: str = ""
    format_name: str = ""
    sample_rate: int = 0
    channels: int = 0
    source: str = "download"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "url": self.url,
            "channel": self.channel,
            "duration": self.duration,
            "upload_date": self.upload_date,
            "thumbnail": self.thumbnail,
            "audio_path": self.audio_path,
            "format_name": self.format_name,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "source": self.source,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioMetadata":
        return cls(
            video_id=str(data.get("video_id", "")),
            title=str(data.get("title", "")),
            url=str(data.get("url", "")),
            channel=str(data.get("channel", "")),
            duration=float(data.get("duration", 0.0)),
            upload_date=str(data.get("upload_date", "")),
            thumbnail=str(data.get("thumbnail", "")),
            audio_path=str(data.get("audio_path", "")),
            format_name=str(data.get("format_name", "")),
            sample_rate=int(data.get("sample_rate", 0)),
            channels=int(data.get("channels", 0)),
            source=str(data.get("source", "download")),
            extra=dict(data.get("extra", {})),
        )


def ensure_tools(*, need_downloader: bool = False) -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise DependencyError(
            "Missing required tools: "
            + ", ".join(missing)
            + ". Install ffmpeg and make sure it is on PATH."
        )
    if need_downloader and not _has_ytdlp():
        raise DependencyError(
            "yt-dlp is required. Install it with `pip install yt-dlp`."
        )


def _has_ytdlp() -> bool:
    try:
        import yt_dlp  # noqa: F401

        return True
    except ImportError:
        return shutil.which("yt-dlp") is not None


def _first_entry(info: dict[str, Any]) -> dict[str, Any]:
    if info.get("_type") == "playlist":
        entries = info.get("entries") or []
        if not entries:
            raise DownloadError("playlist has no entries")
        return entries[0] or {}
    return info


def fetch_video_metadata(url: str, config: PipelineConfig) -> dict[str, Any]:
    ensure_tools(need_downloader=True)
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    if config.cookies_path:
        options["cookiefile"] = str(config.cookies_path)

    try:
        import yt_dlp

        with yt_dlp.YoutubeDL(options) as ydl:
            info = _first_entry(ydl.extract_info(url, download=False))
    except ImportError:
        info = _fetch_metadata_subprocess(url, config)

    if not info:
        raise DownloadError(f"could not read video metadata for {url}")
    return {
        "video_id": str(info.get("id", "")),
        "title": str(info.get("title", "") or ""),
        "channel": str(
            info.get("channel")
            or info.get("uploader")
            or info.get("creator")
            or ""
        ),
        "duration": float(info.get("duration") or 0.0),
        "upload_date": str(info.get("upload_date", "") or ""),
        "thumbnail": str(info.get("thumbnail", "") or ""),
        "webpage_url": str(info.get("webpage_url", "") or url),
    }


def _fetch_metadata_subprocess(url: str, config: PipelineConfig) -> dict[str, Any]:
    command = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--skip-download",
        "--dump-single-json",
    ]
    if config.cookies_path:
        command.extend(["--cookies", str(config.cookies_path)])
    command.append(url)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise DownloadError(
            "yt-dlp metadata lookup failed: "
            + (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error")
        )
    try:
        return _first_entry(json.loads(result.stdout))
    except json.JSONDecodeError as exc:
        raise DownloadError("yt-dlp returned invalid metadata JSON") from exc


def probe_media(path: Path) -> dict[str, Any]:
    if shutil.which("ffprobe") is None:
        raise DependencyError("ffprobe is required. Install ffmpeg.")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise DownloadError(
            f"ffprobe failed for {path}: {result.stderr.strip() or 'unknown error'}"
        )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise DownloadError("ffprobe returned invalid JSON") from exc

    streams = data.get("streams") or []
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if audio_stream is None:
        raise DownloadError(f"no audio stream found in {path}")
    format_info = data.get("format") or {}
    duration = 0.0
    for candidate in (
        format_info.get("duration"),
        audio_stream.get("duration"),
    ):
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break
    if duration <= 0:
        raise DownloadError(f"could not determine audio duration for {path}")
    return {
        "duration": duration,
        "format_name": str(format_info.get("format_name", "")),
        "sample_rate": int(audio_stream.get("sample_rate") or 0),
        "channels": int(audio_stream.get("channels") or 0),
        "codec_name": str(audio_stream.get("codec_name", "")),
    }


def _progress_hook(reporter: ProgressReporter):
    def hook(data: dict[str, Any]) -> None:
        status = data.get("status")
        if status == "downloading":
            downloaded = data.get("downloaded_bytes")
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            if downloaded and total:
                current = int(100 * downloaded / total)
                reporter.update(
                    "download",
                    current,
                    100,
                    message="downloading audio",
                    detail=f"{human_duration(downloaded)} / {human_duration(total)}",
                )
            else:
                reporter.emit(
                    "download",
                    message="downloading audio",
                    detail=str(data.get("_percent_str", "")).strip(),
                )
        elif status == "finished":
            reporter.update("download", 100, 100, message="download finished")

    return hook


def download_audio(
    config: PipelineConfig,
    paths: JobPaths,
    reporter: ProgressReporter,
    *,
    known_metadata: dict[str, Any] | None = None,
) -> AudioMetadata:
    paths.ensure()
    if config.input_audio is not None:
        return _copy_local_audio(config.input_audio, paths, reporter, force=config.force_overwrite)

    existing = find_audio_file(paths.audio_dir)
    if existing is not None and not config.force_overwrite:
        ensure_tools()
        reporter.stage("download", f"reusing audio {existing.name}")
        metadata = _metadata_from_existing(existing, known_metadata, config)
        _write_metadata(paths, metadata)
        return metadata

    ensure_tools(need_downloader=True)
    reporter.stage("download", "starting yt-dlp")
    outtmpl = str(paths.audio_dir / "original.%(ext)s")
    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "progress_hooks": [_progress_hook(reporter)],
        "overwrites": bool(config.force_overwrite),
    }
    if config.cookies_path:
        options["cookiefile"] = str(config.cookies_path)

    try:
        import yt_dlp

        with yt_dlp.YoutubeDL(options) as ydl:
            info = _first_entry(ydl.extract_info(config.video_url, download=True))
    except ImportError:
        info = _download_subprocess(config, paths, reporter)

    audio_path = find_audio_file(paths.audio_dir)
    if audio_path is None:
        raise DownloadError("yt-dlp finished but no audio file was produced")

    probe = probe_media(audio_path)
    metadata = AudioMetadata(
        video_id=str(info.get("id") or (known_metadata or {}).get("video_id", "")),
        title=str(info.get("title") or (known_metadata or {}).get("title", "")),
        url=str(info.get("webpage_url") or config.video_url),
        channel=str(
            info.get("channel")
            or info.get("uploader")
            or (known_metadata or {}).get("channel", "")
        ),
        duration=float(probe["duration"]),
        upload_date=str(info.get("upload_date", "") or ""),
        thumbnail=str(info.get("thumbnail", "") or ""),
        audio_path=str(audio_path),
        format_name=str(probe["format_name"]),
        sample_rate=int(probe["sample_rate"]),
        channels=int(probe["channels"]),
        source="download",
        extra={"codec_name": probe["codec_name"]},
    )
    _write_metadata(paths, metadata)
    reporter.update("download", 100, 100, message="audio ready")
    return metadata


def _download_subprocess(
    config: PipelineConfig,
    paths: JobPaths,
    reporter: ProgressReporter,
) -> dict[str, Any]:
    reporter.stage("download", "yt-dlp module unavailable; using command line")
    command = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "-f",
        "bestaudio/best",
        "-o",
        str(paths.audio_dir / "original.%(ext)s"),
        "--write-info-json",
    ]
    if config.force_overwrite:
        command.append("--force-overwrites")
    if config.cookies_path:
        command.extend(["--cookies", str(config.cookies_path)])
    command.append(config.video_url)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise DownloadError(
            "yt-dlp download failed: "
            + (result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error")
        )
    info_file = next(iter(paths.audio_dir.glob("original.info.json")), None)
    if info_file is None:
        return {"id": "", "title": "", "webpage_url": config.video_url}
    try:
        return _first_entry(json.loads(info_file.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {"id": "", "title": "", "webpage_url": config.video_url}


def _copy_local_audio(
    source: Path,
    paths: JobPaths,
    reporter: ProgressReporter,
    *,
    force: bool = False,
) -> AudioMetadata:
    source = source.expanduser().resolve()
    if not source.exists():
        raise DownloadError(f"local audio file not found: {source}")
    ensure_dir(paths.audio_dir)
    target = paths.audio_dir / f"original{source.suffix.lower()}"
    if force or not target.exists() or target.stat().st_size == 0:
        reporter.stage("download", f"copying local audio {source.name}")
        shutil.copy2(source, target)
    probe = probe_media(target)
    metadata = AudioMetadata(
        video_id="",
        title=source.stem,
        url="",
        duration=float(probe["duration"]),
        audio_path=str(target),
        format_name=str(probe["format_name"]),
        sample_rate=int(probe["sample_rate"]),
        channels=int(probe["channels"]),
        source="local",
        extra={"source_path": str(source), "codec_name": probe["codec_name"]},
    )
    _write_metadata(paths, metadata)
    return metadata


def _metadata_from_existing(
    audio_path: Path,
    known_metadata: dict[str, Any] | None,
    config: PipelineConfig,
) -> AudioMetadata:
    stored = read_json_or_none(audio_path.parent / "metadata.json")
    if stored:
        metadata = AudioMetadata.from_dict(stored)
        metadata.audio_path = str(audio_path)
        return metadata
    probe = probe_media(audio_path)
    known = known_metadata or {}
    return AudioMetadata(
        video_id=str(known.get("video_id", "")),
        title=str(known.get("title", "")),
        url=str(known.get("webpage_url", config.video_url)),
        channel=str(known.get("channel", "")),
        duration=float(probe["duration"]),
        audio_path=str(audio_path),
        format_name=str(probe["format_name"]),
        sample_rate=int(probe["sample_rate"]),
        channels=int(probe["channels"]),
        source="download",
    )


def _write_metadata(paths: JobPaths, metadata: AudioMetadata) -> None:
    atomic_write_json(paths.metadata_file, metadata.to_dict())


def read_metadata(paths: JobPaths) -> AudioMetadata | None:
    data = read_json_or_none(paths.metadata_file)
    if data is None:
        return None
    return AudioMetadata.from_dict(data)


def normalize_audio(
    input_path: Path,
    output_path: Path,
    reporter: ProgressReporter,
    *,
    force: bool = False,
) -> Path:
    ensure_tools()
    if (
        not force
        and output_path.exists()
        and output_path.stat().st_size > 0
        and output_path.stat().st_mtime >= input_path.stat().st_mtime
    ):
        reporter.stage("prepare", f"reusing normalized audio {output_path.name}")
        return output_path

    reporter.stage("prepare", "normalizing audio to 16 kHz mono WAV")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise DownloadError(
            f"ffmpeg normalization failed: {result.stderr.strip() or 'unknown error'}"
        )
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise DownloadError("ffmpeg produced an empty WAV file")
    reporter.stage("prepare", "normalized audio ready")
    return output_path
