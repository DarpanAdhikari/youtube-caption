from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

from .config import default_output_root
from .errors import PipelineError

_UNSAFE_FILENAME = re.compile(r'[/\\:*?"<>|\x00-\x1f]')
_SPACE = re.compile(r"\s+")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_ID = re.compile(r"/(?:embed|shorts|live|v)/([A-Za-z0-9_-]{11})")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_title(title: str, fallback: str = "video", limit: int = 80) -> str:
    value = unicodedata.normalize("NFC", str(title or ""))
    value = _UNSAFE_FILENAME.sub(" ", value)
    value = _SPACE.sub(" ", value).strip(" ._-")
    if len(value) > limit:
        value = value[:limit].rstrip(" ._-")
    return value or fallback


def parse_video_id(url_or_id: str) -> str:
    value = str(url_or_id or "").strip()
    if _VIDEO_ID.fullmatch(value):
        return value

    parsed = urlparse(value if "://" in value else "https://" + value)
    host = (parsed.hostname or "").lower()
    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID.fullmatch(candidate):
            return candidate
    if (
        host == "youtube.com"
        or host.endswith(".youtube.com")
        or host == "youtube-nocookie.com"
        or host.endswith(".youtube-nocookie.com")
    ):
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if _VIDEO_ID.fullmatch(query_id):
            return query_id
        match = _PATH_ID.search(parsed.path)
        if match:
            return match.group(1)

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:11]
    return f"url-{digest}"


def local_video_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    stem = safe_title(path.stem, fallback="audio", limit=40)
    return f"local-{stem}-{digest}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temp_name = handle.name
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_or_none(path: Path) -> Any:
    if not path.exists():
        return None
    return read_json(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def remove_dir_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            import shutil

            shutil.rmtree(child)
        else:
            child.unlink()


def find_audio_file(audio_dir: Path) -> Path | None:
    if not audio_dir.exists():
        return None
    candidates = [
        item
        for item in audio_dir.iterdir()
        if item.is_file()
        and item.stem.startswith("original")
        and item.suffix.lower() not in {".json", ".part", ".ytdl", ".tmp"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_size)


def human_duration(seconds: float | None) -> str:
    total = int(round(float(seconds or 0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


@dataclass(frozen=True)
class JobPaths:
    root: Path
    audio_dir: Path
    chunks_dir: Path
    chunk_audio_dir: Path
    transcripts_dir: Path
    refinement_dir: Path
    output_dir: Path
    state_dir: Path

    @classmethod
    def for_video(cls, output_root: Path, video_id: str, title: str) -> "JobPaths":
        root = output_root.expanduser() / f"{video_id}_{safe_title(title, fallback=video_id)}"
        return cls(
            root=root,
            audio_dir=root / "audio",
            chunks_dir=root / "chunks",
            chunk_audio_dir=root / "chunks" / "audio",
            transcripts_dir=root / "transcripts",
            refinement_dir=root / "refinement",
            output_dir=root / "output",
            state_dir=root / "state",
        )

    @property
    def job_file(self) -> Path:
        return self.state_dir / "job.json"

    @property
    def errors_file(self) -> Path:
        return self.state_dir / "errors.jsonl"

    @property
    def normalized_wav(self) -> Path:
        return self.audio_dir / "normalized.wav"

    @property
    def metadata_file(self) -> Path:
        return self.audio_dir / "metadata.json"

    @property
    def raw_transcript_file(self) -> Path:
        return self.transcripts_dir / "raw.json"

    @property
    def refined_transcript_file(self) -> Path:
        return self.transcripts_dir / "refined.json"

    def chunk_audio_file(self, index: int) -> Path:
        return self.chunk_audio_dir / f"chunk_{index:04d}.wav"

    def chunk_result_file(self, index: int) -> Path:
        return self.chunks_dir / f"chunk_{index:04d}.json"

    def ensure(self) -> "JobPaths":
        for path in (
            self.root,
            self.audio_dir,
            self.chunks_dir,
            self.chunk_audio_dir,
            self.transcripts_dir,
            self.refinement_dir,
            self.output_dir,
            self.state_dir,
        ):
            ensure_dir(path)
        return self

    def archive_state(self) -> Path | None:
        if not self.job_file.exists():
            return None
        target = self.state_dir / f"job.{timestamp_slug()}.json"
        os.replace(self.job_file, target)
        return target

    def reset_run_outputs(self) -> None:
        for path in (
            self.chunks_dir,
            self.transcripts_dir,
            self.refinement_dir,
            self.output_dir,
            self.state_dir,
        ):
            remove_dir_contents(path)
            ensure_dir(path)

    def existing_chunk_files(self) -> list[Path]:
        if not self.chunks_dir.exists():
            return []
        return sorted(
            item
            for item in self.chunks_dir.glob("chunk_*.json")
            if item.is_file()
        )


def find_existing_job(output_root: Path, video_id: str) -> JobPaths | None:
    if not output_root.exists():
        return None
    candidates = []
    for item in output_root.iterdir():
        if not item.is_dir():
            continue
        if item.name == video_id or item.name.startswith(video_id + "_"):
            job = item / "state" / "job.json"
            if job.exists():
                candidates.append(item)
    if not candidates:
        return None
    root = max(candidates, key=lambda item: (item / "state" / "job.json").stat().st_mtime)
    return JobPaths(
        root=root,
        audio_dir=root / "audio",
        chunks_dir=root / "chunks",
        chunk_audio_dir=root / "chunks" / "audio",
        transcripts_dir=root / "transcripts",
        refinement_dir=root / "refinement",
        output_dir=root / "output",
        state_dir=root / "state",
    )


def resolve_output_root(value: str | Path | None) -> Path:
    if value is None:
        return default_output_root()
    return Path(value).expanduser()


def validate_root_is_writable(path: Path) -> None:
    try:
        ensure_dir(path)
        probe = path / ".write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PipelineError(f"output root is not writable: {path} ({exc})") from exc
