import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    IpBlocked,
    InvalidVideoId,
    NoTranscriptFound,
    PoTokenRequired,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
)

from .schemas import AvailableTranscript, Segment, TranscriptResponse

_ytt_api = YouTubeTranscriptApi()

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_ID_PATTERN = re.compile(r"/(?:embed|shorts|live|v)/([A-Za-z0-9_-]{11})")


class TranscriptError(Exception):
    def __init__(self, message: str, error_type: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.status_code = status_code


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()

    if _VIDEO_ID_PATTERN.fullmatch(value):
        return value

    if "://" not in value:
        value = "https://" + value

    parsed = urlparse(value)
    host = (parsed.hostname or "").lower()

    if host in ("youtu.be", "www.youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
        if _VIDEO_ID_PATTERN.fullmatch(candidate):
            return candidate

    if host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if _VIDEO_ID_PATTERN.fullmatch(query_id):
            return query_id

        match = _PATH_ID_PATTERN.search(parsed.path)
        if match:
            return match.group(1)

    raise TranscriptError(
        f"Could not extract a video id from: {url_or_id}",
        "invalid_url",
        400,
    )


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _build_segments(transcript) -> list[Segment]:
    return [
        Segment(start=snippet.start, duration=snippet.duration, text=snippet.text)
        for snippet in transcript
    ]


def _join_text(segments: list[Segment]) -> str:
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


@dataclass
class FullTranscript:
    video_id: str
    language: str
    language_code: str
    is_generated: bool
    segments: list[Segment]

    @property
    def total(self) -> int:
        return len(self.segments)

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        last = self.segments[-1]
        return last.start + last.duration


# Long transcripts are paged out a window at a time. YouTube is only contacted
# once per (video, language) and the result is kept here so following pages and
# range queries are served from memory instead of re-fetching.
_CACHE: dict[tuple[str, tuple[str, ...]], tuple[FullTranscript, float]] = {}
_CACHE_TTL_SECONDS = 3600.0
_CACHE_MAX_ENTRIES = 32


def _cache_key(video_id: str, language_priority: list[str]) -> tuple[str, tuple[str, ...]]:
    return (video_id, tuple(language_priority))


def _cache_get(key: tuple[str, tuple[str, ...]]) -> FullTranscript | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    value, stored_at = entry
    if time.monotonic() - stored_at > _CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return value


def _cache_put(key: tuple[str, tuple[str, ...]], value: FullTranscript) -> None:
    if len(_CACHE) >= _CACHE_MAX_ENTRIES:
        oldest_key = min(_CACHE, key=lambda k: _CACHE[k][1])
        _CACHE.pop(oldest_key, None)
    _CACHE[key] = (value, time.monotonic())


def _fetch_full(video_id: str, language_priority: list[str]) -> FullTranscript:
    try:
        transcript = _ytt_api.fetch(video_id, languages=language_priority)
    except NoTranscriptFound as exc:
        raise TranscriptError(
            f"No transcript found for video '{video_id}' in languages {language_priority}. "
            f"Available transcripts are listed at /languages?url={video_id}",
            "no_transcript_found",
            404,
        ) from exc
    except (TranscriptsDisabled, VideoUnavailable, VideoUnplayable, AgeRestricted) as exc:
        raise TranscriptError(
            f"Transcript cannot be retrieved for video '{video_id}': {exc}",
            "transcript_unavailable",
            404,
        ) from exc
    except InvalidVideoId as exc:
        raise TranscriptError(
            f"Invalid video id: '{video_id}'",
            "invalid_video_id",
            400,
        ) from exc
    except (RequestBlocked, IpBlocked) as exc:
        raise TranscriptError(
            "YouTube is blocking requests from this server's IP. "
            "Configure a proxy to work around this.",
            "youtube_blocked",
            503,
        ) from exc
    except (PoTokenRequired, YouTubeRequestFailed, CouldNotRetrieveTranscript) as exc:
        raise TranscriptError(
            f"Failed to retrieve transcript: {exc}",
            "fetch_failed",
            502,
        ) from exc

    return FullTranscript(
        video_id=transcript.video_id,
        language=transcript.language,
        language_code=transcript.language_code,
        is_generated=transcript.is_generated,
        segments=_build_segments(transcript),
    )


def get_full_transcript(video_id: str, language_priority: list[str]) -> FullTranscript:
    key = _cache_key(video_id, language_priority)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    full = _fetch_full(video_id, language_priority)
    _cache_put(key, full)
    return full


def _intersecting_range(
    segments: list[Segment], start: float, end: float
) -> tuple[int | None, int | None]:
    """Indices of the first/last segment overlapping [start, end).

    Falls back to the nearest line when the range lands in a gap (or is a
    zero-width point), mirroring the client-side range selection.
    """
    if not segments:
        return None, None

    first: int | None = None
    last: int | None = None
    for index, segment in enumerate(segments):
        segment_end = segment.start + (segment.duration or 0.0)
        if segment.start < end and segment_end > start:
            if first is None:
                first = index
            last = index

    if first is not None:
        return first, last

    best_index = 0
    best_distance: float | None = None
    for index, segment in enumerate(segments):
        distance = abs(segment.start - start)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index, best_index


def fetch_transcript(
    url_or_id: str,
    languages: list[str] | None,
    *,
    offset: int = 0,
    limit: int | None = None,
    start: float | None = None,
    end: float | None = None,
    context: int = 0,
    verbatim_only: bool = False,
) -> TranscriptResponse:
    video_id = extract_video_id(url_or_id)
    language_priority = languages or ["ne", "en"]
    full = get_full_transcript(video_id, language_priority)
    total = full.total

    base = dict(
        video_id=full.video_id,
        url=canonical_url(full.video_id),
        language=full.language,
        language_code=full.language_code,
        is_generated=full.is_generated,
        total_segments=total,
        duration=full.duration,
    )

    if verbatim_only:
        return TranscriptResponse(
            **base,
            verbatim=_join_text(full.segments),
            segments=[],
            offset=0,
            limit=0,
            has_more=False,
        )

    if start is not None or end is not None:
        window_start = start if start is not None else 0.0
        window_end = end if end is not None else float("inf")
        context_size = max(0, context)
        first, last = _intersecting_range(full.segments, window_start, window_end)
        if first is None or last is None:
            window: list[Segment] = []
            offset_used = 0
        else:
            slice_start = max(0, first - context_size)
            slice_end = min(total, last + 1 + context_size)
            window = full.segments[slice_start:slice_end]
            offset_used = slice_start
        limit_used: int | None = len(window)
    else:
        offset_used = max(0, offset)
        if limit is None:
            window = full.segments[offset_used:]
            limit_used = None
        else:
            window = full.segments[offset_used : offset_used + limit]
            limit_used = limit

    has_more = offset_used + len(window) < total
    include_verbatim = start is None and end is None and limit is None

    return TranscriptResponse(
        **base,
        verbatim=_join_text(full.segments) if include_verbatim else "",
        segments=window,
        normalized=None,
        changes=None,
        offset=offset_used,
        limit=limit_used,
        has_more=has_more,
    )


def list_languages(url_or_id: str) -> list[AvailableTranscript]:
    video_id = extract_video_id(url_or_id)

    try:
        transcript_list = _ytt_api.list(video_id)
    except (TranscriptsDisabled, VideoUnavailable, VideoUnplayable, AgeRestricted) as exc:
        raise TranscriptError(
            f"Transcripts are not available for video '{video_id}': {exc}",
            "transcript_unavailable",
            404,
        ) from exc
    except InvalidVideoId as exc:
        raise TranscriptError(
            f"Invalid video id: '{video_id}'",
            "invalid_video_id",
            400,
        ) from exc
    except (RequestBlocked, IpBlocked) as exc:
        raise TranscriptError(
            "YouTube is blocking requests from this server's IP. "
            "Configure a proxy to work around this.",
            "youtube_blocked",
            503,
        ) from exc
    except (PoTokenRequired, YouTubeRequestFailed, CouldNotRetrieveTranscript) as exc:
        raise TranscriptError(
            f"Failed to list transcripts: {exc}",
            "fetch_failed",
            502,
        ) from exc

    return [
        AvailableTranscript(
            language=transcript.language,
            language_code=transcript.language_code,
            is_generated=transcript.is_generated,
            is_translatable=transcript.is_translatable,
        )
        for transcript in transcript_list
    ]
