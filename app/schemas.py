from typing import Optional

from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    duration: float
    text: str


class TranscriptResponse(BaseModel):
    video_id: str
    url: str
    language: str
    language_code: str
    is_generated: bool
    verbatim: str
    segments: list[Segment]
    normalized: Optional[str] = None
    changes: Optional[str] = None

    # Paging / range metadata. A long transcript is far too big to hand over in
    # one piece, so callers ask for a window and use these fields to know where
    # they are and whether to ask for more.
    offset: int = 0
    limit: Optional[int] = None
    total_segments: int = 0
    duration: float = 0.0
    has_more: bool = False


class AvailableTranscript(BaseModel):
    language: str
    language_code: str
    is_generated: bool
    is_translatable: bool


class LanguagesResponse(BaseModel):
    video_id: str
    url: str
    transcripts: list[AvailableTranscript]


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
