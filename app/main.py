from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .schemas import ErrorResponse, LanguagesResponse, TranscriptResponse
from .service import TranscriptError, extract_video_id, fetch_transcript, list_languages

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Extract YouTube transcripts (verbatim text + timestamped segments) "
        "from a video URL or ID."
    ),
)


@app.exception_handler(TranscriptError)
async def transcript_error_handler(_, exc: TranscriptError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.error_type},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail), "error_type": "http_error"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request parameters",
            "error_type": "validation_error",
            "errors": exc.errors(),
        },
    )


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.get(
    "/transcript",
    response_model=TranscriptResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["transcript"],
)
async def get_transcript(
    url: str = Query(..., description="Full YouTube URL or a bare 11-char video ID"),
    languages: str = Query(
        None,
        description="Comma-separated language codes in priority order, e.g. 'ne,en'",
    ),
    offset: int = Query(0, ge=0, description="Segment index to start from (paging)"),
    limit: int = Query(
        None,
        ge=1,
        le=5000,
        description="Max segments to return; omit for the whole transcript",
    ),
    start: float = Query(None, ge=0, description="Window start time in seconds"),
    end: float = Query(None, ge=0, description="Window end time in seconds"),
    context: int = Query(
        0,
        ge=0,
        le=100,
        description="Extra segments kept before/after a time window",
    ),
    verbatim_only: bool = Query(
        False,
        description="Return only the joined verbatim text (no segments)",
    ),
):
    return await run_fetch(
        url,
        _parse_languages(languages),
        offset=offset,
        limit=limit,
        start=start,
        end=end,
        context=context,
        verbatim_only=verbatim_only,
    )


@app.get(
    "/transcript/{video_id}",
    response_model=TranscriptResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["transcript"],
)
async def get_transcript_by_id(
    video_id: str,
    languages: str = Query(None, description="Comma-separated language codes"),
    offset: int = Query(0, ge=0),
    limit: int = Query(None, ge=1, le=5000),
    start: float = Query(None, ge=0),
    end: float = Query(None, ge=0),
    context: int = Query(0, ge=0, le=100),
    verbatim_only: bool = Query(False),
):
    return await run_fetch(
        video_id,
        _parse_languages(languages),
        offset=offset,
        limit=limit,
        start=start,
        end=end,
        context=context,
        verbatim_only=verbatim_only,
    )


@app.get(
    "/languages",
    response_model=LanguagesResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["transcript"],
)
async def get_languages(
    url: str = Query(..., description="Full YouTube URL or a bare video ID"),
):
    return await run_list(url)


def _parse_languages(languages: str | None) -> list[str] | None:
    if not languages:
        return None
    return [code.strip() for code in languages.split(",") if code.strip()]


async def run_fetch(
    url: str,
    languages: list[str] | None,
    *,
    offset: int = 0,
    limit: int | None = None,
    start: float | None = None,
    end: float | None = None,
    context: int = 0,
    verbatim_only: bool = False,
) -> TranscriptResponse:
    return await run_in_threadpool(
        fetch_transcript,
        url,
        languages,
        offset=offset,
        limit=limit,
        start=start,
        end=end,
        context=context,
        verbatim_only=verbatim_only,
    )


async def run_list(url: str) -> LanguagesResponse:
    video_id = extract_video_id(url)
    transcripts = await run_in_threadpool(list_languages, url)
    return LanguagesResponse(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        transcripts=transcripts,
    )
