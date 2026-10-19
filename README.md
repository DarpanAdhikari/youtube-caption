# YouTube Transcript API

FastAPI service that extracts a YouTube video's transcript (the **Show transcript** panel) and returns it as JSON: full verbatim text plus timestamped segments.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

`--host 0.0.0.0` binds every interface, so the app is reachable through port
forwarding, a VM, or another machine on the LAN. For local-only use,
`--host 127.0.0.1` restricts it to this machine.

Interactive docs: http://127.0.0.1:8000/docs

## Endpoints

### `GET /health`

```json
{"status": "ok"}
```

### `GET /transcript`

| Param           | Required | Description                                                        |
| --------------- | -------- | ------------------------------------------------------------------ |
| `url`           | yes      | Full YouTube URL or a bare 11-char video ID                        |
| `languages`     | no       | Comma-separated codes in priority order (default `ne,en`)          |
| `offset`        | no       | Segment index to start from (paging), default `0`                  |
| `limit`         | no       | Max segments to return (1–5000); omit for the whole transcript     |
| `start` / `end` | no       | Time window in seconds; returns segments overlapping it            |
| `context`       | no       | Extra segments kept before/after a time window (0–100)             |
| `verbatim_only` | no       | Return only the joined text, with an empty `segments` list         |

```bash
curl "http://127.0.0.1:8000/transcript?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ"
curl "http://127.0.0.1:8000/transcript?url=https://youtu.be/dQw4w9WgXcQ&languages=ne,en"
curl "http://127.0.0.1:8000/transcript?url=dQw4w9WgXcQ&offset=300&limit=300"   # next page
curl "http://127.0.0.1:8000/transcript?url=dQw4w9WgXcQ&start=60&end=90&context=2"
```

Response:

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "language": "English",
  "language_code": "en",
  "is_generated": false,
  "verbatim": "full joined transcript text ...",
  "segments": [
    {"start": 1.36, "duration": 1.68, "text": "[♪♪♪]"}
  ],
  "normalized": null,
  "changes": null,
  "offset": 0,
  "limit": null,
  "total_segments": 61,
  "duration": 211.32,
  "has_more": false
}
```

`normalized` / `changes` are reserved for the future Nepali normalization pipeline.

`verbatim` is only populated on an unpaged, untimed request (or with `verbatim_only=true`);
paged and time-window responses leave it empty so a five-hour transcript is never
serialised as a string just to return one window. `total_segments`, `duration` and
`has_more` always describe the **whole** video, not the returned window. The parsed
transcript is cached in memory per `(video, language)` for an hour, so the first request
hits YouTube and the following pages are served from cache.

### `GET /transcript/{video_id}`

Same as above, with the video ID in the path:

```bash
curl "http://127.0.0.1:8000/transcript/dQw4w9WgXcQ"
```

### `GET /languages`

Lists every transcript available for a video (use this when `404 no_transcript_found` occurs):

```bash
curl "http://127.0.0.1:8000/languages?url=dQw4w9WgXcQ"
```

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "transcripts": [
    {"language": "English", "language_code": "en", "is_generated": false, "is_translatable": true}
  ]
}
```

## Errors

All errors return `{"detail": "...", "error_type": "..."}`.

| HTTP | `error_type`           | Meaning                                     |
| ---- | ---------------------- | ------------------------------------------- |
| 400  | `invalid_url`          | Could not extract a video ID                |
| 400  | `invalid_video_id`     | YouTube rejected the ID                     |
| 404  | `no_transcript_found`  | No transcript in the requested languages    |
| 404  | `transcript_unavailable` | Video/transcripts disabled, private, etc. |
| 502  | `fetch_failed`         | YouTube responded but retrieval failed      |
| 503  | `youtube_blocked`      | YouTube is IP-blocking this server          |

## Notes

- Supported URL forms: `watch?v=`, `youtu.be/`, `shorts/`, `embed/`, `live/`, or a bare video ID.
- Bound to `0.0.0.0`, the API has no authentication — anyone who can reach the port can
  use it. Forward only the port you need and keep it off the public internet.
- YouTube blocks many cloud-provider IPs; if you hit `503 youtube_blocked`, route requests through a proxy (see `youtube-transcript-api` proxy docs).
- This uses an undocumented YouTube endpoint, so behavior may change upstream.
