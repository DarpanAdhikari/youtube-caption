# YouTube Transcript API

A FastAPI service to extract YouTube video transcripts (from the "Show transcript" panel) and return them as structured JSON. It provides full verbatim text, timestamped segments, language discovery, and flexible querying (paging, time windows, and context).

## Features

- Extract transcripts from YouTube URLs or video IDs
- Get available transcript languages for a video
- Return full text or timestamped segments
- Support for language prioritization (e.g., `ne,en`)
- Paging with `offset`/`limit` and time-range filtering with `start`/`end` + `context`
- In-memory caching per `(video, language)` for faster repeat requests
- Interactive API docs via Swagger UI (`/docs`) and ReDoc (`/redoc`)
- Ready for deployment on [Render](https://render.com/)

## Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Uvicorn](https://www.uvicorn.org/) - ASGI server
- [Gunicorn](https://gunicorn.org/) - Production WSGI/ASGI process manager
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) - Transcript extraction

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/DarpanAdhikari/youtube-caption.git
   cd youtube-caption
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

5. **Open API docs**
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Production (Render)

This repo includes a `Procfile` and `render.yaml`. To deploy:

1. Fork/clone this repo and push to GitHub.
2. Go to [Render](https://render.com/) → New → Web Service → Connect your repo.
3. Render will use the `render.yaml` config or auto-detect Python. Ensure build uses `pip install -r requirements.txt` and start command uses Gunicorn with Uvicorn workers (included in `Procfile`).
4. Your service will be available at `https://your-service-name.onrender.com`.

The app binds to `0.0.0.0:$PORT` as required by Render.

## API Reference

### Health Check

**GET** `/health`

Returns server status.

```json
{
  "status": "ok"
}
```

### Get Transcript

**GET** `/transcript`

Extract transcript from a YouTube URL or video ID.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | `string` | Yes | Full YouTube URL (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`, `https://youtu.be/dQw4w9WgXcQ`) or bare 11-char video ID. |
| `languages` | `string` | No | Comma-separated language codes in priority order (default: `ne,en`). Example: `en,fr,es`. |
| `offset` | `integer` | No | Segment index to start from for paging (default: `0`, min: `0`). |
| `limit` | `integer` | No | Max number of segments to return (min: `1`, max: `5000`). Omit to return all segments. |
| `start` | `float` | No | Start time in seconds to filter segments (min: `0`). |
| `end` | `float` | No | End time in seconds to filter segments (min: `0`). |
| `context` | `integer` | No | Extra segments to include before/after the time window (min: `0`, max: `100`, default: `0`). |
| `verbatim_only` | `boolean` | No | If `true`, returns only the joined verbatim text with an empty `segments` array (default: `false`). |

#### Examples

```bash
# Get full transcript
curl "https://your-service.onrender.com/transcript?url=dQw4w9WgXcQ"

# With specific language priority
curl "https://your-service.onrender.com/transcript?url=https://youtu.be/dQw4w9WgXcQ&languages=en"

# Paginated results
curl "https://your-service.onrender.com/transcript?url=dQw4w9WgXcQ&offset=10&limit=20"

# Time window with context
curl "https://your-service.onrender.com/transcript?url=dQw4w9WgXcQ&start=60&end=90&context=2"

# Verbatim text only
curl "https://your-service.onrender.com/transcript?url=dQw4w9WgXcQ&verbatim_only=true"
```

#### Response

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "language": "English",
  "language_code": "en",
  "is_generated": false,
  "verbatim": "Never gonna give you up...",
  "segments": [
    { "start": 0.0, "duration": 3.0, "text": "We're no strangers to love" }
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

**Notes:**
- `verbatim` is populated only on unpaged, untimed requests (or when `verbatim_only=true`). For paged/time-windowed responses, it's typically omitted/empty to keep payloads small.
- `total_segments`, `duration`, and `has_more` always describe the entire transcript, not the returned window.
- `normalized` and `changes` are reserved for future Nepali normalization features.

### Get Transcript by Video ID

**GET** `/transcript/{video_id}`

Same as `/transcript`, but pass the video ID directly in the path.

```bash
curl "https://your-service.onrender.com/transcript/dQw4w9WgXcQ?languages=en"
```

### List Available Languages

**GET** `/languages`

Get all available transcript languages for a given video.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `url` | `string` | Yes | YouTube URL or video ID. |

```bash
curl "https://your-service.onrender.com/languages?url=dQw4w9WgXcQ"
```

#### Response

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "transcripts": [
    {
      "language_code": "en",
      "language": "English",
      "is_generated": false
    },
    {
      "language_code": "ne",
      "language": "Nepali",
      "is_generated": false
    }
  ]
}
```

## Error Handling

Errors are returned as JSON with `detail` and `error_type`:

```json
{
  "detail": "Could not extract a video id from: invalid-url",
  "error_type": "invalid_url"
}
```

Common error types: `invalid_url`, `invalid_video_id`, `no_transcript_found`, `transcript_unavailable`, `youtube_blocked`, `fetch_failed`, `validation_error`.

## Caching

The service caches full transcripts in memory for **1 hour** per `(video_id, language_priority)`. This avoids repeated calls to YouTube for paged/range requests. The cache has a max size of 32 entries with LRU eviction.

## Deployment Notes

- **Port binding**: Must bind to `0.0.0.0:$PORT`. Both `Procfile` and `render.yaml` handle this.
- **Process manager**: Uses `gunicorn` with `uvicorn.workers.UvicornWorker` for production performance.
- **YouTube blocking**: If YouTube blocks requests from the server's IP, you'll get `youtube_blocked` (503). This is common on some cloud providers/shared IPs; proxies or PoToken handling may be needed depending on the video.
- **Ephemeral filesystem**: No files are written to disk; cache is in-memory and resets on redeploy/restart (expected on Render free tier).

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is open-source and available under the MIT License.
