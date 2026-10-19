# YouTube Transcript API

A free, open-source REST API that extracts YouTube video transcripts and returns them as clean, structured JSON. Perfect for building apps, scripts, or browser extensions that need access to YouTube captions.

## What This Is

This API retrieves transcripts from YouTube's "Show transcript" feature and exposes them through simple HTTP endpoints. It returns:

- The full verbatim transcript text
- Timestamped segments (start time, duration, and text)
- Available transcript languages for any video
- Flexible filtering, paging, and time-window queries

## Live Demo

Once deployed, you can try it at: `https://your-service-name.onrender.com`

Interactive API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

## How It Works

1. **Send a request** with a YouTube URL or video ID to one of the endpoints.
2. **Fetches the transcript** using `youtube-transcript-api` to retrieve available captions from YouTube.
3. **Caches intelligently** - full transcripts are cached in memory for 1 hour per (video + language priority), so repeated requests and pagination don't hit YouTube again.
4. **Returns structured JSON** with transcript data, metadata, and pagination info.

This approach avoids scraping HTML directly and works with most publicly available transcripts (auto-generated and manual captions where accessible).

## API Usage

### Health Check

```bash
GET /health
```

```json
{ "status": "ok" }
```

### Get Transcript

```bash
GET /transcript?url=<youtube-url-or-id>
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | string | _required_ | Full YouTube URL (`youtube.com/watch?v=...`, `youtu.be/...`) or 11-char video ID. |
| `languages` | string | `ne,en` | Comma-separated language codes in priority order. Tries in order until it finds a match (e.g., `en,fr,es`). |
| `offset` | integer | `0` | Skip this many segments (for pagination). |
| `limit` | integer | _all_ | Return at most this many segments (1–5000). |
| `start` | float | _none_ | Filter to segments overlapping from this time (seconds). |
| `end` | float | _none_ | Filter to segments overlapping until this time (seconds). |
| `context` | integer | `0` | Include extra segments before/after the time window (0–100). |
| `verbatim_only` | boolean | `false` | Return only joined text; `segments` will be empty. |

**Example:**

```bash
curl "https://your-api.example.com/transcript?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

**Response (simplified):**

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "language": "English",
  "language_code": "en",
  "is_generated": false,
  "verbatim": "...",
  "segments": [
    { "start": 0.0, "duration": 3.0, "text": "We're no strangers to love" }
  ],
  "total_segments": 61,
  "duration": 211.32,
  "has_more": false
}
```

**Notes:**
- For paged or time-windowed requests, `verbatim` is typically empty to keep responses lightweight.
- `total_segments`, `duration`, and `has_more` always reflect the **entire** transcript, not just the returned slice.

### Get Transcript by Video ID

```bash
GET /transcript/{video_id}
```

Same as above, but pass the video ID in the URL path.

### List Available Languages

```bash
GET /languages?url=<youtube-url-or-id>
```

Returns all transcripts YouTube makes available for a video.

```json
{
  "video_id": "dQw4w9WgXcQ",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "transcripts": [
    { "language_code": "en", "language": "English", "is_generated": false },
    { "language_code": "ne", "language": "Nepali", "is_generated": false }
  ]
}
```

## Use With the Browser Extension

If you want to use this API directly in your browser, check out the official companion extension:

**[YouTube Caption Extension](https://github.com/DarpanAdhikari/youtube-caption-extention.git)**

It works with any self-hosted instance of this API, making it easy to fetch, view, or copy YouTube captions right from the page.

## Self-Hosting

This API is designed to be easy to self-host. It's deployment-ready for [Render](https://render.com/), but you can run it anywhere that supports Python/ASGI (Railway, Fly.io, DigitalOcean App Platform, etc.).

### Requirements

- Python 3.10+
- Internet access to reach YouTube
- A host that can bind to the port provided via `$PORT` (cloud platforms usually set this)

### Local Development

```bash
# Clone and setup
git clone https://github.com/DarpanAdhikari/youtube-caption.git
cd youtube-caption

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run dev server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Deploy on Render

This repo includes `Procfile` and `render.yaml`. On Render:

1. Create a new Web Service and connect this repository
2. Render will automatically install dependencies and start with Gunicorn + Uvicorn workers
3. The app binds to `0.0.0.0:$PORT` as required

No extra configuration is needed out of the box.

## Technical Details

- **Framework**: FastAPI (async, with proper error handling and OpenAPI schema)
- **Caching**: In-memory LRU cache (max 32 entries, 1h TTL). Each worker maintains its own cache.
- **Error Handling**: Consistent JSON errors with descriptive `error_type` values (e.g., `invalid_url`, `no_transcript_found`, `youtube_blocked`, `transcript_unavailable`, `fetch_failed`).
- **Limits**: `limit` capped at 5000 segments; `context` capped at 100 segments to prevent oversized responses.
- **Privacy**: No tracking, no logging of URLs by default, and no persistent storage. Requests go directly from your server to YouTube.

## Contributing

We welcome contributions from everyone! Whether it's fixing bugs, improving docs, or adding features, your help is appreciated.

### How to Contribute

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**: Keep them focused and follow existing code style
4. **Test locally**: Ensure the API still works as expected
5. **Commit**: Write clear, concise commit messages
6. **Push and open a PR**: Describe what you changed and why

### Ideas for Contributions

- Support for additional YouTube URL formats
- Alternative transcript sources/fallbacks
- Rate limiting or request throttling
- Optional Redis-backed caching for multi-worker deployments
- Better handling for region-locked/age-restricted cases

## Known Limitations

- Some videos may have no transcripts available (disabled by creator, or age-restricted).
- YouTube may block requests from certain IP ranges (especially shared cloud IPs). If this happens, the API returns HTTP 503 with `error_type: youtube_blocked`.
- Auto-generated transcripts vary by availability per video/language.
- Caching is in-memory only; restarting the service clears the cache.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute it as you see fit.

## Acknowledgments

Built with [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) by [jdepoix](https://github.com/jdepoix).
