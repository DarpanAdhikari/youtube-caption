# Long-form Nepali transcription pipeline

This package is a resumable pipeline for turning long YouTube recordings
(including videos without captions) into Nepali transcripts and subtitle files.
It runs in Colab or on Kaggle and is intentionally separate from the existing
captions API: nothing in `app/` is imported or changed.

## What it does

1. Downloads the best available audio with `yt-dlp`.
2. Converts it to 16 kHz mono WAV with `ffmpeg`.
3. Detects silence and cuts the recording into roughly 10-minute chunks.
4. Transcribes each chunk with a permissive Nepali Whisper model and preserves
   the raw transcript.
5. Applies conservative rule-based text normalization and an optional glossary.
6. Exports plain text, SRT, VTT, JSON, and DOCX, plus a ZIP bundle.

An 8-hour recording is normally 50-60 chunks. Completed chunks, chunk audio, and
the job state are saved to disk, so a Colab or Kaggle disconnect does not require
starting over.

## Quick start (Colab)

Open `notebooks/nepali_transcript_colab.ipynb` in Colab and follow its cells.
Use a GPU runtime. The free T4 tier is the design target; 12 hours of audio is a
practical upper bound for one session, depending on model and runtime limits.

The notebook:

- installs `requirements-pipeline.txt`;
- mounts Google Drive;
- keeps the model cache and finished jobs in Drive;
- transcribes a URL and downloads the ZIP bundle;
- shows how to resume, re-export, and run the benchmark.

## Quick start (Kaggle)

For the reference-backed Phase 0 benchmark, upload
`notebooks/nepali_parliament_benchmark_kaggle.ipynb` to Kaggle, enable Internet
and a T4/P100 GPU, and run all cells. It clones the `phase0-benchmark` branch,
prepares 100 duration-stratified clips from the test split of
`kiranpantha/NepaliParliamentDSv1`, benchmarks four models, and writes
`parliament-100-results.zip` under `/kaggle/working`.

The dataset has Devanagari references but no declared license. Use it for
evaluation only and do not redistribute its audio or references.

## Local command line

```bash
python3 -m venv .venv-pipeline
source .venv-pipeline/bin/activate
pip install -r requirements-pipeline.txt

python -m transcript_pipeline \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --output-root ./DRP-Transcripts \
  --model nepali-medium-ct2 \
  --fallback-model openai-turbo \
  --refine-mode rule
```

`requirements-pipeline.txt` pins `av==15.1.0`. Newer PyAV builds (for example
19.0.2) fail with the installed `faster-whisper` release and raise
`open() got an unexpected keyword argument 'metadata_errors'`.

Useful options:

```bash
# List the curated permissive model registry.
python -m transcript_pipeline --list-models

# Resume after a disconnect.
python -m transcript_pipeline --url "URL" --run-mode resume

# Re-run export from an existing raw transcript.
python -m transcript_pipeline --url "URL" --run-mode export_only

# Re-run refinement only (for example after editing corrections.json).
python -m transcript_pipeline --url "URL" --run-mode refine_only

# Try failed chunks again.
python -m transcript_pipeline --url "URL" --run-mode retry_failed

# Transcribe a local audio/video file instead of a URL.
python -m transcript_pipeline --audio ./recording.mp4 --output-root ./DRP-Transcripts
```

`--run-mode auto` (the default) resumes an unfinished job and re-exports a
completed one. Use `--run-mode fresh --force-overwrite` to deliberately restart
a job with different chunk or model settings.

## Output layout

For a video `VIDEO_ID` with title `TITLE`:

```text
DRP-Transcripts/VIDEO_ID_TITLE/
  audio/
    original.*              # source audio
    metadata.json
    normalized.wav          # removed after success unless --keep-wav
  chunks/
    chunk_0001.json         # raw ASR result for a chunk
    audio/                  # removed after success unless --keep-chunks
  transcripts/
    raw.json
    refined.json
  refinement/
    changes.json
    report.json
  output/
    transcript.txt
    transcript.raw.txt
    subtitles.srt
    subtitles.raw.srt
    subtitles.vtt
    subtitles.raw.vtt
    transcript.json
    transcript.docx         # when python-docx is installed
    metadata.json
    transcript_bundle.zip
  state/
    job.json
    errors.jsonl
```

`raw.json` is the unedited ASR output. The refined files are the default
deliverables, but the raw files are always kept so a refinement can be audited
or redone.

## Models and the benchmark gate

The default is `nepali-medium-ct2`
(`Dragneel/whisper-medium-nepali-openslr-ct2`, Apache-2.0). It is a practical
first choice, not a final quality claim. The model choice should be locked by
the Phase 0 benchmark on real Nepali recordings, especially parliament audio.

Curated registry:

| Key | Repository | License | Backend | Notes |
|---|---|---|---|---|
| `nepali-medium-ct2` | `Dragneel/whisper-medium-nepali-openslr-ct2` | Apache-2.0 | CT2 | 775 MB, reported WER 11.17 |
| `nepali-turbo-ct2` | `ayushkhadkaa/whisper-turbo-nepali-ct2-fp16` | Apache-2.0 | CT2 | 1.6 GB, provenance/WER needs checking |
| `nepali-large-v3` | `Dragneel/whisper-large-v3-nepali-openslr` | Apache-2.0 | Transformers | Reported WER 22.31; convert to CT2 |
| `nepali-large-v3-kiran` | `kiranpantha/whisper-large-v3-nepali` | Apache-2.0 | Transformers | Reported WER 18.73; tokenizer borrowed from OpenAI |
| `nepali-turbo-kiran` | `kiranpantha/whisper-large-v3-turbo-nepali` | MIT | Transformers | Turbo label unverified; inspect before use |
| `openai-turbo` | `large-v3-turbo` | MIT | faster-whisper size | Baseline only |
| `openai-large-v3` | `large-v3` | MIT | faster-whisper size | Baseline only |

Non-commercial model repositories are excluded by policy. Custom Hugging Face
IDs are accepted, but their license/architecture cannot be verified
automatically, so prefer the registry for production work.

### Benchmark

Benchmark local files:

```bash
python -m transcript_pipeline.benchmark \
  --models nepali-medium-ct2,nepali-turbo-ct2,openai-turbo \
  --audio /data/sample1.wav \
  --audio /data/sample2.wav \
  --reference /data/sample1.txt \
  --reference /data/sample2.txt \
  --output ./benchmark
```

Benchmark a directory of clips and references:

```bash
python -m transcript_pipeline.benchmark \
  --models nepali-medium-ct2,nepali-turbo-ct2 \
  --audio-dir /data/parliament-100/audio \
  --reference-dir /data/parliament-100/references \
  --output ./benchmark
```

Each reference is a `.txt` file matched to the audio by stem. Explicit
`--audio` and `--reference` values take precedence and can be mixed with the
directory options.

Benchmark YouTube URLs without transcribing the whole recording:

```bash
python -m transcript_pipeline.benchmark \
  --models nepali-medium-ct2,nepali-turbo-ct2,openai-turbo \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --max-seconds 600 \
  --output ./benchmark
```

`--max-seconds` defaults to 600 for URL runs; use `0` for full audio. For URL
samples, name a reference `VIDEO_ID.txt` so it matches the trimmed sample.
WER/CER are only reported when a reference is supplied. Also spot-check names,
dates, numbers, and institution names, since aggregate error rates can hide the
errors that matter most in parliamentary records.

The Kaggle notebook prepares a held-out in-domain benchmark from the parliament
test split. The Colab notebook includes an uploaded-audio benchmark cell (for
parliament clips plus reference transcripts) and a FLEURS `ne_np` cell that
downloads 10 public reference-backed clips for a quick GPU sanity check.

### Current gate status

A 600-second sample from the 2081 Asar 11 House of Representatives meeting
(`cw5HPUXW4WY`) did not pass manual review with `nepali-medium-ct2`,
`nepali-turbo-ct2`, `openai-turbo`, `openai-large-v3`, or
`nepali-large-v3-kiran`. The last model also emitted mixed English/Nepali text
despite `language=ne`. The sample has no captions or reference transcript, so
WER/CER could not be computed. Do not treat these outputs as publication-ready;
run the reference-backed Kaggle benchmark before integrating the pipeline with
the API.

The in-domain Kaggle benchmark is prepared but not yet run. It samples 100
duration-stratified clips from the held-out test split of
`kiranpantha/NepaliParliamentDSv1` and is set up to compare
`nepali-medium-ct2`, `nepali-turbo-ct2`, `openai-large-v3`, and
`nepali-large-v3`. The dataset license is undeclared, so keep its audio and
references for evaluation only.

A small reference-backed check on 10 FLEURS `ne_np` test clips gave mean
WER/CER of 28.2%/11.9% for `nepali-medium-ct2`, 40.7%/16.1% for
`nepali-turbo-ct2`, 100.6%/27.5% for `openai-large-v3`, and 109.2%/87.3% for
`nepali-large-v3-kiran`. This is a directional sample, not a release threshold;
`nepali-medium-ct2` remains the best tested option, but parliament audio needs
domain adaptation, a glossary, or a stronger model.

## Refinement

`--refine-mode rule` is the default and is safe for offline use:

- Unicode normalization and whitespace cleanup;
- spacing around `।` and `॥`;
- exact-match corrections from `--corrections`;
- optional glossary terms/variants from `--glossary`;
- numbers, dates, URLs, email addresses, and Latin-script spans are protected
  from dictionary replacement.

`--refine-mode neural --corrector-url http://HOST:PORT` sends each segment to an
external Nepali corrector and accepts a correction only when similarity is at
least `--refine-min-similarity` (default `0.90`) and all protected spans survive.
If the corrector is unavailable, the rules are still applied. The neural path is
optional and is not required for the pipeline.

Glossary format:

```json
{
  "terms": ["प्रतिनिधि सभा", "नेपाल सरकार"],
  "variants": {
    "प्रतिनिधी सभा": "प्रतिनिधि सभा"
  }
}
```

Corrections format:

```json
{
  "errors": {
    "परिक्षा": "परीक्षा",
    "अध्यक्षज्यु": "अध्यक्षज्यू"
  }
}
```

## Tests

The unit tests use only the standard library and do not download models or call
YouTube:

```bash
python3 -m pytest -q
```

Heavy dependencies (`faster-whisper`, `torch`, `yt-dlp`, `python-docx`) are
imported only when their feature is used, so the test suite can run in a minimal
environment.

## Operational notes

- Use Google Drive for `--output-root` in Colab; `/content` is temporary. On
  Kaggle, keep outputs under `/kaggle/working` and download the ZIP before the
  session ends.
- The model cache can also live in Drive so a fresh runtime does not download
  the model again.
- `--keep-chunks` and `--keep-wav` trade disk space for faster debugging.
- If all chunks fail, the job is marked failed and `state/errors.jsonl` contains
  the tracebacks.
- If some chunks fail, the job finishes as `completed_with_errors` and successful
  chunks are still exported.
- `--cookies` accepts a Netscape-format `cookies.txt` for restricted videos.
- Respect YouTube's terms, copyright, and the permissions attached to the source
  recording.

## Deferred work

The extension and API integration are deliberately deferred until the transcript
quality gate passes. The intended future path is:

1. Run the pipeline in Colab, on Kaggle, or in a worker.
2. Expose a job-based API (submit, status, download).
3. Teach the existing background worker to proxy that API when YouTube has no
   captions, while leaving the current captions endpoints unchanged.
