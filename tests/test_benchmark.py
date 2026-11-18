from pathlib import Path

import pytest

from transcript_pipeline.audio import AudioMetadata
from transcript_pipeline.benchmark import (
    BenchmarkRow,
    _aggregate_rows,
    _collect_audio_files,
    _download_url_audio,
    _load_references,
    _render_markdown,
    char_error_rate,
    edit_distance,
    normalize_metric_text,
    word_error_rate,
)
from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.progress import NullProgress
import transcript_pipeline.benchmark as benchmark_module


def test_edit_distance():
    assert edit_distance(list("abc"), list("abc")) == 0
    assert edit_distance(list("abc"), list("abd")) == 1
    assert edit_distance(list("abc"), list("ab")) == 1


def test_word_and_char_error_rate():
    reference = "नेपालको संसद् बैठक"
    assert word_error_rate(reference, reference) == 0
    assert word_error_rate(reference, "नेपालको बैठक") > 0
    assert char_error_rate(reference, reference) == 0
    assert char_error_rate(reference, "नेपालको संसद") > 0


def test_normalize_metric_text():
    assert normalize_metric_text("  नेपाल।  संसद्,  ") == "नेपाल संसद्"


def test_download_url_audio_reuses_downloaded_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "downloaded.m4a"
    source.write_bytes(b"fake")

    def fake_download(config, paths, reporter, *, known_metadata=None):
        return AudioMetadata(
            video_id="dQw4w9WgXcQ",
            title="Benchmark",
            duration=10,
            audio_path=str(source),
        )

    monkeypatch.setattr(benchmark_module, "download_audio", fake_download)
    path = _download_url_audio(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        PipelineConfig(),
        tmp_path,
        0,
        NullProgress(),
    )
    assert path == source


def _row(
    model: str = "nepali-medium-ct2",
    *,
    audio: str = "clip.wav",
    duration: float = 10.0,
    elapsed: float = 5.0,
    wer: float | None = 0.2,
    cer: float | None = 0.1,
    error: str | None = None,
) -> BenchmarkRow:
    return BenchmarkRow(
        model=model,
        repo_id="example/model",
        audio=audio,
        duration=duration,
        elapsed=elapsed,
        rtf=(elapsed / duration) if duration else 0.0,
        segments=1,
        config="ct2",
        languages={"ne": 1.0},
        transcript_path="",
        wer=wer,
        cer=cer,
        error=error,
    )


def test_aggregate_rows_reports_mean_median_and_failures():
    rows = [
        _row(audio="a.wav", wer=0.2, cer=0.1),
        _row(audio="b.wav", wer=0.4, cer=0.3),
        _row(
            audio="c.wav",
            duration=0.0,
            elapsed=0.0,
            wer=None,
            cer=None,
            error="boom",
        ),
    ]
    aggregates = _aggregate_rows(rows, [{"model": "nepali-medium-ct2"}])
    assert len(aggregates) == 1
    aggregate = aggregates[0]
    assert aggregate["clips"] == 3
    assert aggregate["scored_clips"] == 2
    assert aggregate["failed_clips"] == 1
    assert aggregate["mean_wer"] == pytest.approx(0.3)
    assert aggregate["median_wer"] == pytest.approx(0.3)
    assert aggregate["mean_cer"] == pytest.approx(0.2)
    assert aggregate["median_cer"] == pytest.approx(0.2)
    assert aggregate["rtf"] == pytest.approx(0.5)


def test_render_markdown_includes_aggregate_table():
    rows = [_row(), _row(audio="b.wav", wer=0.4, cer=0.3)]
    summary = {
        "models": [{"model": "nepali-medium-ct2"}],
        "aggregates": _aggregate_rows(rows, [{"model": "nepali-medium-ct2"}]),
        "rows": [row.to_dict() for row in rows],
    }
    markdown = _render_markdown(summary)
    assert "## Aggregate" in markdown
    assert "Mean WER" in markdown
    assert "30.00%" in markdown


def test_collect_audio_files_filters_supported_extensions(tmp_path: Path):
    (tmp_path / "a.wav").write_bytes(b"a")
    (tmp_path / "b.flac").write_bytes(b"b")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    assert _collect_audio_files([str(tmp_path)]) == [
        tmp_path / "a.wav",
        tmp_path / "b.flac",
    ]


def test_collect_audio_files_rejects_empty_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _collect_audio_files([str(tmp_path)])


def test_load_references_from_directory_and_explicit_override(tmp_path: Path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "clip.txt").write_text("directory reference", encoding="utf-8")
    (refs / "other.txt").write_text("other reference", encoding="utf-8")
    explicit = tmp_path / "clip.txt"
    explicit.write_text("explicit reference", encoding="utf-8")

    loaded = _load_references([str(explicit)], [str(refs)])
    assert loaded["clip"] == "explicit reference"
    assert loaded["other"] == "other reference"
