import json
import zipfile
from pathlib import Path

from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.exports import (
    build_caption_cues,
    export_all,
    render_srt,
    render_vtt,
)
from transcript_pipeline.merge import TranscriptSegment
from transcript_pipeline.paths import JobPaths
from transcript_pipeline.state import new_job


def make_job(tmp_path: Path):
    config = PipelineConfig(output_root=tmp_path)
    paths = JobPaths.for_video(tmp_path, "dQw4w9WgXcQ", "Test Video").ensure()
    job = new_job(
        config,
        video_id="dQw4w9WgXcQ",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Test Video",
        paths=paths,
    )
    job.duration = 20
    return config, paths, job


def test_caption_cues_and_srt_vtt():
    config = PipelineConfig(max_chars_per_line=20, max_lines=2, max_cps=100)
    segments = [
        TranscriptSegment(0, 5, "यो एक लामो वाक्य हो जसलाई उपशीर्षकमा बाँड्नुपर्छ"),
        TranscriptSegment(5, 10, "दोस्रो वाक्य।"),
    ]
    cues = build_caption_cues(segments, config)
    assert cues
    srt = render_srt(cues)
    vtt = render_vtt(cues)
    assert "-->" in srt
    assert "WEBVTT" in vtt
    assert "00:00:00,000" in srt
    assert "00:00:00.000" in vtt


def test_export_all_writes_outputs(tmp_path: Path):
    config, paths, job = make_job(tmp_path)
    segments = [
        TranscriptSegment(0, 5, "पहिलो वाक्य।"),
        TranscriptSegment(5, 10, "दोस्रो वाक्य।"),
    ]
    result = export_all(
        paths,
        job,
        segments,
        segments,
        config,
        quality={"warnings": []},
        refinement_report={"mode": "off"},
    )
    assert result.files["transcript_txt"].exists()
    assert result.files["subtitles_srt"].exists()
    assert result.files["metadata_json"].exists()
    assert result.files["bundle_zip"].exists()
    assert "पहिलो" in result.files["transcript_txt"].read_text(encoding="utf-8")
    metadata = json.loads(result.files["metadata_json"].read_text(encoding="utf-8"))
    assert metadata["counts"]["refined_segments"] == 2
    with zipfile.ZipFile(result.files["bundle_zip"]) as archive:
        names = archive.namelist()
        assert "transcript.txt" in names
        assert "metadata.json" in names
        assert len(names) == len(set(names))
