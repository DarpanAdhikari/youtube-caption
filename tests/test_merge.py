from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.merge import (
    TranscriptSegment,
    assess_quality,
    merge_segments,
    segments_to_text,
    timestamped_paragraphs,
)
from transcript_pipeline.state import ChunkRecord, JobState


def make_job() -> JobState:
    job = JobState(
        video_id="dQw4w9WgXcQ",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Test",
        output_dir="/tmp/test",
        duration=1200,
        model="nepali-medium-ct2",
    )
    job.chunks = [
        ChunkRecord(1, 0, 600, 0, 600, "chunk_1.wav"),
        ChunkRecord(2, 600, 1200, 599.5, 1200, "chunk_2.wav"),
    ]
    return job


def test_merge_deduplicates_overlap_and_is_monotonic():
    job = make_job()
    segments = [
        TranscriptSegment(0, 5, "नमस्ते", chunk_index=1),
        TranscriptSegment(595, 600, "संसद्", chunk_index=1),
        TranscriptSegment(599, 602, "संसद्", chunk_index=2),
        TranscriptSegment(605, 610, "अर्को", chunk_index=2),
    ]
    merged = merge_segments(segments, job)
    assert [segment.text for segment in merged] == ["नमस्ते", "संसद्", "अर्को"]
    assert merged[1].end == 602
    assert merged[2].start >= merged[1].start


def test_segment_outside_planned_chunk_is_dropped():
    job = make_job()
    segments = [
        TranscriptSegment(598, 600, "dropped", chunk_index=2),
        TranscriptSegment(601, 603, "kept", chunk_index=2),
    ]
    merged = merge_segments(segments, job)
    assert [segment.text for segment in merged] == ["kept"]


def test_text_and_paragraphs():
    segments = [
        TranscriptSegment(0, 5, "पहिलो वाक्य।"),
        TranscriptSegment(5, 10, "दोस्रो वाक्य।"),
    ]
    text = segments_to_text(segments)
    assert "पहिलो वाक्य।" in text
    assert "दोस्रो वाक्य।" in text
    paragraphs = timestamped_paragraphs(segments)
    assert paragraphs[0][0] == 0
    assert "पहिलो" in paragraphs[0][1]


def test_quality_flags_repetition():
    job = make_job()
    repeated = " ".join(["शब्द"] * 80)
    segments = [TranscriptSegment(0, 10, repeated)]
    quality = assess_quality(segments, job.duration)
    assert "high repeated n-gram ratio" in " ".join(quality["warnings"])
