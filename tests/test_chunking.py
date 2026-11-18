import pytest

from transcript_pipeline.chunking import (
    Silence,
    plan_chunks,
    prune_tiny_chunks,
    validate_chunk_plan,
)
from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.errors import ChunkingError


def config() -> PipelineConfig:
    return PipelineConfig(
        chunk_seconds=600,
        min_chunk_seconds=480,
        max_chunk_seconds=720,
        chunk_overlap_seconds=0.5,
    )


def test_short_audio_is_one_chunk():
    plans = plan_chunks(300, [], config())
    assert len(plans) == 1
    assert plans[0].start == 0
    assert plans[0].end == 300
    assert plans[0].cut_reason == "whole"


def test_silence_boundaries_are_used():
    silences = [Silence(590, 610), Silence(1190, 1210), Silence(1790, 1810)]
    plans = plan_chunks(3600, silences, config())
    assert len(plans) == 6
    assert plans[0].start == 0
    assert plans[0].end == pytest.approx(600)
    assert plans[0].cut_reason == "silence"
    assert plans[-1].end == 3600
    validate_chunk_plan(plans, 3600)


def test_fixed_cut_adds_overlap():
    plans = plan_chunks(1300, [], config())
    assert len(plans) == 2
    assert plans[1].start == pytest.approx(600)
    assert plans[1].audio_start == pytest.approx(599.5)
    assert plans[1].audio_end == pytest.approx(1300)
    validate_chunk_plan(plans, 1300)


def test_tiny_tail_is_absorbed():
    plans = plan_chunks(1300, [], config())
    assert plans[-1].start == 600
    assert plans[-1].end == 1300


def test_validate_rejects_overlap():
    plans = plan_chunks(3600, [], config())
    broken = [plans[0], plans[1].__class__(
        index=2,
        start=plans[0].end - 1,
        end=plans[0].end + 10,
        audio_start=plans[0].end - 1,
        audio_end=plans[0].end + 10,
        cut_reason="fixed",
    )]
    with pytest.raises(ChunkingError):
        validate_chunk_plan(broken, 3600)


def test_prune_tiny_chunks_merges_last():
    plans = plan_chunks(1300, [], config())
    tiny = plans[:-1] + [
        plans[-1].__class__(
            index=plans[-1].index,
            start=plans[-1].end - 10,
            end=plans[-1].end,
            audio_start=plans[-1].end - 10,
            audio_end=plans[-1].end,
            cut_reason="tail",
        )
    ]
    pruned = prune_tiny_chunks(tiny, 480)
    assert len(pruned) == len(tiny) - 1
