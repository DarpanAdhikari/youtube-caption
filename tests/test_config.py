from pathlib import Path

import pytest

from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.errors import PipelineError


def test_roundtrip_paths():
    config = PipelineConfig(
        output_root=Path("/tmp/out"),
        input_audio=Path("/tmp/in.m4a"),
        glossary_path=Path("/tmp/glossary.json"),
    )
    restored = PipelineConfig.from_dict(config.to_dict())
    assert restored.output_root == Path("/tmp/out")
    assert restored.input_audio == Path("/tmp/in.m4a")
    assert restored.glossary_path == Path("/tmp/glossary.json")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"refine_mode": "magic"},
        {"run_mode": "magic"},
        {"chunk_seconds": 0},
        {"min_chunk_seconds": 700, "chunk_seconds": 600},
        {"max_chunk_seconds": 500, "chunk_seconds": 600},
        {"chunk_overlap_seconds": -1},
        {"max_lines": 0},
        {"refine_min_similarity": 1.1},
    ],
)
def test_validation_rejects_bad_values(kwargs):
    with pytest.raises(PipelineError):
        PipelineConfig(**kwargs).validate()
