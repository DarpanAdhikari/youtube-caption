from pathlib import Path

from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.merge import TranscriptSegment
from transcript_pipeline.refine import (
    CorrectionDictionary,
    Glossary,
    build_initial_prompt,
    find_protected_spans,
    is_protected_token,
    normalize_text,
    refine_segments,
)


def test_normalize_text_spacing_and_danda():
    assert normalize_text("  नेपाल  ।संसद्  ") == "नेपाल। संसद्"


def test_protected_spans_cover_dates_numbers_and_latin():
    text = "२०८०-०१-०१ मा 100 जना YouTube प्रयोगकर्ता"
    kinds = {span.kind for span in find_protected_spans(text)}
    assert "date" in kinds
    assert "number" in kinds
    assert "latin" in kinds
    assert is_protected_token("YouTube")
    assert is_protected_token("१००")


def test_dictionary_applies_only_unprotected_devanagari():
    dictionary = CorrectionDictionary({"परिक्षा": "परीक्षा"})
    corrected, changes = dictionary.apply("यो परिक्षा राम्रो छ")
    assert corrected == "यो परीक्षा राम्रो छ"
    assert changes
    protected, protected_changes = dictionary.apply("100 परिक्षा")
    assert protected == "100 परीक्षा"
    assert protected_changes


def test_refine_segments_rule_mode():
    config = PipelineConfig(refine_mode="rule")
    segments = [TranscriptSegment(0, 2, "यो परिक्षा हो")]
    result = refine_segments(
        segments,
        config,
        dictionary=CorrectionDictionary({"परिक्षा": "परीक्षा"}),
    )
    assert result.segments[0].text == "यो परीक्षा हो"
    assert result.report["changed_segments"] == 1
    assert result.changes[0]["original"] != result.changes[0]["corrected"]


def test_refine_off_is_identity():
    config = PipelineConfig(refine_mode="off")
    segments = [TranscriptSegment(0, 2, "यो परिक्षा हो")]
    result = refine_segments(segments, config)
    assert result.segments[0].text == "यो परिक्षा हो"
    assert result.changes == []


def test_glossary_prompt_and_dictionary(tmp_path: Path):
    path = tmp_path / "glossary.json"
    path.write_text(
        '{"terms": ["प्रतिनिधि सभा", "नेपाल सरकार"], '
        '"variants": {"सभा": "प्रतिनिधि सभा"}}',
        encoding="utf-8",
    )
    glossary = Glossary.from_file(path)
    assert "प्रतिनिधि सभा" in glossary.prompt
    prompt = build_initial_prompt(PipelineConfig(initial_prompt="नेपाली संसद्"), glossary)
    assert "नेपाली संसद्" in prompt
    assert "प्रतिनिधि सभा" in prompt
