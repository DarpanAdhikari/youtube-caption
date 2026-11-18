from pathlib import Path
from types import SimpleNamespace

import pytest

from transcript_pipeline.config import PipelineConfig
from transcript_pipeline.errors import PipelineError
from transcript_pipeline.models import (
    DEFAULT_MODEL_KEY,
    _convert_to_ct2,
    _find_converter,
    ensure_model,
    list_models,
    model_choices_text,
    resolve_model,
)
from transcript_pipeline.progress import NullProgress
import transcript_pipeline.models as models_module


def test_default_is_registered():
    spec = resolve_model(DEFAULT_MODEL_KEY)
    assert spec.key == DEFAULT_MODEL_KEY
    assert spec.license == "apache-2.0"
    assert spec.backend == "ct2"


def test_resolution_by_repo_and_custom_backend():
    assert resolve_model("openai-turbo").repo_id == "large-v3-turbo"
    custom_ct2 = resolve_model("someone/custom-ct2")
    assert custom_ct2.backend == "ct2"
    assert not custom_ct2.needs_conversion
    custom_tf = resolve_model("someone/custom-transformers")
    assert custom_tf.backend == "transformers"
    assert custom_tf.needs_conversion


def test_noncommercial_models_are_rejected():
    with pytest.raises(PipelineError, match="non-commercial"):
        resolve_model("sumanpaudel1997/nepali-asr-whisper-turbo")
    with pytest.raises(PipelineError):
        resolve_model("org/noncommercial-nepali")


def test_faster_whisper_size_needs_no_download():
    spec = resolve_model("openai-turbo")
    ref = ensure_model(spec, PipelineConfig(), NullProgress())
    assert ref == "large-v3-turbo"


def test_registry_and_choices_text():
    keys = {spec.key for spec in list_models()}
    assert DEFAULT_MODEL_KEY in keys
    assert "nepali-large-v3" in keys
    assert "openai-turbo" in model_choices_text()


def test_find_converter_uses_venv_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    converter = tmp_path / "ct2-transformers-converter"
    converter.write_text("", encoding="utf-8")
    monkeypatch.setattr(models_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(models_module.sys, "executable", str(python))

    assert _find_converter() == str(converter)


def test_conversion_copies_preprocessor_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    preprocessor = source / "preprocessor_config.json"
    preprocessor.write_text('{"feature_size": 128}', encoding="utf-8")

    def fake_run(command, capture_output, text):
        output_dir = Path(command[command.index("--output_dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "model.bin").write_bytes(b"model")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(models_module, "_find_converter", lambda: "converter")
    monkeypatch.setattr(models_module.subprocess, "run", fake_run)

    _convert_to_ct2(source, target, "int8", NullProgress())

    copied = target / "preprocessor_config.json"
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == preprocessor.read_text(encoding="utf-8")
