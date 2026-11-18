from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .errors import DependencyError, PipelineError
from .progress import ProgressReporter

TOKENIZER_FILES = (
    "added_tokens.json",
    "merges.txt",
    "normalizer.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

NONCOMMERCIAL_REPOS = {
    "sumanpaudel1997/nepali-asr-whisper-turbo",
    "sumanpaudel1997/nepali-asr-whisper-medium",
    "ailabpvtltd/whisper-large-v3-turbo-nepali-english-codemix-asr",
    "ailabpvtltd/sravaani-nepali-english-codemix-asr",
    "ailabpvtltd/indicconformer-nepali-english-codemix-asr",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    license: str
    backend: str
    architecture: str
    notes: str = ""
    tokenizer_repo: str | None = None
    needs_conversion: bool = False
    default_compute_type: str = "float16"

    @property
    def is_ct2(self) -> bool:
        return self.backend == "ct2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "repo_id": self.repo_id,
            "license": self.license,
            "backend": self.backend,
            "architecture": self.architecture,
            "notes": self.notes,
            "tokenizer_repo": self.tokenizer_repo,
            "needs_conversion": self.needs_conversion,
            "default_compute_type": self.default_compute_type,
        }


DEFAULT_MODEL_KEY = "nepali-medium-ct2"

_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="nepali-medium-ct2",
        repo_id="Dragneel/whisper-medium-nepali-openslr-ct2",
        license="apache-2.0",
        backend="ct2",
        architecture="medium",
        notes="CT2 int8, WER 11.17 on OpenSLR, 775 MB, fastest safe default.",
    ),
    ModelSpec(
        key="nepali-turbo-ct2",
        repo_id="ayushkhadkaa/whisper-turbo-nepali-ct2-fp16",
        license="apache-2.0",
        backend="ct2",
        architecture="turbo",
        notes="CT2 fp16 turbo, 1.6 GB; provenance and WER need benchmark verification.",
    ),
    ModelSpec(
        key="nepali-large-v3",
        repo_id="Dragneel/whisper-large-v3-nepali-openslr",
        license="apache-2.0",
        backend="transformers",
        architecture="large-v3",
        notes="WER 22.31 OpenSLR; complete tokenizer; convert to CT2 before use.",
        needs_conversion=True,
    ),
    ModelSpec(
        key="nepali-large-v3-kiran",
        repo_id="kiranpantha/whisper-large-v3-nepali",
        license="apache-2.0",
        backend="transformers",
        architecture="large-v3",
        notes="WER 18.73 / CER 4.49; tokenizer must be borrowed from OpenAI base.",
        tokenizer_repo="openai/whisper-large-v3",
        needs_conversion=True,
    ),
    ModelSpec(
        key="nepali-turbo-kiran",
        repo_id="kiranpantha/whisper-large-v3-turbo-nepali",
        license="mit",
        backend="transformers",
        architecture="large-v3 (turbo label unverified)",
        notes="Config reports 32 decoder layers and lacks tokenizer files; verify before use.",
        tokenizer_repo="openai/whisper-large-v3",
        needs_conversion=True,
    ),
    ModelSpec(
        key="openai-turbo",
        repo_id="large-v3-turbo",
        license="mit",
        backend="faster-whisper-size",
        architecture="turbo",
        notes="Baseline only; weak Nepali performance.",
    ),
    ModelSpec(
        key="openai-large-v3",
        repo_id="large-v3",
        license="mit",
        backend="faster-whisper-size",
        architecture="large-v3",
        notes="Baseline only; weak Nepali performance.",
    ),
)

REGISTRY: dict[str, ModelSpec] = {model.key: model for model in _MODELS}
_BY_REPO: dict[str, ModelSpec] = {model.repo_id: model for model in _MODELS}


def list_models() -> list[ModelSpec]:
    return list(_MODELS)


def resolve_model(name_or_repo: str) -> ModelSpec:
    value = str(name_or_repo or "").strip()
    if not value:
        raise PipelineError("model name must not be empty")
    if value in REGISTRY:
        return REGISTRY[value]
    if value in _BY_REPO:
        return _BY_REPO[value]
    lower = value.lower()
    if value in NONCOMMERCIAL_REPOS or "noncommercial" in lower or "non-commercial" in lower:
        raise PipelineError(
            f"{value} is a non-commercial model and is excluded by policy. "
            "Choose a permissive model from the registry."
        )
    if "ct2" in lower or "faster-whisper" in lower or "faster_whisper" in lower:
        backend = "ct2"
        needs_conversion = False
    else:
        backend = "transformers"
        needs_conversion = True
    return ModelSpec(
        key=value,
        repo_id=value,
        license="unknown",
        backend=backend,
        architecture="unknown",
        notes="Custom model; license and architecture were not verified.",
        needs_conversion=needs_conversion,
    )


def _slug(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace("@", "__")


def _relative_ct2_path(spec: ModelSpec, quantization: str) -> str:
    if spec.backend == "ct2":
        return f"ct2/{_slug(spec.repo_id)}"
    return f"ct2/{_slug(spec.repo_id)}-{quantization}"


def _runtime_path(root: Path, relative: str) -> Path:
    return root / relative


def _persistent_path(config: PipelineConfig, relative: str) -> Path | None:
    if config.drive_model_cache_dir is None:
        return None
    return config.drive_model_cache_dir / relative


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _restore_persistent(config: PipelineConfig, target: Path, relative: str) -> None:
    persistent = _persistent_path(config, relative)
    if persistent is None or target.exists() or not persistent.exists():
        return
    _copy_tree(persistent, target)


def _save_persistent(config: PipelineConfig, source: Path, relative: str) -> None:
    persistent = _persistent_path(config, relative)
    if persistent is None or persistent.exists() or not source.exists():
        return
    _copy_tree(source, persistent)


def _snapshot_download(repo_id: str, target: Path, reporter: ProgressReporter) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DependencyError(
            "huggingface_hub is required to download models. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    reporter.stage("model", f"downloading {repo_id}")
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    reporter.stage("model", f"downloaded {repo_id}")


def _copy_tokenizer(tokenizer_repo: str, target: Path, reporter: ProgressReporter) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DependencyError(
            "huggingface_hub is required to prepare model tokenizers."
        ) from exc
    reporter.stage("model", f"copying tokenizer files from {tokenizer_repo}")
    target.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_FILES:
        destination = target / filename
        if destination.exists():
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=tokenizer_repo,
                filename=filename,
            )
        except Exception:
            continue
        shutil.copy2(downloaded, destination)


def _copy_preprocessor_config(source: Path, target: Path) -> None:
    source_file = source / "preprocessor_config.json"
    target_file = target / "preprocessor_config.json"
    if source_file.exists() and not target_file.exists():
        shutil.copy2(source_file, target_file)


def _find_converter() -> str:
    executable = shutil.which("ct2-transformers-converter")
    if executable:
        return executable
    sibling = Path(sys.executable).with_name("ct2-transformers-converter")
    if sibling.is_file():
        return str(sibling)
    raise DependencyError(
        "ct2-transformers-converter was not found. Install ctranslate2 and "
        "transformers, then retry."
    )


def _convert_to_ct2(
    source: Path,
    target: Path,
    quantization: str,
    reporter: ProgressReporter,
) -> None:
    converter = _find_converter()
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    reporter.stage(
        "model",
        f"converting {source.name} to CT2 {quantization}",
    )
    command = [
        converter,
        "--model",
        str(source),
        "--output_dir",
        str(target),
        "--quantization",
        quantization,
        "--force",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise DependencyError(
            "CT2 conversion failed: "
            + (result.stderr.strip() or result.stdout.strip() or "unknown error")
        )
    if not (target / "model.bin").exists():
        raise DependencyError("CT2 conversion produced no model.bin")
    _copy_preprocessor_config(source, target)
    reporter.stage("model", f"CT2 model ready at {target}")


def _ensure_source(
    spec: ModelSpec,
    reporter: ProgressReporter,
    config: PipelineConfig,
) -> Path:
    relative = f"source/{_slug(spec.repo_id)}"
    target = _runtime_path(config.model_cache_dir, relative)
    if (target / "config.json").exists():
        return target
    _snapshot_download(spec.repo_id, target, reporter)
    if spec.tokenizer_repo:
        _copy_tokenizer(spec.tokenizer_repo, target, reporter)
    return target


def ensure_model(
    spec: ModelSpec,
    config: PipelineConfig,
    reporter: ProgressReporter,
) -> Path | str:
    if spec.backend == "faster-whisper-size":
        return spec.repo_id

    quantization = spec.default_compute_type or "float16"
    relative = _relative_ct2_path(spec, quantization)
    target = _runtime_path(config.model_cache_dir, relative)
    _restore_persistent(config, target, relative)
    if (target / "model.bin").exists():
        if spec.backend == "transformers":
            source = _ensure_source(spec, reporter, config)
            _copy_preprocessor_config(source, target)
        reporter.stage("model", f"using cached CT2 model {spec.repo_id}")
        return target

    if spec.backend == "ct2":
        _snapshot_download(spec.repo_id, target, reporter)
        if not (target / "model.bin").exists():
            raise DependencyError(
                f"{spec.repo_id} does not contain a CT2 model.bin. "
                "Choose a CT2 repository or a Transformers model for conversion."
            )
        _save_persistent(config, target, relative)
        return target

    source = _ensure_source(spec, reporter, config)
    _convert_to_ct2(source, target, quantization, reporter)
    _save_persistent(config, target, relative)
    return target


def model_choices_text() -> str:
    lines = []
    for spec in _MODELS:
        lines.append(
            f"{spec.key:24} {spec.license:12} {spec.backend:20} {spec.repo_id}"
        )
    return "\n".join(lines)
