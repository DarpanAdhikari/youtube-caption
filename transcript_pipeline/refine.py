from __future__ import annotations

import json
import re
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .merge import TranscriptSegment
from .paths import JobPaths, atomic_write_json
from .progress import ProgressReporter

_URL_RE = re.compile(r"(?:(?:https?|ftp)://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_DATE_RE = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")
_NUMBER_RE = re.compile(
    r"(?<![\w\u0900-\u097F])[-+]?\d+(?:[.,]\d+)*(?:%|₹|Rs|रु)?(?![\w\u0900-\u097F])"
)
_LATIN_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]*$")
_LATIN_SPAN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*")
_DEVANAGARI_ONLY = re.compile(r"^[\u0900-\u097F\s]+$")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([।॥,;:?!])")
_SPACE_AFTER_DANDA = re.compile(r"([।॥])(?=[^\s।॥])")
_TRIM_PUNCT = "।॥?!,;:.()\"'’"


@dataclass(frozen=True)
class ProtectedSpan:
    start: int
    end: int
    kind: str

    def text(self, source: str) -> str:
        return source[self.start : self.end]


def find_protected_spans(text: str) -> list[ProtectedSpan]:
    spans: list[ProtectedSpan] = []
    for pattern, kind in (
        (_URL_RE, "url"),
        (_EMAIL_RE, "email"),
        (_DATE_RE, "date"),
        (_NUMBER_RE, "number"),
        (_LATIN_SPAN_RE, "latin"),
    ):
        for match in pattern.finditer(text):
            spans.append(ProtectedSpan(match.start(), match.end(), kind))
    spans.sort(key=lambda span: (-(span.end - span.start), span.start))
    kept: list[ProtectedSpan] = []
    for span in spans:
        if any(not (span.end <= other.start or span.start >= other.end) for other in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda span: span.start)


def strip_token(token: str) -> str:
    return token.rstrip(_TRIM_PUNCT)


def is_protected_token(token: str, registered: set[str] | None = None) -> bool:
    value = token.strip()
    if not value:
        return True
    if registered and strip_token(value) in registered:
        return True
    if _URL_RE.search(value) or _EMAIL_RE.search(value):
        return True
    if _DATE_RE.search(value) or _NUMBER_RE.search(value):
        return True
    if _LATIN_WORD_RE.match(strip_token(value)):
        return True
    if any(character.isascii() and character.isalpha() for character in strip_token(value)):
        return True
    return False


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text or ""))
    value = _ZERO_WIDTH_RE.sub("", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = _SPACE_BEFORE_PUNCT.sub(r"\1", value)
    value = _SPACE_AFTER_DANDA.sub(r"\1 ", value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    return value.strip()


def _punctuation_suffix(token: str) -> str:
    stripped = strip_token(token)
    return token[len(stripped) :]


@dataclass
class Change:
    original: str
    corrected: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {
            "original": self.original,
            "corrected": self.corrected,
            "kind": self.kind,
        }


class CorrectionDictionary:
    def __init__(self, entries: dict[str, str] | None = None):
        self.entries: dict[str, str] = {}
        for wrong, right in (entries or {}).items():
            self._add(wrong, right)
        self._phrase_keys = sorted(
            (tuple(key.split()) for key in self.entries if len(key.split()) > 1),
            key=len,
            reverse=True,
        )

    def _add(self, wrong: str, right: str) -> None:
        wrong_value = normalize_text(wrong)
        right_value = normalize_text(right)
        if not wrong_value or not right_value:
            return
        if not _DEVANAGARI_ONLY.match(strip_token(wrong_value).replace(" ", "")):
            return
        if not _DEVANAGARI_ONLY.match(strip_token(right_value).replace(" ", "")):
            return
        if wrong_value == right_value:
            return
        self.entries[wrong_value] = right_value

    @classmethod
    def from_file(cls, path: Path | None) -> "CorrectionDictionary":
        if path is None:
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        entries: dict[str, str] = {}
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("wrong") and item.get("right"):
                    entries[str(item["wrong"])] = str(item["right"])
        elif isinstance(data, dict):
            raw_entries = data.get("errors", data.get("corrections", data))
            if isinstance(raw_entries, list):
                for item in raw_entries:
                    if isinstance(item, dict) and item.get("wrong") and item.get("right"):
                        entries[str(item["wrong"])] = str(item["right"])
            elif isinstance(raw_entries, dict):
                for wrong, right in raw_entries.items():
                    if isinstance(wrong, str) and isinstance(right, str):
                        entries[wrong] = right
        return cls(entries)

    def apply(
        self,
        text: str,
        *,
        registered: set[str] | None = None,
    ) -> tuple[str, list[Change]]:
        tokens = text.split()
        if not tokens or not self.entries:
            return text, []

        changes: list[Change] = []
        consumed: set[int] = set()
        replacements: dict[int, tuple[int, str, str]] = {}

        for key_tokens in self._phrase_keys:
            width = len(key_tokens)
            for index in range(0, len(tokens) - width + 1):
                positions = range(index, index + width)
                if any(position in consumed for position in positions):
                    continue
                current = tuple(strip_token(tokens[position]) for position in positions)
                if current != key_tokens:
                    continue
                if any(is_protected_token(tokens[position], registered) for position in positions):
                    continue
                wrong = " ".join(tokens[position] for position in positions)
                right = self.entries[" ".join(key_tokens)]
                suffix = _punctuation_suffix(tokens[index + width - 1])
                replacements[index] = (width, right + suffix, wrong)
                for position in positions:
                    consumed.add(position)

        for index, token in enumerate(tokens):
            if index in consumed:
                continue
            key = strip_token(token)
            replacement = self.entries.get(key)
            if replacement is None or is_protected_token(token, registered):
                continue
            replacements[index] = (1, replacement + _punctuation_suffix(token), token)
            consumed.add(index)

        if not replacements:
            return text, []

        output: list[str] = []
        index = 0
        while index < len(tokens):
            item = replacements.get(index)
            if item is None:
                output.append(tokens[index])
                index += 1
                continue
            width, replacement, wrong = item
            output.append(replacement)
            changes.append(Change(wrong, replacement, "spelling"))
            index += width
        return " ".join(output), changes


@dataclass
class Glossary:
    terms: list[str] = field(default_factory=list)
    variants: dict[str, str] = field(default_factory=dict)
    source: str | None = None

    @property
    def prompt(self) -> str:
        text = " ".join(term.strip() for term in self.terms if term.strip())
        return text[:180].strip()

    @property
    def registered(self) -> set[str]:
        return {strip_token(term) for term in self.terms if term.strip()}

    def dictionary(self) -> CorrectionDictionary:
        return CorrectionDictionary(self.variants)

    @classmethod
    def from_file(cls, path: Path | None) -> "Glossary":
        if path is None:
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        terms: list[str] = []
        variants: dict[str, str] = {}
        if isinstance(data, list):
            terms = [str(item) for item in data if isinstance(item, str)]
        elif isinstance(data, dict):
            raw_terms = data.get("terms", [])
            if isinstance(raw_terms, list):
                terms = [str(item) for item in raw_terms if isinstance(item, str)]
            raw_variants = data.get("variants", {})
            if isinstance(raw_variants, dict):
                variants = {
                    str(wrong): str(right)
                    for wrong, right in raw_variants.items()
                    if isinstance(wrong, str) and isinstance(right, str)
                }
        return cls(terms=terms, variants=variants, source=str(path))


class NepaliCorrectorClient:
    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def correct(self, text: str) -> str | None:
        payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url + "/correct/detailed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None
        corrected = data.get("corrected") if isinstance(data, dict) else None
        return str(corrected) if corrected else None


def _protected_tokens_preserved(original: str, corrected: str) -> bool:
    spans = find_protected_spans(original)
    if not spans:
        return True
    for span in spans:
        value = span.text(original)
        if value and value not in corrected:
            return False
    return True


def _accept_neural(
    original: str,
    corrected: str,
    min_similarity: float,
) -> tuple[bool, float]:
    if not corrected.strip():
        return False, 0.0
    if not _protected_tokens_preserved(original, corrected):
        return False, 0.0
    similarity = SequenceMatcher(None, original, corrected).ratio()
    return similarity >= min_similarity, similarity


@dataclass
class RefinementResult:
    segments: list[TranscriptSegment]
    changes: list[dict[str, Any]]
    report: dict[str, Any]


def refine_segments(
    segments: list[TranscriptSegment],
    config: PipelineConfig,
    *,
    glossary: Glossary | None = None,
    dictionary: CorrectionDictionary | None = None,
    reporter: ProgressReporter | None = None,
) -> RefinementResult:
    if config.refine_mode == "off":
        return RefinementResult(
            segments=list(segments),
            changes=[],
            report={"mode": "off", "changed_segments": 0, "changes": 0},
        )

    dictionary = dictionary or CorrectionDictionary()
    glossary = glossary or Glossary()
    registered = glossary.registered
    client = (
        NepaliCorrectorClient(config.corrector_url)
        if config.refine_mode == "neural" and config.corrector_url
        else None
    )
    refined: list[TranscriptSegment] = []
    changes: list[dict[str, Any]] = []
    changed_segments = 0
    rejected_neural = 0
    similarities: list[float] = []
    total = len(segments)

    for index, segment in enumerate(segments, start=1):
        original = segment.text
        normalized = normalize_text(original)
        rule_text, rule_changes = dictionary.apply(normalized, registered=registered)
        final_text = rule_text
        applied_changes = rule_changes
        accepted_neural = False

        if client is not None:
            neural_text = client.correct(rule_text)
            if neural_text is not None:
                accepted, similarity = _accept_neural(
                    rule_text,
                    normalize_text(neural_text),
                    config.refine_min_similarity,
                )
                similarities.append(similarity)
                if accepted:
                    final_text = normalize_text(neural_text)
                    accepted_neural = True
                else:
                    rejected_neural += 1
            else:
                rejected_neural += 1

        if not final_text.strip():
            final_text = original

        if final_text != original:
            changed_segments += 1
            if not applied_changes and not accepted_neural:
                applied_changes = [Change(original, final_text, "normalization")]
            for change in applied_changes:
                changes.append(
                    {
                        "segment_index": index - 1,
                        "start": round(segment.start, 3),
                        "end": round(segment.end, 3),
                        **change.to_dict(),
                        "similarity": round(
                            SequenceMatcher(None, original, final_text).ratio(),
                            4,
                        ),
                    }
                )
        refined.append(
            TranscriptSegment(
                start=segment.start,
                end=segment.end,
                text=final_text,
                confidence=segment.confidence,
                chunk_index=segment.chunk_index,
            )
        )
        if reporter is not None and (index % 25 == 0 or index == total):
            reporter.update(
                "refine",
                index,
                total,
                message="refining transcript",
                detail=f"{changed_segments} segments changed",
            )

    report = {
        "mode": config.refine_mode,
        "segments": total,
        "changed_segments": changed_segments,
        "changes": len(changes),
        "rejected_neural": rejected_neural,
        "mean_neural_similarity": (
            round(sum(similarities) / len(similarities), 4) if similarities else None
        ),
        "glossary_terms": len(glossary.terms),
        "dictionary_entries": len(dictionary.entries),
    }
    return RefinementResult(segments=refined, changes=changes, report=report)


def build_initial_prompt(
    config: PipelineConfig,
    glossary: Glossary | None,
) -> str | None:
    parts = []
    if config.initial_prompt:
        parts.append(config.initial_prompt.strip())
    if glossary is not None and glossary.prompt:
        parts.append(glossary.prompt)
    text = " ".join(part for part in parts if part).strip()
    return text[:180] if text else None


def save_refinement_outputs(
    paths: JobPaths,
    changes: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    atomic_write_json(paths.refinement_dir / "changes.json", changes)
    atomic_write_json(paths.refinement_dir / "report.json", report)
