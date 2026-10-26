"""Exact synthetic-filename mapping; no fuzzy matching is permitted."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
import re
from typing import Collection, Mapping

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_OUTER = re.compile(r"^(e\d+)_(.+)$")
_VARIANT = re.compile(r"^(.+)_([0-9]+)$")


@dataclass(frozen=True)
class MappingResult:
    original_id: str | None
    real_split: str | None
    variant: str | None
    generation_tokens: tuple[str, ...]
    reason_code: str | None = None

    @property
    def matched(self) -> bool:
        return self.original_id is not None and self.reason_code is None


def artifact_reason(path: str | PurePath) -> str | None:
    """Classify known notebook/macOS artifacts without deleting them."""
    candidate = PurePath(path)
    parts = candidate.parts
    name = candidate.name
    if any(part == "__MACOSX" for part in parts) or name.startswith("._"):
        return "macos_artifact"
    if any(part.startswith(".") for part in parts[:-1]):
        return "hidden_directory_artifact"
    if "-checkpoint" in Path(name).stem:
        return "notebook_checkpoint"
    if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        return "unsupported_extension"
    return None


def _suffix_candidates(body: str, stems: Collection[str]) -> set[str]:
    stem_set = stems if isinstance(stems, set) else set(stems)
    matches = {body} if body in stem_set else set()
    for index, character in enumerate(body):
        if character == "_" and body[index + 1 :] in stem_set:
            matches.add(body[index + 1 :])
    return matches


def map_synthetic_filename(
    filename: str,
    split_stems: Mapping[str, Collection[str]],
) -> MappingResult:
    """Map a filename through exact complete-stem or underscore-suffix lookup."""
    stem = Path(filename).stem
    variant_match = _VARIANT.match(stem)
    if not variant_match:
        return MappingResult(None, None, None, (), "unparseable_variant")
    prefix_and_source, variant = variant_match.groups()
    outer_match = _OUTER.match(prefix_and_source)
    if not outer_match:
        return MappingResult(None, None, variant, (), "unparseable_generation")
    outer, body = outer_match.groups()

    matches: list[tuple[str, str]] = []
    for split, stems in split_stems.items():
        matches.extend((split, value) for value in _suffix_candidates(body, stems))
    unique = sorted(set(matches))
    if not unique:
        return MappingResult(None, None, variant, (outer,), "source_not_found")
    if len(unique) != 1:
        return MappingResult(None, None, variant, (outer,), "ambiguous_source")

    split, original_id = unique[0]
    prefix = body[: -len(original_id)].rstrip("_")
    tokens = (outer, *(part for part in prefix.split("_") if part))
    reason = None if split == "training" else "source_not_training"
    return MappingResult(original_id, split, variant, tokens, reason)
