"""Shared contracts for portable synthetic-image score overlays."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

SCORE_SCHEMA_VERSION = "1.0"


def normalize_scores(
    values: Mapping[str, float], *, higher_is_better: bool = True
) -> dict[str, float]:
    """Min-max normalize finite values to higher-is-better scores in ``[0, 1]``."""
    finite = {key: float(value) for key, value in values.items() if math.isfinite(value)}
    if not finite:
        return {}
    low, high = min(finite.values()), max(finite.values())
    if low == high:
        return {key: 1.0 for key in finite}
    normalized = {key: (value - low) / (high - low) for key, value in finite.items()}
    if not higher_is_better:
        normalized = {key: 1.0 - value for key, value in normalized.items()}
    return normalized


def score_record(
    synthetic_id: str,
    *,
    method: str,
    method_version: str,
    score: float,
    components: Mapping[str, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standard score row; all public scores are higher-is-better."""
    if not math.isfinite(score):
        raise ValueError("Score must be finite")
    if not 0.0 <= score <= 1.0:
        raise ValueError("Normalized score must be between 0 and 1")
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "synthetic_id": synthetic_id,
        "method": method,
        "method_version": method_version,
        "direction": "higher_is_better",
        "score": float(score),
        "components": dict(components or {}),
        "provenance": dict(provenance or {}),
    }


def validate_score_overlay(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate an overlay and reject duplicate synthetic identifiers."""
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        required = {"schema_version", "synthetic_id", "method", "method_version", "direction", "score"}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Score row {line_number} is missing fields: {missing}")
        if row["schema_version"] != SCORE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported score schema on row {line_number}")
        if row["direction"] != "higher_is_better":
            raise ValueError("Score overlays must expose higher-is-better values")
        synthetic_id = str(row["synthetic_id"])
        if synthetic_id in seen:
            raise ValueError(f"Duplicate score for synthetic_id {synthetic_id}")
        value = float(row["score"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid normalized score on row {line_number}")
        seen.add(synthetic_id)
        validated.append(dict(row))
    return validated
