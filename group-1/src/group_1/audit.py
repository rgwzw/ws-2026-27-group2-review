"""Synthetic manifest audit."""
from collections import Counter
from typing import Any, Iterable, Mapping


def audit(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    reasons = Counter(r for row in rows for r in row.get("validity", {}).get("reason_codes", []))
    return {
        "schema_version": "1.0", "records": len(rows),
        "analysis_eligible": sum(bool(r.get("eligibility", {}).get("analysis")) for r in rows),
        "training_eligible": sum(bool(r.get("eligibility", {}).get("segmentation_training")) for r in rows),
        "supervision": dict(sorted(Counter(r.get("supervision", {}).get("status", "missing") for r in rows).items())),
        "reason_codes": dict(sorted(reasons.items())),
        "source_groups": dict(sorted(Counter(r.get("source_group", "missing") for r in rows).items())),
    }
