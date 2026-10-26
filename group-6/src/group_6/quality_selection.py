"""Quality selector accepting any available score subset."""
from typing import Any, Iterable, Mapping


def select_quality(records: Iterable[Mapping[str, Any]], *, budget: int, methods: Iterable[str] = (), seed: int = 42) -> list[dict[str, Any]]:
    methods = tuple(methods)
    ranked = []
    for row in records:
        if not row.get("eligibility", {}).get("segmentation_training"):
            continue
        scores = row.get("scores", {})
        values = [float(scores[m]["score"] if isinstance(scores[m], dict) else scores[m]) for m in methods if m in scores]
        fallback = 1.0 if row.get("supervision", {}).get("status") == "verified" else 0.0
        ranked.append((sum(values) / len(values) if values else fallback, str(row["synthetic_id"])))
    if len(ranked) < budget:
        raise ValueError("Insufficient training-eligible records")
    ranked.sort(key=lambda value: (-value[0], value[1]))
    return [{"schema_version": "1.0", "synthetic_id": key, "selected": True, "rank": rank, "method": "quality", "budget": budget, "seed": seed, "score": score} for rank, (score, key) in enumerate(ranked[:budget], 1)]
