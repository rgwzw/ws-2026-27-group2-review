"""Deterministic, eligibility-aware synthetic subset selection."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random
from typing import Any, Iterable, Mapping

BUDGETS = {"0.25x": 548, "0.50x": 1095}


def budget_size(name: str, real_training_size: int = 2190) -> int:
    if name in BUDGETS and real_training_size == 2190:
        return BUDGETS[name]
    if not name.endswith("x"):
        raise ValueError("Budget must use canonical notation such as 0.25x")
    factor = float(name[:-1])
    return int(factor * real_training_size + 0.5)


def random_training_subset(
    records: Iterable[Mapping[str, Any]], *, budget: str, seed: int = 42
) -> list[dict[str, Any]]:
    """Select a source-group-stratified subset from training-eligible records."""
    size = budget_size(budget)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("eligibility", {}).get("segmentation_training"):
            groups[str(record["source_group"])].append(record)
    total = sum(map(len, groups.values()))
    if total < size:
        raise ValueError(f"Need {size} training-eligible records, found {total}")

    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda row: row["synthetic_id"])
        rng.shuffle(values)
    raw = {group: size * len(values) / total for group, values in groups.items()}
    quota = {group: min(len(groups[group]), int(value)) for group, value in raw.items()}
    remaining = size - sum(quota.values())
    order = sorted(groups, key=lambda group: (-(raw[group] - int(raw[group])), group))
    while remaining:
        progressed = False
        for group in order:
            if quota[group] < len(groups[group]):
                quota[group] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError("Unable to satisfy subset budget")

    chosen = [record for group in sorted(groups) for record in groups[group][: quota[group]]]
    chosen.sort(key=lambda row: row["synthetic_id"])
    return [
        {
            "schema_version": "1.0",
            "synthetic_id": record["synthetic_id"],
            "selected": True,
            "rank": rank,
            "method": f"random_budget_{budget}",
            "budget": f"budget_{budget}",
            "seed": seed,
            "source_group": record["source_group"],
        }
        for rank, record in enumerate(chosen, 1)
    ]


def canonical_checksum(records: Iterable[Mapping[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in records).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
