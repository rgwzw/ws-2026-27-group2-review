"""Aggregate any available subset of valid result bundles."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

from .schema import validate_result


def load_results(root: str | Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(Path(root).rglob("metrics.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        validate_result(result)
        results.append(result)
    return results


def aggregate_results(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for result in results:
        buckets.setdefault((str(result["experiment"]), str(result["split"])), []).append(result)
    aggregates = []
    for (experiment, split), values in sorted(buckets.items()):
        row: dict[str, Any] = {
            "experiment": experiment,
            "split": split,
            "seeds": sorted(int(value["seed"]) for value in values),
            "runs": len(values),
        }
        for metric in ("miou", "road_iou", "sidewalk_iou", "road_as_sidewalk_error_rate"):
            observed = [float(value["metrics"][metric]) for value in values if value["metrics"].get(metric) is not None]
            row[f"{metric}_mean"] = mean(observed) if observed else None
            row[f"{metric}_std"] = stdev(observed) if len(observed) > 1 else 0.0 if observed else None
        aggregates.append(row)
    return aggregates
