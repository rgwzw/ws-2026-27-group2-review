"""Downstream aggregation over any available valid result subset."""
from typing import Any, Iterable, Mapping
from sensation_seminar.results.aggregate import aggregate_results


def analyze_results(results: Iterable[Mapping[str, Any]], *, baseline: str = "random_budget_0.25x") -> list[dict[str, Any]]:
    rows = aggregate_results(results)
    baselines = {(r["split"]): r for r in rows if r["experiment"] == baseline}
    for row in rows:
        base = baselines.get(row["split"])
        row["miou_delta_vs_baseline"] = None if base is None or row["miou_mean"] is None or base["miou_mean"] is None else row["miou_mean"] - base["miou_mean"]
    return rows
