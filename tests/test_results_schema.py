from __future__ import annotations

import json

import pytest

from sensation_seminar.results.aggregate import aggregate_results
from sensation_seminar.results.io import write_result_bundle
from sensation_seminar.results.schema import validate_result


def _result(seed: int = 17) -> dict:
    matrix = [[0] * 9 for _ in range(9)]
    matrix[1][1] = 2
    return {
        "schema_version": "1.0", "run_id": f"run-{seed}", "status": "completed",
        "experiment": "real_only", "seed": seed, "split": "testing",
        "git": {"revision": "abc", "dirty": False}, "environment": {"python": "test"},
        "data": {"dataset": "fixture"},
        "training": {"checkpoint_selected_on_validation": True},
        "metrics": {
            "confusion_matrix": matrix, "miou": 0.5, "road_iou": 1.0,
            "sidewalk_iou": 0.0, "road_as_sidewalk_error_rate": 0.0,
            "sidewalk_as_road_error_rate": None,
        },
    }


def test_result_bundle_and_aggregation(tmp_path) -> None:
    first, second = _result(17), _result(42)
    validate_result(first)
    bundle = write_result_bundle(tmp_path / "run", first, {"fixture": True})
    assert json.loads((bundle / "metrics.json").read_text())["seed"] == 17
    assert (bundle / "summary.csv").is_file()
    aggregate = aggregate_results([first, second])
    assert aggregate[0]["seeds"] == [17, 42]
    assert aggregate[0]["miou_mean"] == 0.5


def test_schema_rejects_test_leakage_and_missing_confusion() -> None:
    result = _result()
    result["training"]["checkpoint_selected_on_validation"] = False
    with pytest.raises(ValueError):
        validate_result(result)
    result = _result()
    del result["metrics"]["confusion_matrix"]
    with pytest.raises(ValueError):
        validate_result(result)
