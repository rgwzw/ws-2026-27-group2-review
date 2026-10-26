"""Validation for portable experiment result bundles."""

from __future__ import annotations

from typing import Any, Mapping

RESULT_SCHEMA_VERSION = "1.0"
REQUIRED = {
    "schema_version", "run_id", "status", "experiment", "seed", "split",
    "git", "environment", "data", "training", "metrics",
}


def validate_result(result: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED - result.keys())
    if missing:
        raise ValueError(f"Result is missing required fields: {missing}")
    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("Unsupported result schema version")
    if result["status"] not in {"running", "completed", "failed"}:
        raise ValueError("Invalid result status")
    if result["split"] not in {"validation", "testing"}:
        raise ValueError("Results may describe validation or testing only")
    if not isinstance(result["seed"], int):
        raise ValueError("Seed must be an integer")
    matrix = result["metrics"].get("confusion_matrix")
    if not isinstance(matrix, list) or len(matrix) != 9 or any(len(row) != 9 for row in matrix):
        raise ValueError("Metrics must preserve a 9x9 confusion matrix")
    if result["split"] == "testing" and not result["training"].get("checkpoint_selected_on_validation"):
        raise ValueError("Test results require a checkpoint selected on validation")
