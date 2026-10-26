"""Metrics derived exclusively from persisted confusion counts."""

from __future__ import annotations

from typing import Sequence

CLASS_NAMES = [
    "background", "road", "sidewalk", "bikelane", "person",
    "vehicle", "bicycle", "traffic_light", "obstacle",
]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metrics_from_confusion(matrix: Sequence[Sequence[int]]) -> dict[str, object]:
    if len(matrix) != len(CLASS_NAMES) or any(len(row) != len(CLASS_NAMES) for row in matrix):
        raise ValueError("Expected a 9x9 confusion matrix")
    per_class: dict[str, dict[str, float | int | None]] = {}
    ious: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[index][index])
        false_negative = sum(matrix[index]) - true_positive
        false_positive = sum(row[index] for row in matrix) - true_positive
        union = true_positive + false_positive + false_negative
        iou = _ratio(true_positive, union)
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        if iou is not None:
            ious.append(iou)
        per_class[name] = {
            "support": sum(matrix[index]),
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "confusion_matrix": [list(map(int, row)) for row in matrix],
        "miou": sum(ious) / len(ious) if ious else None,
        "per_class": per_class,
        "road_iou": per_class["road"]["iou"],
        "sidewalk_iou": per_class["sidewalk"]["iou"],
    }
