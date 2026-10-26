"""Road/sidewalk confusion and safety-oriented metrics."""

from __future__ import annotations

from typing import Sequence


def safety_metrics(matrix: Sequence[Sequence[int]]) -> dict[str, float | None]:
    if len(matrix) != 9 or any(len(row) != 9 for row in matrix):
        raise ValueError("Expected a 9x9 confusion matrix")
    road_support = sum(matrix[1])
    sidewalk_support = sum(matrix[2])
    road_as_sidewalk = matrix[1][2] / road_support if road_support else None
    sidewalk_as_road = matrix[2][1] / sidewalk_support if sidewalk_support else None
    safety_ious = []
    for class_id in (4, 6, 7, 8):
        true_positive = matrix[class_id][class_id]
        union = sum(matrix[class_id]) + sum(row[class_id] for row in matrix) - true_positive
        if union:
            safety_ious.append(true_positive / union)
    return {
        "road_as_sidewalk_error_rate": road_as_sidewalk,
        "sidewalk_as_road_error_rate": sidewalk_as_road,
        "safety_class_miou": sum(safety_ious) / len(safety_ious) if safety_ious else None,
    }
