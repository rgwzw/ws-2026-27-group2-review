"""Dependency-free confusion counts for semantic segmentation."""

from __future__ import annotations

from collections.abc import Iterable


def _flatten(values: Iterable):
    for value in values:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            yield from _flatten(value)
        else:
            yield int(value)


def confusion_matrix(
    targets: Iterable, predictions: Iterable, *, classes: int = 9, ignore_index: int = 255
) -> list[list[int]]:
    """Return rows=ground truth, columns=prediction and exclude ignore pixels."""
    target_values = list(_flatten(targets))
    prediction_values = list(_flatten(predictions))
    if len(target_values) != len(prediction_values):
        raise ValueError("Targets and predictions must contain the same number of pixels")
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, prediction in zip(target_values, prediction_values):
        if target == ignore_index:
            continue
        if not 0 <= target < classes or not 0 <= prediction < classes:
            raise ValueError(f"Class outside [0, {classes - 1}]: target={target}, prediction={prediction}")
        matrix[target][prediction] += 1
    return matrix


def add_confusions(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("Confusion matrices have different shapes")
    return [[a + b for a, b in zip(left_row, right_row)] for left_row, right_row in zip(left, right)]
