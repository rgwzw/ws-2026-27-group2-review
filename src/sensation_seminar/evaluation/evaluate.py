"""Common real validation/test evaluation."""

from __future__ import annotations

from sensation_seminar.metrics.confusion import add_confusions, confusion_matrix
from sensation_seminar.metrics.safety import safety_metrics
from sensation_seminar.metrics.segmentation import metrics_from_confusion


def evaluate(model, loader, *, device: str, ignore_index: int = 255) -> dict[str, object]:
    import torch

    model.eval()
    combined = [[0] * 9 for _ in range(9)]
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            targets = batch["mask"].cpu().tolist()
            combined = add_confusions(
                combined,
                confusion_matrix(targets, predictions, ignore_index=ignore_index),
            )
    result = metrics_from_confusion(combined)
    result.update(safety_metrics(combined))
    return result
