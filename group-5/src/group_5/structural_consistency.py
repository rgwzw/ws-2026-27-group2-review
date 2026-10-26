"""Boundary consistency for verified aligned label maps."""
import numpy as np
from sensation_seminar.synthetic_data.scoring import score_record


def _boundaries(labels: np.ndarray) -> np.ndarray:
    edges = np.zeros(labels.shape, bool)
    edges[1:] |= labels[1:] != labels[:-1]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return edges


def boundary_score(synthetic_id: str, source: np.ndarray, synthetic: np.ndarray, *, supervision_status: str) -> dict:
    if supervision_status != "verified":
        raise ValueError("Pixel boundary comparison requires verified supervision")
    a, b = np.asarray(source), np.asarray(synthetic)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Label maps must be aligned 2D arrays")
    ea, eb = _boundaries(a), _boundaries(b)
    union = np.logical_or(ea, eb).sum()
    score = float(np.logical_and(ea, eb).sum() / union) if union else 1.0
    return score_record(synthetic_id, method="boundary_iou", method_version="1", score=score)
