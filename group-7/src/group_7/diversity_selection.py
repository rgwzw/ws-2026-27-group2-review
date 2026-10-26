"""Deterministic farthest-first feature-space coverage."""
from typing import Mapping
import numpy as np


def select_diverse(embeddings: Mapping[str, np.ndarray], *, budget: int) -> list[str]:
    keys = sorted(embeddings)
    if budget < 1 or len(keys) < budget:
        raise ValueError("Budget must fit available embeddings")
    vectors = {k: np.asarray(embeddings[k], float).ravel() for k in keys}
    if len({v.shape for v in vectors.values()}) != 1:
        raise ValueError("Embedding dimensions differ")
    selected = [keys[0]]
    while len(selected) < budget:
        remaining = [k for k in keys if k not in selected]
        distances = {k: min(float(np.linalg.norm(vectors[k] - vectors[s])) for s in selected) for k in remaining}
        selected.append(min(remaining, key=lambda k: (-distances[k], k)))
    return selected
