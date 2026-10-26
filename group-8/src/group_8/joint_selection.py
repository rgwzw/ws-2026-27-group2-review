"""Independent joint quality and diversity selection."""
from typing import Mapping
import numpy as np


def select_joint(quality: Mapping[str, float], embeddings: Mapping[str, np.ndarray], *, budget: int, quality_weight: float = 0.5, seed: int = 42) -> list[dict]:
    keys = sorted(quality.keys() & embeddings.keys())
    if not 0 <= quality_weight <= 1 or len(keys) < budget:
        raise ValueError("Invalid weight or insufficient common records")
    vectors = {k: np.asarray(embeddings[k], float).ravel() for k in keys}
    selected = [min(keys, key=lambda k: (-float(quality[k]), k))]
    while len(selected) < budget:
        remaining = [k for k in keys if k not in selected]
        distances = {k: min(float(np.linalg.norm(vectors[k] - vectors[s])) for s in selected) for k in remaining}
        scale = max(distances.values()) or 1.0
        selected.append(min(remaining, key=lambda k: (-(quality_weight * float(quality[k]) + (1 - quality_weight) * distances[k] / scale), k)))
    return [{"schema_version": "1.0", "synthetic_id": key, "selected": True, "rank": rank, "method": "joint", "budget": budget, "seed": seed, "quality_weight": quality_weight} for rank, key in enumerate(selected, 1)]
