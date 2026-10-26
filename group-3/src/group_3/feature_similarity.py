"""Feature similarity from externally supplied frozen embeddings."""
from typing import Mapping
import numpy as np
from sensation_seminar.synthetic_data.scoring import score_record


def cosine_similarity_scores(synthetic: Mapping[str, np.ndarray], source: Mapping[str, np.ndarray]) -> list[dict]:
    rows = []
    for key in sorted(synthetic.keys() & source.keys()):
        left, right = np.asarray(synthetic[key], float).ravel(), np.asarray(source[key], float).ravel()
        if left.shape != right.shape or not np.linalg.norm(left) or not np.linalg.norm(right):
            continue
        cosine = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))
        rows.append(score_record(key, method="feature_cosine", method_version="1", score=(cosine + 1) / 2, provenance={"embedding_provider": "caller"}))
    return rows
