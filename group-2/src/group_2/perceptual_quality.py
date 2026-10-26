"""Dependency-light perceptual-quality fallback."""
from typing import Mapping
import numpy as np
from sensation_seminar.synthetic_data.scoring import normalize_scores, score_record


def score_images(images: Mapping[str, np.ndarray]) -> list[dict]:
    raw = {}
    for key, image in images.items():
        array = np.asarray(image, dtype=float)
        if array.ndim < 2 or not np.isfinite(array).all():
            continue
        gray = array.mean(axis=2) if array.ndim == 3 else array
        raw[key] = float(gray.std() + np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
    scores = normalize_scores(raw)
    return [score_record(key, method="perceptual_fallback", method_version="1", score=scores[key], provenance={"metric": "contrast_and_gradient"}) for key in sorted(scores)]
