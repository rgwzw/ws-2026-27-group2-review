"""Teacher-prediction semantic consistency."""
import numpy as np
from sensation_seminar.synthetic_data.scoring import score_record


def consistency_score(synthetic_id: str, source: np.ndarray, synthetic: np.ndarray, *, road_id: int = 0, sidewalk_id: int = 1) -> dict:
    a, b = np.asarray(source), np.asarray(synthetic)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Prediction maps must be aligned 2D arrays")
    valid = a != 255
    agreement = float((a[valid] == b[valid]).mean()) if valid.any() else 0.0
    safety = valid & np.isin(a, [road_id, sidewalk_id])
    safety_agreement = float((a[safety] == b[safety]).mean()) if safety.any() else 1.0
    score = 0.75 * agreement + 0.25 * safety_agreement
    return score_record(synthetic_id, method="semantic_consistency", method_version="1", score=score, components={"agreement": agreement, "road_sidewalk_agreement": safety_agreement})
