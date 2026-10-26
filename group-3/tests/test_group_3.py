import numpy as np
from group_3.feature_similarity import cosine_similarity_scores


def test_known_cosines_and_invalid_vectors():
    rows = cosine_similarity_scores({"same": np.array([1, 0]), "opposite": np.array([-1, 0]), "zero": np.zeros(2)}, {k: np.array([1, 0]) for k in ("same", "opposite", "zero")})
    assert [(row["synthetic_id"], row["score"]) for row in rows] == [("opposite", 0.0), ("same", 1.0)]
