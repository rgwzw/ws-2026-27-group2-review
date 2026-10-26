import numpy as np
from group_7.diversity_selection import select_diverse


def test_farthest_first_is_deterministic():
    embeddings = {"a": np.array([0.]), "b": np.array([1.]), "c": np.array([10.])}
    first = select_diverse(embeddings, budget=2)
    assert [row["synthetic_id"] for row in first] == ["a", "c"]
    assert first == select_diverse(embeddings, budget=2)
    assert all(row["schema_version"] == "1.0" for row in first)
