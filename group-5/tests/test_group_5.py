import numpy as np
import pytest
from group_5.structural_consistency import boundary_score


def test_verified_identical_geometry_ranks_highest():
    labels = np.array([[0, 0], [1, 1]])
    assert boundary_score("same", labels, labels, supervision_status="verified")["score"] == 1.0
    with pytest.raises(ValueError, match="verified"):
        boundary_score("unknown", labels, labels, supervision_status="uncertain")
