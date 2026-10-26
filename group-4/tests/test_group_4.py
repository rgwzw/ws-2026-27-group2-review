import numpy as np
import pytest
from group_4.semantic_consistency import consistency_score


def test_semantic_consistency_penalizes_road_sidewalk_swap():
    source = np.array([[0, 1], [2, 255]])
    same = consistency_score("same", source, source)
    swapped = consistency_score("swap", source, np.array([[1, 0], [2, 9]]))
    assert same["score"] == 1.0 and swapped["score"] < same["score"]
    with pytest.raises(ValueError):
        consistency_score("bad", source, np.zeros((1, 1)))
