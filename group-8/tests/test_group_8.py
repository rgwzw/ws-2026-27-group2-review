import numpy as np
from group_8.joint_selection import select_joint


def test_joint_selection_meets_budget_and_uses_both_inputs():
    quality = {"a": 1., "b": .8, "c": .1}
    embeddings = {"a": np.array([0.]), "b": np.array([.1]), "c": np.array([10.])}
    selected = select_joint(quality, embeddings, budget=2, quality_weight=.2)
    assert [row["synthetic_id"] for row in selected] == ["a", "c"]
    assert len(select_joint(quality, embeddings, budget=2)) == 2
