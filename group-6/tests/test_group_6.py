import pytest
from group_6.quality_selection import select_quality


def test_selector_handles_missing_scores_and_stable_ties():
    rows = [{"synthetic_id": key, "eligibility": {"segmentation_training": eligible}, "supervision": {"status": "verified"}, "scores": scores} for key, eligible, scores in [("b", True, {"q": {"score": .5}}), ("a", True, {"q": {"score": .5}}), ("x", False, {"q": {"score": 1}})]]
    assert [r["synthetic_id"] for r in select_quality(rows, budget=2, methods=["q"])] == ["a", "b"]
    with pytest.raises(ValueError):
        select_quality(rows, budget=3)
