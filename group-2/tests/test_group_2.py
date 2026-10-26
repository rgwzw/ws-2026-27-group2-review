import numpy as np
from group_2.perceptual_quality import score_images


def test_perceptual_fallback_is_deterministic_and_skips_invalid():
    images = {"flat": np.zeros((3, 3)), "detail": np.eye(3), "invalid": np.array([np.nan])}
    first = score_images(images)
    assert first == score_images(images)
    assert [row["synthetic_id"] for row in first] == ["detail", "flat"]
    assert first[0]["score"] > first[1]["score"]
