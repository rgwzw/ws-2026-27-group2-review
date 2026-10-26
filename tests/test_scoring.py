import json

import pytest

from sensation_seminar.synthetic_data.manifest import load_manifest
from sensation_seminar.synthetic_data.scoring import (
    normalize_scores,
    score_record,
    validate_score_overlay,
)


def test_normalization_and_overlay_join(tmp_path):
    assert normalize_scores({"a": 4, "b": 2}, higher_is_better=False) == {"a": 0.0, "b": 1.0}
    synthetic_id = "sha256:" + "a" * 64
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "schema_version": "1.1",
        "synthetic_id": synthetic_id,
        "eligibility": {"analysis": True, "segmentation_training": False},
    }) + "\n", encoding="utf-8")
    row = score_record(synthetic_id, method="fixture", method_version="1", score=0.75)
    overlay = tmp_path / "scores.jsonl"
    overlay.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_manifest(manifest, overlays=[overlay], eligibility_filter="analysis")
    assert loaded[0]["scores"]["fixture"]["score"] == 0.75
    assert validate_score_overlay([row]) == [row]


def test_invalid_and_duplicate_scores_are_rejected():
    with pytest.raises(ValueError, match="between 0 and 1"):
        score_record("id", method="x", method_version="1", score=2.0)
    row = score_record("id", method="x", method_version="1", score=0.5)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_score_overlay([row, row])
