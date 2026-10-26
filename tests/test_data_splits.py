from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.data
@pytest.mark.skipif(os.environ.get("SENSATION_RUN_DATA_TESTS") != "1", reason="set SENSATION_RUN_DATA_TESTS=1")
def test_authoritative_split_counts_pairing_and_disjointness() -> None:
    root = Path("datasets/sensation_ds_v2_1_1152x768")
    expected = {"training": 2190, "validation": 281, "testing": 281}
    stem_sets = {}
    for split, count in expected.items():
        images = {path.stem for path in (root / split / "images").iterdir() if path.is_file()}
        masks = {path.stem for path in (root / split / "masks").iterdir() if path.is_file()}
        assert len(images) == count
        assert images == masks
        stem_sets[split] = images
    assert stem_sets["training"].isdisjoint(stem_sets["validation"])
    assert stem_sets["training"].isdisjoint(stem_sets["testing"])
    assert stem_sets["validation"].isdisjoint(stem_sets["testing"])
