from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from sensation_seminar.synthetic_data.manifest import build_manifest_records, load_manifest, write_jsonl
from sensation_seminar.synthetic_data.mapping import artifact_reason, map_synthetic_filename
from sensation_seminar.synthetic_data.validation import eligibility, supervision_for


def test_exact_nested_mapping_and_nontraining_rejection() -> None:
    splits = {"training": {"0026", "_abc"}, "validation": {"held"}, "testing": set()}
    mapped = map_synthetic_filename("e2_eft3__abc_01.jpg", splits)
    assert mapped.original_id == "_abc"
    assert mapped.generation_tokens == ("e2", "eft3")
    assert mapped.reason_code is None
    assert map_synthetic_filename("e1_held_00.png", splits).reason_code == "source_not_training"
    assert map_synthetic_filename("e1_missing_00.jpg", splits).reason_code == "source_not_found"


def test_ambiguous_suffix_and_artifacts() -> None:
    result = map_synthetic_filename(
        "e1_prefix_scene_00.jpg",
        {"training": {"scene", "prefix_scene"}, "validation": set(), "testing": set()},
    )
    assert result.reason_code == "ambiguous_source"
    assert artifact_reason("group7/images/.ipynb_checkpoints/e1_a_00.jpg") == "hidden_directory_artifact"
    assert artifact_reason("group9/images/__MACOSX/._e1_a_00.jpg") == "macos_artifact"
    assert artifact_reason("group2/images/e1_a_00-checkpoint.jpg") == "notebook_checkpoint"


def test_supervision_controls_training_eligibility() -> None:
    config = {
        "source_size": [1152, 768],
        "groups": {"group3": {"status": "verified", "mask_strategy": "identity", "target_size": [1152, 768]}},
        "default": {"status": "uncertain", "mask_strategy": "unknown", "target_size": None},
    }
    verified = supervision_for("group3", "x", (1152, 768), config)
    wrong_size = supervision_for("group3", "x", (768, 768), config)
    assert eligibility(artifact=None, decode_error=None, mapping_error=None, supervision=verified) == {
        "analysis": True,
        "segmentation_training": True,
    }
    assert wrong_size["status"] == "invalid"
    assert eligibility(artifact=None, decode_error=None, mapping_error=None, supervision=wrong_size)[
        "segmentation_training"
    ] is False


def _rgb(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "navy").save(path)


def test_tiny_manifest_roundtrip(tmp_path: Path) -> None:
    data = tmp_path / "datasets"
    real = data / "sensation_ds_v2_1_1152x768"
    for split in ("training", "validation", "testing"):
        (real / split / "images").mkdir(parents=True)
        (real / split / "masks").mkdir(parents=True)
    _rgb(real / "training/images/0026.jpg", (12, 8))
    Image.new("L", (12, 8), 1).save(real / "training/masks/0026.png")
    _rgb(data / "synthetic-data/group3/images/e1_0026_00.jpg", (12, 8))
    config = {
        "source_size": [12, 8],
        "groups": {"group3": {"status": "verified", "mask_strategy": "identity", "target_size": [12, 8]}},
        "default": {"status": "uncertain", "mask_strategy": "unknown", "target_size": None},
    }
    records = list(build_manifest_records(data, config, repository_root=tmp_path))
    assert len(records) == 1
    assert records[0]["eligibility"]["segmentation_training"] is True
    assert records[0]["source_relpath"].startswith("datasets/")

    output = tmp_path / "manifest.jsonl"
    assert write_jsonl(records, output) == 1
    assert load_manifest(output, eligibility_filter="segmentation_training") == records
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "1.1"
