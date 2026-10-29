"""Tests for a common real/synthetic sample schema.

These tests protect the combined training DataLoader against regressions such
as:

    KeyError: 'original_id'

PyTorch's default collate function requires dictionary samples in the same
batch to expose the same keys. The seminar training pipeline deliberately
mixes real and synthetic samples, so both datasets must return the common
schema:

    image
    mask
    sample_id
    original_id
    kind
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from sensation_seminar.datasets.combined import CombinedDataset
from sensation_seminar.datasets.real import RealSegmentationDataset
from sensation_seminar.datasets.synthetic import SyntheticSegmentationDataset


EXPECTED_SAMPLE_KEYS = {
    "image",
    "mask",
    "sample_id",
    "original_id",
    "kind",
}


def _tensor_transform(image: Image.Image, mask: Image.Image):
    """Convert a same-size image/mask fixture to tensors."""
    image_array = np.asarray(image).copy()
    mask_array = np.asarray(mask).copy()

    image_tensor = (
        torch.from_numpy(image_array)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )
    mask_tensor = torch.from_numpy(mask_array).long()

    return image_tensor, mask_tensor


def _create_fixture_dataset(tmp_path: Path):
    """Create one real and one aligned synthetic sample."""
    data_root = tmp_path / "datasets" / "sensation_ds_v2_1_1152x768"

    real_image_dir = data_root / "training" / "images"
    real_mask_dir = data_root / "training" / "masks"

    real_image_dir.mkdir(parents=True)
    real_mask_dir.mkdir(parents=True)

    original_id = "sample001"

    real_image_path = real_image_dir / f"{original_id}.jpg"
    real_mask_path = real_mask_dir / f"{original_id}.png"

    # Small deterministic fixture. Both image and mask are 8x6.
    Image.new(
        "RGB",
        (8, 6),
        color=(40, 80, 120),
    ).save(real_image_path)

    Image.new(
        "L",
        (8, 6),
        color=1,
    ).save(real_mask_path)

    synthetic_dir = tmp_path / "datasets" / "synthetic-data" / "group3"
    synthetic_dir.mkdir(parents=True)

    synthetic_path = synthetic_dir / "e3_sample001_00.jpg"

    Image.new(
        "RGB",
        (8, 6),
        color=(60, 90, 130),
    ).save(synthetic_path)

    record = {
        "synthetic_id": "sha256:fixture-synthetic-001",
        "original_id": original_id,
        "source_relpath": str(synthetic_path),
        "real_mask_relpath": str(real_mask_path),
        "real_split": "training",
        "eligibility": {
            "analysis": True,
            "segmentation_training": True,
        },
        "supervision": {
            "status": "verified",
            "mask_strategy": "identity",
            "source_size": [8, 6],
            "synthetic_size": [8, 6],
        },
    }

    real_dataset = RealSegmentationDataset(
        data_root,
        "training",
        transform=_tensor_transform,
    )

    synthetic_dataset = SyntheticSegmentationDataset(
        [record],
        transform=_tensor_transform,
    )

    return real_dataset, synthetic_dataset


def test_real_sample_contains_common_metadata_schema(tmp_path: Path) -> None:
    """Real samples must expose original_id as well as sample_id."""
    real_dataset, _ = _create_fixture_dataset(tmp_path)

    sample = real_dataset[0]

    assert set(sample) == EXPECTED_SAMPLE_KEYS
    assert sample["sample_id"] == "sample001"
    assert sample["original_id"] == "sample001"
    assert sample["kind"] == "real"


def test_synthetic_sample_contains_common_metadata_schema(tmp_path: Path) -> None:
    """Synthetic samples must use exactly the same top-level keys."""
    _, synthetic_dataset = _create_fixture_dataset(tmp_path)

    sample = synthetic_dataset[0]

    assert set(sample) == EXPECTED_SAMPLE_KEYS
    assert sample["sample_id"] == "sha256:fixture-synthetic-001"
    assert sample["original_id"] == "sample001"
    assert sample["kind"] == "synthetic"


def test_real_and_synthetic_samples_collate_in_same_batch(tmp_path: Path) -> None:
    """A mixed batch must collate without KeyError or custom collate_fn."""
    real_dataset, synthetic_dataset = _create_fixture_dataset(tmp_path)

    combined = CombinedDataset(
        real_dataset,
        synthetic_dataset,
    )

    # There is exactly one real and one synthetic sample. batch_size=2
    # therefore guarantees that both sample types are collated together.
    loader = DataLoader(
        combined,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert set(batch) == EXPECTED_SAMPLE_KEYS

    assert batch["image"].shape[0] == 2
    assert batch["mask"].shape[0] == 2

    assert list(batch["kind"]) == [
        "real",
        "synthetic",
    ]

    assert list(batch["original_id"]) == [
        "sample001",
        "sample001",
    ]

    assert list(batch["sample_id"]) == [
        "sample001",
        "sha256:fixture-synthetic-001",
    ]
