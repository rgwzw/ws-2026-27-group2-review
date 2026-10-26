"""Training-only synthetic dataset with enforced supervision alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from PIL import Image

from .transforms import apply_supervision_transform

PairTransform = Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]


class SyntheticSegmentationDataset:
    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        repository_root: str | Path = ".",
        transform: PairTransform | None = None,
    ):
        self.repository_root = Path(repository_root)
        self.transform = transform
        self.records = list(records)
        for record in self.records:
            if not record.get("eligibility", {}).get("segmentation_training"):
                raise ValueError(f"Synthetic record is not training eligible: {record.get('synthetic_id')}")
            if record.get("real_split") != "training":
                raise ValueError("Synthetic supervision may reference the real training split only")

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repository_root / path

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(self._path(record["source_relpath"])) as source:
            image = source.convert("RGB")
        with Image.open(self._path(record["real_mask_relpath"])) as source:
            mask = source.copy()
        image, mask = apply_supervision_transform(image, mask, record["supervision"])
        if self.transform:
            image, mask = self.transform(image, mask)
        return {
            "image": image,
            "mask": mask,
            "sample_id": record["synthetic_id"],
            "original_id": record["original_id"],
            "kind": "synthetic",
        }
