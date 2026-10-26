"""Read-only access to the authoritative real SENSATION-DS splits."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

PairTransform = Callable[[Image.Image, Image.Image], tuple[Image.Image, Image.Image]]


class RealSegmentationDataset:
    def __init__(self, root: str | Path, split: str, transform: PairTransform | None = None):
        if split not in {"training", "validation", "testing"}:
            raise ValueError(f"Unknown authoritative split: {split}")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        image_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"
        self.samples = []
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            mask_path = mask_dir / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path.name}")
            self.samples.append((image_path, mask_path))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.copy()
        if self.transform:
            image, mask = self.transform(image, mask)
        return {"image": image, "mask": mask, "sample_id": image_path.stem, "kind": "real"}
