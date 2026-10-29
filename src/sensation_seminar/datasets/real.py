"""Read-only access to the authoritative real SENSATION-DS splits."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image


PairTransform = Callable[
    [Image.Image, Image.Image],
    tuple[object, object],
]


class RealSegmentationDataset:
    """Read-only dataset for the real SENSATION-DS splits.

    The existing dataset split is authoritative and must not be changed.

    Every returned sample uses the same metadata schema as the synthetic
    dataset so that real and synthetic samples can safely be combined in
    one PyTorch DataLoader.

    Returned dictionary keys
    ------------------------
    image
        Image or transformed image tensor.

    mask
        Segmentation mask or transformed mask tensor.

    sample_id
        Identifier of the concrete sample.

        For real images this is the image stem.

    original_id
        Identifier of the corresponding original SENSATION-DS image.

        For real samples this is identical to ``sample_id``.

        For synthetic samples, the corresponding synthetic dataset uses
        this field to point back to the source real image.

    kind
        ``"real"`` for samples from this dataset.
    """

    VALID_SPLITS = {
        "training",
        "validation",
        "testing",
    }

    VALID_IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: PairTransform | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unknown split {split!r}. "
                f"Expected one of {sorted(self.VALID_SPLITS)}."
            )

        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_dir}"
            )

        if not self.mask_dir.is_dir():
            raise FileNotFoundError(
                f"Mask directory does not exist: {self.mask_dir}"
            )

        images = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.VALID_IMAGE_EXTENSIONS
        )

        if not images:
            raise ValueError(
                f"No images found in {self.image_dir}"
            )

        self.samples: list[tuple[Path, Path]] = []

        for image_path in images:
            mask_path = self.mask_dir / f"{image_path.stem}.png"

            if not mask_path.is_file():
                raise FileNotFoundError(
                    "Missing segmentation mask for "
                    f"{image_path.name}: {mask_path}"
                )

            self.samples.append(
                (
                    image_path,
                    mask_path,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        image_path, mask_path = self.samples[index]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        with Image.open(mask_path) as source:
            # Do not convert the mask to RGB or interpolate it here.
            # Its class IDs must remain unchanged.
            mask = source.copy()

        if self.transform is not None:
            image, mask = self.transform(
                image,
                mask,
            )

        sample_id = image_path.stem

        return {
            "image": image,
            "mask": mask,

            # For a real sample, the concrete sample and the original
            # SENSATION-DS source are the same image.
            "sample_id": sample_id,
            "original_id": sample_id,

            "kind": "real",
        }
    