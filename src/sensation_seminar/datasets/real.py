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
    """Read-only dataset for the authoritative real SENSATION-DS splits.

    The existing training, validation, and testing split is authoritative
    and must not be modified by this class.

    Every returned sample intentionally follows the same metadata schema as
    :class:`SyntheticSegmentationDataset`. This is important because real and
    synthetic samples are mixed inside one PyTorch DataLoader. PyTorch's
    default collate function expects all dictionaries in a batch to contain
    the same keys.

    Returned keys
    -------------
    image
        The RGB image or transformed image tensor.

    mask
        The segmentation mask or transformed mask tensor.

    sample_id
        Identifier of the concrete sample. For a real image this is the image
        filename stem.

    original_id
        Identifier of the corresponding original SENSATION-DS training image.
        For real images this is identical to ``sample_id``. Synthetic samples
        use this field to point back to their real source image.

    kind
        Always ``"real"`` for this dataset.
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
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unknown authoritative split: {split!r}. "
                f"Expected one of {sorted(self.VALID_SPLITS)}."
            )

        self.root = Path(root)
        self.split = split
        self.transform = transform

        image_dir = self.root / split / "images"
        mask_dir = self.root / split / "masks"

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory does not exist: {image_dir}"
            )

        if not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Mask directory does not exist: {mask_dir}"
            )

        self.samples: list[tuple[Path, Path]] = []

        for image_path in sorted(image_dir.iterdir()):
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in self.VALID_IMAGE_EXTENSIONS
            ):
                continue

            mask_path = mask_dir / f"{image_path.stem}.png"

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for image {image_path.name}: {mask_path}"
                )

            self.samples.append((image_path, mask_path))

        if not self.samples:
            raise ValueError(
                f"No image/mask pairs found for split {split!r} below {self.root}"
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
            # Preserve class IDs exactly. Do not convert segmentation masks
            # to RGB or apply image-style interpolation here.
            mask = source.copy()

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        sample_id = image_path.stem

        return {
            "image": image,
            "mask": mask,
            "sample_id": sample_id,

            # Real samples are their own source/original image. Keeping this
            # key also on real samples makes the dictionary schema identical
            # to SyntheticSegmentationDataset and prevents DataLoader collate
            # failures when real and synthetic samples share a batch.
            "original_id": sample_id,

            "kind": "real",
        }
