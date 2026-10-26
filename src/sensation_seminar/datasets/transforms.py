"""Geometry-safe image/mask transforms implemented with Pillow."""

from __future__ import annotations

from collections.abc import Mapping

from PIL import Image


def apply_supervision_transform(
    image: Image.Image, mask: Image.Image, supervision: Mapping[str, object]
) -> tuple[Image.Image, Image.Image]:
    """Align a real source mask to a stored synthetic image."""
    if supervision.get("status") != "verified":
        raise ValueError("Synthetic supervision must be verified before training")
    strategy = supervision.get("mask_strategy")
    synthetic_size = tuple(supervision.get("synthetic_size") or ())
    if image.size != synthetic_size:
        raise ValueError(f"Synthetic image size {image.size} differs from manifest {synthetic_size}")
    if strategy == "identity":
        if mask.size != image.size:
            raise ValueError("Identity supervision requires equal image and mask sizes")
        return image, mask
    if strategy == "resize":
        return image, mask.resize(image.size, resample=Image.Resampling.NEAREST)
    raise NotImplementedError(f"Verified mask strategy is not implemented: {strategy}")


def paired_resize(
    image: Image.Image, mask: Image.Image, target_size: tuple[int, int]
) -> tuple[Image.Image, Image.Image]:
    """Resize an already aligned pair; labels always use nearest-neighbor."""
    return (
        image.resize(target_size, resample=Image.Resampling.BILINEAR),
        mask.resize(target_size, resample=Image.Resampling.NEAREST),
    )
