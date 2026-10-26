"""Portable prediction export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def save_label_predictions(predictions: Iterable[tuple[str, np.ndarray]], output_dir: str | Path) -> list[Path]:
    """Save two-dimensional class-id arrays as lossless PNG files."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for sample_id, prediction in predictions:
        array = np.asarray(prediction)
        if array.ndim != 2 or array.min(initial=0) < 0 or array.max(initial=0) > 255:
            raise ValueError("Predictions must be 2D uint8-compatible class IDs")
        path = root / f"{sample_id}.png"
        temporary = path.with_suffix(".png.tmp")
        Image.fromarray(array.astype(np.uint8), mode="L").save(temporary, format="PNG")
        temporary.replace(path)
        written.append(path)
    return written
