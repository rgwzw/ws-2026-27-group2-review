"""Image validation, eligibility, and supervision-geometry classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError

VERIFIED_STRATEGIES = {"identity", "resize", "resize_with_padding", "crop", "custom"}


def inspect_image(path: str | Path) -> tuple[int | None, int | None, str | None, str | None]:
    """Read image metadata and verify decoding without changing the file."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            return width, height, image.mode, None
    except (OSError, UnidentifiedImageError) as error:
        return None, None, None, f"decode_error:{type(error).__name__}"


def supervision_for(
    source_group: str,
    source_relpath: str,
    synthetic_size: tuple[int, int] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the most specific configured geometry rule and validate its size."""
    normalized_path = "/" + source_relpath.replace("\\", "/").strip("/") + "/"
    group_config = config.get("groups", {}).get(source_group, {})
    selected: Mapping[str, Any] | None = None
    for rule in group_config.get("rules", []):
        if rule.get("path_contains") in normalized_path:
            selected = rule
            break
    if selected is None and "status" in group_config:
        selected = group_config
    if selected is None:
        selected = config.get("default", {})

    source_size = list(config.get("source_size", [1152, 768]))
    target_size = selected.get("target_size")
    status = str(selected.get("status", "uncertain"))
    strategy = str(selected.get("mask_strategy", "unknown"))
    notes = str(selected.get("notes", ""))
    actual = list(synthetic_size) if synthetic_size else None

    if status == "verified" and strategy not in VERIFIED_STRATEGIES:
        status, notes = "invalid", f"Unsupported verified strategy: {strategy}. {notes}".strip()
    if status == "verified" and target_size != actual:
        status = "invalid"
        notes = f"Expected synthetic size {target_size}, observed {actual}. {notes}".strip()
    if status == "verified" and strategy == "identity" and source_size != actual:
        status = "invalid"
        notes = f"Identity supervision requires source size {source_size}. {notes}".strip()

    return {
        "status": status,
        "mask_strategy": strategy,
        "source_size": source_size,
        "synthetic_size": actual,
        "target_size": target_size,
        "notes": notes,
    }


def eligibility(
    *,
    artifact: str | None,
    decode_error: str | None,
    mapping_error: str | None,
    supervision: Mapping[str, Any],
) -> dict[str, bool]:
    """Keep analysis and segmentation-training eligibility deliberately separate."""
    analysis = artifact is None and decode_error is None
    training = analysis and mapping_error is None and supervision.get("status") == "verified"
    return {"analysis": analysis, "segmentation_training": training}
