"""Factory for the seminar's single common downstream architecture."""

from __future__ import annotations

from typing import Any, Mapping


def build_deeplabv3plus(config: Mapping[str, Any] | None = None):
    """Build the fixed DeepLabV3Plus benchmark, importing heavy dependencies lazily."""
    settings = {
        "architecture": "DeepLabV3Plus",
        "encoder_name": "timm-mobilenetv3_large_075",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 9,
    }
    if config:
        settings.update(config)
    if settings["architecture"] != "DeepLabV3Plus":
        raise ValueError("The seminar benchmark architecture must be DeepLabV3Plus")
    if settings["encoder_name"] != "timm-mobilenetv3_large_075" or settings["classes"] != 9:
        raise ValueError("The common encoder and nine-class output may not be changed")
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the project training dependencies to instantiate the model") from error
    return smp.DeepLabV3Plus(
        encoder_name=settings["encoder_name"],
        encoder_weights=settings["encoder_weights"],
        in_channels=settings["in_channels"],
        classes=settings["classes"],
    )
