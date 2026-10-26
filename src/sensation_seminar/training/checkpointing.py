"""Resumable checkpoint state kept outside Git."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, *, model, optimizer, epoch: int, best_metric: float) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        temporary,
    )
    temporary.replace(path)
    return path


def load_checkpoint(path: str | Path, *, model, optimizer=None, map_location="cpu") -> dict[str, Any]:
    import torch

    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state
