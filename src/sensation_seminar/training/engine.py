"""Shared PyTorch training loop for the fixed segmentation benchmark."""

from __future__ import annotations


def train_epoch(model, loader, optimizer, *, device: str, ignore_index: int = 255) -> float:
    import torch

    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].long().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, masks, ignore_index=ignore_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
    if not batches:
        raise ValueError("Training loader is empty")
    return total_loss / batches


def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ModuleNotFoundError:
        return "cpu"
