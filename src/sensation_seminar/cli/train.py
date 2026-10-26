"""Train the fixed seminar baseline from a named experiment configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sensation_seminar.config import load_merged
from sensation_seminar.datasets.combined import CombinedDataset
from sensation_seminar.datasets.real import RealSegmentationDataset
from sensation_seminar.datasets.synthetic import SyntheticSegmentationDataset
from sensation_seminar.datasets.transforms import paired_resize
from sensation_seminar.evaluation.evaluate import evaluate
from sensation_seminar.models.segmentation import build_deeplabv3plus
from sensation_seminar.results.io import environment_state, git_state, write_result_bundle
from sensation_seminar.synthetic_data.manifest import load_manifest
from sensation_seminar.training.checkpointing import load_checkpoint, save_checkpoint
from sensation_seminar.training.engine import choose_device, train_epoch
from sensation_seminar.training.reproducibility import seed_everything


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _subset_ids(path: Path) -> set[str]:
    return {json.loads(line)["synthetic_id"] for line in path.read_text(encoding="utf-8").splitlines() if line}


def _tensor_transform(target_size: tuple[int, int]):
    def transform(image, mask):
        import numpy
        import torch

        image, mask = paired_resize(image, mask, target_size)
        image_tensor = torch.from_numpy(numpy.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(numpy.asarray(mask).copy()).long()
        return image_tensor, mask_tensor

    return transform


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment", required=True)
    result.add_argument("--seed", type=int, choices=[17, 42, 73], required=True)
    result.add_argument("--data-root", type=Path, default=Path("datasets/sensation_ds_v2_1_1152x768"))
    result.add_argument("--output-root", type=Path, default=Path("baseline/results"))
    result.add_argument("--checkpoint-root", type=Path, default=Path("baseline/checkpoints"))
    result.add_argument("--resume", type=Path)
    result.add_argument("--device")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    experiment_path = Path("configs/experiments") / f"{args.experiment}.yaml"
    config = load_merged(
        "configs/data/sensation_ds.yaml",
        "configs/model/deeplabv3plus_mobilenet.yaml",
        "configs/training/common.yaml",
        experiment_path,
    )
    seed_everything(args.seed)
    device = choose_device(args.device or config["device"])
    target_size = tuple(config["input_size"])
    transform = _tensor_transform(target_size)
    real_train = RealSegmentationDataset(args.data_root, "training", transform)
    train_data = real_train
    manifest_path = Path(config["synthetic_manifest"]) if config.get("synthetic_manifest") else None
    subset_path = Path(config["synthetic_subset"]) if config.get("synthetic_subset") else None
    if manifest_path and subset_path:
        selected = _subset_ids(subset_path)
        records = [row for row in load_manifest(manifest_path) if row["synthetic_id"] in selected]
        if len(records) != config["synthetic_samples"]:
            raise ValueError("Subset and configured synthetic sample count differ")
        train_data = CombinedDataset(real_train, SyntheticSegmentationDataset(records, transform=transform))

    import torch

    loader = torch.utils.data.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"]
    )
    validation_loader = torch.utils.data.DataLoader(
        RealSegmentationDataset(args.data_root, "validation", transform),
        batch_size=config["batch_size"], num_workers=config["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        RealSegmentationDataset(args.data_root, "testing", transform),
        batch_size=config["batch_size"], num_workers=config["num_workers"],
    )
    model = build_deeplabv3plus(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    start_epoch, best, stale = 0, -1.0, 0
    if args.resume:
        state = load_checkpoint(args.resume, model=model, optimizer=optimizer, map_location=device)
        start_epoch, best = int(state["epoch"]) + 1, float(state["best_metric"])
    checkpoint = args.checkpoint_root / f"{args.experiment}_seed{args.seed}.pt"
    for epoch in range(start_epoch, config["epochs"]):
        train_epoch(model, loader, optimizer, device=device, ignore_index=config["ignore_index"])
        validation = evaluate(model, validation_loader, device=device, ignore_index=config["ignore_index"])
        score = float(validation["miou"] or 0.0)
        if score > best:
            best, stale = score, 0
            save_checkpoint(checkpoint, model=model, optimizer=optimizer, epoch=epoch, best_metric=best)
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]:
                break
    load_checkpoint(checkpoint, model=model, map_location=device)
    test_metrics = evaluate(model, test_loader, device=device, ignore_index=config["ignore_index"])
    run_id = f"{args.experiment}_seed{args.seed}"
    result = {
        "schema_version": "1.0", "run_id": run_id, "status": "completed",
        "experiment": args.experiment, "seed": args.seed, "split": "testing",
        "git": git_state(), "environment": environment_state(),
        "data": {"dataset": "sensation_ds_v2_1_1152x768", "manifest_sha256": _sha256(manifest_path), "subset_sha256": _sha256(subset_path)},
        "training": {"checkpoint_selected_on_validation": True, "best_validation_miou": best},
        "metrics": test_metrics,
    }
    write_result_bundle(
        args.output_root / run_id, result, config,
        {"uri": str(checkpoint), "sha256": _sha256(checkpoint)},
    )
    print(args.output_root / run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
