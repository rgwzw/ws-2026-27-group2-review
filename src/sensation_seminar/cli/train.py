"""Train the fixed seminar baseline from a named experiment configuration."""

from __future__ import annotations

import argparse
import csv
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
from sensation_seminar.results.io import (
    environment_state,
    git_state,
    write_result_bundle,
)
from sensation_seminar.synthetic_data.manifest import load_manifest
from sensation_seminar.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from sensation_seminar.training.engine import choose_device, train_epoch
from sensation_seminar.training.reproducibility import seed_everything


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _subset_ids(path: Path) -> set[str]:
    return {
        json.loads(line)["synthetic_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def _tensor_transform(target_size: tuple[int, int]):
    def transform(image, mask):
        import numpy
        import torch

        image, mask = paired_resize(image, mask, target_size)

        image_tensor = (
            torch.from_numpy(numpy.asarray(image).copy())
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        mask_tensor = torch.from_numpy(
            numpy.asarray(mask).copy()
        ).long()

        return image_tensor, mask_tensor

    return transform


def _initialize_history(path: Path, *, resume: bool) -> None:
    """Create the training history CSV if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if resume and path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "validation_miou",
                "best_validation_miou",
                "early_stopping_stale",
                "early_stopping_patience",
            ]
        )


def _append_history(
    path: Path,
    *,
    epoch: int,
    train_loss: float,
    validation_miou: float,
    best_validation_miou: float,
    stale: int,
    patience: int,
) -> None:
    """Append one epoch to training_history.csv."""
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{validation_miou:.6f}",
                f"{best_validation_miou:.6f}",
                stale,
                patience,
            ]
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)

    result.add_argument(
        "--experiment",
        required=True,
    )

    result.add_argument(
        "--seed",
        type=int,
        choices=[17, 42, 73],
        required=True,
    )

    result.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "datasets/sensation_ds_v2_1_1152x768"
        ),
    )

    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("baseline/results"),
    )

    result.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("baseline/checkpoints"),
    )

    result.add_argument(
        "--resume",
        type=Path,
    )

    result.add_argument(
        "--device",
    )

    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)

    experiment_path = (
        Path("configs/experiments")
        / f"{args.experiment}.yaml"
    )

    config = load_merged(
        "configs/data/sensation_ds.yaml",
        "configs/model/deeplabv3plus_mobilenet.yaml",
        "configs/training/common.yaml",
        experiment_path,
    )

    seed_everything(args.seed)

    device = choose_device(
        args.device or config["device"]
    )

    target_size = tuple(config["input_size"])
    transform = _tensor_transform(target_size)

    print("=" * 72)
    print("SENSATION Seminar Training")
    print("=" * 72)
    print(f"Experiment : {args.experiment}")
    print(f"Seed       : {args.seed}")
    print(f"Device     : {device}")
    print(f"Input size : {target_size}")
    print(f"Epochs     : {config['epochs']}")
    print(f"Batch size : {config['batch_size']}")
    print(
        "Patience   : "
        f"{config['early_stopping_patience']}"
    )
    print("=" * 72)
    print(flush=True)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    real_train = RealSegmentationDataset(
        args.data_root,
        "training",
        transform,
    )

    train_data = real_train

    manifest_path = (
        Path(config["synthetic_manifest"])
        if config.get("synthetic_manifest")
        else None
    )

    subset_path = (
        Path(config["synthetic_subset"])
        if config.get("synthetic_subset")
        else None
    )

    if manifest_path and subset_path:
        selected = _subset_ids(subset_path)

        records = [
            row
            for row in load_manifest(manifest_path)
            if row["synthetic_id"] in selected
        ]

        if len(records) != config["synthetic_samples"]:
            raise ValueError(
                "Subset and configured synthetic "
                "sample count differ"
            )

        synthetic_train = SyntheticSegmentationDataset(
            records,
            transform=transform,
        )

        train_data = CombinedDataset(
            real_train,
            synthetic_train,
        )

        print(
            f"Real training samples      : "
            f"{len(real_train)}"
        )
        print(
            f"Synthetic training samples : "
            f"{len(synthetic_train)}"
        )
        print(
            f"Combined training samples  : "
            f"{len(train_data)}"
        )
    else:
        print(
            f"Real training samples      : "
            f"{len(real_train)}"
        )
        print("Synthetic training samples : 0")

    print(flush=True)

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    import torch

    loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    validation_loader = torch.utils.data.DataLoader(
        RealSegmentationDataset(
            args.data_root,
            "validation",
            transform,
        ),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    test_loader = torch.utils.data.DataLoader(
        RealSegmentationDataset(
            args.data_root,
            "testing",
            transform,
        ),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    print("Building model...", flush=True)

    model = build_deeplabv3plus(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    print("Model ready.", flush=True)
    print()

    # ------------------------------------------------------------------
    # Run paths
    # ------------------------------------------------------------------

    run_id = f"{args.experiment}_seed{args.seed}"
    run_directory = args.output_root / run_id

    history_path = (
        run_directory
        / "training_history.csv"
    )

    checkpoint = (
        args.checkpoint_root
        / f"{args.experiment}_seed{args.seed}.pt"
    )

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    start_epoch = 0
    best = -1.0
    stale = 0

    if args.resume:
        print(
            f"Resuming checkpoint: {args.resume}",
            flush=True,
        )

        state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )

        start_epoch = int(state["epoch"]) + 1
        best = float(state["best_metric"])

        # Current checkpoint format does not persist stale.
        # Therefore it starts again at zero when resuming.
        stale = int(state.get("stale", 0))

        print(
            f"Resume from epoch {start_epoch + 1}, "
            f"best validation mIoU={best:.4f}",
            flush=True,
        )
        print()

    _initialize_history(
        history_path,
        resume=bool(args.resume),
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    patience = config["early_stopping_patience"]

    print("Starting training.")
    print("=" * 72)
    print(flush=True)

    for epoch in range(
        start_epoch,
        config["epochs"],
    ):
        epoch_number = epoch + 1

        print(
            f"Epoch {epoch_number}/{config['epochs']}",
            flush=True,
        )

        train_loss = train_epoch(
            model,
            loader,
            optimizer,
            device=device,
            ignore_index=config["ignore_index"],
        )

        validation = evaluate(
            model,
            validation_loader,
            device=device,
            ignore_index=config["ignore_index"],
        )

        score = float(
            validation["miou"] or 0.0
        )

        improved = score > best

        if improved:
            best = score
            stale = 0

            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best,
            )

            checkpoint_status = "saved"
        else:
            stale += 1
            checkpoint_status = "not improved"

        _append_history(
            history_path,
            epoch=epoch_number,
            train_loss=train_loss,
            validation_miou=score,
            best_validation_miou=best,
            stale=stale,
            patience=patience,
        )

        print(
            f"  train loss       : {train_loss:.4f}"
        )
        print(
            f"  validation mIoU  : {score:.4f}"
        )
        print(
            f"  best val mIoU    : {best:.4f}"
        )
        print(
            f"  checkpoint       : {checkpoint_status}"
        )
        print(
            f"  early stopping   : "
            f"{stale}/{patience}"
        )
        print(flush=True)

        if stale >= patience:
            print(
                "Early stopping triggered "
                f"after epoch {epoch_number}.",
                flush=True,
            )
            print()
            break

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    if not checkpoint.exists():
        raise RuntimeError(
            "No training checkpoint was created."
        )

    print("=" * 72)
    print(
        "Training finished. Loading best checkpoint.",
        flush=True,
    )
    print(
        f"Best validation mIoU: {best:.4f}",
        flush=True,
    )
    print(
        "Running final evaluation on the real test set...",
        flush=True,
    )

    load_checkpoint(
        checkpoint,
        model=model,
        map_location=device,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        ignore_index=config["ignore_index"],
    )

    test_miou = test_metrics.get("miou")

    if test_miou is not None:
        print(
            f"Test mIoU: {float(test_miou):.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Result bundle
    # ------------------------------------------------------------------

    result = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "completed",
        "experiment": args.experiment,
        "seed": args.seed,
        "split": "testing",
        "git": git_state(),
        "environment": environment_state(),
        "data": {
            "dataset": (
                "sensation_ds_v2_1_1152x768"
            ),
            "manifest_sha256": _sha256(
                manifest_path
            ),
            "subset_sha256": _sha256(
                subset_path
            ),
        },
        "training": {
            "checkpoint_selected_on_validation": True,
            "best_validation_miou": best,
            "training_history": str(
                history_path
            ),
        },
        "metrics": test_metrics,
    }

    write_result_bundle(
        run_directory,
        result,
        config,
        {
            "uri": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
    )

    print()
    print("=" * 72)
    print("Training completed successfully.")
    print(f"Run ID      : {run_id}")
    print(f"Results     : {run_directory}")
    print(f"History     : {history_path}")
    print(f"Checkpoint  : {checkpoint}")
    print("=" * 72)
    print(flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
