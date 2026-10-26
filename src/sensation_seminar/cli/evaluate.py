"""Evaluate a saved common-model checkpoint on a real split."""

from __future__ import annotations

import argparse
from pathlib import Path

from sensation_seminar.cli.train import _sha256, _tensor_transform
from sensation_seminar.config import load_merged
from sensation_seminar.datasets.real import RealSegmentationDataset
from sensation_seminar.evaluation.evaluate import evaluate
from sensation_seminar.models.segmentation import build_deeplabv3plus
from sensation_seminar.results.io import environment_state, git_state, write_result_bundle
from sensation_seminar.training.checkpointing import load_checkpoint
from sensation_seminar.training.engine import choose_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, choices=[17, 42, 73], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "testing"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/sensation_ds_v2_1_1152x768"))
    parser.add_argument("--output-root", type=Path, default=Path("baseline/results"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    config = load_merged(
        "configs/model/deeplabv3plus_mobilenet.yaml",
        "configs/training/common.yaml",
        Path("configs/experiments") / f"{args.experiment}.yaml",
    )
    import torch

    device = choose_device(args.device)
    transform = _tensor_transform(tuple(config["input_size"]))
    loader = torch.utils.data.DataLoader(
        RealSegmentationDataset(args.data_root, args.split, transform),
        batch_size=config["batch_size"], num_workers=config["num_workers"],
    )
    model = build_deeplabv3plus(config).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    metrics = evaluate(model, loader, device=device, ignore_index=config["ignore_index"])
    run_id = f"{args.experiment}_seed{args.seed}_{args.split}"
    result = {
        "schema_version": "1.0", "run_id": run_id, "status": "completed",
        "experiment": args.experiment, "seed": args.seed, "split": args.split,
        "git": git_state(), "environment": environment_state(),
        "data": {"dataset": "sensation_ds_v2_1_1152x768", "manifest_sha256": None, "subset_sha256": None},
        "training": {"checkpoint_selected_on_validation": True}, "metrics": metrics,
    }
    write_result_bundle(
        args.output_root / run_id, result, config,
        {"uri": str(args.checkpoint), "sha256": _sha256(args.checkpoint)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
