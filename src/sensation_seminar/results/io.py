"""Write self-describing result bundles without embedding checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

from .schema import validate_result


def git_state(repository_root: str | Path = ".") -> dict[str, object]:
    def run(*arguments: str) -> str:
        process = subprocess.run(
            ["git", "-c", f"safe.directory={Path(repository_root).resolve().as_posix()}", *arguments],
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )
        return process.stdout.strip()

    return {"revision": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def environment_state() -> dict[str, object]:
    state: dict[str, object] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        state.update({"pytorch": torch.__version__, "cuda": torch.version.cuda})
    except ModuleNotFoundError:
        state.update({"pytorch": None, "cuda": None})
    return state


def write_result_bundle(
    output: str | Path,
    result: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    checkpoint_reference: Mapping[str, Any] | None = None,
) -> Path:
    validate_result(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "config.yaml").write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = checkpoint_reference or {"uri": None, "sha256": None}
    (output / "checkpoint_reference.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    row = {
        "run_id": result["run_id"],
        "experiment": result["experiment"],
        "seed": result["seed"],
        "split": result["split"],
        "miou": result["metrics"].get("miou"),
        "road_iou": result["metrics"].get("road_iou"),
        "sidewalk_iou": result["metrics"].get("sidewalk_iou"),
        "road_as_sidewalk_error_rate": result["metrics"].get("road_as_sidewalk_error_rate"),
        "sidewalk_as_road_error_rate": result["metrics"].get("sidewalk_as_road_error_rate"),
    }
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return output
