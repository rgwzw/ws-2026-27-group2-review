"""Write self-describing result bundles without embedding checkpoints."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

from .schema import validate_result


def git_state(
    repository_root: str | Path = ".",
) -> dict[str, object]:
    """Return the current Git revision and dirty state."""

    def run(*arguments: str) -> str:
        process = subprocess.run(
            [
                "git",
                "-c",
                (
                    "safe.directory="
                    f"{Path(repository_root).resolve().as_posix()}"
                ),
                *arguments,
            ],
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )

        return process.stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain")),
    }


def environment_state() -> dict[str, object]:
    """Return relevant information about the Python/PyTorch environment."""

    state: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        state.update(
            {
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
            }
        )
    except ModuleNotFoundError:
        state.update(
            {
                "pytorch": None,
                "cuda": None,
            }
        )

    return state


def _write_json(
    path: Path,
    data: Mapping[str, Any],
) -> None:
    """Write formatted JSON to a file."""

    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_result_bundle(
    output: str | Path,
    result: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    checkpoint_reference: Mapping[str, Any] | None = None,
) -> Path:
    """Write the final result files for one experiment run.

    The run directory is allowed to exist already.

    This is intentional because other parts of the training pipeline may
    create the run directory before training has completed, for example to
    store:

    - training_history.csv
    - intermediate logs
    - diagnostic information

    Existing directories therefore do not constitute an error.

    The final result files are written or replaced when training has
    completed successfully.
    """

    validate_result(result)

    output = Path(output)

    # The run directory may already exist, for example because
    # training_history.csv was created at the start of training.
    #
    # exist_ok=True is therefore required here.
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not output.is_dir():
        raise NotADirectoryError(
            f"Result output path exists but is not a directory: {output}"
        )

    # Logs may also already exist.
    logs_directory = output / "logs"
    logs_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Resolved configuration
    # ------------------------------------------------------------------

    # The historical filename is config.yaml. The current configuration
    # object is serialized as JSON, which is also valid YAML syntax.
    _write_json(
        output / "config.yaml",
        resolved_config,
    )

    # ------------------------------------------------------------------
    # Metrics and run metadata
    # ------------------------------------------------------------------

    _write_json(
        output / "metrics.json",
        result,
    )

    # ------------------------------------------------------------------
    # Checkpoint reference
    # ------------------------------------------------------------------

    checkpoint = checkpoint_reference or {
        "uri": None,
        "sha256": None,
    }

    _write_json(
        output / "checkpoint_reference.json",
        checkpoint,
    )

    # ------------------------------------------------------------------
    # Compact CSV summary
    # ------------------------------------------------------------------

    metrics = result["metrics"]

    row = {
        "run_id": result["run_id"],
        "experiment": result["experiment"],
        "seed": result["seed"],
        "split": result["split"],
        "miou": metrics.get("miou"),
        "road_iou": metrics.get("road_iou"),
        "sidewalk_iou": metrics.get("sidewalk_iou"),
        "road_as_sidewalk_error_rate": metrics.get(
            "road_as_sidewalk_error_rate"
        ),
        "sidewalk_as_road_error_rate": metrics.get(
            "sidewalk_as_road_error_rate"
        ),
    }

    summary_path = output / "summary.csv"

    with summary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row),
        )

        writer.writeheader()
        writer.writerow(row)

    return output