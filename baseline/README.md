# Instructor baseline workflow

This workflow creates the shared results students compare against. Run commands from the repository root. Do not commit datasets, manifests, checkpoints, full predictions, or run logs.

## 1. Environment setup

Create a Python 3.11+ environment and install `python -m pip install -e ".[dev]"`. On TinyGPU, use the site-supported CUDA/PyTorch build. Record the resolved environment through the result bundle rather than editing configs per machine.

## 2. Check the immutable dataset

Place the provided data at `datasets/sensation_ds_v2_1_1152x768` and `datasets/synthetic-data`. Run:

```bash
SENSATION_RUN_DATA_TESTS=1 pytest -m data
pytest
```

The expected real pair counts are 2,190 training, 281 validation, and 281 testing. Never regenerate or rearrange these splits.

## 3. Create the synthetic manifest

```bash
python -m sensation_seminar.cli.build_manifest --data-root datasets --output synthetic-data-seminar/manifest.jsonl
```

Review validity, `eligibility`, and `supervision` counts. Filename matches with uncertain geometry are analysis-eligible but not training-eligible.

## 4. Check synthetic eligibility

Inspect `configs/data/synthetic_supervision.yaml` and the manifest audit. Only `supervision.status: verified` records may have `eligibility.segmentation_training: true`. Do not relax a rule without documented generator evidence and a reproducible mask transform.

## 5. Create the fixed random subset

```bash
python -m sensation_seminar.cli.make_subset --manifest synthetic-data-seminar/manifest.jsonl --method random --budget 0.25x --seed 42
```

This creates exactly 548 synthetic samples. Archive its checksum and provide the same subset manifest to every group. `0.25x` is relative to 2,190 real training images, not to the synthetic pool.

## 6. Train the real-only baseline

For each seed in `17 42 73` run:

```bash
python -m sensation_seminar.cli.train --experiment real_only --seed 17
```

Use `--resume <checkpoint>` after preemption. Do not alter the common model or recipe between runs.

## 7. Train the mandatory random-budget baseline

For the same three seeds run:

```bash
python -m sensation_seminar.cli.train --experiment random_budget_0.25x --seed 17
```

The optional `random_budget_0.50x` experiment uses 1,095 synthetic images and is not required before teaching starts.

## 8. Validation

Training selects checkpoints only by real validation mIoU. Review logs and the recorded best validation value. Synthetic data and real test data must never influence checkpoint or subset selection.

## 9. Final test evaluation

The training CLI evaluates the locked checkpoint once on the real test split. To evaluate an archived checkpoint explicitly, use `python -m sensation_seminar.cli.evaluate --experiment real_only --seed 17 --checkpoint <path> --split testing`.

## 10. Aggregate results

```bash
python -m sensation_seminar.cli.aggregate --results-root baseline/results --output baseline/aggregate.json
```

Verify all six mandatory runs, three seeds per experiment, and report mean, standard deviation, per-class IoU, road/sidewalk metrics, and both road/sidewalk confusion rates.

## 11. Provide results to students

Distribute the resolved configs, `metrics.json`, `summary.csv`, aggregate table, subset manifest and checksums, and checkpoint references if access is appropriate. Students normally train only `<their_method>_budget_0.25x`; they do not retrain instructor baselines unless explicitly asked.
