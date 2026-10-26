# Quality-Aware Synthetic Data for Road Scene Understanding

This repository is the shared software framework for the seminar **Road Scene Understanding for Visually Impaired**. It studies whether synthetic-image quality, semantic consistency, structural preservation, diversity, and subset selection improve semantic segmentation on real SENSATION-DS scenes.

## Ground rules

- `datasets/sensation_ds_v2_1_1152x768/{training,validation,testing}` is authoritative and read-only.
- Synthetic data under `datasets/synthetic-data/` may augment training only.
- Validation and testing always contain real images; testing is reserved for final evaluation.
- A filename match is not enough for supervision. Synthetic records require verified image/mask geometry before segmentation training.
- Groups share one DeepLabV3Plus benchmark and result schema. Cross-group inputs are optional.

## Quick start

Use Python 3.11 or newer and install the project in a virtual environment:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

Build the synthetic index and the fixed instructor comparison subset:

```bash
python -m sensation_seminar.cli.build_manifest --data-root datasets --output synthetic-data-seminar/manifest.jsonl
python -m sensation_seminar.cli.make_subset --manifest synthetic-data-seminar/manifest.jsonl --method random --budget 0.25x --seed 42
```

Run the mandatory baselines for seeds 17, 42, and 73:

```bash
python -m sensation_seminar.cli.train --experiment real_only --seed 17
python -m sensation_seminar.cli.train --experiment random_budget_0.25x --seed 17
```

Repeat with seeds `42` and `73`. The secondary `random_budget_0.50x` baseline is supported but not required before the seminar.

## Repository map

- `architecture.md`: scientific and software design.
- `configs/`: immutable data definitions and reproducible experiment settings.
- `src/sensation_seminar/`: shared indexing, datasets, model, training, metrics, evaluation, and result code.
- `synthetic-data-seminar/`: manifest schema and generated metadata workspace; images are not copied.
- `baseline/`: instructor workflow and result location.
- `group-1/` ... `group-9/`: independent student tasks, code, tests, and result contracts.
- `tests/`: CPU-fast shared tests. Optional checks use `data`, `gpu`, `slow`, and `integration` markers.

The normal test suite neither downloads model weights nor modifies `datasets/`. See `baseline/README.md` for the full instructor workflow and each group README for its independent task and fallback behavior.
