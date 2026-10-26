# Normalized synthetic-data workspace

This directory indexes the read-only files in `datasets/synthetic-data/`; it does not contain copied images.

## Build

```bash
python -m sensation_seminar.cli.build_manifest \
  --data-root datasets \
  --output synthetic-data-seminar/manifest.jsonl
```

The generated `manifest.jsonl` is intentionally ignored by Git. It uses `manifest.schema.json` version 1.1 and repository-relative paths. Rebuild it whenever source data or supervision configuration changes, and record its checksum in each experiment.

## Eligibility and geometry

`eligibility.analysis` permits non-training analysis of a decodable, non-artifact image. `eligibility.segmentation_training` additionally requires an exact training-source match and `supervision.status == "verified"`. Filename correspondence alone never makes a record trainable.

Supervision rules live in `configs/data/synthetic_supervision.yaml`. `identity` and `resize` rules are applied to the original real mask at load time; masks always use nearest-neighbor interpolation. Uncertain 4:3 or square canvases remain available for quality/diversity analysis but are excluded from segmentation training until their crop/padding/custom operation is documented and implemented.

## Generated layout

```text
synthetic-data-seminar/
├── manifest.schema.json
├── manifest.jsonl
├── audit/
├── scores/<method>.jsonl
└── subsets/<method>_budget_0.25x.jsonl
```

Score overlays are keyed by `synthetic_id` and declare method/version/direction. Subset manifests record the selected IDs, method, canonical budget name, fixed seed, rank, and manifest checksum. The instructor random comparison is always `random_budget_0.25x` (548 samples relative to 2,190 real training images) with selection seed 42.

Never edit, rename, delete, or reorganize files under `datasets/` from this workspace.
