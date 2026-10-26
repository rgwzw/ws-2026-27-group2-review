# Seminar Software Architecture

## 1. Seminar Goal

The repository will support one coordinated study of this question:

> How do synthetic image quality, semantic consistency, diversity, and data selection influence semantic segmentation performance on real SENSATION-DS road scenes?

All downstream comparisons keep the segmentation model, real-data split, training recipe, metrics, and result format fixed. The independent variable is the synthetic subset. The primary model is `segmentation_models_pytorch.DeepLabV3Plus` with encoder `tu-mobilenetv3_large_075` and ImageNet weights. Synthetic images may augment training only; validation and testing always use the existing real SENSATION-DS splits.

The nine projects form one pipeline rather than nine independent model projects: audit -> individual quality/consistency signals -> quality and diversity selection -> combined selection -> safety-oriented synthesis of results.

## 2. Existing Repository Assessment

The repository currently contains only the tracked GitLab template `README.md`. `prompt.md` and `datasets/` are local, untracked inputs; there is no existing Python package, training code, metric implementation, test suite, CI configuration, or ignore policy to reuse.

The actual paths differ from the names in the task description:

```text
datasets/
├── sensation_ds_v2_1_1152x768/
│   ├── README.MD
│   ├── training/{images,masks}/
│   ├── validation/{images,masks}/
│   └── testing/{images,masks}/
└── synthetic-data/
    └── group1/ ... group9/
```

The real split is complete and authoritative:

| Split | image/mask pairs | Observed size | Purpose |
|---|---:|---|---|
| training | 2,190 | 1152 x 768 | training, source matching, selection features |
| validation | 281 | 1152 x 768 | early stopping and locked model selection |
| testing | 281 | 1152 x 768 | final real-only evaluation |

Every image stem has a mask stem, and there are no stem overlaps between splits. Masks contain documented class IDs 0-8 and ignore ID 255: background, road, sidewalk, bikelane, person, vehicle, bicycle, traffic light, and obstacle. The dataset README mentions auxiliary class CSVs, but they are not present; therefore the checked-in label configuration must encode this documented schema and tests must verify mask values.

Each synthetic source has 8,760 regular images, normally four variants for each training source. The layouts and generation properties are not uniform:

| Source | Layout / format | Important observation |
|---|---|---|
| group1 | `images/01`, `images/02`; JPEG | two run IDs, two variants per run; sampled size 768 x 512 |
| group2 | flat JPEG | 8,760 regular files plus 38 `-checkpoint` artifacts; nested prefixes such as `e9`, `eft3`, `eft4` |
| group3 | flat JPEG | four variants; sampled size 1152 x 768 |
| group4 | flat PNG | four variants; `generated_index.csv`; sampled size 768 x 576 |
| group5 | `01`, `02/images`, `03/images`; JPEG | generation manifests/configs in 02/03; mixed generation layouts |
| group6 | flat JPEG | run ID `e29`; sampled size 768 x 768 |
| group7 | flat JPEG | four extra `.ipynb_checkpoints` files |
| group8 | flat JPEG | four variants; sampled size 1152 x 768 |
| group9 | flat JPEG | one extra `__MACOSX` resource-fork file |

Existing group4/group5 metadata is useful enrichment, but its absolute HPC paths and, in group4, references such as `masks_ade_v2`, are not valid paths in this checkout. Actual repository-relative files and exact stem matching remain authoritative. Input files under `datasets/` must never be renamed, moved, or overwritten.

## 3. Data Flow

```text
authoritative real training images + masks       read-only synthetic sources
                  \                              /
                   exact source-ID reconciliation
                               |
              synthetic-data-seminar/manifest.jsonl
                               |
       audit and score overlays keyed by synthetic_id
                               |
      deterministic, size-matched subset manifests
                               |
   common DeepLabV3Plus training (real + synthetic)
                               |
       real validation -> locked checkpoint choice
                               |
            one final real test evaluation
                               |
       standard run metrics, predictions, summaries
                               |
           Group 9 comparison and paper figures
```

Synthetic samples use the matched real **training** mask as supervision. Image and mask receive one paired geometric transform; mask interpolation is nearest-neighbor. A source match does not prove that generation preserved semantics, which is precisely what Groups 4 and 5 measure. Validation/test images, masks, features, or outcomes must not influence scoring, subset selection, thresholds, or training.

## 4. Proposed Directory Structure

```text
.
├── README.md
├── pyproject.toml
├── .gitignore
├── .gitlab-ci.yml
├── configs/
│   ├── data/sensation_ds.yaml
│   ├── model/deeplabv3plus_mobilenet.yaml
│   ├── training/common.yaml
│   └── experiments/{real_only,random_025,random_050}.yaml
├── src/sensation_seminar/
│   ├── config.py
│   ├── datasets/{real.py,synthetic.py,combined.py,transforms.py}
│   ├── synthetic_data/{manifest.py,mapping.py,validation.py,scoring.py,selection.py}
│   ├── models/segmentation.py
│   ├── training/{engine.py,reproducibility.py,checkpointing.py}
│   ├── evaluation/{evaluate.py,predictions.py}
│   ├── metrics/{confusion.py,segmentation.py,safety.py}
│   ├── results/{schema.py,io.py,aggregate.py}
│   └── cli/{build_manifest.py,make_subset.py,train.py,evaluate.py,aggregate.py}
├── scripts/hpc/{train.slurm,evaluate.slurm}
├── synthetic-data-seminar/
│   ├── README.md
│   ├── manifest.schema.json
│   ├── manifest.jsonl                 # generated, not committed
│   ├── audit/                         # generated reports
│   ├── scores/<method>.jsonl          # generated score overlays
│   └── subsets/<experiment>.jsonl     # generated selections
├── baseline/
│   ├── README.md
│   └── results/<run_id>/{config.yaml,metrics.json,summary.csv,logs/,predictions/}
├── group-1/ ... group-9/
│   ├── README.md
│   ├── src/group_<n>/<method>.py
│   ├── tests/test_group_<n>.py
│   └── results/{README.md,<run_id>/...}
└── tests/
    ├── fixtures/
    ├── test_data_splits.py
    ├── test_manifest.py
    ├── test_metrics.py
    ├── test_results_schema.py
    └── test_smoke_pipeline.py
```

Directories containing large or reproducible outputs are ignored. Only documentation, schemas, configurations, small fixtures, and deliberately curated summary tables/figures are candidates for Git. Checkpoints and full prediction maps live in configured scratch/project storage, with their URI and checksum recorded in results.

## 5. Synthetic Data Normalization

Normalization is a non-destructive index, not a copy. This avoids roughly 78,840 duplicate images, works without privileged link creation on Windows, and keeps the original byte stream available. Paths in the manifest are POSIX-style and repository-relative; a configurable data root relocates the checkout on HPC. Symlink materialization may be an optional CLI mode, never the default.

### Discovery and mapping

1. Recursively enumerate supported image extensions under each `datasets/synthetic-data/groupN`, preserving the exact relative path and filename.
2. Flag paths containing dot-directories, `__MACOSX`, Apple `._` files, or `-checkpoint` as artifacts rather than silently deleting them.
3. Parse the final `_<digits>` token from the right as the variant and the first `e<digits>_` token as a generation/run token.
4. Resolve the original by exact lookup against the 2,190 training stems. First try the remaining body; then try suffixes beginning only after an underscore. Accept exactly one match. This safely maps group2 names such as `e2_e9_0026_01.jpg` and `e2_eft3__8rh..._00.jpg`. Never use fuzzy or substring similarity.
5. A match to validation/test, zero matches, or multiple matches is invalid and is never training-eligible. Preserve the reason in the manifest.
6. Enrich records by exact generated filename from existing CSV/JSON metadata. Do not trust recorded absolute paths. Validate source image/mask existence, decoding, mode, dimensions, extension, and allowed mask IDs.
7. Assign `synthetic_id = sha256(source_group + "\0" + source_relpath)` and optionally compute a content SHA-256 for duplicate detection.

### Canonical `manifest.jsonl` schema

Each line has:

```json
{
  "schema_version": "1.0",
  "synthetic_id": "sha256:...",
  "source_group": "group2",
  "source_relpath": "datasets/synthetic-data/group2/images/e2_e9_0026_01.jpg",
  "original_filename": "e2_e9_0026_01.jpg",
  "extension": ".jpg",
  "original_id": "0026",
  "real_image_relpath": "datasets/sensation_ds_v2_1_1152x768/training/images/0026.jpg",
  "real_mask_relpath": "datasets/sensation_ds_v2_1_1152x768/training/masks/0026.png",
  "real_split": "training",
  "generation_tokens": ["e2", "e9"],
  "variant": "01",
  "width": 512,
  "height": 512,
  "mode": "RGB",
  "content_sha256": null,
  "validity": {"status": "valid", "reason_codes": []},
  "generation_metadata": {},
  "scores": {},
  "diversity": {},
  "selection_status": null
}
```

The canonical manifest is a reproducible snapshot and is not rewritten concurrently. Group score files are JSONL overlays with `schema_version`, `synthetic_id`, `method`, `method_version`, `score` (higher is always better), optional components, and provenance. Subset files contain `synthetic_id`, rank, selected flag, selection method, budget, seed, and score provenance. `load_manifest(..., overlays=...)` performs validated joins, so the effective manifest exposes optional scores/diversity/selection without forcing groups to edit one shared large file. Audit summaries report discovered, valid, invalid, artifact, duplicate, and source-coverage counts.

## 6. Baseline Architecture

The shared factory builds `DeepLabV3Plus(encoder_name="tu-mobilenetv3_large_075", encoder_weights="imagenet", classes=9)`. Ignore ID 255 is excluded from loss and metrics. A single configuration controls input size, augmentation, optimizer, scheduler, batch size, epochs, mixed precision, workers, early stopping, and checkpoint resume. Exact defaults are benchmarked and frozen by instructors before student experiments; groups may not silently change them.

The initial baseline matrix is:

- A: real training only.
- B25: real training plus 548 synthetic records (25% of 2,190, rounded half up).
- B50: real training plus 1,095 synthetic records (50%); recommended secondary baseline.

Random subsets use a committed/generated subset manifest, seed 42, valid rows only, and stratification by source group. Training is repeated with seeds 17, 42, and 73. The same B25/B50 subsets are reused as controls rather than resampled per method. Each run saves the resolved configuration, Git revision, environment versions, manifest/subset checksums, metrics, logs, and checkpoint reference. The best validation-mIoU checkpoint is evaluated on test only after the run is locked.

## 7. Common Experiment Protocol

- Real split boundaries are immutable. Only `training/` contributes labels to optimization or selection.
- Primary budget is 548 synthetic images. The required comparison is real-only, matched random-548, and selected-548. Budget 1,095 is an extension.
- Selection methods may use synthetic images, paired real training images/masks, and pretrained frozen models. They may not inspect real validation/test outcomes to select samples.
- All comparisons use the same optimizer recipe, augmentation, stopping rule, seeds, and maximum update policy. The loader defines epoch length/sampling weights so adding synthetic data does not create an undocumented compute advantage.
- Report all three seeds individually plus mean, standard deviation, and paired difference from the same-seed random baseline. Record unique original-source coverage and source-group composition.
- Validation is used for early stopping; test is evaluated once per finalized run. Group 9 does not choose a winner from test scores.

Required metrics are mIoU over classes 0-8, per-class IoU, per-class F1/precision/recall, road IoU (ID 1), sidewalk IoU (ID 2), and road-as-sidewalk error: `count(gt=road and pred=sidewalk) / count(gt=road)`. Ignore pixels are excluded. Also report sidewalk-as-road and macro safety-class IoU for person, bicycle, traffic light, and obstacle. Confusion counts are stored so aggregate metrics can be recomputed.

## 8. Group 1-9 Responsibilities

### Group 1 - Synthetic Data Audit and Characterization

- **Question:** Which structural defects, coverage gaps, duplicates, and generation differences exist?
- **Implementation/input:** audit the canonical manifest, image decodability, dimensions, artifacts, mappings, source coverage, variants, and hashes.
- **Output:** audit JSON/CSV, exclusion overlay, coverage tables, and figures consumed by every group.
- **Minimum:** reliable validity report and source/group/variant statistics. **Extension:** perceptual duplicates and distribution diagnostics.
- **Experiment/result:** compare random-548 drawn before versus after audit exclusions; quantify both downstream change and retained coverage.
- **Test:** fixture containing flat, nested, artifact, unmatched, and group2-style names produces exact reason codes and counts.

### Group 2 - Perceptual Image Quality

- **Question:** Does visual quality predict downstream usefulness?
- **Implementation/input:** score valid images with justified no-reference metrics (for example BRISQUE/NIQE) and paired metrics only where alignment makes them meaningful (SSIM/LPIPS).
- **Output:** normalized score overlay, metric agreement analysis, top/bottom examples.
- **Minimum:** one reproducible metric and top-548 subset. **Extension:** multiple metrics, calibration, and ablation.
- **Experiment/result:** top-548 versus the exact random-548 control; correlation with per-sample loss or downstream delta where feasible.
- **Test:** deterministic fixture scores, higher-is-better normalization, invalid-row handling, and output schema.

### Group 3 - Feature-Based Semantic Similarity

- **Question:** Does feature similarity between a synthetic image and its source predict utility?
- **Implementation/input:** batch frozen DINOv2 or CLIP embeddings, cache them, and compute a documented similarity.
- **Output:** embedding metadata/cache reference, similarity overlay, rankings, distributions.
- **Minimum:** one backbone and top-548 selection. **Extension:** second backbone or layer/pooling comparison.
- **Experiment/result:** similarity-selected-548 versus random-548 and perceptual-quality selection.
- **Test:** tiny mocked/frozen embeddings give known cosine scores, stable ordering, and cache-key validation.

### Group 4 - Semantic Consistency

- **Question:** Are road-scene classes preserved from source to synthetic image?
- **Implementation/input:** apply one frozen segmentation teacher to paired images and compare predictions, with special road/sidewalk checks; never train the teacher on validation/test.
- **Output:** global and class-wise consistency overlay, change maps for a small curated sample.
- **Minimum:** agreement/mIoU-style score and top-548 subset. **Extension:** confidence-aware or class-risk-weighted consistency.
- **Experiment/result:** consistency-selected-548 versus matched random and Group 3 similarity.
- **Test:** synthetic logits/masks with known swaps yield exact consistency and road/sidewalk penalties.

### Group 5 - Structural Consistency

- **Question:** Does preservation of layout and boundaries predict utility beyond semantics?
- **Implementation/input:** paired source/synthetic data; measure edges, segmentation boundaries, spatial class regions, or justified depth/geometry proxies.
- **Output:** structural score overlay, component ablation, qualitative overlays.
- **Minimum:** one robust boundary/layout measure and top-548 subset. **Extension:** multi-scale or learned geometry measure.
- **Experiment/result:** structural-selected-548 versus random-548 and semantic-consistency selection.
- **Test:** controlled translated/deformed fixture ranks identical geometry above distorted geometry.

### Group 6 - Quality-Based Selection

- **Question:** Which combination of audit, perceptual, feature, semantic, and structural quality best selects useful data?
- **Implementation/input:** consume Groups 1-5 overlays; implement transparent normalization, weighting/ranking, missing-score policy, and thresholds.
- **Output:** versioned selector, selected subset manifests, weight/threshold ablations.
- **Minimum:** fixed, documented composite and top-548. **Extension:** train-only learned weights or Pareto/robustness analysis.
- **Experiment/result:** composite-selected-548 versus random-548 and each best single signal.
- **Test:** known score table verifies joins, missing data, tie-breaking, budget, and deterministic ranking.

### Group 7 - Diversity-Aware Selection

- **Question:** Does feature-space coverage improve over random selection at fixed size?
- **Implementation/input:** cached embeddings; clustering, k-center, or another scalable coverage method with source-aware diagnostics.
- **Output:** diversity annotations, cluster summary, subset manifests, coverage plots.
- **Minimum:** one deterministic diversity selector at 548. **Extension:** alternative embedding/algorithm and 1,095 budget.
- **Experiment/result:** diverse-548 versus random-548 and quality-only-548.
- **Test:** toy clusters select the expected representatives with stable seed/tie behavior.

### Group 8 - Joint Quality-Diversity Selection

- **Question:** Is a quality/diversity trade-off better than either objective alone?
- **Implementation/input:** Group 6 quality and Group 7 diversity outputs; implement a documented multi-objective or quality-constrained coverage selector.
- **Output:** joint subsets, trade-off curve, ablation table, chosen operating point fixed on training-side evidence.
- **Minimum:** one joint selector at 548. **Extension:** Pareto frontier, sensitivity, and 1,095 budget.
- **Experiment/result:** joint-548 versus the exact random-, quality-, and diversity-only 548 subsets.
- **Test:** toy inputs meet budget, quality floor, coverage expectation, deterministic ties, and schema.

### Group 9 - Downstream and Safety-Oriented Evaluation

- **Question:** Which strategies improve overall and safety-critical real-scene performance, and are gains stable?
- **Implementation/input:** validate all standard result bundles, aggregate seeds, compute paired deltas/uncertainty, and create common tables/plots without changing submitted metrics.
- **Output:** comparison CSV/JSON, paper-ready plots, failure analysis, reproducibility matrix.
- **Minimum:** schema validation and full baseline/group comparison. **Extension:** confidence intervals, significance/effect-size analysis, source/class subgroup study, replication of the best method.
- **Experiment/result:** three-seed confirmation of the nominated best quality, diversity, and joint methods; conclusions distinguish validation decisions from held-out test findings.
- **Test:** miniature result bundles aggregate to known means/deltas and reject leakage, missing seeds, mismatched budgets, or configs.

## 9. Shared Interfaces

The shared package owns contracts; group packages provide methods:

```python
load_manifest(path, *, overlays=(), valid_only=True) -> list[SyntheticRecord]
score_synthetic_image(record, context) -> ScoreRecord
select_subset(records, *, budget, seed, method) -> list[SelectionRecord]
build_datasets(config, subset) -> tuple[Dataset, Dataset, Dataset]
save_results(run_result, output_dir) -> None
validate_results(path) -> ValidationReport
```

Paths are `pathlib.Path`; configs contain repository-relative paths or CLI overrides. Scores must declare direction, version, model/checkpoint identity, preprocessing, and missing-value behavior. Selectors must return exactly the requested number of unique `synthetic_id` values or fail clearly, use stable `synthetic_id` tie-breaking, and never copy image files. Shared configuration and dataclasses prevent group-specific CSV dialects.

## 10. Result Schema

Every run directory contains a resolved `config.yaml`, `metrics.json`, `summary.csv`, and `logs/`; predictions and checkpoint references are optional. `metrics.json` requires:

- `schema_version`, `run_id`, UTC timestamp, status, group/method, split;
- Git revision and dirty flag; Python/PyTorch/CUDA/package versions;
- model/loss/training parameters and seed;
- real-data version/path fingerprint, manifest checksum, subset checksum, budget and composition;
- epoch/checkpoint selection rule and checkpoint URI/checksum;
- confusion matrix, sample/pixel counts, mIoU, per-class metrics, safety metrics;
- validation-best and final-test values clearly separated; warnings/failures.

`summary.csv` is one flat row per `(run_id, seed, split)` with stable columns for automated comparison. Aggregates never overwrite raw runs. Full checkpoints, dense predictions, and logs are not committed; small final `comparison.csv`, `aggregate.json`, and figures may be committed after instructor review.

## 11. Testing Strategy

`pytest` is the central command. Default tests are CPU-only, use temporary directories and tiny generated fixtures, do not traverse the full local dataset, download weights, or train a real network. Mark optional checks `data`, `gpu`, `slow`, and `integration`; instructors can run `pytest`, `pytest -m data --data-root ...`, or a documented smoke command.

Shared tests verify immutable split discovery/pairing, label IDs and ignore handling, exact/adversarial filename mapping, manifest/schema joins, deterministic subsets, confusion-derived metrics, result validation, and a two-batch mocked train/evaluate pipeline. Each group test described above is also a progress gate: import + valid method output + deterministic minimal behavior + standard schema constitutes minimum implementation progress. CI runs formatting/type checks if adopted and the fast suite. Tests must never write inside `datasets/`.

## 12. HPC Execution Strategy

All primary workflows are non-interactive CLIs driven by YAML plus overrides. Data, cache, and output roots are CLI/environment parameters; no user-specific absolute paths are stored. Slurm scripts select GPU/resources, activate a pinned environment, stage no authoritative data, and assign collision-free output paths from method, budget, seed, timestamp, and scheduler job ID.

Training records seeds and deterministic settings, seeds workers, supports AMP, checkpoints optimizer/scheduler/scaler state, and resumes safely after preemption. Atomic result writes and a final `status` distinguish incomplete runs. Embeddings/scores use checksummed caches. Logs go to both stdout and run storage. Device selection supports `--device auto|cpu|cuda[:n]`; notebooks may inspect results but are never required to create them.

## 13. Git Workflow

Every created or modified file is inspected, staged by explicit path, and committed alone with a meaningful message. Before each commit, `git diff -- <file>`, `git diff --cached`, and `git status --short` confirm scope. Generated manifests, caches, datasets, environments, checkpoints, predictions, temporary files, and bulk results remain ignored. If correctness genuinely requires an inseparable multi-file change, work stops for instructor approval rather than violating the one-file-per-commit rule.

## 14. Implementation Plan for Phase 2

After explicit architecture approval, implementation proceeds in this dependency order. Each numbered path is a separate commit; brace notation below is an ordered repetition, not a multi-file commit.

1. `.gitignore` (`chore: ignore data and experiment artifacts`).
2. `README.md` (`docs: replace GitLab template with seminar overview`).
3. `pyproject.toml` (`build: define seminar package and test dependencies`).
4. `configs/data/sensation_ds.yaml`.
5. `src/sensation_seminar/config.py`.
6. `src/sensation_seminar/synthetic_data/mapping.py`.
7. `src/sensation_seminar/synthetic_data/validation.py`.
8. `src/sensation_seminar/synthetic_data/manifest.py`.
9. `src/sensation_seminar/cli/build_manifest.py`.
10. `synthetic-data-seminar/manifest.schema.json`.
11. `synthetic-data-seminar/README.md`.
12. `tests/test_manifest.py`, followed separately by `tests/test_data_splits.py`.
13. `src/sensation_seminar/synthetic_data/scoring.py`, then `selection.py`, then `cli/make_subset.py`.
14. `src/sensation_seminar/datasets/real.py`, `synthetic.py`, `combined.py`, then `transforms.py` (four separate commits).
15. `configs/model/deeplabv3plus_mobilenet.yaml`, then `src/sensation_seminar/models/segmentation.py`.
16. `src/sensation_seminar/metrics/confusion.py`, `segmentation.py`, `safety.py`, then `tests/test_metrics.py`.
17. `src/sensation_seminar/training/reproducibility.py`, `checkpointing.py`, `engine.py`, then `src/sensation_seminar/cli/train.py`.
18. `src/sensation_seminar/evaluation/evaluate.py`, `predictions.py`, then `src/sensation_seminar/cli/evaluate.py`.
19. `src/sensation_seminar/results/schema.py`, `io.py`, `aggregate.py`, `src/sensation_seminar/cli/aggregate.py`, then `tests/test_results_schema.py`.
20. `configs/training/common.yaml`, then experiment configs `real_only.yaml`, `random_025.yaml`, and `random_050.yaml`.
21. `baseline/README.md`, `scripts/hpc/train.slurm`, `scripts/hpc/evaluate.slurm`, then `tests/test_smoke_pipeline.py`.
22. For `n = 1..9`, commit in order: `group-n/README.md`; `group-n/src/group_n/<method>.py` (`audit`, `perceptual_quality`, `feature_similarity`, `semantic_consistency`, `structural_consistency`, `quality_selection`, `diversity_selection`, `joint_selection`, `result_analysis`); `group-n/tests/test_group_n.py`; `group-n/results/README.md`.
23. `.gitlab-ci.yml` after the complete fast suite is locally green.

Generated `manifest.jsonl`, score/subset overlays, baseline runs, and group runs are then produced by instructors or students, not committed as implementation files. Baseline A/B25 (and then B50) must be completed and schema-validated before student-selected downstream experiments begin.

Phase 2 must not begin until this architecture is explicitly approved.
