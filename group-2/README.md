# Group 2 — Perceptual Image Quality

## Status and research question

**Fallback first version, metric version `1`.** This implementation scores synthetic
images and selects a reproducible 548-image subset. It does not train or evaluate a
segmentation model. We investigate whether selecting images by perceptual quality
improves downstream segmentation at a fixed synthetic-data budget.

The second metric, metric agreement/calibration analysis, top/bottom figures and
downstream experiments are **not completed**. This first version is not evidence
that the selected images improve segmentation.

## Inputs and data paths

- A canonical JSONL manifest in shared schema `1.1`.
- Synthetic images referred to by that manifest, from all source groups.
- No real images or masks are read by this scorer or selector. Creating the shared
  canonical manifest is a separate prerequisite; it indexes the real split names.

`--data-root` means the directory **containing `synthetic-data/`**, default `datasets`.
For example, canonical `datasets/synthetic-data/group3/images/example.png` resolves
to `<data-root>/synthetic-data/group3/images/example.png`. Absolute canonical paths
containing `synthetic-data/` are relocated using that same suffix. The canonical
IDs and input manifest are never rewritten. Preserve the original internal paths;
there is no filename guessing. Manifest/output arguments are relative to the
current working directory unless absolute.

`source_group=group2` refers to a previous seminar's image source, not our team.
Our scorer uses every `eligibility.analysis=true` row, irrespective of source group.

## Method and selection

1. Load the canonical manifest with shared `load_manifest`. Reject duplicate IDs.
2. Read only analysis-eligible images, one at a time. Decode with Pillow
   `convert("RGB")`, retain uint8 intensity range 0–255, and calculate in float64.
   There is no resizing, cropping, EXIF rotation or intensity rescaling.
3. Compute grayscale as the unweighted arithmetic mean of RGB channels and use the
   repository's contrast-gradient formula:

   ```text
   raw = std(gray)
       + mean(abs(diff(gray, axis=0)))
       + mean(abs(diff(gray, axis=1)))
   ```

4. Min-max normalize across all successfully scored analysis rows to `[0, 1]`.
   Higher is better. A constant pool receives `1.0` for every image, matching the
   shared normalizer. Sort score rows by `synthetic_id` for reproducible output.
5. Keep only records with both eligibility flags true and a valid score. Sort by
   descending score, then ascending `synthetic_id`, and select **exactly 548**.
   No source-group quotas or random sampling are applied. The subset's `seed=42`
   follows the shared metadata convention; this algorithm has no random operation.
6. Validate scores using shared `validate_score_overlay`; write both JSONL files
   using shared `write_jsonl`. The 548-candidate check runs before either file is
   written. Too few scored training candidates cause a clear error (CLI exit 2),
   with no new outputs; previous output files, if any, remain untouched.

Missing, corrupt or smaller-than-2×2 images are skipped with a `SKIPPED` diagnostic
on stderr. They receive no score and cannot be selected. Non-finite/invalid array
fixtures are also rejected. Malformed score rows, unknown IDs and duplicate IDs
are errors. Outputs must be different from the input manifest and outside the
data root. Existing output files at the requested paths are replaced on success.

## Local setup and tests

Run commands from the repository root with Python 3.11 or newer. A small scoring
environment needs only NumPy and Pillow; tests additionally need pytest and
jsonschema. There is no requirement to install Torch or download model weights.

```bash
python3 -m venv group-2/.venv
group-2/.venv/bin/python -m pip install numpy Pillow pytest jsonschema
PYTHONDONTWRITEBYTECODE=1 group-2/.venv/bin/python -m pytest -q -p no:cacheprovider group-2/tests tests/test_scoring.py tests/test_selection.py tests/test_manifest.py
```

All test images/manifests/outputs are temporary **TEST ONLY** fixtures. The end-to-end
test generates 550 four-by-four PNGs to verify the actual 548 selection budget; it
does not use seminar data. Tests cover the formula and direction, deterministic
scores/selection, tie-breaking, invalid/missing images, eligibility filtering,
insufficient candidates, shared schema validation, overlay joins, repeatable CLI
output and compatibility with the shared synthetic dataset constructor.

There is no reduced-budget production option. A tiny image sample is useful for
development, but cannot produce a formal 548-row subset; use the fixture tests
above instead of presenting sample results as a complete seminar selection.

## Run when the canonical manifest and full candidate data are ready

```bash
PYTHONPATH=src:group-2/src group-2/.venv/bin/python -m group_2.cli \
  --data-root datasets \
  --manifest synthetic-data-seminar/manifest.jsonl \
  --scores-output group-2/results/fallback-v1/scores.jsonl \
  --subset-output group-2/results/fallback-v1/perceptual_quality_budget_0.25x.jsonl
```

Set `--data-root /absolute/path/to/datasets` to read a relocated local copy.
`PYTHONPATH` is necessary because the root package installer does not install
`group_2`; pytest adds this path automatically. Use `--help` to inspect the CLI.

This command performs scoring and selection only. Before treating the output as
a full-pool result, use a complete canonical manifest, confirm all expected images
are present, and inspect any skipped-file diagnostics. Merely having 548 readable
candidates does not establish that the full candidate pool was evaluated.

## Outputs and shared interfaces

| Output | Contract and content |
| --- | --- |
| `scores.jsonl` | Shared score schema `1.0`: `synthetic_id`, `method=perceptual_fallback`, `method_version=1` (string), `direction=higher_is_better`, normalized `score`, raw score in `components`, preprocessing/normalization in `provenance`. One row per successfully scored analysis image. |
| `perceptual_quality_budget_0.25x.jsonl` | Exactly 548 shared-style subset rows: schema `1.0`, ID, `selected=true`, rank 1–548, method, `budget=budget_0.25x`, seed and source group. Also records score/method/version and `manifest_sha256` (SHA-256 of the exact input manifest bytes). |

The canonical manifest remains the source of paths and supervision metadata. The
shared training code joins subset IDs back to it; no images are copied into the
subset. The canonical `manifest.schema.json` validates canonical rows, not these
score/subset rows. Scores use the shared Python validator; subsets follow the
shared selector's fields and the training consumer's ID join. No public code or
configuration was changed. A Group 2 experiment YAML is not supplied in this
version; whoever starts shared training must configure its canonical/subset paths.

Generated results live under `group-2/results/fallback-v1/`, which is ignored by the
existing repository rules. Fixture outputs are never committed as formal results.

## Limitations and later work

- Contrast and gradients measure detail, not human perceptual quality. Noise,
  excessive sharpening and artifacts can receive high scores. Brightness/contrast,
  native image size and source-group differences can bias the ranking.
- Scores depend on the normalization pool, so scores from a small sample and the
  full pool are not directly comparable. Ties use IDs, not source-image diversity.
- Eligibility and supervision metadata are trusted; this code does not re-audit
  real masks or prove synthetic/source geometry alignment. Full canonical JSON
  Schema validation is exercised in fixtures; runtime loading uses the shared
  loader's version checks rather than a full dataset audit.
- The scorer holds metadata and scalar scores in memory, but only one decoded image
  at a time. This first version has no score cache, resume or parallel workers.
- The two-person extension still needs a second justified metric, calibration and
  agreement analysis. SSIM/LPIPS are optional and require justified alignment and
  paired real training images.
- The final project protocol compares `perceptual_quality_budget_0.25x` with
  instructor-provided `real_only` and the exact `random_budget_0.25x`, using training
  seeds 17, 42 and 73. Those downstream runs and result bundles are separate work;
  this local first version does not execute them.
