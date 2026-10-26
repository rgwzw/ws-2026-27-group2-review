# Group 2 — Perceptual Image Quality

## Group Goal
Assign reproducible visual-quality scores to analysis-eligible synthetic images.
## Research Question
Does visual quality predict segmentation utility at fixed training budget?
## Why This Matters
Plausible images can still contain distortions, but visual quality is a hypothesis—not a synonym for downstream utility.
## Input Data
Required: canonical manifest and synthetic images. Paired source images are needed only for aligned full-reference metrics.
## Required Implementation
Implement BRISQUE/NIQE or the shipped contrast-gradient fallback, normalize higher-is-better, record model/preprocessing provenance, and use SSIM/LPIPS only when geometry justifies it.
## Step-by-Step Work Plan
Filter analysis rows; score deterministically; calibrate direction; inspect top/bottom examples; select the best 548 training-eligible records; run the benchmark.
## Required Experiment
Compare `perceptual_quality_budget_0.25x` against `real_only` and the exact `random_budget_0.25x` for seeds 17, 42, and 73.
## Expected Results
Score JSONL, agreement CSV, examples/figures, subset JSONL, and standard result bundles.
## Required Tests
Check deterministic fixtures, normalization direction, invalid images, and score schema.
## One-Student Scope
One reproducible no-reference metric and one fixed-budget experiment.
## Two-Student Extension
Add another metric, calibration, and metric-agreement/downstream correlation analysis.
## Definition of Done
Scores carry provenance; selection is exact/reproducible; tests and result validation pass.
## Optional Dependencies
Group 1 exclusions may refine analysis; paired metrics are optional.
## Fallback
Use manifest eligibility and the dependency-light local fallback in `perceptual_quality.py`.
