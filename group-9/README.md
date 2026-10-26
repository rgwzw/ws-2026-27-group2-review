# Group 9 — Downstream and Safety-Oriented Evaluation

## Group Goal
Compare available methods fairly and report safety-critical behavior.
## Research Question
Which completed strategies improve real-scene performance reliably across seeds?
## Why This Matters
Common aggregation separates robust downstream evidence from attractive proxy scores.
## Input Data
Required: valid baseline result bundles. Any non-empty subset of group result bundles is optional; raw submitted metrics are immutable.
## Required Implementation
Validate schemas/config/budgets/splits, aggregate seeds, compute paired deltas/uncertainty, create common tables/plots, and flag leakage or incomparability.
## Step-by-Step Work Plan
Discover bundles; validate; group comparable runs; aggregate overall/per-class/safety metrics; compute deltas; inspect failures; build reproducibility matrix.
## Required Experiment
Confirm nominated quality, diversity and joint methods over seeds 17, 42 and 73 when those results exist; otherwise report baselines and available methods.
Students do not retrain instructor `real_only` or `random_budget_0.25x` baselines unless explicitly requested.
## Expected Results
`comparison.csv`, `aggregate.json`, paper figures, failure analysis, and reproducibility matrix.
## Required Tests
Miniature bundles must aggregate to known values and reject leakage, malformed matrices, mismatched budgets/configs or required-seed claims.
## One-Student Scope
Schema validation, flexible baseline comparison, tables and safety metrics.
## Two-Student Extension
Confidence intervals, effect sizes, subgroup analysis, and best-method replication.
## Definition of Done
Baseline-only mode works; available results aggregate unchanged; validation/test conclusions are separated; tests pass.
## Optional Dependencies
Any completed standard result bundle from Groups 1–8 may be included.
## Fallback
Aggregate instructor baseline bundles alone and label absent comparisons explicitly.
