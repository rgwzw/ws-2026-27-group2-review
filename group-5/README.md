# Group 5 — Structural Consistency

## Group Goal
Measure preservation of boundaries, layout, and scene geometry.
## Research Question
Does structural preservation add predictive value beyond semantic identity?
## Why This Matters
Segmentation depends on spatial class boundaries, not merely class presence.
## Input Data
Required: canonical pairs, supervision metadata, and locally derived edges/label maps. Pixel comparison requires verified alignment.
## Required Implementation
Implement a robust boundary/layout score, refuse unjustified pixel alignment, document components, and generate qualitative overlays.
## Step-by-Step Work Plan
Partition by supervision; derive structures; score verified pairs; use non-pixel summaries for others; rank eligible rows; select 548; benchmark.
## Required Experiment
Compare `structural_consistency_budget_0.25x` with `real_only` and exact `random_budget_0.25x` for all required seeds.
## Expected Results
Score/component JSONL, ablation CSV, overlays, subset JSONL, and result bundles.
## Required Tests
Identical geometry must outrank translated/deformed fixtures; uncertain supervision must be rejected for pixel comparison.
## One-Student Scope
One boundary/layout metric and fixed-budget benchmark.
## Two-Student Extension
Add multi-scale or learned depth/geometry proxies and ablations.
## Definition of Done
Alignment assumptions are explicit; exact subset and validated results exist; tests pass.
## Optional Dependencies
Group 4 outputs may support comparisons.
## Fallback
Compute local edges or label boundaries; do not depend on another group.
