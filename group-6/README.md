# Group 6 — Quality-Based Selection

## Group Goal
Select a fixed-budget training subset from any available quality evidence.
## Research Question
Which transparent quality combination improves over random selection?
## Why This Matters
The paper needs a reproducible decision rule, not independent rankings alone.
## Input Data
Required: canonical manifest and at least one local or optional standard score overlay. Training eligibility is mandatory.
## Required Implementation
Validate joins, normalize higher-is-better, document weights/missing policy, use stable ID tie-breaking, and return exactly 548 unique rows.
## Step-by-Step Work Plan
Discover overlays; validate provenance; normalize/combine; ablate weights; select; freeze subset; run three seeds.
## Required Experiment
Compare `quality_budget_0.25x` with `real_only` and exact `random_budget_0.25x`; compare individual signals when available.
Students train only the group method for seeds 17, 42, and 73; instructors provide both shared baselines unless retraining is explicitly requested.
## Expected Results
Versioned selector, weight/threshold CSV, subset JSONL, figures, and standard result bundles.
## Required Tests
Test missing overlays, joins, ties, ineligible rows, budget failure, and determinism.
## One-Student Scope
One signal, documented fallback, exact top-548 selector and benchmark.
## Two-Student Extension
Multiple signals, robust weights or training-only learned combination.
## Definition of Done
Any non-empty signal set works; selection is exact; no validation tuning; tests/results validate.
## Optional Dependencies
Any subset of Groups 1–5 overlays may be used.
## Fallback
Use verified-supervision status as shipped transparent signal when no overlay exists.
