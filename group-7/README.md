# Group 7 — Diversity-Aware Selection

## Group Goal
Cover synthetic feature space at a fixed training budget.
## Research Question
Does representative feature-space coverage outperform random sampling?
## Why This Matters
Redundant high-quality images may contribute less than complementary examples.
## Input Data
Required: training-eligible manifest records and locally cached frozen-model embeddings.
## Required Implementation
Implement deterministic clustering, k-center, or shipped farthest-first selection with stable ties and coverage diagnostics.
## Step-by-Step Work Plan
Embed eligible rows; validate cache/dimensions; select representatives; measure coverage/source balance; freeze 548 IDs; benchmark.
## Required Experiment
Compare `diversity_budget_0.25x` with `real_only` and exact `random_budget_0.25x` across required seeds.
## Expected Results
Embedding provenance, diversity annotations, cluster/coverage CSV and plots, subset JSONL, result bundles.
## Required Tests
Toy clusters must select expected representatives with exact budget and deterministic ties.
## One-Student Scope
One embedding and one deterministic coverage method.
## Two-Student Extension
Alternative algorithm/backbone and secondary 0.50x budget.
## Definition of Done
Cache is reproducible; selection exact; diagnostics and validated results exist; tests pass.
## Optional Dependencies
Quality overlays may be analyzed but are unnecessary.
## Fallback
Generate frozen embeddings locally and run `select_diverse` without other group outputs.
