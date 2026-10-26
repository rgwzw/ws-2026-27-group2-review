# Group 4 — Semantic Consistency

## Group Goal
Quantify preservation of road-scene classes between source and synthetic images.
## Research Question
Does teacher-prediction consistency, especially road/sidewalk consistency, predict utility?
## Why This Matters
Unsafe semantic changes can make visually plausible augmentation harmful.
## Input Data
Required: manifest pairs and predictions from one frozen segmentation teacher. Teacher training must not use validation/test data.
## Required Implementation
Compute global and class-wise agreement or mIoU, expose road/sidewalk penalties, provenance, and qualitative change maps.
## Step-by-Step Work Plan
Freeze teacher; predict paired images; align only with justified transforms; calculate metrics; rank eligible records; inspect changes; benchmark 548 samples.
## Required Experiment
Compare `semantic_consistency_budget_0.25x` with `real_only` and exact `random_budget_0.25x` across three seeds.
Students train only the group method for seeds 17, 42, and 73; instructors provide both shared baselines unless retraining is explicitly requested.
## Expected Results
Consistency overlay, class CSV, change maps, subset JSONL, and standard result bundles.
## Required Tests
Known masks/logits with class swaps must produce exact global and road/sidewalk penalties; mismatched shapes fail.
## One-Student Scope
One teacher, one consistency score, and fixed-budget benchmark.
## Two-Student Extension
Add confidence-aware or safety-weighted ablations.
## Definition of Done
Teacher provenance and split safety are documented; tests and results validate.
## Optional Dependencies
Group 3 similarity is an optional comparison.
## Fallback
Use locally generated frozen-teacher predictions; no other overlay is needed.
