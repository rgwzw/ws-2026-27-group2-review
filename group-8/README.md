# Group 8 — Joint Quality-Diversity Selection

## Group Goal
Balance one quality signal with feature-space coverage.
## Research Question
Can a quality/diversity trade-off outperform either objective alone?
## Why This Matters
Selection should avoid both harmful samples and redundant subsets.
## Input Data
Required: training-eligible manifest rows, one compatible higher-is-better quality score, and one embedding per candidate.
## Required Implementation
Implement a documented multi-objective rule or quality-constrained coverage, stable ties, missing-data policy, and operating-point selection using training-side evidence only.
## Step-by-Step Work Plan
Validate/join inputs; normalize; sweep trade-off; measure quality/coverage; freeze one 548-record operating point; run controls and three seeds.
## Required Experiment
Compare `joint_budget_0.25x` with `real_only`, exact random, and quality/diversity controls when available.
Students train only the group method for seeds 17, 42, and 73; instructors provide both shared baselines unless retraining is explicitly requested.
## Expected Results
Joint subsets, trade-off CSV/curve, ablations, chosen-point rationale, and standard result bundles.
## Required Tests
Toy inputs must meet budget, quality/coverage expectations, schema and deterministic ties.
## One-Student Scope
One simple quality signal, one embedding, one fixed trade-off and benchmark.
## Two-Student Extension
Pareto/sensitivity analysis and secondary budget.
## Definition of Done
No hard Group 6/7 import; exact subset; operating point is leakage-free; tests/results validate.
## Optional Dependencies
Compatible outputs from Groups 2–7 can replace local inputs.
## Fallback
Compute Group 2’s simple fallback score and local frozen embeddings, then call `select_joint`.
