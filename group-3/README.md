# Group 3 — Feature-Based Semantic Similarity

## Group Goal
Measure semantic similarity between each synthetic image and its real training source.
## Research Question
Does frozen-feature similarity identify more useful augmentation samples?
## Why This Matters
It tests content preservation beyond low-level visual quality.
## Input Data
Required: paired manifest paths and one frozen DINOv2 or CLIP embedding provider; only real training sources are used.
## Required Implementation
Batch and cache embeddings with checksummed model/preprocessing metadata, compute cosine similarity, normalize it, and rank training-eligible rows.
## Step-by-Step Work Plan
Resolve pairs; embed each unique image; validate cache keys; compute scores; inspect distributions; select 548; train/evaluate three seeds.
## Required Experiment
Compare `feature_similarity_budget_0.25x` with instructor `real_only` and exact `random_budget_0.25x`.
## Expected Results
Embedding metadata/cache reference, score JSONL, plots, subset JSONL, and standard result bundles.
## Required Tests
Mocked vectors must yield known cosine values, stable ordering, invalid-vector handling, and cache-key checks.
## One-Student Scope
One backbone, one pooling rule, and the fixed-budget benchmark.
## Two-Student Extension
Compare a second backbone or pooling/layer choice.
## Definition of Done
Frozen model identity is recorded; no validation/test leakage; tests and result validation pass.
## Optional Dependencies
Group 2 scores may be compared, never required.
## Fallback
Accept caller-supplied cached embeddings and use `cosine_similarity_scores` independently.
