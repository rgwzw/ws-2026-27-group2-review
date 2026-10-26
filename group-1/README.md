# Group 1 — Synthetic Data Audit and Characterization

## Group Goal
Characterize every synthetic record without changing source data.
## Research Question
Which defects, coverage gaps, dimensions, duplicates, and supervision geometries occur?
## Why This Matters
The paper needs a defensible population and explicit exclusions before scoring or training.
## Input Data
Required: canonical manifest and referenced synthetic images. Shared manifest/validation code is authoritative. No other group is required.
## Required Implementation
Extend `audit.py` with coverage, dimension, hash/duplicate, artifact, mapping, and `supervision.status` reports. Never promote uncertain supervision to training eligibility.
## Step-by-Step Work Plan
Validate the manifest; summarize reasons, sources, sizes and supervision; inspect outliers; export tables and figures; create an exclusion-based 548-record subset from training-eligible rows.
## Required Experiment
Compare `audit_budget_0.25x` with instructor `real_only` and the exact `random_budget_0.25x` control for seeds 17, 42, and 73.
## Expected Results
`audit.json`, coverage CSVs, exclusion overlay, subset JSONL, figures, and standard segmentation result bundles.
## Required Tests
Test flat/nested paths, artifacts, unmatched names, exact reason counts, and deterministic output on tiny fixtures.
## One-Student Scope
Validity, dimensions, mappings, supervision and coverage plus the fixed-budget experiment.
## Two-Student Extension
Add perceptual duplicate detection and distribution diagnostics.
## Definition of Done
The group test passes; every exclusion is explained; subset size is exact; results validate.
## Optional Dependencies
Optional score overlays may be used only for diagnostics.
## Fallback
Use the canonical manifest and local image metadata; no group output is necessary.
