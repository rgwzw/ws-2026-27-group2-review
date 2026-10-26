#!/usr/bin/env bash
#SBATCH --job-name=sensation-evaluate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SEMINAR_REPOSITORY:?Set SEMINAR_REPOSITORY to the repository checkout}"
: "${SEMINAR_EXPERIMENT:?Set SEMINAR_EXPERIMENT}"
: "${SEMINAR_SEED:?Set SEMINAR_SEED}"
: "${SEMINAR_CHECKPOINT:?Set SEMINAR_CHECKPOINT}"
: "${SEMINAR_SPLIT:=testing}"
: "${SEMINAR_OUTPUT_ROOT:=baseline/results}"

cd "$SEMINAR_REPOSITORY"
python -m sensation_seminar.cli.evaluate \
  --experiment "$SEMINAR_EXPERIMENT" \
  --seed "$SEMINAR_SEED" \
  --checkpoint "$SEMINAR_CHECKPOINT" \
  --split "$SEMINAR_SPLIT" \
  --output-root "$SEMINAR_OUTPUT_ROOT"
