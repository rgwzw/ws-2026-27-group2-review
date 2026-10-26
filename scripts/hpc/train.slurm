#!/usr/bin/env bash
#SBATCH --job-name=sensation-baseline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${SEMINAR_REPOSITORY:?Set SEMINAR_REPOSITORY to the repository checkout}"
: "${SEMINAR_EXPERIMENT:=real_only}"
: "${SEMINAR_SEED:=17}"
: "${SEMINAR_OUTPUT_ROOT:=baseline/results}"
: "${SEMINAR_CHECKPOINT_ROOT:=baseline/checkpoints}"

cd "$SEMINAR_REPOSITORY"
python -m sensation_seminar.cli.train \
  --experiment "$SEMINAR_EXPERIMENT" \
  --seed "$SEMINAR_SEED" \
  --output-root "$SEMINAR_OUTPUT_ROOT" \
  --checkpoint-root "$SEMINAR_CHECKPOINT_ROOT"
