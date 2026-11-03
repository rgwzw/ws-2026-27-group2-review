#!/bin/bash -l
#SBATCH --job-name=group2-quality
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=group-2/logs/score-quality-%j.out
#SBATCH --chdir=/home/woody/srsu/srsu119v/ws-2026-27

set -euo pipefail

module load python/3.12-conda
source .venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD/group-2/src"
export PYTHONDONTWRITEBYTECODE=1

python -m group_2.cli \
  --data-root datasets \
  --manifest synthetic-data-seminar/manifest.jsonl \
  --scores-output group-2/results/fallback-v1/scores.jsonl \
  --subset-output group-2/results/fallback-v1/perceptual_quality_budget_0.25x.jsonl

wc -l group-2/results/fallback-v1/scores.jsonl
wc -l group-2/results/fallback-v1/perceptual_quality_budget_0.25x.jsonl
