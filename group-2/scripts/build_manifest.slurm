#!/bin/bash -l
#SBATCH --job-name=group2-manifest
#SBATCH --partition=work
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=group-2/logs/build-manifest-%j.out
#SBATCH --chdir=/home/woody/srsu/srsu119v/ws-2026-27

set -euo pipefail

module load python/3.12-conda
source .venv/bin/activate

export PYTHONPATH="$PWD/src:$PWD/group-2/src"
export PYTHONDONTWRITEBYTECODE=1

python -m sensation_seminar.cli.build_manifest \
  --data-root datasets \
  --output synthetic-data-seminar/manifest.jsonl

wc -l synthetic-data-seminar/manifest.jsonl
