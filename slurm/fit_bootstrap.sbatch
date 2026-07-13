#!/bin/bash
#SBATCH --job-name=boot-refit
#SBATCH --partition=smallgpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-559%16
#SBATCH --output=/scratch/gautschi/manors/bootstrap/logs/refit_%A_%a.out
set -euo pipefail

# One (draw, patch) tile per array task. 40 draws x 14 patches = 560 tiles.
# Index decode lives in scripts.fit_bootstrap_draw.tile (tested); this script
# only forwards $SLURM_ARRAY_TASK_ID. RESUME=1 skips tiles whose JSON exists,
# so a walltime-truncated array re-submits for just the missing tiles.
#
# Prereq: none beyond the HF dataset. Output JSONs are tiny; rsync
# results/bootstrap/models back to the laptop after the array completes.
#   Submit:  sbatch slurm/fit_bootstrap.sbatch
#   Resume:  RESUME=1 sbatch slurm/fit_bootstrap.sbatch

REPO="${REPO:-$HOME/Ising-sim2real}"
RESUME="${RESUME:-0}"
cd "$REPO"
mkdir -p /scratch/gautschi/manors/bootstrap/logs

export HF_HOME="${HF_HOME:-/scratch/gautschi/manors/bootstrap/hf_cache}"

args=(--index "$SLURM_ARRAY_TASK_ID")
[ "$RESUME" = 1 ] && args+=(--resume)

echo "[$(date +%H:%M:%S)] tile $SLURM_ARRAY_TASK_ID start (RESUME=$RESUME)"
uv run python -m scripts.fit_bootstrap_draw "${args[@]}"
echo "[$(date +%H:%M:%S)] tile $SLURM_ARRAY_TASK_ID done"
