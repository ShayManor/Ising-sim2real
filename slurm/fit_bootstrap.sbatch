#!/bin/bash
#SBATCH --job-name=boot-refit
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gautschi/manors/bootstrap/logs/refit_%A_%a.out
set -euo pipefail

# Joint-bootstrap refit driver. 40 draws x 14 patches = 560 tiles, split into
# NCHUNK contiguous chunks -- one array task per chunk. Each chunk runs its tiles
# PARALLEL-at-a-time (xargs -P) so a 96-core highmem job packs ~4 fits (each fit's
# process pool caps at ~24 workers). Index decode + resample live in
# scripts.fit_bootstrap_draw (tested); this script just fans indices to it.
#
# highmem-qpart QOS: max 8 submitted, cpu=384 -> 4 x 96-core jobs run concurrently
# (the whole array 0-3 at once), giving 16 concurrent fits, no GPU waste.
#
# RESUME=1 skips tiles whose JSON already exists (walltime-truncated re-submit).
#   Submit:  HF_HOME=/scratch/gautschi/manors/hf_cache sbatch slurm/fit_bootstrap.sbatch
#   Resume:  RESUME=1 HF_HOME=... sbatch slurm/fit_bootstrap.sbatch

REPO="${REPO:-$HOME/Ising-sim2real}"
cd "$REPO"
RESUME="${RESUME:-0}"
NTILES="${NTILES:-560}"
NCHUNK="${NCHUNK:-4}"
PARALLEL="${PARALLEL:-4}"          # concurrent fits per job (4 x ~24 cores = ~96)
mkdir -p /scratch/gautschi/manors/bootstrap/logs

# Force the absolute warm-cache path: ~/.bashrc sets HF_HOME=$SCRATCH/hf_cache,
# but $SCRATCH is UNSET in non-login sbatch/srun shells -> HF_HOME=/hf_cache
# (nonexistent) -> offline cache-miss. Do NOT use ${HF_HOME:-...} (the broken
# value is already set, so :- won't override it).
export HF_HOME=/scratch/gautschi/manors/hf_cache
# Fully offline: the fit-set data is already cached, and per-tile HF API calls
# (list_repo_files + etag HEADs) 429 at 16-way concurrency. Requires a warm cache
# + a pre-generated results/bootstrap/models/patches.json (see _patches); build
# both online once before submitting.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# One BLAS/OpenMP thread per process: PARALLEL fits x ~24 pool workers already
# saturate the 96 allocated cores, so threaded BLAS inside each worker would
# oversubscribe.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

per=$(( (NTILES + NCHUNK - 1) / NCHUNK ))
start=$(( SLURM_ARRAY_TASK_ID * per ))
end=$(( start + per )); [ "$end" -gt "$NTILES" ] && end="$NTILES"
[ "$start" -ge "$NTILES" ] && { echo "chunk $SLURM_ARRAY_TASK_ID: no tiles (start=$start >= $NTILES)"; exit 0; }

resume_flag=""; [ "$RESUME" = 1 ] && resume_flag="--resume"
echo "[$(date +%H:%M:%S)] chunk $SLURM_ARRAY_TASK_ID: tiles [$start,$end) PARALLEL=$PARALLEL RESUME=$RESUME"

seq "$start" "$((end - 1))" | xargs -P "$PARALLEL" -I{} bash -c '
  i="$1"
  echo "[$(date +%H:%M:%S)] tile $i start"
  if uv run python -m scripts.fit_bootstrap_draw --index "$i" '"$resume_flag"'; then
    echo "[$(date +%H:%M:%S)] tile $i done"
  else
    echo "[$(date +%H:%M:%S)] tile $i FAILED (rc=$?)"
  fi
' _ {}

echo "[$(date +%H:%M:%S)] chunk $SLURM_ARRAY_TASK_ID complete"
