#!/bin/bash -l
#SBATCH --job-name=ising_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/ising%j.out
#SBATCH -e /scratch/gautschi/manors/logs/ising%j.err
set -euo pipefail

#   decoders : mwpm (SI1000 DEM baseline), mwpm-rl (RL DEM baseline),
#              ising (NVIDIA pre-decoder + MWPM) with BOTH fast and accurate models
#   coverage : all 14 patches x {X,Z} x all 15 round counts = 420 configs
#   shots    : all 50,000 per config (override with SHOTS=N for a quick pass)
#   metric   : per-config LER + LER per cycle + decode latency -> CSV
#   data     : pulled from the HF dataset (SOURCE=hf) -- detection events,
#              circuits, and shipped DEMs -- so NO 12 GB local Willow tree is
#              needed. Set SOURCE=local to read a local tree instead.
#
# Timeout-safe: work is sharded by (distance, basis) and each shard writes its own
# CSV the moment it finishes, so a 12h wall never loses completed shards. Re-submit
# the same script to resume -- finished shards are skipped.
#
# Prereqs (run once on a login node, with internet):
#   uv sync          # installs the cu124 torch build + deps into .venv
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
REPO="${SLURM_SUBMIT_DIR:-$PWD}"                             # repo root (submit from here)
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/ising_eval}"     # results land here (NOT in git)
SHOTS="${SHOTS:-}"                                           # empty = all 50k; set e.g. 5000 for a quick pass
MODELS="${MODELS:-fast accurate}"                           # ising model variants to sweep
DEVICE="${DEVICE:-auto}"                                     # auto -> CUDA (cu124 build on Linux), else CPU
DISTANCES="${DISTANCES:-3 5 7}"
BASES="${BASES:-X Z}"
SOURCE="${SOURCE:-auto}"                                     # auto = local Willow tree if present, else HF
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
WILLOW_RAW_DIR="${WILLOW_RAW_DIR:-$REPO/google_105Q_surface_code_d3_d5_d7}"
export WILLOW_RAW_DIR

# Torch (cu124 on Linux) uses the GPU; let numpy/pymatching use the CPU allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
# Cache HF downloads (parquet + circuits + DEMs) on scratch, not $HOME.
export HF_HOME="${HF_HOME:-$OUTDIR/hf_cache}"

cd "$REPO"
mkdir -p "$OUTDIR"
shots_flag=(); [ -n "$SHOTS" ] && shots_flag=(--shots "$SHOTS")

# --- pick source: use the local Willow tree if it's there, else pull from HF ---
if [ "$SOURCE" = "auto" ]; then
  if [ -d "$WILLOW_RAW_DIR" ] && [ -n "$(ls -A "$WILLOW_RAW_DIR" 2>/dev/null)" ]; then
    SOURCE=local
  else
    SOURCE=hf
  fi
fi

# Prefetch the public dataset once (idempotent; already-cached files are skipped).
# No token needed -- the repo is public, so download anonymously even if a stale
# token is present. This is what makes the run work on nodes without the local data.
if [ "$SOURCE" = "hf" ]; then
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  echo "pulling dataset $HF_REPO into $HF_HOME (public, no token) ..."
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], repo_type="dataset", token=False)
print("dataset ready at", path)
PY
fi

# --- progress tracking -------------------------------------------------------
# TOTAL shards = (#distances * #bases) * (1 classical + #ising-models).
nd=$(set -- $DISTANCES; echo $#); nb=$(set -- $BASES; echo $#); nm=$(set -- $MODELS; echo $#)
TOTAL=$(( nd * nb * (1 + nm) ))
DONE=0; RUN_N=0; RUN_T=0; WALL0=$(date +%s)
fmt () { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

eta () {  # print elapsed + ETA based on the mean time of shards that actually ran
  local el=$(( $(date +%s) - WALL0 )) rem="?"
  [ "$RUN_N" -gt 0 ] && rem=$(fmt $(( (RUN_T / RUN_N) * (TOTAL - DONE) )))
  echo "    progress $DONE/$TOTAL shards | elapsed $(fmt $el) | est. remaining $rem"
}

echo "repo=$REPO outdir=$OUTDIR device=$DEVICE shots=${SHOTS:-all} models='$MODELS'"
echo "source=$SOURCE hf_repo=$HF_REPO hf_home=$HF_HOME threads=$OMP_NUM_THREADS"
echo "plan: $TOTAL shards = ${nd}d x ${nb}b x (1 classical + ${nm} ising); per-config bar streams to the .err log"

run_shard () {  # $1=out.csv  $2..=ising-eval args
  local out="$1"; shift
  DONE=$(( DONE + 1 ))
  if [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] [$DONE/$TOTAL] skip (done): $(basename "$out")"; eta; return; fi
  echo "[$(date +%H:%M:%S)] [$DONE/$TOTAL] start:     $(basename "$out")"
  local t0; t0=$(date +%s)
  uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" \
    --device "$DEVICE" "${shots_flag[@]}" --out "$out" "$@"
  local dt=$(( $(date +%s) - t0 ))
  RUN_N=$(( RUN_N + 1 )); RUN_T=$(( RUN_T + dt ))
  echo "[$(date +%H:%M:%S)] [$DONE/$TOTAL] finished:  $(basename "$out") in $(fmt $dt)"
  eta
}

# --- pass 1: classical baselines + ising models, sharded by (distance, basis) ---
for d in $DISTANCES; do
  for b in $BASES; do
    run_shard "$OUTDIR/classical_d${d}_${b}.csv" \
      --decoders mwpm,mwpm-rl --distances "$d" --bases "$b"
    for m in $MODELS; do
      run_shard "$OUTDIR/ising_${m}_d${d}_${b}.csv" \
        --decoders ising --model "$m" --distances "$d" --bases "$b"
    done
  done
done

# --- merge every shard CSV into one results file (single header) -------------
merged="$OUTDIR/eval_all.csv"
first=1
: > "$merged"
for f in "$OUTDIR"/classical_*.csv "$OUTDIR"/ising_*.csv; do
  [ -s "$f" ] || continue
  if [ "$first" = 1 ]; then cat "$f" > "$merged"; first=0; else tail -n +2 "$f" >> "$merged"; fi
done
echo "merged $(($(wc -l < "$merged") - 1)) rows -> $merged"
echo "[$(date +%H:%M:%S)] ALL DONE: $DONE/$TOTAL shards in $(fmt $(( $(date +%s) - WALL0 )))"
