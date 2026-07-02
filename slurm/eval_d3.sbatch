#!/bin/bash -l
#SBATCH --job-name=ising_eval_d3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
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
#   speed    : (distance, basis) groups run in parallel, one pinned per GPU
#              (CUDA_VISIBLE_DEVICES), so both L40S are used. A device preflight
#              prints use_gpu/gpus up front -- no silent CPU fallback eating the
#              wall clock. --device cuda hard-fails if no GPU is visible.
#
# Timeout-safe: work is sharded by (distance, basis) and each shard writes its own
# CSV the moment it finishes, so a 12h wall never loses completed shards. Re-submit
# the same script to resume -- finished shards are skipped.
#
# Prereqs (run once on a login node, with internet):
#   uv sync                              # cu124 torch build + deps into .venv
#   uv run python scripts/setup_ising.py # vendored NVIDIA Ising code + weights
#                                        # (needs git-lfs; the sbatch retries this
#                                        #  in-job but git-lfs may be login-only)
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
# Repo root: walk up from the submit dir (or $PWD) to the pyproject.toml, so this
# works whether sbatch is submitted from the repo root or from slurm/. Override REPO=.
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [ "$REPO" != "/" ] && [ ! -e "$REPO/pyproject.toml" ]; do REPO="$(dirname "$REPO")"; done
[ -e "$REPO/pyproject.toml" ] || { echo "ERROR: repo root (pyproject.toml) not found above ${SLURM_SUBMIT_DIR:-$PWD}; set REPO=/path/to/Ising-sim2real"; exit 1; }
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/ising_eval}"     # results land here (NOT in git)
SHOTS="${SHOTS:-}"                                           # empty = all 50k; set e.g. 5000 for a quick pass
MODELS="${MODELS:-fast accurate}"                           # ising model variants to sweep
DEVICE="${DEVICE:-auto}"                                     # auto -> CUDA (cu124 build on Linux), else CPU
ISING_DEVICE="${ISING_DEVICE:-cpu}"                          # Ising ALWAYS on CPU: the CUDA forward corrupts at
                                                            # shard scale (correct in isolation, chance-level across
                                                            # a full shard -- see slurm/smoke_test.slurm). Classical
                                                            # (pymatching) is CPU regardless of DEVICE.
# Ising round cap: the CPU forward cost grows ~linearly with rounds and a high-round
# Ising total LER saturates to ~0.5 (per-cycle is the useful metric, best estimated
# at low/moderate rounds), so capping keeps the signal while making the sweep fit in
# 12h. Classical decoders are cheap (pymatching) and run at ALL rounds. Set empty to
# run Ising at every round (much slower -- time it with slurm/smoke_test.slurm first).
ISING_MAX_ROUNDS="${ISING_MAX_ROUNDS:-70}"
NPAR="${NPAR:-4}"                                            # concurrent CPU Ising shards (cores split across them)
DISTANCES="${DISTANCES:-3}"
BASES="${BASES:-X Z}"
SOURCE="${SOURCE:-auto}"                                     # auto = local Willow tree if present, else HF
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
PUSH_RESULTS="${PUSH_RESULTS:-1}"                           # 1 = push CSVs to git after the run; 0 = off
RESULTS_DIR="${RESULTS_DIR:-results}"                      # in-repo dir the CSVs are committed to
RESUME="${RESUME:-0}"                                       # 0 = always re-run (default); 1 = skip shards whose CSV exists
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

# --- Ising decoder setup: vendored NVIDIA code + weights are gitignored, so a fresh
# checkout has neither. The ising shards need both. Set up once if missing; the
# NVIDIA HF weights are GATED, so this uses the git-lfs path (GitHub, public) and
# fails fast with a login-node instruction rather than crashing every ising shard.
have_ising_weights () { ls models/ising/*.pt >/dev/null 2>&1 || ls models/ising/*.safetensors >/dev/null 2>&1; }
if [ -n "$MODELS" ] && { [ ! -d third_party/Ising-Decoding/code ] || ! have_ising_weights; }; then
  echo "setting up vendored Ising decoder (scripts/setup_ising.py) ..."
  if ! uv run python scripts/setup_ising.py; then
    echo "ERROR: Ising setup failed (needs git-lfs + internet on this node)."
    echo "Run it once on a LOGIN node and resubmit:  uv run python scripts/setup_ising.py"
    echo "(or set MODELS='' to run the classical baselines only)"
    exit 1
  fi
fi

WALL0=$(date +%s)
fmt () { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

# --- device preflight: confirm what we'll actually run on (no silent CPU fall) ---
GPU_COUNT=$(uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
case "$DEVICE" in
  cpu) USE_GPU=0 ;;
  *)   # auto / cuda*
    if [ "${GPU_COUNT:-0}" -gt 0 ]; then USE_GPU=1
    elif [ "$DEVICE" = auto ]; then echo "WARNING: no CUDA visible -> running on CPU"; USE_GPU=0
    else echo "ERROR: --device '$DEVICE' but torch sees 0 CUDA devices"; exit 1; fi ;;
esac
WORKERS=$([ "$USE_GPU" = 1 ] && echo "$GPU_COUNT" || echo 1)   # one group per GPU, else serial

echo "device=$DEVICE use_gpu=$USE_GPU gpus=$GPU_COUNT workers=$WORKERS source=$SOURCE shots=${SHOTS:-all} models='$MODELS'"
echo "plan: ${nd:=$(set -- $DISTANCES; echo $#)}d x ${nb:=$(set -- $BASES; echo $#)}b groups; per-config bar streams to the .err log"

run_shard () {  # $1=device $2=tag $3=out.csv  $4..=ising-eval args
  local dev="$1" tag="$2" out="$3"; shift 3
  if [ "${RESUME:-0}" = 1 ] && [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] [$tag] skip (resume): $(basename "$out")"; return; fi
  local t0; t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [$tag] start:    $(basename "$out")"
  uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" \
    --device "$dev" "${shots_flag[@]}" --out "$out" "$@"
  echo "[$(date +%H:%M:%S)] [$tag] finished: $(basename "$out") in $(fmt $(( $(date +%s) - t0 )))"
}

# --- classical shards (pymatching, CPU, fast): all rounds, run serially --------
for d in $DISTANCES; do for b in $BASES; do
  run_shard "$DEVICE" "cls d$d$b" "$OUTDIR/classical_d${d}_${b}.csv" --decoders mwpm,mwpm-rl --distances "$d" --bases "$b"
done; done

# --- ising shards (CPU): run NPAR at once, each capped to CORES/NPAR threads ----
# One torch process per (distance,basis,model). Running NPAR concurrently with a
# per-process thread cap keeps all cores busy without oversubscription -- the lever
# that fits the sweep in the wall clock. Ising rounds are capped by ISING_MAX_ROUNDS.
CORES="${SLURM_CPUS_PER_TASK:-96}"
ising_jobs=(); for d in $DISTANCES; do for b in $BASES; do for m in $MODELS; do ising_jobs+=("$d:$b:$m"); done; done; done
njobs=${#ising_jobs[@]}
[ "$NPAR" -gt "$njobs" ] && NPAR="$njobs"; [ "${NPAR:-0}" -lt 1 ] && NPAR=1
ITHREADS=$(( CORES / NPAR )); [ "$ITHREADS" -lt 1 ] && ITHREADS=1
mr_flag=(); [ -n "$ISING_MAX_ROUNDS" ] && mr_flag=(--max-rounds "$ISING_MAX_ROUNDS")
echo "ising: $njobs shards, $NPAR concurrent x $ITHREADS threads on $CORES cores; max-rounds=${ISING_MAX_ROUNDS:-all}"

ising_shard () {  # $1="d:b:m" -- subshell so OMP_NUM_THREADS is per-shard
  local d b m; IFS=: read -r d b m <<< "$1"
  export OMP_NUM_THREADS="$ITHREADS" MKL_NUM_THREADS="$ITHREADS"
  run_shard "$ISING_DEVICE" "isg d$d$b/$m" "$OUTDIR/ising_${m}_d${d}_${b}.csv" \
    --decoders ising --model "$m" --distances "$d" --bases "$b" "${mr_flag[@]}"
}

# Launch in waves of NPAR (portable across bash versions; no `wait -n` needed).
for (( i=0; i<njobs; i+=NPAR )); do
  for (( k=0; k<NPAR && i+k<njobs; k++ )); do ising_shard "${ising_jobs[i+k]}" & done
  wait   # barrier: finish this wave before the next
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

# --- commit + push the CSVs to the current branch so they can be pulled elsewhere ---
# Opt out with PUSH_RESULTS=0. CSVs land in <repo>/results/ on the current branch;
# pull locally with `git pull`. Non-fatal: a push hiccup never fails the eval job.
if [ "${PUSH_RESULTS:-1}" = 1 ]; then
  bash scripts/push_results.sh "$REPO" "$OUTDIR" "${RESULTS_DIR:-results}" || true
fi

echo "[$(date +%H:%M:%S)] ALL DONE: ${njobs:-0} ising shards in $(fmt $(( $(date +%s) - WALL0 )))"
