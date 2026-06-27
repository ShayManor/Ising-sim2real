#!/bin/bash -l
#SBATCH --job-name=ising_eval_d7
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
REPO="${SLURM_SUBMIT_DIR:-$PWD}"                            # repo root (submit from here)
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/ising_eval}"     # results land here (NOT in git)
SHOTS="${SHOTS:-}"                                           # empty = all 50k; set e.g. 5000 for a quick pass
MODELS="${MODELS:-fast accurate}"                           # ising model variants to sweep
DEVICE="${DEVICE:-auto}"                                     # auto -> CUDA (cu124 build on Linux), else CPU
DISTANCES="${DISTANCES:-7}"
BASES="${BASES:-X Z}"
SOURCE="${SOURCE:-auto}"                                     # auto = local Willow tree if present, else HF
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
PUSH_RESULTS="${PUSH_RESULTS:-1}"                           # 1 = push CSVs to git after the run; 0 = off
RESULTS_BRANCH="${RESULTS_BRANCH:-eval-results}"           # git branch the CSVs are pushed to
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
  if [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] [$tag] skip (done): $(basename "$out")"; return; fi
  local t0; t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [$tag] start:    $(basename "$out")"
  uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" \
    --device "$dev" "${shots_flag[@]}" --out "$out" "$@"
  echo "[$(date +%H:%M:%S)] [$tag] finished: $(basename "$out") in $(fmt $(( $(date +%s) - t0 )))"
}

run_group () {  # $1=distance $2=basis $3=gpu-slot -- classical + ising shards for one (d,b)
  local d="$1" b="$2" slot="$3" dev tag g0; g0=$(date +%s)
  if [ "$USE_GPU" = 1 ]; then export CUDA_VISIBLE_DEVICES="$slot"; dev=cuda; tag="gpu$slot d$d$b"; else dev="$DEVICE"; tag="cpu d$d$b"; fi
  echo "[$(date +%H:%M:%S)] [$tag] group start"
  run_shard "$dev" "$tag" "$OUTDIR/classical_d${d}_${b}.csv" --decoders mwpm,mwpm-rl --distances "$d" --bases "$b"
  for m in $MODELS; do
    run_shard "$dev" "$tag" "$OUTDIR/ising_${m}_d${d}_${b}.csv" --decoders ising --model "$m" --distances "$d" --bases "$b"
  done
  echo "[$(date +%H:%M:%S)] [$tag] group DONE in $(fmt $(( $(date +%s) - g0 )))"
}

# --- run (distance, basis) groups in GPU-pinned waves (one group per GPU) -----
groups=(); for d in $DISTANCES; do for b in $BASES; do groups+=("$d:$b"); done; done
n=${#groups[@]}
for (( s=0; s<n; s+=WORKERS )); do
  for (( k=0; k<WORKERS && s+k<n; k++ )); do
    IFS=: read -r gd gb <<< "${groups[s+k]}"
    run_group "$gd" "$gb" "$k" &
  done
  wait   # barrier: finish this wave (and surface any failure) before the next
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

# --- push the CSVs to a git branch so they can be pulled elsewhere (non-fatal) ---
# Opt out with PUSH_RESULTS=0. Pull locally: git fetch origin && git switch eval-results
if [ "${PUSH_RESULTS:-1}" = 1 ]; then
  bash scripts/push_results.sh "$REPO" "$OUTDIR" "${RESULTS_BRANCH:-eval-results}" || true
fi

echo "[$(date +%H:%M:%S)] ALL DONE: $n groups in $(fmt $(( $(date +%s) - WALL0 )))"
