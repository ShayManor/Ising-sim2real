#!/bin/bash -l
#SBATCH --job-name=ising_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=24:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/ising%j.out
#SBATCH -e /scratch/gautschi/manors/logs/ising%j.err
set -euo pipefail

# =============================================================================
# One eval job for the WHOLE decoder panel across d3, d5, d7 on real Willow data.
# Supersedes slurm/eval_d3.sbatch + slurm/eval_d7.sbatch (single distance each).
#
#   decoders : DEM-based classical panel + NVIDIA Ising pre-decoder --
#              mwpm (SI1000 DEM), mwpm-rl (RL DEM), beliefmatching (correlated
#              matching / Libra family), bposd, bplsd (ldpc), tesseract, all off
#              the shipped SI1000 DEM; plus ising with each MODEL (fast/accurate).
#   coverage : DISTANCES x {X,Z} x round counts, all 14 patches per distance
#              (the eval harness sweeps every patch automatically).
#   shots    : all 50,000 per config (override with SHOTS=N for a quick pass).
#   metric   : per-config LER + LER per cycle + decode latency -> one CSV/shard.
#
# ONE-DECODER MODE (rerun/overwrite a single decoder after a change):
#   ONLY=bposd     sbatch slurm/eval.sbatch     # just bposd across d3,d5,d7
#   ONLY=ising     sbatch slurm/eval.sbatch     # just ising (with its MODELS)
#   ONLY=tesseract sbatch slurm/eval.sbatch
# ONLY restricts the plan to that one decoder and overwrites ONLY its shard CSVs
# in OUTDIR (every other decoder's CSVs are left untouched, then merged back in),
# so you can re-run one decoder without recomputing the rest.
#
# Sharding: work is split by (decoder, distance, basis) -- each shard writes its
# own CSV the moment it finishes, so a wall-clock timeout never loses a finished
# shard. Re-submit with RESUME=1 to skip shards whose CSV already exists.
#
# Prereqs (run once on a login node, with internet):
#   uv sync --extra decoders             # base env + ldpc/beliefmatching/tesseract
#   uv run python scripts/setup_ising.py # vendored NVIDIA Ising code + weights
#                                        # (only needed if MODELS is non-empty)
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [ "$REPO" != "/" ] && [ ! -e "$REPO/pyproject.toml" ]; do REPO="$(dirname "$REPO")"; done
[ -e "$REPO/pyproject.toml" ] || { echo "ERROR: repo root (pyproject.toml) not found above ${SLURM_SUBMIT_DIR:-$PWD}; set REPO=/path/to/Ising-sim2real"; exit 1; }
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/ising_eval}"     # results land here (NOT in git)
SHOTS="${SHOTS:-}"                                           # empty = all 50k; set e.g. 5000 for a quick pass
DISTANCES="${DISTANCES:-3 5 7}"
BASES="${BASES:-X Z}"
DECODERS="${DECODERS:-mwpm mwpm-rl beliefmatching bposd bplsd tesseract}"  # DEM-based panel
MODELS="${MODELS:-fast accurate}"                           # ising model variants ('' = skip ising)
ONLY="${ONLY:-}"                                            # run just this one decoder (classical name or 'ising')
# Expensive decoders (BP post-processing / A* search / neural) are capped by rounds:
# per-cycle LER is best estimated at low/moderate rounds and their cost grows with
# rounds, so this keeps the sweep in the wall clock. Cheap matching (mwpm/mwpm-rl)
# always runs at ALL rounds. Set MAX_ROUNDS='' to uncap the expensive ones too.
MAX_ROUNDS="${MAX_ROUNDS:-70}"
NPAR="${NPAR:-6}"                                           # concurrent CPU shards (cores split across them)
DEVICE="${DEVICE:-cpu}"                                     # panel + ising both run on CPU here
SOURCE="${SOURCE:-auto}"                                    # auto = local Willow tree if present, else HF
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
PUSH_RESULTS="${PUSH_RESULTS:-1}"                          # 1 = push CSVs to git after the run; 0 = off
RESULTS_DIR="${RESULTS_DIR:-results}"                     # in-repo dir the CSVs are committed to
RESUME="${RESUME:-0}"                                      # 0 = always re-run (default); 1 = skip shards whose CSV exists
WILLOW_RAW_DIR="${WILLOW_RAW_DIR:-$REPO/google_105Q_surface_code_d3_d5_d7}"
export WILLOW_RAW_DIR

# --- ONLY: restrict the plan to a single decoder (overwrite just its shards) ---
if [ -n "$ONLY" ]; then
  if [ "$ONLY" = ising ]; then
    DECODERS=""
    [ -n "$MODELS" ] || { echo "ERROR: ONLY=ising but MODELS is empty"; exit 1; }
  else
    case " mwpm mwpm-rl beliefmatching bposd bplsd tesseract " in
      *" $ONLY "*) DECODERS="$ONLY"; MODELS="" ;;
      *) echo "ERROR: ONLY='$ONLY' is not a known decoder (mwpm|mwpm-rl|beliefmatching|bposd|bplsd|tesseract|ising)"; exit 1 ;;
    esac
  fi
  echo "ONLY mode: running just '$ONLY' across ${DISTANCES// /,} (overwrites its shards)."
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export HF_HOME="${HF_HOME:-$OUTDIR/hf_cache}"

cd "$REPO"
mkdir -p "$OUTDIR"
shots_flag=(); [ -n "$SHOTS" ] && shots_flag=(--shots "$SHOTS")

# --- panel-extra preflight: the ldpc/beliefmatching/tesseract adapters need the
# optional `decoders` extra. Fail fast with a login-node fix rather than crashing
# every panel shard after hours of queueing.
needs_extra=0
for d in $DECODERS; do case "$d" in beliefmatching|bposd|bplsd|tesseract) needs_extra=1 ;; esac; done
if [ "$needs_extra" = 1 ]; then
  if ! uv run python -c "import ldpc, beliefmatching, tesseract_decoder" 2>/dev/null; then
    echo "setting up decoder panel extra (uv sync --extra decoders) ..."
    uv sync --extra decoders || { echo "ERROR: 'uv sync --extra decoders' failed; run it once on a login node."; exit 1; }
  fi
fi

# --- pick source: use the local Willow tree if it's there, else pull from HF ---
if [ "$SOURCE" = "auto" ]; then
  if [ -d "$WILLOW_RAW_DIR" ] && [ -n "$(ls -A "$WILLOW_RAW_DIR" 2>/dev/null)" ]; then
    SOURCE=local
  else
    SOURCE=hf
  fi
fi
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

# --- Ising setup: vendored NVIDIA code + weights are gitignored. Only needed when
# ising is in the plan (MODELS non-empty). Fail fast with a login-node instruction.
have_ising_weights () { ls models/ising/*.pt >/dev/null 2>&1 || ls models/ising/*.safetensors >/dev/null 2>&1; }
if [ -n "$MODELS" ] && { [ ! -d third_party/Ising-Decoding/code ] || ! have_ising_weights; }; then
  echo "setting up vendored Ising decoder (scripts/setup_ising.py) ..."
  if ! uv run python scripts/setup_ising.py; then
    echo "ERROR: Ising setup failed (needs git-lfs + internet on this node)."
    echo "Run it once on a LOGIN node and resubmit:  uv run python scripts/setup_ising.py"
    echo "(or set MODELS='' to run the classical panel only)"
    exit 1
  fi
fi

WALL0=$(date +%s)
fmt () { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

is_expensive () { case "$1" in bposd|bplsd|tesseract|ising) return 0 ;; *) return 1 ;; esac; }

run_shard () {  # $1=tag $2=out.csv  $3..=ising-eval args
  local tag="$1" out="$2"; shift 2
  if [ "${RESUME:-0}" = 1 ] && [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] [$tag] skip (resume): $(basename "$out")"; return; fi
  local t0; t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [$tag] start:    $(basename "$out")"
  uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" --device "$DEVICE" \
    "${shots_flag[@]}" --out "$out" "$@"
  echo "[$(date +%H:%M:%S)] [$tag] finished: $(basename "$out") in $(fmt $(( $(date +%s) - t0 )))"
}

# --- build the shard list: (decoder, distance, basis) + (ising model, distance, basis)
jobs=()
for d in $DISTANCES; do for b in $BASES; do
  for dec in $DECODERS; do jobs+=("cls:$dec:$d:$b"); done
  for m in $MODELS;   do jobs+=("isg:$m:$d:$b");   done
done; done
njobs=${#jobs[@]}
[ "$njobs" -gt 0 ] || { echo "ERROR: empty plan (DECODERS and MODELS both empty?)"; exit 1; }
[ "$NPAR" -gt "$njobs" ] && NPAR="$njobs"; [ "${NPAR:-0}" -lt 1 ] && NPAR=1
CORES="${SLURM_CPUS_PER_TASK:-96}"
STHREADS=$(( CORES / NPAR )); [ "$STHREADS" -lt 1 ] && STHREADS=1
echo "plan: $njobs shards, $NPAR concurrent x $STHREADS threads on $CORES cores; source=$SOURCE shots=${SHOTS:-all} max-rounds(expensive)=${MAX_ROUNDS:-all}"

dispatch () {  # $1="kind:name:d:b" -- subshell so thread env is per-shard
  local kind name d b; IFS=: read -r kind name d b <<< "$1"
  export OMP_NUM_THREADS="$STHREADS" MKL_NUM_THREADS="$STHREADS"
  local mr_flag=()
  if [ "$kind" = isg ]; then
    [ -n "$MAX_ROUNDS" ] && mr_flag=(--max-rounds "$MAX_ROUNDS")
    run_shard "isg d$d$b/$name" "$OUTDIR/ising_${name}_d${d}_${b}.csv" \
      --decoders ising --model "$name" --distances "$d" --bases "$b" "${mr_flag[@]}"
  else
    if [ -n "$MAX_ROUNDS" ] && is_expensive "$name"; then mr_flag=(--max-rounds "$MAX_ROUNDS"); fi
    run_shard "$name d$d$b" "$OUTDIR/${name}_d${d}_${b}.csv" \
      --decoders "$name" --distances "$d" --bases "$b" "${mr_flag[@]}"
  fi
}

# Launch in waves of NPAR (portable; no `wait -n` needed).
for (( i=0; i<njobs; i+=NPAR )); do
  for (( k=0; k<NPAR && i+k<njobs; k++ )); do dispatch "${jobs[i+k]}" & done
  wait   # barrier: finish this wave before the next
done

# --- merge every shard CSV into one results file (single header) -------------
merged="$OUTDIR/eval_all.csv"
first=1
: > "$merged"
for f in "$OUTDIR"/*_d[357]_[XZ].csv; do
  [ -s "$f" ] || continue
  [ "$f" = "$merged" ] && continue
  if [ "$first" = 1 ]; then cat "$f" > "$merged"; first=0; else tail -n +2 "$f" >> "$merged"; fi
done
echo "merged $(($(wc -l < "$merged") - 1)) rows -> $merged"

# --- commit + push the CSVs (opt out with PUSH_RESULTS=0). Non-fatal. ----------
if [ "${PUSH_RESULTS:-1}" = 1 ]; then
  bash scripts/push_results.sh "$REPO" "$OUTDIR" "${RESULTS_DIR:-results}" || true
fi

echo "[$(date +%H:%M:%S)] ALL DONE: $njobs shards in $(fmt $(( $(date +%s) - WALL0 )))"
