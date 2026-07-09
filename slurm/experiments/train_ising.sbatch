#!/bin/bash -l
#SBATCH --job-name=ising_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH -o /scratch/gautschi/manors/logs/train%j.out
#SBATCH -e /scratch/gautschi/manors/logs/train%j.err
set -euo pipefail

# =============================================================================
# Train the vendored NVIDIA Ising pre-decoder from a swappable ablation profile,
# then run a real-Willow d7 eval on the freshly trained checkpoint -- all in one
# job. Reuses the vendored training pipeline (third_party/Ising-Decoding) and
# this project's own eval harness (ising-eval) unmodified; profiles are the only
# new tracked artifact (third_party/ is entirely gitignored, so anything dropped
# into its conf/ would not survive a fresh clone).
#
#   profile  : PROFILE (default fast_repro) selects
#              slurm/experiments/profiles/<PROFILE>.{yaml,env}
#                .yaml -> Hydra overrides (model_id, distance/n_rounds, code_rotation,
#                         25-param noise_model) -- the noise-model/model-size ablation axis
#                .env   -> PREDECODER_TRAIN_EPOCHS/SAMPLES/LR_MILESTONES, GPUS --
#                         the hyperparam ablation axis
#   training : runs third_party/Ising-Decoding's own code/scripts/local_run.sh,
#              pointed at THIS project's uv-managed venv (PREDECODER_PYTHON) instead
#              of installing a second Python environment -- our .venv already has
#              torch/omegaconf/hydra-core/safetensors. Auto-resumes from the last
#              checkpoint if resubmitted with the same PROFILE (FRESH_START=1 to
#              force a clean run).
#   eval     : after training completes, finds the newest checkpoint under
#              outputs/<PROFILE>/models/best_model/ (falling back to models/),
#              and runs ising-eval --model custom --model-id <from profile.yaml>
#              --checkpoint <that .pt> against real Willow d7 (X and Z), plus the
#              mwpm classical baseline for comparison. Same source/push-to-git
#              conventions as slurm/eval_d7.sbatch, sized for this job's 64 cores.
#
# Timeout-safe: if training doesn't finish inside the wall clock, SLURM kills the
# job here and the eval step below never runs -- resubmit the same PROFILE to
# resume training from its last checkpoint.
#
# Prereqs (run once on a login node, with internet):
#   uv sync                              # this project's own venv (torch/hydra/etc.)
#   uv run python scripts/setup_ising.py # vendored NVIDIA Ising code + base weights
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [ "$REPO" != "/" ] && [ ! -e "$REPO/pyproject.toml" ]; do REPO="$(dirname "$REPO")"; done
[ -e "$REPO/pyproject.toml" ] || { echo "ERROR: repo root (pyproject.toml) not found above ${SLURM_SUBMIT_DIR:-$PWD}; set REPO=/path/to/Ising-sim2real"; exit 1; }

PROFILE="${PROFILE:-fast_repro}"
PROFILES_DIR="$REPO/slurm/experiments/profiles"
PROFILE_YAML="$PROFILES_DIR/$PROFILE.yaml"
PROFILE_ENV="$PROFILES_DIR/$PROFILE.env"
[ -f "$PROFILE_YAML" ] || { echo "ERROR: missing profile config $PROFILE_YAML"; exit 1; }
[ -f "$PROFILE_ENV" ] || { echo "ERROR: missing profile env $PROFILE_ENV"; exit 1; }

FRESH_START="${FRESH_START:-0}"
GPUS="${GPUS:-1}"
SOURCE="${SOURCE:-auto}"                                     # auto = local Willow tree if present, else HF
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
WILLOW_RAW_DIR="${WILLOW_RAW_DIR:-$REPO/google_105Q_surface_code_d3_d5_d7}"
export WILLOW_RAW_DIR
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/ising_train/$PROFILE}"  # eval CSVs land here (NOT in git)
EVAL_SHOTS="${EVAL_SHOTS:-}"                                  # empty = all 50k; set e.g. 5000 for a quick pass
PUSH_RESULTS="${PUSH_RESULTS:-1}"                             # 1 = push eval CSVs to git after the run; 0 = off
RESULTS_DIR="${RESULTS_DIR:-results}"

export HF_HOME="${HF_HOME:-$OUTDIR/hf_cache}"
cd "$REPO"
mkdir -p "$OUTDIR"

echo "profile=$PROFILE gpus=$GPUS fresh_start=$FRESH_START source=$SOURCE"
echo "profile config: $PROFILE_YAML"
echo "profile env:    $PROFILE_ENV"

# shellcheck disable=SC1090
source "$PROFILE_ENV"

# --- pick source: local Willow tree if present, else pull from HF -------------
if [ "$SOURCE" = "auto" ]; then
  if [ -d "$WILLOW_RAW_DIR" ] && [ -n "$(ls -A "$WILLOW_RAW_DIR" 2>/dev/null)" ]; then SOURCE=local; else SOURCE=hf; fi
fi
if [ "$SOURCE" = "hf" ]; then
  export HF_HUB_DISABLE_PROGRESS_BARS=1
  echo "pulling dataset $HF_REPO into $HF_HOME (public, no token) ..."
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("dataset ready at", snapshot_download(sys.argv[1], repo_type="dataset", token=False))
PY
fi

# --- vendored Ising code + base weights (gitignored) --------------------------
have_ising_weights () { ls models/ising/*.pt >/dev/null 2>&1 || ls models/ising/*.safetensors >/dev/null 2>&1; }
if [ ! -d third_party/Ising-Decoding/code ] || ! have_ising_weights; then
  echo "setting up vendored Ising decoder (scripts/setup_ising.py) ..."
  uv run python scripts/setup_ising.py || { echo "ERROR: Ising setup failed (needs git-lfs + internet)."; exit 1; }
fi

# =============================================================================
# 1) Train -- reuse the vendored pipeline's own local_run.sh, pointed at our venv.
# =============================================================================
uv sync --quiet
PREDECODER_PYTHON="$REPO/.venv/bin/python"
echo "[$(date +%H:%M:%S)] training: PREDECODER_PYTHON=$PREDECODER_PYTHON"

EXPERIMENT_NAME="$PROFILE" CONFIG_NAME="$PROFILE" WORKFLOW=train \
EXTRA_PARAMS="--config-path=$PROFILES_DIR" \
PREDECODER_PYTHON="$PREDECODER_PYTHON" \
GPUS="$GPUS" FRESH_START="$FRESH_START" \
  bash third_party/Ising-Decoding/code/scripts/local_run.sh

echo "[$(date +%H:%M:%S)] training done"

# =============================================================================
# 2) Locate the freshly trained checkpoint.
# =============================================================================
TRAIN_OUT="third_party/Ising-Decoding/outputs/$PROFILE/models"
CKPT="$(ls -t "$TRAIN_OUT"/best_model/*.pt 2>/dev/null | head -1)"
[ -n "$CKPT" ] || CKPT="$(ls -t "$TRAIN_OUT"/*.pt 2>/dev/null | head -1)"
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint found under $TRAIN_OUT"; exit 1; }
echo "checkpoint: $CKPT"

MODEL_ID="$(uv run python -c "
from omegaconf import OmegaConf
print(OmegaConf.load('$PROFILE_YAML').model_id)
")"
echo "model_id: $MODEL_ID"

# =============================================================================
# 3) Post-train eval: real Willow d7, X and Z, this checkpoint vs. mwpm.
# =============================================================================
CORES="${SLURM_CPUS_PER_TASK:-64}"
ITHREADS=$(( CORES / 2 )); [ "$ITHREADS" -lt 1 ] && ITHREADS=1

run_shard () {  # $1=tag $2=out.csv $3..=ising-eval args
  local tag="$1" out="$2"; shift 2
  echo "[$(date +%H:%M:%S)] [$tag] start:    $(basename "$out")"
  local t0; t0=$(date +%s)
  uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" --device cpu \
    --distances 7 --out "$out" "$@"
  echo "[$(date +%H:%M:%S)] [$tag] finished: $(basename "$out") in $(( $(date +%s) - t0 ))s"
}

shots_flag=(); [ -n "$EVAL_SHOTS" ] && shots_flag=(--shots "$EVAL_SHOTS")

# classical baseline (cheap, serial)
run_shard "mwpm" "$OUTDIR/classical_d7.csv" --decoders mwpm --bases X,Z "${shots_flag[@]}"

# trained model, one shard per basis, run concurrently on this job's cores
for basis in X Z; do
  (
    export OMP_NUM_THREADS="$ITHREADS" MKL_NUM_THREADS="$ITHREADS"
    run_shard "ising/$basis" "$OUTDIR/ising_${PROFILE}_d7_${basis}.csv" \
      --decoders ising --model custom --model-id "$MODEL_ID" --checkpoint "$CKPT" \
      --bases "$basis" "${shots_flag[@]}"
  ) &
done
wait

merged="$OUTDIR/eval_all.csv"
first=1; : > "$merged"
for f in "$OUTDIR"/classical_*.csv "$OUTDIR"/ising_*.csv; do
  [ -s "$f" ] || continue
  if [ "$first" = 1 ]; then cat "$f" > "$merged"; first=0; else tail -n +2 "$f" >> "$merged"; fi
done
echo "merged $(($(wc -l < "$merged") - 1)) rows -> $merged"

if [ "${PUSH_RESULTS:-1}" = 1 ]; then
  bash scripts/push_results.sh "$REPO" "$OUTDIR" "${RESULTS_DIR:-results}" || true
fi

echo "[$(date +%H:%M:%S)] ALL DONE: profile=$PROFILE checkpoint=$CKPT"
