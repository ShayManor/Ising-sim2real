#!/bin/bash -l
#SBATCH --job-name=ising_sensitivity
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
# RQ3 per-parameter sensitivity sweep. The fit rung, re-run once per PERTURBED
# noise-model set: baseline + 23 free params each overestimated (x2/+floor, see
# scripts/gen_sensitivity_models.py). Every set is a subdir of SETTINGS_DIR
# holding 14 per-patch model JSONs; the fit rung reads it via FITTED_MODELS_DIR.
#
# Same panel/scoring as eval_synth.sbatch (RUNG=fit), but ONE operating point
# (r=30, both bases, all 14 patches, all distances incl d7). All (setting x
# decoder x config) shards go into ONE flat pool so the node stays saturated
# with no per-setting idle tail. RAW DATA ONLY -- the ranking-flip stats are a
# LOCAL step (scripts/build_sensitivity.py).
#
# Gautschi: highmem, 24h, 96 cores (per user direction + eval_willow.sbatch
# precedent; the sweep is 24 fit-rung passes, too much for smallgpu's 12h).
#
# Prereqs (once, on a login node): uv sync --extra decoders;
#   uv run python scripts/setup_ising.py; scripts/gen_sensitivity_models.py then
#   rsync results/sensitivity/models to $SETTINGS_DIR.
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [ "$REPO" != "/" ] && [ ! -e "$REPO/pyproject.toml" ]; do REPO="$(dirname "$REPO")"; done
[ -e "$REPO/pyproject.toml" ] || { echo "ERROR: repo root (pyproject.toml) not found above ${SLURM_SUBMIT_DIR:-$PWD}; set REPO=/path/to/Ising-sim2real"; exit 1; }
SETTINGS_DIR="${SETTINGS_DIR:-$REPO/results/sensitivity/models}"  # baseline/ + <param>/ subdirs
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/sensitivity}"          # per-setting CSVs land in OUTDIR/<setting>/
SYNTH_SHOTS="${SYNTH_SHOTS:-20000}"                               # shots per config, cheap decoders + ising
SLOW_SHOTS="${SLOW_SHOTS:-1500}"                                  # bposd/tesseract/beliefmatching (cost driver)
SEED="${SEED:-1234}"
ROUNDS="${ROUNDS:-30}"                                            # single operating point for the whole panel
DISTANCES="${DISTANCES:-3 5 7}"
BASES="${BASES:-X Z}"
DECODERS="${DECODERS:-mwpm beliefmatching bposd bplsd tesseract}"
MODELS="${MODELS:-fast accurate}"
SLOW_DECODERS="${SLOW_DECODERS:-bposd tesseract beliefmatching}"  # shot-capped at SLOW_SHOTS
ONLY_SETTINGS="${ONLY_SETTINGS:-}"                                # space list to restrict settings (smoke/rerun)
NPAR="${NPAR:-48}"
DEVICE="${DEVICE:-cpu}"
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
RESUME="${RESUME:-0}"                                             # 1 = skip shards whose CSV already exists
OUTCOMES="${OUTCOMES:-0}"                                         # per-shot NPZ off by default (ranking uses n_errors)

[ -d "$SETTINGS_DIR" ] || { echo "ERROR: SETTINGS_DIR '$SETTINGS_DIR' not found; run scripts/gen_sensitivity_models.py and rsync it here"; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export HF_HOME="${HF_HOME:-$OUTDIR/hf_cache}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "$REPO"
mkdir -p "$OUTDIR"

# --- setting list (baseline first, then each perturbed param) ----------------
settings=()
for d in "$SETTINGS_DIR"/*/; do
  s="$(basename "$d")"
  [ -n "$ONLY_SETTINGS" ] && { case " $ONLY_SETTINGS " in *" $s "*) : ;; *) continue ;; esac; }
  settings+=("$s")
done
[ "${#settings[@]}" -gt 0 ] || { echo "ERROR: no setting subdirs in $SETTINGS_DIR"; exit 1; }

# --- panel-extra + Ising preflight (same as eval_synth.sbatch) ---------------
needs_extra=0
for dd in $DECODERS; do case "$dd" in beliefmatching|bposd|bplsd|tesseract) needs_extra=1 ;; esac; done
if [ "$needs_extra" = 1 ] && ! uv run python -c "import ldpc, beliefmatching, tesseract_decoder" 2>/dev/null; then
  echo "setting up decoder panel extra (uv sync --extra decoders) ..."
  uv sync --extra decoders || { echo "ERROR: 'uv sync --extra decoders' failed; run it once on a login node."; exit 1; }
fi
have_ising_weights () { ls models/ising/*.pt >/dev/null 2>&1 || ls models/ising/*.safetensors >/dev/null 2>&1; }
if [ -n "$MODELS" ] && { [ ! -d third_party/Ising-Decoding/code ] || ! have_ising_weights; }; then
  echo "setting up vendored Ising decoder (scripts/setup_ising.py) ..."
  uv run python scripts/setup_ising.py || { echo "ERROR: Ising setup failed; run scripts/setup_ising.py on a login node."; exit 1; }
fi

# --- prefetch per-config artifacts once (fit rung needs the parquet events +
# the si1000_noisy circuit template too; warms the shared HF cache so the flat
# pool does not race a cold cache -- see eval_synth.sbatch's note). -----------
echo "prefetching circuits+DEMs+fit-rung artifacts (r=$ROUNDS only) from $HF_REPO into $HF_HOME ..."
uv run python - "$HF_REPO" "$ROUNDS" <<'PY'
import sys
from ising_sim2real.ingest.hf import discover_configs_hf, _download, _stem
repo, rounds = sys.argv[1], int(sys.argv[2])
configs = [c for c in discover_configs_hf(repo) if c.rounds == rounds]
for cfg in configs:
    stem = _stem(cfg)
    _download(repo, f"circuits/{stem}.stim")
    _download(repo, f"dems/{stem}.si1000.dem.gz")
    _download(repo, f"data/{stem}.parquet")
    _download(repo, f"circuits/{stem}.si1000_noisy.stim")
print(f"prefetched {len(configs)} configs at r={rounds}")
PY

WALL0=$(date +%s)
fmt () { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }
is_slow () { case " $SLOW_DECODERS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
patches_for () { case "$1" in
  3) echo "q2_7 q4_5 q4_9 q6_3 q6_7 q6_11 q8_5 q8_9 q10_7" ;;
  5) echo "q4_7 q6_5 q6_9 q8_7" ;;
  7) echo "q6_7" ;;
esac; }

run_shard () {  # $1=tag $2=out.csv $3=shots  $4..=ising-eval args (FITTED_MODELS_DIR must be exported)
  local tag="$1" out="$2" shots="$3"; shift 3
  if [ "${RESUME:-0}" = 1 ] && [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] [$tag] skip (resume)"; return; fi
  local t0; t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [$tag] start"
  uv run ising-eval --source synth --rung fit --seed "$SEED" --rounds "$ROUNDS" \
    --synth-shots "$shots" --hf-repo "$HF_REPO" --device "$DEVICE" --no-progress \
    --out "$out" "$@"
  echo "[$(date +%H:%M:%S)] [$tag] done in $(fmt $(( $(date +%s) - t0 )))"
}

# --- build one flat shard list across ALL settings ---------------------------
jobs=()
for s in "${settings[@]}"; do
  for d in 7 5 3; do
    case " $DISTANCES " in *" $d "*) : ;; *) continue ;; esac
    for b in $BASES; do for p in $(patches_for "$d"); do
      for dec in $DECODERS; do jobs+=("$s:cls:$dec:$d:$b:$p"); done
      for m in $MODELS; do jobs+=("$s:isg:$m:$d:$b:$p"); done
    done; done
  done
done
njobs=${#jobs[@]}
[ "$NPAR" -gt "$njobs" ] && NPAR="$njobs"; [ "${NPAR:-0}" -lt 1 ] && NPAR=1
CORES="${SLURM_CPUS_PER_TASK:-96}"
STHREADS=$(( CORES / NPAR )); [ "$STHREADS" -lt 1 ] && STHREADS=1
echo "plan: ${#settings[@]} settings, $njobs shards, $NPAR concurrent x $STHREADS threads on $CORES cores; r=$ROUNDS shots=$SYNTH_SHOTS slow=$SLOW_SHOTS"

dispatch () {  # $1="setting:kind:name:d:b:patch"
  local s kind name d b p; IFS=: read -r s kind name d b p <<< "$1"
  export OMP_NUM_THREADS="$STHREADS" MKL_NUM_THREADS="$STHREADS"
  export FITTED_MODELS_DIR="$SETTINGS_DIR/$s"
  local od="$OUTDIR/$s"; mkdir -p "$od"
  case "$kind" in
    isg) run_shard "$s/isg d$d$b/$p/$name" "$od/ising_${name}_d${d}_${b}_${p}.csv" "$SYNTH_SHOTS" \
           --decoders ising --model "$name" --distances "$d" --bases "$b" --patches "$p" ;;
    cls) local shots="$SYNTH_SHOTS"; is_slow "$name" && shots="$SLOW_SHOTS"
         run_shard "$s/$name d$d$b/$p" "$od/${name}_d${d}_${b}_${p}.csv" "$shots" \
           --decoders "$name" --distances "$d" --bases "$b" --patches "$p" ;;
  esac
}

for j in "${jobs[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 1; done
  dispatch "$j" &
done
wait

# --- per-setting merge (one eval_all.csv each) -------------------------------
for s in "${settings[@]}"; do
  od="$OUTDIR/$s"; merged="$od/eval_all.csv"; first=1; : > "$merged"
  for f in "$od"/*_d[357]_[XZ]_*.csv; do
    [ -s "$f" ] || continue; [ "$f" = "$merged" ] && continue
    if [ "$first" = 1 ]; then cat "$f" > "$merged"; first=0; else tail -n +2 "$f" >> "$merged"; fi
  done
  echo "merged $(($(wc -l < "$merged") - 1)) rows -> $merged"
done

echo "[$(date +%H:%M:%S)] ALL DONE: $njobs shards in $(fmt $(( $(date +%s) - WALL0 )))"
