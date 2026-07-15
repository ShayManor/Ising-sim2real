#!/bin/bash -l
#SBATCH --job-name=ising_2x2
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
# 2x2 prior/channel decomposition (reviewer's "single most valuable experiment").
# Separates Delta_channel (the noise actually got worse) from Delta_prior (the
# decoder's prior is mischaracterized) for the fit rung:
#
#     syndrome source \ decoder prior |   theta (baseline)   theta' (perturbed)
#     --------------------------------+--------------------------------------------
#     theta  (baseline)               |   L(t,t)  [diagonal]   L(t,t')  <- THIS JOB
#     theta' (perturbed)              |   L(t',t) <- THIS JOB   L(t',t') [diagonal]
#
# The two DIAGONAL cells are already computed by eval_sensitivity.sbatch
# (results/sensitivity/{baseline,<param>}). This job computes ONLY the two
# OFF-DIAGONAL (source/prior-mismatch) cells per swept param, via the fit rung's
# PRIOR_MODELS_DIR decoupling (synthetic.py): sample from FITTED_MODELS_DIR, decode
# with a prior built from PRIOR_MODELS_DIR.  Delta_prior = L(t,t') - L(t,t);
# Delta_channel = L(t',t') - L(t,t).  RAW DATA ONLY -- decomposition is LOCAL.
#
# Targeted per the review: the most-disruptive Y-containing Pauli channels from
# RQ3 + one inert control, panel WITHOUT ising (channel/prior question is about
# the matched decoders), all distances, r=30.
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [ "$REPO" != "/" ] && [ ! -e "$REPO/pyproject.toml" ]; do REPO="$(dirname "$REPO")"; done
[ -e "$REPO/pyproject.toml" ] || { echo "ERROR: repo root (pyproject.toml) not found above ${SLURM_SUBMIT_DIR:-$PWD}; set REPO=/path/to/Ising-sim2real"; exit 1; }
MODELS_DIR="${MODELS_DIR:-$REPO/results/sensitivity/models}"      # baseline/ + <param>/ subdirs (from gen_sensitivity_models.py)
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/twobytwo}"            # per-cell CSVs land in OUTDIR/<cell>/
# the Y-containing Pauli channels RQ3 flagged as most disruptive + 1 inert control
PARAMS_2X2="${PARAMS_2X2:-p_idle_spam_Y p_idle_cnot_Y p_cnot_ZY p_cnot_ZZ}"
BASELINE="${BASELINE:-baseline}"                                 # subdir name of the unperturbed set
SYNTH_SHOTS="${SYNTH_SHOTS:-20000}"
SLOW_SHOTS="${SLOW_SHOTS:-1500}"
SEED="${SEED:-1234}"
ROUNDS="${ROUNDS:-30}"
DISTANCES="${DISTANCES:-3 5 7}"
BASES="${BASES:-X Z}"
DECODERS="${DECODERS:-mwpm beliefmatching bposd bplsd tesseract}"
MODELS="${MODELS:-}"                                             # no ising in the decomposition by default
SLOW_DECODERS="${SLOW_DECODERS:-bposd tesseract beliefmatching}"
ONLY_CELLS="${ONLY_CELLS:-}"                                     # space list of cell names to restrict (smoke/rerun)
NPAR="${NPAR:-48}"
DEVICE="${DEVICE:-cpu}"
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
RESUME="${RESUME:-0}"

[ -d "$MODELS_DIR/$BASELINE" ] || { echo "ERROR: '$MODELS_DIR/$BASELINE' not found; run scripts/gen_sensitivity_models.py and rsync it here"; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-96}"
# HF_HOME must live on scratch. Guard against an inherited /hf_cache: the login
# profile sets HF_HOME=$SCRATCH/hf_cache, but $SCRATCH is empty in a non-interactive
# submit shell, so --export=ALL can propagate the unwritable root path "/hf_cache".
case "${HF_HOME:-}" in /scratch/*) ;; *) HF_HOME="$OUTDIR/hf_cache" ;; esac
export HF_HOME
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "$REPO"
mkdir -p "$OUTDIR"

# --- build the off-diagonal cell list: (src, prior) with src != prior ---------
# cell name: "S-<src>__P-<prior>"; each swept param P gives baseline|P and P|baseline.
cells=()   # each entry "cellname:src:prior"
for P in $PARAMS_2X2; do
  [ -d "$MODELS_DIR/$P" ] || { echo "ERROR: swept-param dir '$MODELS_DIR/$P' missing"; exit 1; }
  cells+=("S-${BASELINE}__P-${P}:${BASELINE}:${P}")
  cells+=("S-${P}__P-${BASELINE}:${P}:${BASELINE}")
done
# optional restriction
if [ -n "$ONLY_CELLS" ]; then
  filtered=(); for c in "${cells[@]}"; do cn="${c%%:*}"
    case " $ONLY_CELLS " in *" $cn "*) filtered+=("$c") ;; esac; done
  cells=("${filtered[@]}")
fi
[ "${#cells[@]}" -gt 0 ] || { echo "ERROR: no cells to run"; exit 1; }

# --- panel-extra + prefetch (same as eval_sensitivity.sbatch) ----------------
needs_extra=0
for dd in $DECODERS; do case "$dd" in beliefmatching|bposd|bplsd|tesseract) needs_extra=1 ;; esac; done
if [ "$needs_extra" = 1 ] && ! uv run python -c "import ldpc, beliefmatching, tesseract_decoder" 2>/dev/null; then
  echo "setting up decoder panel extra (uv sync --extra decoders) ..."
  uv sync --extra decoders || { echo "ERROR: 'uv sync --extra decoders' failed; run it once on a login node."; exit 1; }
fi
have_ising_weights () { ls models/ising/*.pt >/dev/null 2>&1 || ls models/ising/*.safetensors >/dev/null 2>&1; }
if [ -n "$MODELS" ] && { [ ! -d third_party/Ising-Decoding/code ] || ! have_ising_weights; }; then
  uv run python scripts/setup_ising.py || { echo "ERROR: Ising setup failed"; exit 1; }
fi

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

run_shard () {  # $1=tag $2=out.csv $3=shots  $4..=ising-eval args (FITTED_MODELS_DIR + PRIOR_MODELS_DIR exported)
  local tag="$1" out="$2" shots="$3"; shift 3
  if [ "${RESUME:-0}" = 1 ] && [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] [$tag] skip (resume)"; return; fi
  local t0; t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [$tag] start"
  uv run ising-eval --source synth --rung fit --seed "$SEED" --rounds "$ROUNDS" \
    --synth-shots "$shots" --hf-repo "$HF_REPO" --device "$DEVICE" --no-progress \
    --out "$out" "$@"
  echo "[$(date +%H:%M:%S)] [$tag] done in $(fmt $(( $(date +%s) - t0 )))"
}

# --- one flat shard pool across ALL cells ------------------------------------
jobs=()
for c in "${cells[@]}"; do
  cn="${c%%:*}"
  for d in 7 5 3; do
    case " $DISTANCES " in *" $d "*) : ;; *) continue ;; esac
    for b in $BASES; do for p in $(patches_for "$d"); do
      for dec in $DECODERS; do jobs+=("$cn:cls:$dec:$d:$b:$p"); done
      for m in $MODELS; do jobs+=("$cn:isg:$m:$d:$b:$p"); done
    done; done
  done
done
njobs=${#jobs[@]}
[ "$NPAR" -gt "$njobs" ] && NPAR="$njobs"; [ "${NPAR:-0}" -lt 1 ] && NPAR=1
CORES="${SLURM_CPUS_PER_TASK:-96}"
STHREADS=$(( CORES / NPAR )); [ "$STHREADS" -lt 1 ] && STHREADS=1
echo "plan: ${#cells[@]} off-diagonal cells, $njobs shards, $NPAR concurrent x $STHREADS threads; r=$ROUNDS"

# map cell name -> src:prior (associative lookup)
declare -A SRC PRIOR
for c in "${cells[@]}"; do IFS=: read -r cn s pr <<< "$c"; SRC[$cn]="$s"; PRIOR[$cn]="$pr"; done

dispatch () {  # $1="cell:kind:name:d:b:patch"
  local cn kind name d b p; IFS=: read -r cn kind name d b p <<< "$1"
  export OMP_NUM_THREADS="$STHREADS" MKL_NUM_THREADS="$STHREADS"
  export FITTED_MODELS_DIR="$MODELS_DIR/${SRC[$cn]}"     # syndrome SOURCE
  export PRIOR_MODELS_DIR="$MODELS_DIR/${PRIOR[$cn]}"    # decode PRIOR (mismatched)
  local od="$OUTDIR/$cn"; mkdir -p "$od"
  case "$kind" in
    isg) run_shard "$cn/isg d$d$b/$p/$name" "$od/ising_${name}_d${d}_${b}_${p}.csv" "$SYNTH_SHOTS" \
           --decoders ising --model "$name" --distances "$d" --bases "$b" --patches "$p" ;;
    cls) local shots="$SYNTH_SHOTS"; is_slow "$name" && shots="$SLOW_SHOTS"
         run_shard "$cn/$name d$d$b/$p" "$od/${name}_d${d}_${b}_${p}.csv" "$shots" \
           --decoders "$name" --distances "$d" --bases "$b" --patches "$p" ;;
  esac
}

for j in "${jobs[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 1; done
  dispatch "$j" &
done
wait

# --- per-cell merge ----------------------------------------------------------
for c in "${cells[@]}"; do
  cn="${c%%:*}"; od="$OUTDIR/$cn"; merged="$od/eval_all.csv"; first=1; : > "$merged"
  for f in "$od"/*_d[357]_[XZ]_*.csv; do
    [ -s "$f" ] || continue; [ "$f" = "$merged" ] && continue
    if [ "$first" = 1 ]; then cat "$f" > "$merged"; first=0; else tail -n +2 "$f" >> "$merged"; fi
  done
  echo "merged $(($(wc -l < "$merged") - 1)) rows -> $merged"
done

echo "[$(date +%H:%M:%S)] ALL DONE: $njobs shards in $(fmt $(( $(date +%s) - WALL0 )))"
