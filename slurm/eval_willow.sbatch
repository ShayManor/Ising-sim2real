#!/bin/bash -l
#SBATCH --job-name=ising_eval_willow
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
# The REAL-Willow row of the noise-fidelity ladder: the WHOLE decoder panel across
# d3, d5, d7 on real hardware data. Companion job eval_synth.sbatch produces the
# synthetic rows (uniform/SI1000/25-param-fit/syndrome-DEM); a LOCAL analysis script
# then compares rankings (Kendall tau / Spearman / bootstrap) across all rows. This
# job only PRODUCES RAW DATA -- no rank stats are computed here.
# Supersedes slurm/eval_d3.sbatch + slurm/eval_d7.sbatch (single distance each).
#
#   decoders : DEM-based classical panel + NVIDIA Ising pre-decoder --
#              mwpm (SI1000 DEM), mwpm-rl (RL DEM), beliefmatching (correlated
#              matching / Libra family), bposd, bplsd (ldpc), tesseract, all off
#              the shipped SI1000 DEM; plus ising with each MODEL (fast/accurate).
#   coverage : DISTANCES x {X,Z} x round counts, all 14 patches per distance
#              (the eval harness sweeps every patch automatically).
#   shots    : all 50,000 per config (override with SHOTS=N for a quick pass).
#   recorded : per (config, decoder): raw error count (n_errors), total + per-cycle
#              LER, decode latency -> CSV; PLUS per-shot logical-error outcomes
#              (bit-packed .npz in OUTDIR/outcomes) so every downstream statistic
#              (tau, bootstrap CIs, rank-agreement CIs) is reproducible LOCALLY
#              without re-running a decode. Set OUTCOMES=0 to skip the .npz dump.
#
# ONE-DECODER MODE (rerun/overwrite a single decoder after a change):
#   ONLY=bposd     sbatch slurm/eval_willow.sbatch   # just bposd across d3,d5,d7
#   ONLY=ising     sbatch slurm/eval_willow.sbatch   # just ising (with its MODELS)
#   ONLY=tesseract sbatch slurm/eval_willow.sbatch
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
# The panel decoders run a SINGLE-THREADED per-shot loop, so concurrency (not
# threads-per-shard) is what uses the node. Run ~one shard per core; shards are
# sub-sharded per patch (and per round for the two search-explosive decoders) so a
# slow decoder no longer serializes 9 patches x 6 rounds on one core.
NPAR="${NPAR:-48}"                                          # concurrent shards (pool, not barrier waves)
# BP+OSD (osd_cs, accurate) and Tesseract (A* search) explode with code size/rounds
# -- ~10 h for a single d7 config at 50k shots. Cap THEIR shots so a shard is bounded
# (n_errors + per-shot .npz keep the CIs honest); matching decoders keep all 50k.
# Set SLOW_SHOTS='' to uncap (only if you have days of wall clock).
SLOW_SHOTS="${SLOW_SHOTS:-10000}"
SLOW_DECODERS="${SLOW_DECODERS:-bposd tesseract}"          # sub-sharded per round + shot-capped
DEVICE="${DEVICE:-cpu}"                                     # panel + ising both run on CPU here
SOURCE="${SOURCE:-auto}"                                    # auto = local Willow tree if present, else HF
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
PUSH_RESULTS="${PUSH_RESULTS:-1}"                          # 1 = push CSVs to git after the run; 0 = off
RESULTS_DIR="${RESULTS_DIR:-results}"                     # in-repo dir the CSVs are committed to
RESUME="${RESUME:-0}"                                      # 0 = always re-run (default); 1 = skip shards whose CSV exists
OUTCOMES="${OUTCOMES:-1}"                                  # 1 = dump per-shot outcomes (.npz) for local stats; 0 = CSV only
OUTCOMES_DIR="${OUTCOMES_DIR:-$OUTDIR/outcomes}"          # where the per-shot .npz land (on scratch, NOT git)
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
  echo "pulling dataset $HF_REPO into $HF_HOME (uses HF_TOKEN if set; anonymous is rate-limited) ..."
  uv run python - "$HF_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
# token=None -> use ambient HF_TOKEN / cached login if present, else anonymous.
# Anonymous bulk downloads are throttled and can stall, so an authenticated
# submitting environment (HF_TOKEN in the login profile) is strongly preferred.
path = snapshot_download(sys.argv[1], repo_type="dataset", token=None, max_workers=16)
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
is_slow () { case " $SLOW_DECODERS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Patches per distance (fixed for this dataset: d3 has 9, d5 has 4, d7 has 1). Used
# to sub-shard so each patch decodes on its own core instead of 9 serially on one.
patches_for () { case "$1" in
  3) echo "q2_7 q4_5 q4_9 q6_3 q6_7 q6_11 q8_5 q8_9 q10_7" ;;
  5) echo "q4_7 q6_5 q6_9 q8_7" ;;
  7) echo "q6_7" ;;
esac; }
# Round counts <= MAX_ROUNDS (the slow decoders shard one round each; d7 has a single
# patch, so per-round is the only way to parallelize it).
STD_ROUNDS="1 10 13 30 50 70 90 110 130 150 170 190 210 230 250"
rounds_le () { for r in $STD_ROUNDS; do { [ -z "$MAX_ROUNDS" ] || [ "$r" -le "$MAX_ROUNDS" ]; } && echo "$r"; done; }

oc_flag=(); [ "$OUTCOMES" = 1 ] && oc_flag=(--outcomes-dir "$OUTCOMES_DIR")
run_shard () {  # $1=tag $2=out.csv  $3..=ising-eval args (caller supplies --shots)
  local tag="$1" out="$2"; shift 2
  if [ "${RESUME:-0}" = 1 ] && [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] [$tag] skip (resume): $(basename "$out")"; return; fi
  local t0; t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [$tag] start:    $(basename "$out")"
  uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" --device "$DEVICE" \
    ${oc_flag[@]+"${oc_flag[@]}"} --out "$out" "$@"
  echo "[$(date +%H:%M:%S)] [$tag] finished: $(basename "$out") in $(fmt $(( $(date +%s) - t0 )))"
}

# --- build the shard list. High distance first so the heaviest shards start early.
# cheap decoders  -> one shard per (decoder, distance, basis, patch), all rounds.
# slow decoders   -> one shard per (decoder, distance, basis, patch, round), shot-capped.
# ising           -> one shard per (model, distance, basis, patch), rounds<=MAX_ROUNDS.
jobs=()
for d in 7 5 3; do
  case " $DISTANCES " in *" $d "*) : ;; *) continue ;; esac
  for b in $BASES; do for p in $(patches_for "$d"); do
    for dec in $DECODERS; do
      if is_slow "$dec"; then
        for r in $(rounds_le); do jobs+=("slow:$dec:$d:$b:$p:$r"); done
      else
        jobs+=("cls:$dec:$d:$b:$p:-")
      fi
    done
    for m in $MODELS; do jobs+=("isg:$m:$d:$b:$p:-"); done
  done; done
done
njobs=${#jobs[@]}
[ "$njobs" -gt 0 ] || { echo "ERROR: empty plan (DECODERS and MODELS both empty?)"; exit 1; }
[ "$NPAR" -gt "$njobs" ] && NPAR="$njobs"; [ "${NPAR:-0}" -lt 1 ] && NPAR=1
CORES="${SLURM_CPUS_PER_TASK:-96}"
STHREADS=$(( CORES / NPAR )); [ "$STHREADS" -lt 1 ] && STHREADS=1
echo "plan: $njobs shards, $NPAR concurrent x $STHREADS threads on $CORES cores; source=$SOURCE shots=${SHOTS:-all} slow_shots=${SLOW_SHOTS:-all} max-rounds(expensive)=${MAX_ROUNDS:-all}"

dispatch () {  # $1="kind:name:d:b:patch:round" -- subshell so thread env is per-shard
  local kind name d b p r; IFS=: read -r kind name d b p r <<< "$1"
  export OMP_NUM_THREADS="$STHREADS" MKL_NUM_THREADS="$STHREADS"
  local sh=(); [ -n "$SHOTS" ] && sh=(--shots "$SHOTS")
  case "$kind" in
    isg)
      local mr=(); [ -n "$MAX_ROUNDS" ] && mr=(--max-rounds "$MAX_ROUNDS")
      run_shard "isg d$d$b/$p/$name" "$OUTDIR/ising_${name}_d${d}_${b}_${p}.csv" \
        --decoders ising --model "$name" --distances "$d" --bases "$b" --patches "$p" \
        ${mr[@]+"${mr[@]}"} ${sh[@]+"${sh[@]}"} ;;
    slow)
      local shots="${SLOW_SHOTS:-$SHOTS}"          # SLOW_SHOTS caps these; '' = use SHOTS
      local ss=(); [ -n "$shots" ] && ss=(--shots "$shots")
      run_shard "$name d$d$b/$p/r$r" "$OUTDIR/${name}_d${d}_${b}_${p}_r${r}.csv" \
        --decoders "$name" --distances "$d" --bases "$b" --patches "$p" --rounds "$r" \
        ${ss[@]+"${ss[@]}"} ;;
    cls)
      local mr=(); if [ -n "$MAX_ROUNDS" ] && is_expensive "$name"; then mr=(--max-rounds "$MAX_ROUNDS"); fi
      run_shard "$name d$d$b/$p" "$OUTDIR/${name}_d${d}_${b}_${p}.csv" \
        --decoders "$name" --distances "$d" --bases "$b" --patches "$p" \
        ${mr[@]+"${mr[@]}"} ${sh[@]+"${sh[@]}"} ;;
  esac
}

# Continuous pool (portable, bash 3.2+): keep <= NPAR shards running by polling the
# live background-job count before launching the next. Shards run for minutes, so 1s
# polling is free -- and this avoids `wait -n` (needs bash >= 4.3, not on all nodes).
# A failed background shard does not trip set -e in the parent; it is logged, leaves
# no CSV, and is retried on RESUME.
for j in "${jobs[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do sleep 1; done
  dispatch "$j" &
done
wait

# --- merge every shard CSV into one results file (single header) -------------
merged="$OUTDIR/eval_all.csv"
first=1
: > "$merged"
for f in "$OUTDIR"/*_d[357]_[XZ]_*.csv; do
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
