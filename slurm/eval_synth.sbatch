#!/bin/bash -l
#SBATCH --job-name=ising_eval_synth
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

# =============================================================================
# A SYNTHETIC row of the noise-fidelity ladder: one submission = one rung
# (RUNG=uniform|si1000|syndrome|fit). Same panel/scoring/sharding as eval_willow.sbatch, but
# detectors/observables are SAMPLED from the rung's DEM instead of read off real
# hardware (--source synth; see src/ising_sim2real/ingest/synthetic.py). This job
# only PRODUCES RAW DATA -- rank stats (Kendall tau / Spearman / bootstrap) are a
# separate LOCAL step over the resulting CSVs.
#
#   decoders : mwpm + the DEM-based panel (beliefmatching/bposd/bplsd/tesseract),
#              all off the rung DEM (matched-prior decoding: sample and decode with
#              the SAME DEM); plus ising with each MODEL (fast/accurate).
#              mwpm-rl is DROPPED (no RL DEM has a synthetic analog).
#   coverage : DISTANCES x {X,Z} x round counts, all 14 patches per distance.
#   shots    : SYNTH_SHOTS sampled per config for cheap decoders; SLOW_SHOTS
#              (fewer) sampled directly for the slow decoders -- there is no fixed
#              50k to subsample from, we choose how many synthetic shots to draw.
#   recorded : identical CSV schema to eval_willow.sbatch, plus per-shot outcomes
#              NPZ (OUTCOMES=1) for local bootstrap/rank-CI recomputation.
#
# Gautschi limits (HARD): smallgpu only, <=12h wall, <=64 cores. Only smallgpu has
# a GPU here; decode is CPU-bound so --gres=gpu:1 is requested but unused.
#
# ONE-DECODER MODE: ONLY=bposd sbatch --export=...,RUNG=si1000 slurm/eval_synth.sbatch
#
# Prereqs (once, on a login node, with internet):
#   uv sync --extra decoders
#   uv run python scripts/setup_ising.py   # only needed if MODELS is non-empty
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [ "$REPO" != "/" ] && [ ! -e "$REPO/pyproject.toml" ]; do REPO="$(dirname "$REPO")"; done
[ -e "$REPO/pyproject.toml" ] || { echo "ERROR: repo root (pyproject.toml) not found above ${SLURM_SUBMIT_DIR:-$PWD}; set REPO=/path/to/Ising-sim2real"; exit 1; }
RUNG="${RUNG:?set RUNG=uniform|si1000|syndrome|fit}"
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/synth_${RUNG}}"  # results land here (NOT in git)
# SYNTH_P only applies to RUNG=uniform (the syndrome rung has no scalar
# parameter -- probabilities come from real Willow detection-event statistics).
SYNTH_P="${SYNTH_P:-2e-3}"                                   # uniform-rung per-error probability
SYNTH_SHOTS="${SYNTH_SHOTS:-20000}"                          # shots SAMPLED per config, cheap decoders
SEED="${SEED:-1234}"                                         # base seed (deterministic per-config re-run)
DISTANCES="${DISTANCES:-3 5 7}"
BASES="${BASES:-X Z}"
DECODERS="${DECODERS:-mwpm beliefmatching bposd bplsd tesseract}"  # DEM-based panel (no mwpm-rl on synthetic)
MODELS="${MODELS:-fast accurate}"                            # ising model variants ('' = skip ising)
ONLY="${ONLY:-}"                                             # run just this one decoder
# Expensive decoders capped by rounds -- see eval_willow.sbatch for rationale.
MAX_ROUNDS="${MAX_ROUNDS:-30}"                                # conservative default to fit 12h/64cpu (spec S5)
# ROUNDS overrides rounds_le() with an exact list (space-separated), e.g. to target
# only the fit rung's held-out eval-set rounds (110-250) without also re-running the
# fit-set rounds MAX_ROUNDS would otherwise include as a 1..N upper bound.
ROUNDS="${ROUNDS:-}"
NPAR="${NPAR:-32}"                                            # concurrent shards, <=64 cores / >=1 core each
SLOW_SHOTS="${SLOW_SHOTS:-5000}"                              # fewer SAMPLED shots for the slow decoders
# beliefmatching is included here (unlike eval_willow.sbatch's original default) --
# it is NOT in is_expensive(), so as a "cls" decoder it would run ALL 15 round counts
# uncapped by MAX_ROUNDS at full SYNTH_SHOTS in one all-or-nothing shard (~hours,
# observed live: 2026-07-08 job 13248448/13248451, zero beliefmatching CSVs after
# 50min while it ate most of the NPAR pool). Sub-sharding it here caps rounds+shots
# like bposd/tesseract, matching the fix already applied to the real-Willow job
# (see memory gautschi-willow-eval-job.md).
SLOW_DECODERS="${SLOW_DECODERS:-bposd tesseract beliefmatching}"  # sub-sharded per round + shot-capped
DEVICE="${DEVICE:-cpu}"                                       # panel + ising both run on CPU here
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
PUSH_RESULTS="${PUSH_RESULTS:-0}"                             # 0 = do NOT auto-commit; rsync CSVs back instead
RESULTS_DIR="${RESULTS_DIR:-results}"
RESUME="${RESUME:-0}"                                         # 1 = skip shards whose CSV already exists
OUTCOMES="${OUTCOMES:-1}"                                     # 1 = dump per-shot outcomes (.npz) for local stats
OUTCOMES_DIR="${OUTCOMES_DIR:-$OUTDIR/outcomes}"

case "$RUNG" in uniform|si1000|syndrome|fit) : ;; *) echo "ERROR: RUNG='$RUNG' must be uniform|si1000|syndrome|fit"; exit 1 ;; esac

# --- ONLY: restrict the plan to a single decoder (overwrite just its shards) ---
if [ -n "$ONLY" ]; then
  if [ "$ONLY" = ising ]; then
    DECODERS=""
    [ -n "$MODELS" ] || { echo "ERROR: ONLY=ising but MODELS is empty"; exit 1; }
  else
    case " mwpm beliefmatching bposd bplsd tesseract " in
      *" $ONLY "*) DECODERS="$ONLY"; MODELS="" ;;
      *) echo "ERROR: ONLY='$ONLY' is not a known synthetic-panel decoder (mwpm|beliefmatching|bposd|bplsd|tesseract|ising)"; exit 1 ;;
    esac
  fi
  echo "ONLY mode: running just '$ONLY' across ${DISTANCES// /,} (overwrites its shards)."
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export HF_HOME="${HF_HOME:-$OUTDIR/hf_cache}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

cd "$REPO"
mkdir -p "$OUTDIR"

# --- panel-extra preflight (same as eval_willow.sbatch) -----------------------
needs_extra=0
for d in $DECODERS; do case "$d" in beliefmatching|bposd|bplsd|tesseract) needs_extra=1 ;; esac; done
if [ "$needs_extra" = 1 ]; then
  if ! uv run python -c "import ldpc, beliefmatching, tesseract_decoder" 2>/dev/null; then
    echo "setting up decoder panel extra (uv sync --extra decoders) ..."
    uv sync --extra decoders || { echo "ERROR: 'uv sync --extra decoders' failed; run it once on a login node."; exit 1; }
  fi
fi

# --- Ising setup: vendored NVIDIA code + weights are gitignored. -------------
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

# --- prefetch circuits+DEMs for every config (small; avoids 32 shards racing
# on first access -- huggingface_hub caches per-file, this just warms it once).
# RUNG=fit additionally needs data/<stem>.parquet (real detection events, via
# load_config_from_hf) and circuits/<stem>.si1000_noisy.stim (the noisy-circuit
# template, via fetch_si1000_noisy_circuit) -- both prefetched here too when
# RUNG=fit, otherwise the SAME 32-shard cold-cache race this prefetch step
# exists to avoid recurs on exactly those two file types (observed live:
# 2026-07-11, job 13408854 stalled near-total for 2.5h+ with ~0% CPU across
# all workers, root-caused to lock contention on these two never-prefetched
# per-config artifacts, not a decoder/algorithm bug). ---
echo "prefetching circuits+DEMs from $HF_REPO into $HF_HOME ..."
uv run python - "$HF_REPO" "$RUNG" <<'PY'
import sys
from ising_sim2real.ingest.hf import discover_configs_hf, _download, _stem
repo = sys.argv[1]
rung = sys.argv[2]
configs = discover_configs_hf(repo)
for cfg in configs:
    stem = _stem(cfg)
    _download(repo, f"circuits/{stem}.stim")
    _download(repo, f"dems/{stem}.si1000.dem.gz")
    if rung == "fit":
        _download(repo, f"data/{stem}.parquet")
        _download(repo, f"circuits/{stem}.si1000_noisy.stim")
print(f"prefetched {len(configs)} configs' circuits+DEMs" + (" +fit-rung artifacts" if rung == "fit" else ""))
PY

WALL0=$(date +%s)
fmt () { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

is_expensive () { case "$1" in bposd|bplsd|tesseract|ising) return 0 ;; *) return 1 ;; esac; }
is_slow () { case " $SLOW_DECODERS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

patches_for () { case "$1" in
  3) echo "q2_7 q4_5 q4_9 q6_3 q6_7 q6_11 q8_5 q8_9 q10_7" ;;
  5) echo "q4_7 q6_5 q6_9 q8_7" ;;
  7) echo "q6_7" ;;
esac; }
STD_ROUNDS="1 10 13 30 50 70 90 110 130 150 170 190 210 230 250"
rounds_le () { if [ -n "$ROUNDS" ]; then for r in $ROUNDS; do echo "$r"; done; return; fi
  for r in $STD_ROUNDS; do { [ -z "$MAX_ROUNDS" ] || [ "$r" -le "$MAX_ROUNDS" ]; } && echo "$r"; done; }

oc_flag=(); [ "$OUTCOMES" = 1 ] && oc_flag=(--outcomes-dir "$OUTCOMES_DIR")
run_shard () {  # $1=tag $2=out.csv $3=synth-shots  $4..=ising-eval args
  local tag="$1" out="$2" shots="$3"; shift 3
  if [ "${RESUME:-0}" = 1 ] && [ -s "$out" ]; then echo "[$(date +%H:%M:%S)] [$tag] skip (resume): $(basename "$out")"; return; fi
  local t0; t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] [$tag] start:    $(basename "$out")"
  uv run ising-eval --source synth --rung "$RUNG" --synth-p "$SYNTH_P" --seed "$SEED" \
    --synth-shots "$shots" --hf-repo "$HF_REPO" --device "$DEVICE" \
    ${oc_flag[@]+"${oc_flag[@]}"} --out "$out" "$@"
  echo "[$(date +%H:%M:%S)] [$tag] finished: $(basename "$out") in $(fmt $(( $(date +%s) - t0 )))"
}

# --- build the shard list (same shape as eval_willow.sbatch) ------------------
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
CORES="${SLURM_CPUS_PER_TASK:-64}"
STHREADS=$(( CORES / NPAR )); [ "$STHREADS" -lt 1 ] && STHREADS=1
echo "plan: $njobs shards, $NPAR concurrent x $STHREADS threads on $CORES cores; rung=$RUNG p=$SYNTH_P seed=$SEED synth-shots=$SYNTH_SHOTS slow-shots=$SLOW_SHOTS max-rounds(expensive)=${MAX_ROUNDS:-all}"

dispatch () {  # $1="kind:name:d:b:patch:round" -- subshell so thread env is per-shard
  local kind name d b p r; IFS=: read -r kind name d b p r <<< "$1"
  export OMP_NUM_THREADS="$STHREADS" MKL_NUM_THREADS="$STHREADS"
  case "$kind" in
    isg)
      local mr=()
      if [ -n "$ROUNDS" ]; then mr=(--rounds "$(echo "$ROUNDS" | tr ' ' ',')")
      elif [ -n "$MAX_ROUNDS" ]; then mr=(--max-rounds "$MAX_ROUNDS"); fi
      run_shard "isg d$d$b/$p/$name" "$OUTDIR/ising_${name}_d${d}_${b}_${p}.csv" "$SYNTH_SHOTS" \
        --decoders ising --model "$name" --distances "$d" --bases "$b" --patches "$p" \
        ${mr[@]+"${mr[@]}"} ;;
    slow)
      run_shard "$name d$d$b/$p/r$r" "$OUTDIR/${name}_d${d}_${b}_${p}_r${r}.csv" "$SLOW_SHOTS" \
        --decoders "$name" --distances "$d" --bases "$b" --patches "$p" --rounds "$r" ;;
    cls)
      local mr=()
      if [ -n "$ROUNDS" ]; then mr=(--rounds "$(echo "$ROUNDS" | tr ' ' ',')")
      elif [ -n "$MAX_ROUNDS" ] && is_expensive "$name"; then mr=(--max-rounds "$MAX_ROUNDS"); fi
      run_shard "$name d$d$b/$p" "$OUTDIR/${name}_d${d}_${b}_${p}.csv" "$SYNTH_SHOTS" \
        --decoders "$name" --distances "$d" --bases "$b" --patches "$p" \
        ${mr[@]+"${mr[@]}"} ;;
  esac
}

# Continuous pool (portable, bash 3.2+) -- see eval_willow.sbatch for rationale.
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

# --- commit + push the CSVs (opt out with PUSH_RESULTS=0, the default here). ---
if [ "${PUSH_RESULTS:-0}" = 1 ]; then
  bash scripts/push_results.sh "$REPO" "$OUTDIR" "${RESULTS_DIR:-results}" || true
fi

echo "[$(date +%H:%M:%S)] ALL DONE: $njobs shards in $(fmt $(( $(date +%s) - WALL0 )))"
