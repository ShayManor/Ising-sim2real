#!/bin/bash -l
#SBATCH --job-name=ising_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/smoke%j.out
#SBATCH -e /scratch/gautschi/manors/logs/smoke%j.err
set -euo pipefail

# =============================================================================
# CPU correctness + runtime-projection gate. Run this BEFORE eval_d3/eval_d7.
#
# Ising runs on CPU (the CUDA forward corrupts at shard scale: correct in
# isolation, chance-level across a full 135-config shard -- a state accumulation
# no cheap probe reproduces, so we don't chase it, we avoid it). This gate
# certifies the path we actually use:
#   1. classical MWPM decodes            -> data/ingest OK
#   2. Ising on CPU is below chance      -> model/adapter OK
#   3. the full d3 Ising sweep FITS the wall clock -- measured, not guessed: time
#      ONE heavy config at the real per-shard thread budget (CORES/NPAR) and
#      project the whole sweep. This is the check a fast correctness-only smoke
#      was missing.
# Exit 0 = safe to launch the sweep; exit 1 = do NOT (reason printed).
#
# Prereqs (once, on a login node): uv sync; uv run python scripts/setup_ising.py
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [ "$REPO" != "/" ] && [ ! -e "$REPO/pyproject.toml" ]; do REPO="$(dirname "$REPO")"; done
[ -e "$REPO/pyproject.toml" ] || { echo "ERROR: repo root (pyproject.toml) not found above ${SLURM_SUBMIT_DIR:-$PWD}; set REPO=/path/to/Ising-sim2real"; exit 1; }
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/ising_smoke}"   # smoke CSVs (NOT in git)
MODEL="${MODEL:-fast}"                                       # ising variant to probe
SOURCE="${SOURCE:-auto}"                                     # auto = local tree if present, else HF
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
WILLOW_RAW_DIR="${WILLOW_RAW_DIR:-$REPO/google_105Q_surface_code_d3_d5_d7}"
export WILLOW_RAW_DIR

# Correctness probe (cheap; CPU LER is shot-invariant so 5k is plenty).
CORR_SHOTS="${CORR_SHOTS:-5000}"
CORR_ROUNDS="${CORR_ROUNDS:-13}"
CORR_LIMIT="${CORR_LIMIT:-4}"
CHANCE_MAX="${CHANCE_MAX:-0.35}"   # Ising must beat this (chance = 0.5)
MWPM_MAX="${MWPM_MAX:-0.20}"       # classical baseline must be at least this strong

# Runtime projection -- MUST mirror the eval scripts' knobs to be meaningful.
NPAR="${NPAR:-4}"                                   # concurrent ising shards in the eval
ISING_MAX_ROUNDS="${ISING_MAX_ROUNDS:-70}"          # eval's round cap ('' = all rounds)
PROBE_ROUNDS="${PROBE_ROUNDS:-${ISING_MAX_ROUNDS:-250}}"  # heaviest config the sweep runs
TIME_BUDGET_H="${TIME_BUDGET_H:-11}"                # project under this (margin below 12h wall)
SWEEP_PATCHES="${SWEEP_PATCHES:-9}"                 # d3 patch count (worst case of the two jobs)
SWEEP_BASES="${SWEEP_BASES:-2}"                     # X,Z
SWEEP_MODELS="${SWEEP_MODELS:-2}"                   # fast,accurate

export HF_HOME="${HF_HOME:-$OUTDIR/hf_cache}"
cd "$REPO"
mkdir -p "$OUTDIR"

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

# --- Ising decoder setup: vendored NVIDIA code + weights (gitignored) ---------
have_ising_weights () { ls models/ising/*.pt >/dev/null 2>&1 || ls models/ising/*.safetensors >/dev/null 2>&1; }
if [ ! -d third_party/Ising-Decoding/code ] || ! have_ising_weights; then
  echo "setting up vendored Ising decoder (scripts/setup_ising.py) ..."
  uv run python scripts/setup_ising.py || { echo "ERROR: Ising setup failed (needs git-lfs + internet)."; exit 1; }
fi

CORES="${SLURM_CPUS_PER_TASK:-96}"
[ "$NPAR" -lt 1 ] && NPAR=1
ITHREADS=$(( CORES / NPAR )); [ "$ITHREADS" -lt 1 ] && ITHREADS=1
echo "source=$SOURCE model=$MODEL cores=$CORES npar=$NPAR ithreads=$ITHREADS max-rounds=${ISING_MAX_ROUNDS:-all}"
echo "correctness: r$CORR_ROUNDS x$CORR_LIMIT @ ${CORR_SHOTS}sh   |   timing: 1x r$PROBE_ROUNDS @ 50k, $ITHREADS threads"
echo "thresholds: ising<$CHANCE_MAX  mwpm<$MWPM_MAX  project<${TIME_BUDGET_H}h"

CPU_MWPM="$OUTDIR/smoke_mwpm_cpu.csv"
CPU_ISING="$OUTDIR/smoke_ising_cpu.csv"
TIMING_CSV="$OUTDIR/smoke_timing_cpu.csv"

# --- correctness probes (cheap) ----------------------------------------------
echo "[$(date +%H:%M:%S)] correctness: mwpm"
uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" --device cpu \
  --distances 3 --bases Z --rounds "$CORR_ROUNDS" --shots "$CORR_SHOTS" --limit "$CORR_LIMIT" \
  --decoders mwpm --out "$CPU_MWPM"
echo "[$(date +%H:%M:%S)] correctness: ising"
uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" --device cpu \
  --distances 3 --bases Z --rounds "$CORR_ROUNDS" --shots "$CORR_SHOTS" --limit "$CORR_LIMIT" \
  --decoders ising --model "$MODEL" --out "$CPU_ISING"

# --- timing probe: ONE heavy config at the eval's per-shard thread budget -----
echo "[$(date +%H:%M:%S)] timing: 1 cfg r$PROBE_ROUNDS @ 50k, $ITHREADS threads"
OMP_NUM_THREADS="$ITHREADS" MKL_NUM_THREADS="$ITHREADS" \
uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" --device cpu \
  --distances 3 --bases Z --rounds "$PROBE_ROUNDS" --shots 50000 --limit 1 \
  --decoders ising --model "$MODEL" --out "$TIMING_CSV"

# --- evaluate the gate --------------------------------------------------------
echo "----- smoke gate -----"
CHANCE_MAX="$CHANCE_MAX" MWPM_MAX="$MWPM_MAX" CPU_MWPM="$CPU_MWPM" CPU_ISING="$CPU_ISING" \
TIMING_CSV="$TIMING_CSV" PROBE_ROUNDS="$PROBE_ROUNDS" ISING_MAX_ROUNDS="$ISING_MAX_ROUNDS" \
NPAR="$NPAR" TIME_BUDGET_H="$TIME_BUDGET_H" SWEEP_PATCHES="$SWEEP_PATCHES" \
SWEEP_BASES="$SWEEP_BASES" SWEEP_MODELS="$SWEEP_MODELS" \
uv run python - <<'PY'
import csv, math, os, sys

# All round counts the sweep decodes (r01 is auto-skipped: Ising needs T>=2).
ROUND_SET = [10, 13, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250]

def rows(path, decoder):
    if not path or not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        if r["decoder"] != decoder or r["note"].startswith(("skipped", "ising-failed")):
            continue
        out.append(r)
    return out

def mean_ler(path, decoder):
    vs = []
    for r in rows(path, decoder):
        try:
            v = float(r["ler"])
        except ValueError:
            continue
        if not math.isnan(v):
            vs.append(v)
    return sum(vs) / len(vs) if vs else None

chance = float(os.environ["CHANCE_MAX"]); mwpm_max = float(os.environ["MWPM_MAX"])
mwpm = mean_ler(os.environ["CPU_MWPM"], "mwpm")
cpu = mean_ler(os.environ["CPU_ISING"], "ising")
print(f"  mwpm (cpu)   = {mwpm}")
print(f"  ising (cpu)  = {cpu}")

# --- runtime projection from the heavy timing config -------------------------
probe_rounds = int(os.environ["PROBE_ROUNDS"])
cap = os.environ["ISING_MAX_ROUNDS"].strip()
cap = int(cap) if cap else None
npar = max(1, int(os.environ["NPAR"]))
budget_s = float(os.environ["TIME_BUDGET_H"]) * 3600
patches = int(os.environ["SWEEP_PATCHES"])
n_shards = int(os.environ["SWEEP_BASES"]) * int(os.environ["SWEEP_MODELS"])

timing = rows(os.environ["TIMING_CSV"], "ising")
probe_secs = None
for r in timing:
    if int(r["rounds"]) == probe_rounds:
        try:
            probe_secs = float(r["decode_seconds"])
        except ValueError:
            pass
        break

proj_h = None
if probe_secs and probe_secs > 0:
    per_round = probe_secs / probe_rounds                       # decode-sec per round at 50k, ITHREADS
    sum_rounds = sum(r for r in ROUND_SET if cap is None or r <= cap)
    waves = math.ceil(n_shards / npar)                          # shard groups run back-to-back
    proj_s = patches * sum_rounds * per_round * waves           # wall-clock for the heaviest job (d3)
    proj_h = proj_s / 3600
    print(f"  timing: r{probe_rounds} 50k = {probe_secs:.1f}s ({per_round:.3f}s/round)")
    print(f"  projected d3 Ising sweep: {patches} patches x sum_rounds({sum_rounds}) x {waves} wave(s) "
          f"= {proj_h:.1f}h  (budget {budget_s/3600:.0f}h)")

fail = []
if mwpm is None or mwpm >= mwpm_max:
    fail.append(f"classical baseline weak/missing: mwpm={mwpm} (want <{mwpm_max}) -> data/ingest broken")
if cpu is None or cpu >= chance:
    fail.append(f"Ising at chance on CPU: ising={cpu} (want <{chance}) -> model/adapter broken")
if proj_h is None:
    fail.append("timing probe produced no decode_seconds -> cannot project runtime")
elif proj_h * 3600 >= budget_s:
    fail.append(f"projected {proj_h:.1f}h exceeds {budget_s/3600:.0f}h -> lower ISING_MAX_ROUNDS or raise NPAR")

if fail:
    print("SMOKE TEST FAILED:")
    for f in fail:
        print("  - " + f)
    sys.exit(1)
print("SMOKE TEST PASSED: CPU path correct and the sweep is projected to fit the wall clock.")
PY
echo "[$(date +%H:%M:%S)] smoke done"
