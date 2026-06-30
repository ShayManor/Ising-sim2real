#!/bin/bash -l
#SBATCH --job-name=ising_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:12:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/smoke%j.out
#SBATCH -e /scratch/gautschi/manors/logs/smoke%j.err
set -euo pipefail

# =============================================================================
# Fail-fast device sanity gate for the Ising pre-decoder. Run this BEFORE a full
# eval_d{3,5,7} sweep.
#
# Why this exists: the Ising decode is correct on CPU/MPS (d3 LER ~0.13) but a
# full CUDA run silently produced chance-level d3/d5 (~0.42/0.50) while d7 stayed
# correct -- a CUDA-only execution bug that a full sweep does not catch until 6h
# of compute later. This script decodes a few d3 configs on CPU AND (if present)
# CUDA at low shots and FAILS LOUDLY if:
#   * the classical MWPM baseline is weak (data/ingest broken), or
#   * Ising on CPU is at chance (model/adapter broken), or
#   * Ising on CUDA disagrees with CPU (the device bug above).
# Exit 0 = safe to launch the full sweep; exit 1 = do NOT trust a GPU sweep.
#
# Prereqs (once, on a login node): uv sync; uv run python scripts/setup_ising.py
# =============================================================================

# --- knobs (env-overridable) -------------------------------------------------
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
while [ "$REPO" != "/" ] && [ ! -e "$REPO/pyproject.toml" ]; do REPO="$(dirname "$REPO")"; done
[ -e "$REPO/pyproject.toml" ] || { echo "ERROR: repo root (pyproject.toml) not found above ${SLURM_SUBMIT_DIR:-$PWD}; set REPO=/path/to/Ising-sim2real"; exit 1; }
OUTDIR="${OUTDIR:-/scratch/gautschi/manors/ising_smoke}"   # smoke CSVs (NOT in git)
MODEL="${MODEL:-fast}"                                       # ising variant to probe
ROUNDS="${ROUNDS:-13}"                                       # single round count (fast)
SHOTS="${SHOTS:-2000}"                                       # low shot count -> seconds per config
LIMIT="${LIMIT:-4}"                                          # configs per shard (a few d3 patches)
SOURCE="${SOURCE:-auto}"                                     # auto = local tree if present, else HF
HF_REPO="${HF_REPO:-ShayManor/willow-surface-code-detection-events}"
WILLOW_RAW_DIR="${WILLOW_RAW_DIR:-$REPO/google_105Q_surface_code_d3_d5_d7}"
export WILLOW_RAW_DIR

# Pass/fail thresholds (override if your noise regime differs).
CHANCE_MAX="${CHANCE_MAX:-0.35}"   # Ising must beat this (chance = 0.5) to count as decoding
MWPM_MAX="${MWPM_MAX:-0.20}"       # classical baseline must be at least this strong
DRIFT_MAX="${DRIFT_MAX:-0.08}"     # max allowed |CUDA - CPU| Ising LER divergence

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
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

# --- device preflight ---------------------------------------------------------
GPU_COUNT=$(uv run python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
echo "source=$SOURCE model=$MODEL rounds=$ROUNDS shots=$SHOTS limit=$LIMIT gpus=$GPU_COUNT"
echo "thresholds: ising<$CHANCE_MAX  mwpm<$MWPM_MAX  |cuda-cpu|<$DRIFT_MAX"

probe () {  # $1=device $2=out.csv $3=decoders [$4=model-flag...]
  local dev="$1" out="$2"; shift 2
  echo "[$(date +%H:%M:%S)] probe $dev -> $(basename "$out")"
  uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" --device "$dev" \
    --distances 3 --bases Z --rounds "$ROUNDS" --shots "$SHOTS" --limit "$LIMIT" \
    --out "$out" "$@"
}

CPU_ISING="$OUTDIR/smoke_ising_cpu.csv"
CUDA_ISING="$OUTDIR/smoke_ising_cuda.csv"
CPU_MWPM="$OUTDIR/smoke_mwpm_cpu.csv"

probe cpu "$CPU_MWPM"  --decoders mwpm
probe cpu "$CPU_ISING" --decoders ising --model "$MODEL"
if [ "${GPU_COUNT:-0}" -gt 0 ]; then
  probe cuda "$CUDA_ISING" --decoders ising --model "$MODEL"
else
  echo "WARNING: no CUDA visible -> skipping the GPU vs CPU divergence check"
fi

# --- evaluate the gate --------------------------------------------------------
echo "----- smoke gate -----"
GPU_COUNT="$GPU_COUNT" CHANCE_MAX="$CHANCE_MAX" MWPM_MAX="$MWPM_MAX" DRIFT_MAX="$DRIFT_MAX" \
CPU_ISING="$CPU_ISING" CUDA_ISING="$CUDA_ISING" CPU_MWPM="$CPU_MWPM" \
uv run python - <<'PY'
import csv, math, os, sys

def mean_ler(path, decoder):
    if not path or not os.path.exists(path):
        return None
    vals = []
    for r in csv.DictReader(open(path)):
        if r["decoder"] != decoder:
            continue
        if r["note"].startswith(("skipped", "ising-failed")):
            continue
        try:
            v = float(r["ler"])
        except ValueError:
            continue
        if not math.isnan(v):
            vals.append(v)
    return sum(vals) / len(vals) if vals else None

chance = float(os.environ["CHANCE_MAX"]); mwpm_max = float(os.environ["MWPM_MAX"])
drift_max = float(os.environ["DRIFT_MAX"]); have_gpu = int(os.environ["GPU_COUNT"]) > 0

mwpm = mean_ler(os.environ["CPU_MWPM"], "mwpm")
cpu = mean_ler(os.environ["CPU_ISING"], "ising")
cuda = mean_ler(os.environ["CUDA_ISING"], "ising") if have_gpu else None

print(f"  mwpm (cpu)   = {mwpm}")
print(f"  ising (cpu)  = {cpu}")
print(f"  ising (cuda) = {cuda}")

fail = []
if mwpm is None or mwpm >= mwpm_max:
    fail.append(f"classical baseline weak/missing: mwpm={mwpm} (want <{mwpm_max}) -> data/ingest broken")
if cpu is None or cpu >= chance:
    fail.append(f"Ising at chance on CPU: ising={cpu} (want <{chance}) -> model/adapter broken")
if have_gpu:
    if cuda is None or cuda >= chance:
        fail.append(f"Ising at chance on CUDA: ising={cuda} (want <{chance}) -> GPU path broken; run the sweep on --device cpu")
    elif cpu is not None and abs(cuda - cpu) >= drift_max:
        fail.append(f"CUDA disagrees with CPU: |{cuda:.3f}-{cpu:.3f}|={abs(cuda-cpu):.3f} (want <{drift_max}) -> GPU path broken; run the sweep on --device cpu")

if fail:
    print("SMOKE TEST FAILED:")
    for f in fail:
        print("  - " + f)
    sys.exit(1)
print("SMOKE TEST PASSED: device path is sound; safe to launch the full sweep.")
PY
echo "[$(date +%H:%M:%S)] smoke done"
