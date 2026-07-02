#!/bin/bash -l
#SBATCH --job-name=ising_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
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
# of compute later.
#
# IMPORTANT: the bug only shows at production scale. A 2000-shot probe does ONE
# forward chunk and passes on CUDA (0.13, correct); the 50000-shot sweep chunks
# the forward (chunk ~16384) and the multi-chunk CUDA path returns chance. So this
# smoke MUST run full shots -- a low-shot probe is a false pass. It decodes a few
# d3 configs on CPU AND (if present) CUDA at full shots and FAILS LOUDLY if:
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
ROUNDS="${ROUNDS:-13}"                                       # single round count (fast per config)
# CPU LER is shot-invariant (~0.13 at 2k or 50k), so the CPU ground truth can be
# cheap. The CUDA probe MUST be full shots: the bug lives in the chunked forward
# (chunk ~16384), so a sub-chunk CUDA probe is a false pass.
SHOTS_CPU="${SHOTS_CPU:-5000}"                               # cheap CPU ground truth
SHOTS_CUDA="${SHOTS_CUDA:-50000}"                            # FULL shots (matches production)
LIMIT="${LIMIT:-4}"                                          # CPU-baseline configs (a few d3 patches, r13)
# The CUDA bug is a state-accumulation across a long shard, NOT a single forward:
# CUDA d3/r13 is correct in isolation (0.13) but chance in the full 135-config
# shard. So the CUDA probe must run a SHARD-LIKE workload in ONE process -- low +
# heavy rounds across several patches -- so the heavy configs can corrupt the
# later light ones. We gate only the LOW_ROUND configs: a working Ising at high
# rounds has total LER ~= 0.5 by construction, so only low rounds are chance-testable.
ROUNDS_CUDA="${ROUNDS_CUDA:-13,250}"                         # light + heavy, interleaved by patch
LIMIT_CUDA="${LIMIT_CUDA:-8}"                                # ~4 patches x {13,250} in one process
LOW_ROUND="${LOW_ROUND:-13}"                                 # the chance-testable round
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
echo "source=$SOURCE model=$MODEL shots_cpu=$SHOTS_CPU shots_cuda=$SHOTS_CUDA gpus=$GPU_COUNT"
echo "cpu baseline: r$ROUNDS x$LIMIT   cuda probe: r$ROUNDS_CUDA x$LIMIT_CUDA (gate low round r$LOW_ROUND)"
echo "thresholds: ising<$CHANCE_MAX  mwpm<$MWPM_MAX  |cuda-cpu|<$DRIFT_MAX"

probe () {  # $1=device $2=shots $3=rounds $4=limit $5=out.csv $6=decoders [$7=model-flag...]
  local dev="$1" shots="$2" rounds="$3" limit="$4" out="$5"; shift 5
  echo "[$(date +%H:%M:%S)] probe $dev shots=$shots rounds=$rounds -> $(basename "$out")"
  uv run ising-eval --source "$SOURCE" --hf-repo "$HF_REPO" --device "$dev" \
    --distances 3 --bases Z --rounds "$rounds" --shots "$shots" --limit "$limit" \
    --out "$out" "$@"
}

CPU_ISING="$OUTDIR/smoke_ising_cpu.csv"
CUDA_ISING="$OUTDIR/smoke_ising_cuda.csv"
CPU_MWPM="$OUTDIR/smoke_mwpm_cpu.csv"

probe cpu "$SHOTS_CPU" "$ROUNDS" "$LIMIT" "$CPU_MWPM"  --decoders mwpm
probe cpu "$SHOTS_CPU" "$ROUNDS" "$LIMIT" "$CPU_ISING" --decoders ising --model "$MODEL"
if [ "${GPU_COUNT:-0}" -gt 0 ]; then
  # ONE process, shard-like (light + heavy rounds across patches) so accumulation shows.
  probe cuda "$SHOTS_CUDA" "$ROUNDS_CUDA" "$LIMIT_CUDA" "$CUDA_ISING" --decoders ising --model "$MODEL"
else
  echo "WARNING: no CUDA visible -> skipping the GPU vs CPU divergence check"
fi

# --- evaluate the gate --------------------------------------------------------
echo "----- smoke gate -----"
GPU_COUNT="$GPU_COUNT" CHANCE_MAX="$CHANCE_MAX" MWPM_MAX="$MWPM_MAX" DRIFT_MAX="$DRIFT_MAX" LOW_ROUND="$LOW_ROUND" \
CPU_ISING="$CPU_ISING" CUDA_ISING="$CUDA_ISING" CPU_MWPM="$CPU_MWPM" \
uv run python - <<'PY'
import csv, math, os, sys

def lers(path, decoder, low_round_only=False, low_round=13):
    """LER values for a decoder; optionally only rows at the chance-testable round."""
    if not path or not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        if r["decoder"] != decoder:
            continue
        if r["note"].startswith(("skipped", "ising-failed")):
            continue
        if low_round_only and int(r["rounds"]) != low_round:
            continue
        try:
            v = float(r["ler"])
        except ValueError:
            continue
        if not math.isnan(v):
            out.append(v)
    return out

def mean(xs):
    return sum(xs) / len(xs) if xs else None

chance = float(os.environ["CHANCE_MAX"]); mwpm_max = float(os.environ["MWPM_MAX"])
drift_max = float(os.environ["DRIFT_MAX"]); have_gpu = int(os.environ["GPU_COUNT"]) > 0
low_round = int(os.environ["LOW_ROUND"])

mwpm = mean(lers(os.environ["CPU_MWPM"], "mwpm"))
cpu = mean(lers(os.environ["CPU_ISING"], "ising"))
# Only the low-round CUDA configs are chance-testable; we take the WORST one so a
# single accumulation-corrupted config (a late r13 after heavy r250s) trips the gate.
cuda_low = lers(os.environ["CUDA_ISING"], "ising", low_round_only=True, low_round=low_round)
cuda_worst = max(cuda_low) if cuda_low else None
cuda_mean = mean(cuda_low)

print(f"  mwpm (cpu)          = {mwpm}")
print(f"  ising (cpu, r{low_round})     = {cpu}")
print(f"  ising (cuda, r{low_round}) n={len(cuda_low)} mean={cuda_mean} worst={cuda_worst}")

fail = []
if mwpm is None or mwpm >= mwpm_max:
    fail.append(f"classical baseline weak/missing: mwpm={mwpm} (want <{mwpm_max}) -> data/ingest broken")
if cpu is None or cpu >= chance:
    fail.append(f"Ising at chance on CPU: ising={cpu} (want <{chance}) -> model/adapter broken")
if have_gpu:
    if cuda_worst is None or cuda_worst >= chance:
        fail.append(f"Ising at chance on CUDA (worst r{low_round}={cuda_worst}, want <{chance}) -> GPU path corrupts at shard scale; run the sweep on --device cpu")
    elif cpu is not None and cuda_mean is not None and abs(cuda_mean - cpu) >= drift_max:
        fail.append(f"CUDA disagrees with CPU: |{cuda_mean:.3f}-{cpu:.3f}|={abs(cuda_mean-cpu):.3f} (want <{drift_max}) -> run the sweep on --device cpu")

if fail:
    print("SMOKE TEST FAILED:")
    for f in fail:
        print("  - " + f)
    sys.exit(1)
print("SMOKE TEST PASSED: device path is sound; safe to launch the full sweep.")
PY
echo "[$(date +%H:%M:%S)] smoke done"
