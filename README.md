# Ising-sim2real

Benchmarks a panel of open surface-code decoders on real Google Willow hardware
data and on synthetic circuit-level noise of rising fidelity, and measures whether
the cheap synthetic benchmark predicts the real-hardware result.

## Status

Scaffolded. The **NVIDIA Ising pre-decoder runs end to end** (CPU and GPU). The
ingest pipeline and decoder adapters are stubbed interfaces, not yet implemented.

## Layout

```
src/ising_sim2real/
  device.py           torch device resolver (auto / cpu / cuda / mps)
  paths.py            canonical repo paths
  ising/
    loader.py         load NVIDIA Ising weights and build the model   [implemented]
    adapter.py        detection events -> (B,4,T,D,D) lattice          [scaffold]
  ingest/             Willow loader, m2d detectors, config discovery   [scaffold]
  decoders/base.py    common Decoder interface for the panel           [scaffold]
scripts/
  setup_ising.py      clone NVIDIA/Ising-Decoding + pull weights
  run_ising.py        load a model and run a forward pass
tests/
  test_ising_smoke.py model loads + CPU forward pass
third_party/          vendored NVIDIA/Ising-Decoding clone   (gitignored)
models/  data/  outputs/                                     (gitignored)
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `git` with `git-lfs`.

```bash
uv sync                 # base env (CPU torch); add --extra dev for pytest/ruff
uv run python scripts/setup_ising.py    # clone NVIDIA repo + pull pretrained weights
```

`setup_ising.py` shallow-clones `NVIDIA/Ising-Decoding` into `third_party/` at a
pinned commit and pulls the two `.pt` weights via git-lfs into `models/ising/`.
Use `--from-huggingface` to fetch the fp16 safetensors from Hugging Face instead.

## Running the Ising pre-decoder

```bash
uv run python scripts/run_ising.py --model fast      --device auto
uv run python scripts/run_ising.py --model accurate  --device cpu  --batch 16
```

Two public models (`code/tests/test_inference_public_model.py` in the vendored repo
confirms the mapping):

| name       | model_id | receptive field R | params    |
|------------|----------|-------------------|-----------|
| `fast`     | 1        | 9                 | 912,772   |
| `accurate` | 4        | 13                | 1,797,764 |

Input layout is `(B, 4, T, D, D)` → output `(B, 4, T, D, D)`. The network is fully
convolutional, so `T`/`D` are free at inference. Per the project guardrail,
`run_ising.py` always reports code distance `d` and receptive field `R` together.

### CPU and GPU

`--device auto` uses CUDA when available and falls back to CPU; `cpu`, `cuda`,
`cuda:N`, and `mps` (Apple Metal) are also accepted. The base `uv sync` installs a
CPU torch build. For an NVIDIA GPU, install a CUDA build of torch, e.g.:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

No code changes are needed — `device.py` moves the model to whichever device you
request.

## Tests

```bash
uv sync --extra dev
uv run pytest          # smoke test: each model loads + forward pass on CPU
```
