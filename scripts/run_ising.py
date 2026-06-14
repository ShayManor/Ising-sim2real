#!/usr/bin/env python3
"""Load a pretrained Ising pre-decoder and run a forward pass.

This proves the model loads and runs on CPU (and on CUDA/MPS when present),
independent of the ingest pipeline. It feeds a synthetic binary syndrome batch of
shape (B, 4, T, D, D) and reports the output shape, parameter count, and latency.

Once the ingest pipeline + adapter (method steps 1 and 4) are built, the synthetic
input here is replaced by real Willow detection events via
`ising.adapter.detection_events_to_lattice`.

Usage:
    python scripts/run_ising.py --model fast --device auto
    python scripts/run_ising.py --model accurate --device cpu --batch 16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the package importable when run directly from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from ising_sim2real.device import describe_device, resolve_device  # noqa: E402
from ising_sim2real.ising.loader import ISING_MODELS, load_ising_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(ISING_MODELS), default="fast")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:N|mps")
    parser.add_argument("--batch", type=int, default=8, help="number of shots B")
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="rounds T for the synthetic input (default: model receptive field)",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=None,
        help="lattice size D for the synthetic input (default: model receptive field)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit weights path")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    model, info = load_ising_model(args.model, device=device, checkpoint=args.checkpoint)

    # The pre-decoder is fully convolutional, so T and D are free at inference;
    # default them to the receptive field R so the input is large enough.
    rounds = args.rounds or info.receptive_field
    distance = args.distance or info.receptive_field

    # Detection events are binary; sample a {0,1} syndrome volume.
    x = torch.randint(
        0, 2, (args.batch, info.input_channels, rounds, distance, distance),
        device=device, dtype=torch.float32,
    )

    with torch.no_grad():
        # Warm-up (allocates kernels / lazy init), then time the real pass.
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    print(f"model:            {info.name} (model_id={info.model_id})")
    # Guardrail: always report code distance d and receptive field R together.
    print(f"receptive field:  R={info.receptive_field}   (input lattice d={distance}, rounds T={rounds})")
    print(f"parameters:       {info.num_params:,}")
    print(f"checkpoint:       {info.checkpoint}")
    print(f"device:           {describe_device(device)}")
    print(f"input  shape:     {tuple(x.shape)}")
    print(f"output shape:     {tuple(y.shape)}  (channels: {info.out_channels})")
    print(f"forward latency:  {elapsed * 1e3:.2f} ms for B={args.batch}"
          f"  ({elapsed / args.batch * 1e3:.3f} ms/shot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
