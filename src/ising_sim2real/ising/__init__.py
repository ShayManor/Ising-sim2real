"""NVIDIA Ising pre-decoder integration: loading the pretrained weights and
mapping Willow detection events into the model's lattice layout.
"""

from ising_sim2real.ising.loader import (
    ISING_MODELS,
    IsingModelInfo,
    load_ising_model,
)

__all__ = ["ISING_MODELS", "IsingModelInfo", "load_ising_model"]
