"""Unit test for willow.py's patch_key -- exists specifically because
`orientation` alone collides across distances (see slurm/eval_synth.sbatch's
patches_for(): "q6_7" is used by both a d3 patch and the d7 patch)."""

from __future__ import annotations


def test_patch_key_disambiguates_shared_orientation_across_distances():
    from ising_sim2real.ingest.willow import patch_key

    assert patch_key(3, "q6_7") != patch_key(7, "q6_7")


def test_patch_key_is_deterministic():
    from ising_sim2real.ingest.willow import patch_key

    assert patch_key(5, "q4_7") == patch_key(5, "q4_7")
