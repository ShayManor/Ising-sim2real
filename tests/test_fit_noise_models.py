"""Unit test for fit_noise_models.py's config-selection logic (offline, no
network -- the actual per-patch fit is exercised in Task 10's real-data
integration test, not here)."""

from __future__ import annotations

from ising_sim2real.ingest.willow import WillowConfig


def test_fit_set_configs_for_patch():
    from scripts.fit_noise_models import FIT_ROUNDS, fit_set_configs_for_patch

    configs = fit_set_configs_for_patch("q4_5", distance=3)
    assert len(configs) == 2 * len(FIT_ROUNDS)
    assert all(c.orientation == "q4_5" for c in configs)
    assert all(c.distance == 3 for c in configs)
    assert {c.rounds for c in configs} == set(FIT_ROUNDS)
    assert {c.basis for c in configs} == {"X", "Z"}


def test_fit_set_and_eval_set_are_disjoint_and_cover_all_rounds():
    from scripts.fit_noise_models import EVAL_ROUNDS, FIT_ROUNDS, STD_ROUNDS

    assert set(FIT_ROUNDS) & set(EVAL_ROUNDS) == set()
    assert set(FIT_ROUNDS) | set(EVAL_ROUNDS) == set(STD_ROUNDS)


def test_all_patches_keeps_both_distances_for_shared_orientation(monkeypatch):
    """q6_7 must survive as TWO distinct entries (d3 and d7), not collapse into
    one -- this is the regression test for the collision found during planning."""
    from scripts.fit_noise_models import _all_patches
    import scripts.fit_noise_models as mod

    fake_configs = [
        WillowConfig(distance=3, basis="Z", rounds=1, orientation="q6_7"),
        WillowConfig(distance=7, basis="Z", rounds=1, orientation="q6_7"),
        WillowConfig(distance=3, basis="Z", rounds=1, orientation="q4_5"),
    ]
    monkeypatch.setattr(mod, "discover_configs_hf", lambda repo: fake_configs)

    patches = _all_patches("fake-repo")
    assert ("q6_7", 3) in patches
    assert ("q6_7", 7) in patches
    assert ("q4_5", 3) in patches
    assert len(patches) == 3
