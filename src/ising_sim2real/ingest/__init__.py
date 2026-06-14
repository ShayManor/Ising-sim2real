"""Ingest pipeline (SCAFFOLD ONLY).

Loads Google Willow circuits + measurement records and derives detection events
and observable flips. These modules define the interfaces the pipeline will fill
in; the bodies raise NotImplementedError until step 1 of the method is built.

Method order (see CLAUDE.md):
    willow.py     -> load circuits and measurement records
    detectors.py  -> measurement -> detector conversion via Stim
    dataset.py    -> enumerate (distance, basis, orientation) configs
"""
