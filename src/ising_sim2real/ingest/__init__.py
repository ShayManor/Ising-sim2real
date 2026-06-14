"""Ingest pipeline for Google Willow hardware data.

Loads Willow circuits and pre-computed detection events, and derives detection
events and observable flips ready for any decoder in the panel.

Modules:
    willow.py     — load circuits and pre-computed detection data from disk
    detectors.py  — load_detection_data (b8 files) + measurements_to_detectors (m2d)
    dataset.py    — discover and filter (distance, basis, orientation, rounds) configs
"""
