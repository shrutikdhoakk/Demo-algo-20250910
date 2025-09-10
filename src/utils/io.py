"""Utility helpers for configuration and output.

Currently only configuration reading is provided.  Additional I/O
functions can be added here as the project evolves.
"""

from __future__ import annotations

import yaml


def read_config(path: str) -> dict:
    """Load a YAML configuration file into a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)