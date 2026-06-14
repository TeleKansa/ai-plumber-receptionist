"""Vertical config loader.

A vertical is a JSON file in verticals/<name>.json carrying the industry-specific
pieces the engine needs: defaults (greeting, tone, style fallbacks), an
instructions_template (string.Template with $placeholders the engine fills), and
the function tool schema. This loader is the ONLY bridge between core and a
vertical; core code stays industry-agnostic.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

VERTICALS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verticals")


@lru_cache(maxsize=None)
def load_vertical(name: str) -> dict:
    """Load and cache verticals/<name>.json as a plain dict (read-only by convention)."""
    path = os.path.join(VERTICALS_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
