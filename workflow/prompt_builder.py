"""Plumbing-tenant entrypoint for prompt building.

Thin compatibility shim over the industry-agnostic core engine: it binds the
"plumbing" vertical (verticals/plumbing.json) and exposes the same surface the
rest of the app already imports (PromptBuilder, prompt_profile_defaults,
DEFAULT_GREETING). All plumbing wording now lives in the vertical config; all
assembly logic lives in core.engine — nothing plumbing-specific remains here.
"""

from __future__ import annotations

from typing import Optional

from core.engine import build_instructions, profile_defaults
from core.vertical import load_vertical

VERTICAL_NAME = "plumbing"

# Backwards-compatible constant (main.py uses it as the greeting fallback).
DEFAULT_GREETING = load_vertical(VERTICAL_NAME)["defaults"]["greeting"]


def prompt_profile_defaults(tenant: Optional[dict] = None) -> dict:
    """Default prompt profile for a plumbing tenant (used by storage.repository)."""
    return profile_defaults(load_vertical(VERTICAL_NAME), tenant)


class PromptBuilder:
    def build(
        self,
        caller_number: str,
        tenant: Optional[dict] = None,
        profile: Optional[dict] = None,
        intake_policy: Optional[dict] = None,
    ) -> str:
        return build_instructions(
            load_vertical(VERTICAL_NAME),
            caller_number,
            tenant=tenant,
            profile=profile,
            intake_policy=intake_policy,
        )
