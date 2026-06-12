"""Vertical config loading — the ONLY place industry-specific behavior enters the engine.

A vertical is a JSON file in /verticals/<name>.json carrying: brand prompt, greeting,
tool schemas, and lead-delivery spec. Core engine code must contain zero
industry-specific strings or logic (enforced by tests/test_no_leakage.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERTICALS_DIR = Path(__file__).resolve().parent.parent / "verticals"


@dataclass(frozen=True)
class Vertical:
    name: str
    system_prompt: str          # may contain {caller_number}
    greeting_instruction: str   # verbatim response.create instructions for the opening line
    tools: list                 # OpenAI Realtime function tool schemas
    submit_tool_name: str       # function call that finalizes a lead
    delivery: dict              # lead delivery spec (channel, env var names, body fields)


def load_vertical(name: str) -> Vertical:
    path = VERTICALS_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return Vertical(
        name=data["name"],
        system_prompt=data["system_prompt"],
        greeting_instruction=data["greeting_instruction"],
        tools=data["tools"],
        submit_tool_name=data["submit_tool_name"],
        delivery=data["delivery"],
    )


def build_lead_body(vertical: Vertical, info: dict, caller_number: str) -> str:
    """Render the lead notification body from the vertical's delivery spec."""
    d = vertical.delivery
    missing = d.get("missing_value", "N/A")
    lines = [d["header"], ""]
    for f in d["fields"]:
        if f.get("fallback") == "caller_number":
            value = info.get(f["key"], caller_number)
        else:
            value = info.get(f["key"], missing)
        lines.append(f"{f['label']}: {value}")
    return "\n".join(lines)
