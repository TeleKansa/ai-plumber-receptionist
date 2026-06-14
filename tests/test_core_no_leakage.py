"""Enforce that the core engine stays industry-agnostic (contract §1.1).

No plumbing-specific (or any vertical-specific) wording or the plumbing tool name
may appear in core/*.py. Industry behavior must live only in verticals/<name>.json.
"""

import pathlib

CORE_DIR = pathlib.Path(__file__).resolve().parent.parent / "core"

# Hard tells that a vertical's domain logic has leaked into the engine.
BANNED_SUBSTRINGS = [
    "plumb",            # plumbing, plumber
    "leak",
    "drain",
    "toilet",
    "water heater",
    "shut the water",
    "shutoff",
    "dispatcher",
    "service address",
    "submit_service_request",  # the plumbing tool name — must be vertical-driven
]


def test_core_contains_no_vertical_specific_strings():
    offenders = []
    for path in sorted(CORE_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for term in BANNED_SUBSTRINGS:
            if term in text:
                offenders.append(f"{path.relative_to(CORE_DIR.parent)} contains '{term}'")
    assert not offenders, "core/ must be industry-agnostic; found vertical-specific content:\n" + "\n".join(offenders)
