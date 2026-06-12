"""Contract §1.1 / charter P1: zero plumber-specific logic may remain in core."""

import pathlib
import re

CORE = pathlib.Path(__file__).resolve().parent.parent / "core"
FORBIDDEN = re.compile(r"plumb|PLUMB|dispatcher|submit_service_request", re.IGNORECASE)


def test_core_has_no_vertical_strings():
    offenders = []
    for f in CORE.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                offenders.append(f"{f.name}:{i}: {line.strip()}")
    assert not offenders, "vertical-specific strings found in core:\n" + "\n".join(offenders)
