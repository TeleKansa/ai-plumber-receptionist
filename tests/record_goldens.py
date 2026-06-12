"""Record golden behavior from the CURRENT code into tests/goldens/plumbing_call.json.

Run from repo root: python3 tests/record_goldens.py
Must be run on the pre-refactor code to capture the benchmark (provenance noted in output).
"""

import asyncio
import importlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.update({
    "OPENAI_API_KEY": "sk-test-dummy",
    "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
    "TWILIO_AUTH_TOKEN": "x" * 32,
    "TWILIO_PHONE_NUMBER": "+15550001111",
    "PLUMBER_PHONE_NUMBER": "+19998887777",
})

from tests.harness import run_scripted_call  # noqa: E402


def main():
    mod = importlib.import_module("main")
    record = asyncio.run(run_scripted_call(mod))
    record["recorded_from_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    out = ROOT / "tests" / "goldens" / "plumbing_call.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"goldens written: {out}")
    print(f"recorded_from_commit: {record['recorded_from_commit']}")
    print(f"oai frames: {len(record['oai_sent'])}, twilio ws frames: {len(record['twilio_ws_sent'])}, sms: {len(record['sms'])}")


if __name__ == "__main__":
    main()
