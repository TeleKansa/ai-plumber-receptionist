"""P1 acceptance: the refactored engine must behave byte-identically to the
pre-refactor code, as recorded in tests/goldens/plumbing_call.json (recorded at
commit fb0e720, before any refactor). Charter Verification Protocol: scripted
media-stream test."""

import asyncio
import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.update({
    "OPENAI_API_KEY": "sk-test-dummy",
    "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
    "TWILIO_AUTH_TOKEN": "x" * 32,
    "TWILIO_PHONE_NUMBER": "+15550001111",
    "PLUMBER_PHONE_NUMBER": "+19998887777",
})

import main  # noqa: E402  (tenant entrypoint sets VERTICAL/PUBLIC_HOST)
import core.engine as engine  # noqa: E402
from tests.harness import run_scripted_call, CALLER  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "goldens" / "plumbing_call.json").read_text())


@pytest.fixture(scope="module")
def record():
    return asyncio.new_event_loop().run_until_complete(run_scripted_call(engine))


def test_twiml_identical(record):
    assert record["twiml"] == GOLDEN["twiml"]


def test_openai_url_identical(record):
    assert record["oai_connect_url"] == GOLDEN["oai_connect_url"]


def test_session_update_identical(record):
    new, old = record["oai_sent"][0], GOLDEN["oai_sent"][0]
    assert new == old, "session.update payload diverged"
    # explicit: instructions byte-identical
    assert new["session"]["instructions"] == old["session"]["instructions"]
    assert new["session"]["tools"] == old["session"]["tools"]


def test_audio_in_frame_identical(record):
    assert record["oai_sent"][1] == GOLDEN["oai_sent"][1], "caller-audio transcode diverged"


def test_greeting_identical(record):
    assert record["oai_sent"][2] == GOLDEN["oai_sent"][2], "greeting payload diverged"


def test_all_oai_frames_identical(record):
    assert record["oai_sent"] == GOLDEN["oai_sent"]


def test_audio_out_to_twilio_identical(record):
    assert record["twilio_ws_sent"] == GOLDEN["twilio_ws_sent"], "AI-audio transcode diverged"


def test_sms_lead_identical(record):
    assert record["sms"] == GOLDEN["sms"], "lead SMS body/recipients diverged"


def test_session_endstate_identical(record):
    assert record["session_state"] == GOLDEN["session_state"]


def test_helper_functions_identical():
    assert engine.make_instructions(CALLER) == GOLDEN["oai_sent"][0]["session"]["instructions"]
    assert engine.build_session_update(CALLER) == GOLDEN["oai_sent"][0]
