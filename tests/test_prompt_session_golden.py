"""Golden regression baseline for the live-call session assembly (Phase C #3 safety net).

Captures the EXACT current output of build_session_update (instructions + tools +
realtime overrides) and the initial greeting response, for representative plumber
configs, BEFORE the core/vertical refactor. The core/vertical split MUST keep this
byte-identical (charter: plumber behavior preserved exactly). If this test fails
during the refactor, plumber behavior changed — investigate; do NOT blindly re-record.

Re-record intentionally with:  RECORD_GOLDEN=1 python -m pytest tests/test_prompt_session_golden.py
"""

import json
import os
from pathlib import Path

# Dummy env so `import main` (builds a Twilio client + Settings at import) works standalone.
for _k, _v in {
    "OPENAI_API_KEY": "sk-test",
    "TWILIO_ACCOUNT_SID": "AC" + "0" * 32,
    "TWILIO_AUTH_TOKEN": "0" * 32,
    "TWILIO_PHONE_NUMBER": "+15555550100",
    "PLUMBER_PHONE_NUMBER": "+15555550101",
    "HOST": "localhost",
    "PUBLIC_HOST": "localhost",
    "OAI_URL": "wss://example/realtime",
    "OPENAI_REALTIME_URL": "wss://example/realtime",
    "OPENAI_REALTIME_MODEL": "gpt-realtime-2",
    "ADMIN_PASSWORD": "secret",
}.items():
    os.environ.setdefault(_k, _v)

import main  # noqa: E402
from workflow.intake_policy import default_intake_policy  # noqa: E402

GOLDEN = Path(__file__).parent / "goldens" / "prompt_session_baseline.json"
CALLER = "+17327890675"
MODEL = "gpt-realtime-2"


def _resolve_greeting(tenant, profile):
    # Mirrors the live path (main.py): profile.greeting -> tenant.greeting -> DEFAULT_GREETING.
    return (profile.get("greeting") if profile else (tenant or {}).get("greeting")) or main.DEFAULT_GREETING


def _fixtures():
    plumber_tenant = {
        "slug": "default",
        "name": "Default Plumbing",
        "business_name": "Default Plumbing",
        "greeting": "Plumbing office, what's going on?",
    }
    # Mirrors production tenant_id=1 shape: property_role + additional_notes, plus a conditional question.
    policy_property_role = {
        "enabled": True,
        "extra_questions_json": json.dumps(
            [
                {
                    "key": "property_role",
                    "label": "Homeowner or renter",
                    "question_text": "Quick one — do you own the place or rent?",
                    "collection_mode": "ask_once",
                    "required": False,
                    "include_in_sms": True,
                    "include_in_admin": True,
                    "active": True,
                },
                {
                    "key": "additional_notes",
                    "label": "Additional notes",
                    "question_text": "Anything else the plumber should know before I send this over?",
                    "collection_mode": "ask_once",
                    "required": False,
                    "include_in_sms": True,
                    "include_in_admin": True,
                    "active": True,
                },
            ]
        ),
        "conditional_questions_json": json.dumps(
            [
                {
                    "key": "water_heater_age",
                    "label": "Water heater age",
                    "question_text": "Roughly how old is the water heater?",
                    "collection_mode": "passive",
                    "condition_type": "issue_contains",
                    "condition_keywords": ["water heater", "hot water"],
                    "active": True,
                }
            ]
        ),
        "sms_include_extra_fields_json": "[]",
        "admin_display_fields_json": "[]",
        "notes": "",
    }
    custom_profile = {
        "business_name": "Acme Plumbing",
        "greeting": "Acme plumbing, what's the problem?",
        "tone": "warmer, still efficient",
        "verbosity": "brief; one thing at a time",
        "closing_line": "You're all set, we'll ring you back.",
        "avoid_phrases_json": json.dumps(["no problem", "as an AI"]),
        "preferred_terms_json": json.dumps(["service address", "callback number"]),
        "extra_instructions_text": "If the caller mentions a coupon, say we'll note it.",
        "realtime_model": "",
    }
    return {
        "plumber_default": dict(tenant=plumber_tenant, profile=None, policy=default_intake_policy()),
        "plumber_property_role_intake": dict(tenant=plumber_tenant, profile=None, policy=policy_property_role),
        "plumber_custom_profile": dict(tenant=plumber_tenant, profile=custom_profile, policy=default_intake_policy()),
    }


def _generate():
    out = {}
    for name, f in _fixtures().items():
        session_update = main.build_session_update(CALLER, f["tenant"], f["profile"], f["policy"], MODEL)
        greeting = main.build_initial_greeting_response(_resolve_greeting(f["tenant"], f["profile"]))
        out[name] = {"session_update": session_update, "greeting": greeting}
    return out


def test_prompt_session_matches_golden():
    current = _generate()
    if os.environ.get("RECORD_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    assert GOLDEN.exists(), "Golden missing; run once with RECORD_GOLDEN=1 to capture the pre-refactor baseline."
    expected = json.loads(GOLDEN.read_text())
    assert current == expected, (
        "Live-call session output changed vs the golden baseline. The core/vertical refactor "
        "must keep plumber output byte-identical, so this should NOT differ. Investigate the diff; "
        "only re-record (RECORD_GOLDEN=1) if the change is genuinely intended and reviewed."
    )
