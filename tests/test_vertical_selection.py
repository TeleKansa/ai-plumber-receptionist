"""Tenant → vertical selection (Phase C #3, step 2 wiring).

Default / legacy tenants must keep getting the plumbing vertical (byte-identical,
also guarded by the golden test). A tenant bound to shoreline (by slug or by an
explicit vertical field) must get the shoreline vertical + its tool.
"""

import os

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

CALLER = "+12395550147"
MODEL = "gpt-realtime-2"


def _session(tenant):
    return main.build_session_update(CALLER, tenant, None, None, MODEL)["session"]


def test_default_tenant_uses_plumbing():
    s = _session({"slug": "default", "name": "Default Plumbing"})
    assert s["tools"][0]["name"] == "submit_service_request"
    assert "plumbing" in s["instructions"].lower()


def test_no_tenant_defaults_to_plumbing():
    s = _session(None)
    assert s["tools"][0]["name"] == "submit_service_request"


def test_shoreline_slug_selects_shoreline():
    s = _session({"slug": "shorelinecost", "name": "Shoreline Cost"})
    assert s["tools"][0]["name"] == "submit_project_inquiry"
    assert "Shoreline Cost" in s["instructions"]


def test_explicit_vertical_field_wins():
    s = _session({"slug": "anything", "name": "X", "vertical": "shoreline"})
    assert s["tools"][0]["name"] == "submit_project_inquiry"
