"""Tests for shoreline lead delivery (contract §3 payload + §4 webhook)."""

import asyncio

from workflow.lead_delivery import (
    DeliveryResult,
    auth_headers,
    build_shoreline_lead,
    deliver_lead_webhook,
    deliver_shoreline_lead,
    webhook_url,
)

SHORELINE_VERTICAL = {
    "delivery": {
        "url_env": "SHORELINE_LEAD_WEBHOOK_URL",
        "auth_header": "x-loopline-webhook-secret",
        "secret_env": "SHORELINE_LEAD_WEBHOOK_SECRET",
    }
}

ARGS = {
    "name": "Sam",
    "callback": "239-555-0147",
    "email": "sam@example.com",
    "zip_code": "33904",
    "project_type": "seawall repair",
    "water_setting": "coastal or saltwater",
    "approx_size_ft": "about 60 feet",
    "condition": "moderate damage",
    "access": "road or driveway",
    "timeline": "within 30 days",
    "urgency": "HIGH",
    "ownership_confirmed": True,
    "consent": True,
    "extra_fields": {"additional_notes": "gate code 1234"},
}


def test_build_shoreline_lead_maps_contract_schema():
    lead = build_shoreline_lead(ARGS, call_sid="CA1", from_number="+12395550147")
    assert lead["source"] == "phone"
    assert lead["caller_name"] == "Sam"
    assert lead["callback_phone"] == "239-555-0147"
    assert lead["zip"] == "33904"
    assert lead["project_type"] == "seawall repair"
    assert lead["timeline"] == "within 30 days"
    assert lead["urgency"] == "HIGH"
    assert lead["ownership_confirmed"] is True
    assert lead["consent"] is True
    assert lead["consent_timestamp"]  # set when consent true
    assert lead["qualification_status"] == "qualified"
    assert lead["notes"] == "gate code 1234"
    # schema completeness: all contract §3 keys present
    for key in [
        "lead_id", "timestamp", "source", "caller_name", "callback_phone", "email",
        "zip", "market", "project_type", "water_setting", "approx_size_ft", "condition",
        "access", "timeline", "urgency", "ownership_confirmed", "consent",
        "consent_timestamp", "qualification_status", "transfer_outcome", "transferred_to",
        "recording_url", "transcript_summary", "notes",
    ]:
        assert key in lead, f"missing §3 field: {key}"


def test_callback_falls_back_to_caller_number():
    lead = build_shoreline_lead({"consent": True}, from_number="+12395550147")
    assert lead["callback_phone"] == "+12395550147"


def test_callback_alias_phrase_uses_caller_number():
    lead = build_shoreline_lead(
        {"name": "Sam", "callback": "this number is good", "consent": True},
        from_number="+12395550147",
    )
    assert lead["callback_phone"] == "+12395550147"


def test_no_consent_leaves_consent_timestamp_blank():
    lead = build_shoreline_lead({"consent": False})
    assert lead["consent"] is False
    assert lead["consent_timestamp"] == ""


def test_webhook_url_reads_env(monkeypatch):
    monkeypatch.setenv("SHORELINE_LEAD_WEBHOOK_URL", "https://example.test/leads")
    assert webhook_url({"url_env": "SHORELINE_LEAD_WEBHOOK_URL"}) == "https://example.test/leads"
    assert webhook_url({}) == ""


def test_deliver_skips_when_no_url():
    result = asyncio.run(deliver_lead_webhook("", {"x": 1}))
    assert isinstance(result, DeliveryResult)
    assert result.delivered is False
    assert result.skipped_reason == "no_webhook_url_configured"


def test_deliver_success_on_2xx():
    async def fake_post(url, payload, headers):
        assert url == "https://example.test/leads"
        assert payload["source"] == "phone"
        return 200

    payload = build_shoreline_lead(ARGS, from_number="+12395550147")
    result = asyncio.run(deliver_lead_webhook("https://example.test/leads", payload, post_func=fake_post))
    assert result.delivered is True
    assert result.status_code == 200


def test_deliver_failure_on_5xx():
    async def fake_post(url, payload, headers):
        return 503

    result = asyncio.run(deliver_lead_webhook("https://example.test/leads", {"x": 1}, post_func=fake_post))
    assert result.delivered is False
    assert result.status_code == 503
    assert "503" in (result.error or "")


def test_deliver_shoreline_lead_consent_declined_never_delivers():
    res = asyncio.run(deliver_shoreline_lead({"name": "X", "consent": False}, vertical=SHORELINE_VERTICAL))
    assert res["delivered"] is False
    assert res["skipped_reason"] == "consent_declined"
    assert res["consent"] is False


def test_deliver_shoreline_lead_delivers_with_url_and_secret_header(monkeypatch):
    monkeypatch.setenv("SHORELINE_LEAD_WEBHOOK_URL", "https://example.test/leads")
    monkeypatch.setenv("SHORELINE_LEAD_WEBHOOK_SECRET", "s3cr3t")
    seen = {}

    async def fake_post(url, payload, headers):
        assert payload["consent"] is True
        seen["headers"] = headers
        return 200

    res = asyncio.run(deliver_shoreline_lead(dict(ARGS), vertical=SHORELINE_VERTICAL, post_func=fake_post))
    assert res["delivered"] is True
    assert res["consent"] is True
    assert res["status_code"] == 200
    assert seen["headers"].get("x-loopline-webhook-secret") == "s3cr3t"


def test_auth_headers_from_env(monkeypatch):
    monkeypatch.setenv("SHORELINE_LEAD_WEBHOOK_SECRET", "s3cr3t")
    assert auth_headers(SHORELINE_VERTICAL["delivery"]) == {"x-loopline-webhook-secret": "s3cr3t"}


def test_auth_headers_empty_when_unset(monkeypatch):
    monkeypatch.delenv("SHORELINE_LEAD_WEBHOOK_SECRET", raising=False)
    assert auth_headers(SHORELINE_VERTICAL["delivery"]) == {}
    assert auth_headers({}) == {}


def test_deliver_shoreline_lead_skips_when_url_unset(monkeypatch):
    monkeypatch.delenv("SHORELINE_LEAD_WEBHOOK_URL", raising=False)
    res = asyncio.run(deliver_shoreline_lead(dict(ARGS), vertical=SHORELINE_VERTICAL))
    assert res["delivered"] is False
    assert res["skipped_reason"] == "no_webhook_url_configured"
