"""Lead delivery for verticals whose leads go to an external buyer (e.g. shoreline).

Plumbing leads are delivered by SMS via workflow/notifications + service_request; this
module is the separate path for webhook-delivered verticals. It builds the contract
§3 lead JSON and POSTs it to the webhook URL named by the vertical's delivery spec
(`url_env`). The URL VALUE is a deploy-time env var the owner sets once ShorelineCost's
endpoint exists; if unset, delivery is skipped gracefully (caller logs + queues).

Not wired into the live call handler yet — that is a separate, guarded step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from workflow.validation import looks_like_phone


@dataclass(frozen=True)
class DeliveryResult:
    delivered: bool
    channel: str
    status_code: Optional[int] = None
    error: Optional[str] = None
    skipped_reason: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_shoreline_lead(
    args: dict,
    *,
    call_sid: str = "",
    from_number: str = "",
    source: str = "phone",
    lead_id: str = "",
    now_iso: Optional[str] = None,
) -> dict:
    """Map submit_project_inquiry args to the contract §3 lead schema.

    System/buyer-side fields (lead_id, market, recording_url, transcript_summary)
    are left blank for ShorelineCost to fill unless provided.
    """
    args = args or {}
    extra = args.get("extra_fields") or {}
    consent = bool(args.get("consent"))
    ts = now_iso or _now_iso()
    # Callback safety net (mirrors the plumbing path): if the AI submitted a phrase like
    # "this number is good" instead of a real number, fall back to the caller's number.
    raw_callback = str(args.get("callback") or "").strip()
    callback_phone = raw_callback if looks_like_phone(raw_callback) else (from_number or raw_callback or "")
    return {
        "lead_id": lead_id,
        "timestamp": ts,
        "source": source,
        "caller_name": args.get("name", ""),
        "callback_phone": callback_phone,
        "email": args.get("email", ""),
        "zip": args.get("zip_code", ""),
        "market": "",
        "project_type": args.get("project_type", ""),
        "water_setting": args.get("water_setting", ""),
        "approx_size_ft": args.get("approx_size_ft"),
        "condition": args.get("condition", ""),
        "access": args.get("access", ""),
        "timeline": args.get("timeline", ""),
        "urgency": args.get("urgency") or "NORMAL",
        "ownership_confirmed": bool(args.get("ownership_confirmed")),
        "consent": consent,
        "consent_timestamp": ts if consent else "",
        "qualification_status": "qualified",
        "transfer_outcome": args.get("transfer_outcome", "none"),
        "transferred_to": args.get("transferred_to"),
        "recording_url": "",
        "transcript_summary": "",
        "notes": str(extra.get("additional_notes") or args.get("notes") or ""),
        "call_sid": call_sid,
    }


def webhook_url(delivery_spec: Optional[dict]) -> str:
    """Resolve the configured webhook URL from the env var named in the vertical spec."""
    spec = delivery_spec or {}
    env_name = spec.get("url_env")
    return os.getenv(env_name, "") if env_name else ""


def auth_headers(delivery_spec: Optional[dict]) -> dict:
    """Build the auth header from the vertical spec: {auth_header: <value of secret_env>}.

    The secret VALUE comes only from the named env var — never from config or git. Returns
    empty if the header name or secret env var is not configured / not set.
    """
    spec = delivery_spec or {}
    header = spec.get("auth_header")
    secret_env = spec.get("secret_env")
    if not header or not secret_env:
        return {}
    secret = os.getenv(secret_env, "")
    return {header: secret} if secret else {}


async def deliver_lead_webhook(url: str, payload: dict, *, headers: Optional[dict] = None, post_func=None) -> DeliveryResult:
    """POST the lead JSON to the webhook. `post_func(url, payload, headers) -> status_code` is
    injectable for tests; default uses httpx. No URL configured → skipped (not an error)."""
    if not url:
        return DeliveryResult(delivered=False, channel="webhook", skipped_reason="no_webhook_url_configured")
    headers = headers or {}
    try:
        if post_func is None:
            import httpx

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload, headers=headers)
                status_code = resp.status_code
        else:
            status_code = await post_func(url, payload, headers)
        status_code = int(status_code)
        delivered = 200 <= status_code < 300
        return DeliveryResult(
            delivered=delivered,
            channel="webhook",
            status_code=status_code,
            error=None if delivered else f"HTTP {status_code}",
        )
    except Exception as exc:  # network/timeout/etc.
        return DeliveryResult(delivered=False, channel="webhook", error=str(exc))


async def deliver_shoreline_lead(
    args: dict,
    *,
    vertical: Optional[dict] = None,
    call_sid: str = "",
    from_number: str = "",
    post_func=None,
) -> dict:
    """Build the §3 lead and deliver per the vertical's delivery spec.

    consent=false → never delivered to buyers (logged only, per contract §3). Returns a
    plain dict the call handler can log + use to decide the function_call_output.
    """
    payload = build_shoreline_lead(args, call_sid=call_sid, from_number=from_number)
    if not payload["consent"]:
        return {
            "delivered": False,
            "channel": "none",
            "skipped_reason": "consent_declined",
            "consent": False,
            "status_code": None,
            "error": None,
            "payload": payload,
        }
    delivery_spec = (vertical or {}).get("delivery")
    url = webhook_url(delivery_spec)
    result = await deliver_lead_webhook(url, payload, headers=auth_headers(delivery_spec), post_func=post_func)
    return {
        "delivered": result.delivered,
        "channel": result.channel,
        "skipped_reason": result.skipped_reason,
        "consent": True,
        "status_code": result.status_code,
        "error": result.error,
        "payload": payload,
    }
