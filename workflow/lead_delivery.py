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
    return {
        "lead_id": lead_id,
        "timestamp": ts,
        "source": source,
        "caller_name": args.get("name", ""),
        "callback_phone": args.get("callback") or from_number or "",
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


async def deliver_lead_webhook(url: str, payload: dict, *, post_func=None) -> DeliveryResult:
    """POST the lead JSON to the webhook. `post_func(url, payload) -> status_code` is
    injectable for tests; default uses httpx. No URL configured → skipped (not an error)."""
    if not url:
        return DeliveryResult(delivered=False, channel="webhook", skipped_reason="no_webhook_url_configured")
    try:
        if post_func is None:
            import httpx

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, json=payload)
                status_code = resp.status_code
        else:
            status_code = await post_func(url, payload)
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
