import json
from typing import Optional

from storage import repository


def _coerce_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "[]")
        except json.JSONDecodeError:
            parsed = [part.strip() for part in value.replace("\r", "\n").replace(",", "\n").split("\n")]
        return _coerce_list(parsed)
    return []


def _dedupe_numbers(items: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for item in items:
        number = item.get("to_number") or ""
        normalized = repository.normalize_phone_number(number)
        key = normalized or number
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def policy_snapshot(policy: Optional[dict]) -> dict:
    if not policy:
        return {}
    return {
        "id": policy.get("id"),
        "tenant_id": policy.get("tenant_id"),
        "normal_sms_recipients": _coerce_list(policy.get("normal_sms_recipients_json")),
        "emergency_sms_recipients": _coerce_list(policy.get("emergency_sms_recipients_json")),
        "backup_sms_recipients": _coerce_list(policy.get("backup_sms_recipients_json")),
        "send_normal_leads": bool(policy.get("send_normal_leads", True)),
        "send_emergency_leads": bool(policy.get("send_emergency_leads", True)),
        "include_extra_fields": bool(policy.get("include_extra_fields", True)),
        "include_additional_notes": bool(policy.get("include_additional_notes", True)),
        "emergency_keywords": _coerce_list(policy.get("emergency_keywords_json")),
        "emergency_rules_json": policy.get("emergency_rules_json") or "[]",
    }


def notification_recipients(policy: Optional[dict], priority: str) -> tuple[list[dict], list[str]]:
    snapshot = policy_snapshot(policy)
    notes = []
    if priority == "emergency":
        if not snapshot.get("send_emergency_leads", True):
            return [], ["Emergency notifications disabled by policy."]
        emergency = snapshot.get("emergency_sms_recipients") or []
        if emergency:
            primary = [{"to_number": number, "recipient_type": "emergency"} for number in emergency]
        else:
            notes.append("No emergency recipients configured; using normal recipients.")
            primary = [{"to_number": number, "recipient_type": "normal"} for number in snapshot.get("normal_sms_recipients") or []]
        return _dedupe_numbers(primary), notes

    if not snapshot.get("send_normal_leads", True):
        return [], ["Normal notifications disabled by policy."]
    primary = [{"to_number": number, "recipient_type": "normal"} for number in snapshot.get("normal_sms_recipients") or []]
    return _dedupe_numbers(primary), notes


def backup_recipients(policy: Optional[dict], already_attempted: Optional[set[str]] = None) -> list[dict]:
    attempted = already_attempted or set()
    items = []
    for number in _coerce_list((policy or {}).get("backup_sms_recipients_json")):
        key = repository.normalize_phone_number(number) or number
        if key in attempted:
            continue
        items.append({"to_number": number, "recipient_type": "backup"})
    return _dedupe_numbers(items)
