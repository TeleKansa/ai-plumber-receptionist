from dataclasses import dataclass
from typing import Optional

from workflow.intake_policy import ADDITIONAL_NOTES_KEY, sms_extra_field_rows_with_keys


@dataclass(frozen=True)
class SmsSendResult:
    success: bool
    provider_message_sid: Optional[str] = None
    error: Optional[str] = None


def build_sms_body(info: dict, from_number: str, intake_policy: Optional[dict] = None, notification_policy: Optional[dict] = None) -> str:
    priority = str(info.get("priority") or "normal").strip().lower()
    title = "EMERGENCY PLUMBING LEAD" if priority == "emergency" else "NEW PLUMBING LEAD"
    body = (
        f"{title}\n\n"
        f"Name: {info.get('name', 'N/A')}\n"
        f"Phone: {info.get('callback', from_number)}\n"
        f"Issue: {info.get('issue', 'N/A')}\n"
        f"Urgency: {info.get('urgency', 'N/A')}\n"
        f"Address: {info.get('address', 'N/A')}"
    )
    if priority in {"urgent", "emergency"}:
        body = f"{body}\nPriority: {priority.upper()}\nReason: {info.get('priority_reason') or 'Rule-based priority match'}"

    include_extra = True if notification_policy is None else bool(notification_policy.get("include_extra_fields", True))
    include_additional = True if notification_policy is None else bool(notification_policy.get("include_additional_notes", True))
    extra_rows = []
    for key, label, value in sms_extra_field_rows_with_keys(intake_policy, info):
        if key == ADDITIONAL_NOTES_KEY:
            if include_additional:
                extra_rows.append((label, value))
        elif include_extra:
            extra_rows.append((label, value))
    if extra_rows:
        extra_text = "\n".join(f"{label}: {value}" for label, value in extra_rows)
        body = f"{body}\n\nExtra:\n{extra_text}"
    return body
