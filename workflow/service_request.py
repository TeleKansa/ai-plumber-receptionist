from dataclasses import dataclass
from typing import Optional

from storage import repository
from workflow.notifications import SmsSendResult
from workflow.validation import validate_service_request_args


@dataclass(frozen=True)
class ServiceRequestResult:
    output: dict
    should_hangup: bool
    closing_instructions: Optional[str] = None


def _normalize_sms_result(result) -> SmsSendResult:
    if isinstance(result, SmsSendResult):
        return result
    if isinstance(result, bool):
        return SmsSendResult(success=result)
    return SmsSendResult(success=bool(result))


def _validation_guidance(errors: dict[str, str]) -> str:
    if "name" in errors:
        return "Ask for the caller's name again. A first name is enough. Do not ask for last name."
    return (
        "The submitted service request is missing or has an invalid field. "
        "Ask only for the first missing or invalid field, then stop. "
        "Do not invent the customer name, address, or any other field."
    )


async def process_service_request(
    call_sid: str,
    args: dict,
    from_number: str,
    plumber_phone_number: str,
    send_sms_func,
    caller_text: str = "",
    tenant_id: Optional[int] = None,
):
    errors = validate_service_request_args(args, caller_text=caller_text)
    if errors:
        guidance = _validation_guidance(errors)
        repository.record_call_event(
            call_sid,
            "validation_failed",
            {
                "errors": errors,
                "missing_fields": list(errors.keys()),
                "args": args,
                "caller_text": caller_text,
                "guidance": guidance,
            },
        )
        return ServiceRequestResult(
            output={
                "success": False,
                "reason": "validation_failed",
                "missing_fields": list(errors.keys()),
                "errors": errors,
                "guidance": guidance,
            },
            should_hangup=False,
            closing_instructions=guidance,
        )

    existing_lead = repository.get_lead_by_call_sid(call_sid)
    if existing_lead:
        notification = repository.get_notification_for_lead(existing_lead["id"])
        repository.record_call_event(
            call_sid,
            "duplicate_submit",
            {
                "lead_id": existing_lead["id"],
                "notification_status": notification.get("status") if notification else None,
            },
        )
        return ServiceRequestResult(
            output={
                "success": notification is None or notification.get("status") == "sent",
                "reason": "already_submitted",
                "lead_saved": True,
                "lead_id": existing_lead["id"],
                "notification_status": notification.get("status") if notification else None,
            },
            should_hangup=True,
            closing_instructions='Say only: "Okay, you\'re all set. We\'ll call you back soon." Then stop.',
        )

    name_provenance = "supported_by_transcript" if caller_text.strip() else "unverified_no_transcript"
    if name_provenance == "unverified_no_transcript":
        repository.record_call_event(
            call_sid,
            "name_provenance_unverified",
            {"name": args.get("name"), "name_provenance": name_provenance},
        )

    lead, notification = repository.create_lead_with_pending_notification(call_sid, args, plumber_phone_number or "", tenant_id=tenant_id)
    repository.record_call_event(
        call_sid,
        "lead_created",
        {"lead_id": lead["id"], "tenant_id": lead.get("tenant_id"), "name_provenance": name_provenance},
    )

    if not plumber_phone_number:
        error = "Tenant notification SMS number is not configured"
        repository.mark_notification_failed(notification["id"], error)
        repository.record_call_event(
            call_sid,
            "notification_config_missing",
            {"lead_id": lead["id"], "notification_id": notification["id"], "tenant_id": lead.get("tenant_id"), "error": error},
        )
        return ServiceRequestResult(
            output={
                "success": False,
                "reason": "notification_failed",
                "lead_saved": True,
                "lead_id": lead["id"],
            },
            should_hangup=True,
            closing_instructions='Say only: "I\'ve captured your information. We\'ll make sure it\'s flagged for follow-up." Then stop.',
        )

    sms_result = _normalize_sms_result(await send_sms_func(call_sid, args, from_number))
    if sms_result.success:
        repository.mark_notification_sent(notification["id"], sms_result.provider_message_sid)
        repository.record_call_event(
            call_sid,
            "sms_sent",
            {"lead_id": lead["id"], "notification_id": notification["id"], "provider_message_sid": sms_result.provider_message_sid},
        )
        return ServiceRequestResult(
            output={"success": True, "lead_saved": True, "lead_id": lead["id"]},
            should_hangup=True,
            closing_instructions=None,
        )

    error = sms_result.error or "SMS provider returned failure"
    repository.mark_notification_failed(notification["id"], error)
    repository.record_call_event(
        call_sid,
        "sms_failed",
        {"lead_id": lead["id"], "notification_id": notification["id"], "error": error},
    )
    return ServiceRequestResult(
        output={
            "success": False,
            "reason": "notification_failed",
            "lead_saved": True,
            "lead_id": lead["id"],
        },
        should_hangup=True,
        closing_instructions='Say only: "I\'ve captured your information. We\'ll make sure it\'s flagged for follow-up." Then stop.',
    )
