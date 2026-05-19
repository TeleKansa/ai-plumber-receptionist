from dataclasses import dataclass
import re
from typing import Optional

from storage import repository
from workflow.intake_policy import applicable_questions, missing_extra_guidance, missing_policy_extra_fields
from workflow.notifications import SmsSendResult
from workflow.validation import looks_like_phone, validate_service_request_args


@dataclass(frozen=True)
class ServiceRequestResult:
    output: dict
    should_hangup: bool
    closing_instructions: Optional[str] = None


CALLBACK_ALIAS_VALUES = {
    "call me at this number",
    "call me back at this number",
    "call me back on this number",
    "caller number",
    "current number",
    "my number",
    "number on file",
    "same number",
    "the number im calling from",
    "the number i m calling from",
    "the number i'm calling from",
    "this number",
    "this number is fine",
    "this number is good",
    "use my number",
    "use this number",
    "yes this number",
}


def _normalize_words(value: object) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u2019", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_callback_alias(value: object) -> bool:
    normalized = _normalize_words(value)
    if normalized in CALLBACK_ALIAS_VALUES:
        return True
    return (
        "this number" in normalized
        and any(word in normalized for word in {"good", "fine", "works", "callback", "call"})
    )


def _normalize_callback_alias(args: dict, from_number: str) -> tuple[dict, Optional[dict]]:
    normalized_args = dict(args or {})
    submitted_callback = normalized_args.get("callback")
    if not _is_callback_alias(submitted_callback):
        return normalized_args, None
    if not looks_like_phone(from_number):
        return normalized_args, None
    normalized_args["callback"] = from_number
    return normalized_args, {
        "submitted_callback": submitted_callback,
        "normalized_callback": from_number,
        "source": "caller_from_number",
    }


def _normalize_sms_result(result) -> SmsSendResult:
    if isinstance(result, SmsSendResult):
        return result
    if isinstance(result, bool):
        return SmsSendResult(success=result)
    return SmsSendResult(success=bool(result))


def _validation_guidance(errors: dict[str, str]) -> str:
    if "issue" in errors:
        return (
            'Ask exactly one calm question, then stop and wait: "What\'s going on with the plumbing?" '
            'Do not say "I need to know" and do not repeat yourself.'
        )
    if "urgency" in errors:
        return (
            'Ask exactly one calm question, then stop and wait: "Is water still coming out or flooding right now?"'
        )
    if "address" in errors:
        return 'Ask exactly one calm question, then stop and wait: "What\'s the service address?"'
    if "callback" in errors:
        return (
            'Ask exactly one calm question, then stop and wait: "Is this number good for callback, '
            'or is there a better number?" If they confirm this number, submit the caller phone number, not the phrase.'
        )
    if "name" in errors:
        return "Ask exactly one question: Could I get your name? A first name is enough. Do not ask for last name."
    return (
        "The submitted service request is missing or has an invalid field. "
        "Ask exactly one question for the missing or invalid field, then stop and wait. "
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
    intake_state: Optional[dict] = None,
):
    args, callback_alias_payload = _normalize_callback_alias(args, from_number)
    if callback_alias_payload:
        repository.record_call_event(
            call_sid,
            "callback_alias_normalized",
            callback_alias_payload,
            tenant_id=tenant_id,
        )

    resolved_tenant_id = tenant_id
    if resolved_tenant_id is None:
        tenant = repository.get_call_tenant(call_sid)
        resolved_tenant_id = tenant["id"] if tenant else None
    intake_policy = repository.get_intake_policy(resolved_tenant_id) if resolved_tenant_id else None

    errors = validate_service_request_args(args, caller_text=caller_text)
    if errors:
        field_priority = ["issue", "urgency", "address", "callback", "name"]
        next_field = next((field for field in field_priority if field in errors), next(iter(errors)))
        errors = {next_field: errors[next_field]}
        guidance = _validation_guidance(errors)
        repository.record_call_event(
            call_sid,
            "validation_failed",
            {
                "errors": errors,
                "missing_fields": list(errors.keys()),
                "args": args,
                "caller_text": caller_text,
                "intake_policy_id": intake_policy.get("id") if intake_policy else None,
                "guidance": guidance,
                "next_question_key": next_field,
            },
        )
        return ServiceRequestResult(
            output={
                "success": False,
                "reason": "validation_failed",
                "missing_fields": list(errors.keys()),
                "errors": errors,
                "next_question_key": next_field,
                "guidance": guidance,
            },
            should_hangup=False,
            closing_instructions=guidance,
        )

    active_question_keys = [question["key"] for question in applicable_questions(intake_policy, args)]
    missing_extra_fields = missing_policy_extra_fields(args, intake_policy, intake_state)
    if missing_extra_fields:
        missing_extra_fields = [missing_extra_fields[0]]
        reason = missing_extra_fields[0].get("reason") or "intake_policy_missing_extra_fields"
        guidance = missing_extra_guidance(missing_extra_fields)
        if intake_state is not None:
            intake_state["pending_extra_question"] = {
                "key": missing_extra_fields[0]["key"],
                "label": missing_extra_fields[0]["label"],
                "question_text": missing_extra_fields[0]["question_text"],
                "collection_mode": missing_extra_fields[0]["collection_mode"],
                "asked": False,
                "caller_response_after_pending": False,
                "caller_response_text": "",
            }
        repository.record_call_event(
            call_sid,
            "intake_question_required",
            {
                "pending_extra_question": missing_extra_fields[0],
                "active_intake_policy_question_keys": active_question_keys,
                "intake_policy_id": intake_policy.get("id") if intake_policy else None,
                "guidance": guidance,
            },
        )
        repository.record_call_event(
            call_sid,
            reason,
            {
                "missing_extra_fields": missing_extra_fields,
                "active_intake_policy_question_keys": active_question_keys,
                "args": args,
                "caller_text": caller_text,
                "intake_policy_id": intake_policy.get("id") if intake_policy else None,
                "guidance": guidance,
            },
        )
        return ServiceRequestResult(
            output={
                "success": False,
                "reason": reason,
                "missing_extra_fields": missing_extra_fields,
                "guidance": guidance,
            },
            should_hangup=False,
            closing_instructions=guidance,
        )

    if intake_state is not None:
        intake_state.pop("pending_extra_question", None)

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

    lead, notification = repository.create_lead_with_pending_notification(
        call_sid,
        args,
        plumber_phone_number or "",
        tenant_id=resolved_tenant_id,
    )
    repository.record_call_event(
        call_sid,
        "lead_created",
        {
            "lead_id": lead["id"],
            "tenant_id": lead.get("tenant_id"),
            "name_provenance": name_provenance,
            "active_intake_policy_question_keys": active_question_keys,
            "extra_fields": args.get("extra_fields") or {},
        },
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
