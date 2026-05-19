"""
AI Plumber Receptionist — OpenAI Realtime API
==============================================
Twilio voice → /media-stream WebSocket → OpenAI Realtime (bidirectional bridge)

Audio: Twilio mulaw 8kHz ↔ pcm16 24kHz ↔ OpenAI (transcoding via audioop)
Function calling: response.output_item.done (GA API event name)
"""

import asyncio
import audioop
import base64
import json
import logging
from typing import Optional

import websockets
import websockets.exceptions
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse
from twilio.rest import Client as TwilioClient

from admin.routes import create_admin_router
from config.settings import get_settings
from storage.database import init_db
from storage import repository
from workflow.notifications import SmsSendResult, build_sms_body as build_notification_sms_body
from workflow.prompt_builder import DEFAULT_GREETING, PromptBuilder
from workflow.service_request import process_service_request
from workflow.sms_result import build_service_request_output

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("plumber")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

settings = get_settings()

OPENAI_API_KEY       = settings.openai_api_key
TWILIO_ACCOUNT_SID   = settings.twilio_account_sid
TWILIO_AUTH_TOKEN    = settings.twilio_auth_token
TWILIO_PHONE_NUMBER  = settings.twilio_phone_number
PLUMBER_PHONE_NUMBER = settings.plumber_phone_number
HOST                 = settings.host
OAI_URL              = settings.oai_url

twilio = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
app    = FastAPI()
app.include_router(create_admin_router(settings))

# call_sid → {from_number, complete}
sessions: dict[str, dict] = {}
prompt_builder = PromptBuilder()

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def make_instructions(
    caller_number: str,
    tenant: Optional[dict] = None,
    prompt_profile: Optional[dict] = None,
    intake_policy: Optional[dict] = None,
) -> str:
    return prompt_builder.build(caller_number, tenant=tenant, profile=prompt_profile, intake_policy=intake_policy)


# ---------------------------------------------------------------------------
# OpenAI function tool
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "name": "submit_service_request",
        "description": "Submit the service request once all 5 fields are collected: issue, urgency, address, callback number, and customer name.",
        "parameters": {
            "type": "object",
            "properties": {
                "issue":    {"type": "string", "description": "Plumbing problem description"},
                "urgency":  {"type": "string", "description": "Urgency — active leak/flooding or not"},
                "address":  {"type": "string", "description": "Full service address"},
                "callback": {
                    "type": "string",
                    "description": "Callback phone number. If the caller confirms 'this number is good' or similar, submit the actual caller phone number, not the phrase.",
                },
                "name":     {"type": "string", "description": "Customer name"},
                "extra_fields": {
                    "type": "object",
                    "description": "Tenant-specific intake policy answers, keyed by field key. Include required and ask_once question answers here. Do not set values to 'unknown' unless the caller actually said they do not know, declined, or gave no answer after you asked. Do not infer homeowner/renter.",
                },
            },
            "required": ["issue", "urgency", "address", "callback", "name"],
        },
    }
]


def build_session_update(
    caller_number: str,
    tenant: Optional[dict] = None,
    prompt_profile: Optional[dict] = None,
    intake_policy: Optional[dict] = None,
) -> dict:
    session = {
        "type":        "realtime",
        "instructions": make_instructions(caller_number, tenant, prompt_profile, intake_policy),
        "tools":       TOOLS,
        "tool_choice": "auto",
    }
    return {
        "type": "session.update",
        "session": session,
    }


def build_initial_greeting_response(greeting_text: str) -> dict:
    return {
        "type": "response.create",
        "response": {
            "instructions": (
                f'Say only: "{greeting_text}" Then stop. '
                "Do not add a second question. Do not repeat yourself. Wait for the caller."
            ),
        },
    }


def _event_text(value: object, limit: int = 800) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}..."


def response_create_event_payload(reason: str, response_create: dict, session: dict) -> dict:
    response = response_create.get("response") or {}
    instructions = response.get("instructions") or ""
    return {
        "reason": reason,
        "has_response_instructions": bool(instructions),
        "instructions": _event_text(instructions),
        "caller_spoke_since_initial_greeting": bool(session.get("caller_spoke_since_initial_greeting")),
        "first_caller_speech_started": bool(session.get("first_caller_speech_started")),
        "previous_response_create_reason": session.get("last_response_create_reason"),
        "complete": bool(session.get("complete")),
        "pending_hangup": bool(session.get("pending_hangup")),
        "submit_service_request_seen": bool(session.get("submit_service_request_seen")),
        "lead_id": session.get("lead_id"),
    }


def response_create_reason_for_service_result(output: dict, should_hangup: bool) -> str:
    if should_hangup:
        return "closing"
    reason = output.get("reason")
    if reason == "validation_failed":
        return "validation_followup"
    if reason in {"intake_policy_missing_extra_fields", "intake_policy_unanswered_extra_field"}:
        return "intake_missing_extra"
    return "other"


def set_media_stream_exit_reason(
    lifecycle: dict,
    reason: str,
    detail: Optional[dict] = None,
    overwrite: bool = False,
) -> str:
    current = lifecycle.get("exit_reason") or "unknown"
    if overwrite or current == "unknown":
        lifecycle["exit_reason"] = reason
        lifecycle["exit_detail"] = detail or {}
    return lifecycle.get("exit_reason") or "unknown"


def safe_record_call_event(call_sid: Optional[str], event_type: str, payload: dict, tenant_id: Optional[int] = None) -> None:
    if not call_sid:
        return
    try:
        repository.record_call_event(call_sid, event_type, payload, tenant_id=tenant_id)
    except Exception as exc:
        log.exception(f"[{call_sid}] Failed to record call_event {event_type}: {exc}")


def safe_mark_call_ended(call_sid: Optional[str], status: str) -> None:
    if not call_sid:
        return
    try:
        repository.mark_call_ended(call_sid, status)
    except Exception as exc:
        log.exception(f"[{call_sid}] Failed to mark call ended status={status}: {exc}")


def media_stream_session_snapshot(
    call_sid: Optional[str],
    stream_sid: Optional[str],
    session: Optional[dict],
    lifecycle: Optional[dict],
    response_active: bool = False,
    assistant_speaking: bool = False,
) -> dict:
    session = session or {}
    lifecycle = lifecycle or {}
    return {
        "call_sid": call_sid,
        "stream_sid": stream_sid,
        "tenant_id": session.get("tenant_id"),
        "media_stream_exit_reason": lifecycle.get("exit_reason") or "unknown",
        "media_stream_exit_detail": lifecycle.get("exit_detail") or {},
        "openai_reader_exit_reason": lifecycle.get("openai_reader_exit_reason"),
        "openai_close_code": lifecycle.get("openai_close_code"),
        "openai_close_reason": lifecycle.get("openai_close_reason"),
        "complete": bool(session.get("complete")),
        "pending_hangup": bool(session.get("pending_hangup")),
        "closing_response_started": bool(session.get("closing_response_started")),
        "hangup_scheduled": bool(session.get("hangup_scheduled")),
        "response_active": bool(response_active),
        "assistant_speaking": bool(assistant_speaking),
        "last_ai_transcript": _event_text(session.get("last_ai_transcript"), limit=500),
        "last_response_create_reason": session.get("last_response_create_reason"),
        "submit_service_request_seen": bool(session.get("submit_service_request_seen")),
        "lead_id": session.get("lead_id"),
        "hangup_reason": session.get("hangup_reason"),
    }


def disconnect_payload(exc: BaseException) -> dict:
    return {
        "disconnect_code": getattr(exc, "code", None),
        "disconnect_reason": getattr(exc, "reason", None),
        "exception_type": type(exc).__name__,
        "exception": str(exc) or None,
    }


def record_twilio_websocket_disconnect(
    call_sid: Optional[str],
    stream_sid: Optional[str],
    exc: BaseException,
    lifecycle: dict,
    session: Optional[dict],
    response_active: bool,
    assistant_speaking: bool,
) -> dict:
    payload = disconnect_payload(exc)
    set_media_stream_exit_reason(lifecycle, "websocket_disconnect", payload)
    snapshot = media_stream_session_snapshot(call_sid, stream_sid, session, lifecycle, response_active, assistant_speaking)
    event_payload = {**payload, "snapshot": snapshot}
    log.warning(f"[{call_sid}] Twilio websocket disconnected: {payload}")
    safe_record_call_event(call_sid, "twilio_websocket_disconnected", event_payload)
    safe_mark_call_ended(call_sid, "websocket_disconnected")
    return event_payload


def record_twilio_stream_stopped(
    call_sid: Optional[str],
    stream_sid: Optional[str],
    stop_payload: dict,
    lifecycle: dict,
    session: Optional[dict],
    response_active: bool,
    assistant_speaking: bool,
) -> dict:
    payload = stop_payload or {}
    set_media_stream_exit_reason(lifecycle, "twilio_stop", payload)
    snapshot = media_stream_session_snapshot(call_sid, stream_sid, session, lifecycle, response_active, assistant_speaking)
    event_payload = {"stop": payload, "snapshot": snapshot}
    log.info(f"[{call_sid}] Twilio stream stopped payload={payload}")
    safe_record_call_event(call_sid, "twilio_stream_stopped", event_payload)
    safe_record_call_event(call_sid, "media_stream_stopped", payload)
    safe_mark_call_ended(call_sid, "stream_stopped")
    return event_payload


def record_openai_websocket_closed(
    call_sid: Optional[str],
    stream_sid: Optional[str],
    exc: BaseException,
    lifecycle: dict,
    session: Optional[dict],
    response_active: bool,
    assistant_speaking: bool,
) -> dict:
    payload = {
        "close_code": getattr(exc, "code", None),
        "close_reason": getattr(exc, "reason", None),
        "exception_type": type(exc).__name__,
        "exception": str(exc) or None,
    }
    lifecycle["openai_reader_exit_reason"] = "openai_closed"
    lifecycle["openai_close_code"] = payload["close_code"]
    lifecycle["openai_close_reason"] = payload["close_reason"]
    snapshot = media_stream_session_snapshot(call_sid, stream_sid, session, lifecycle, response_active, assistant_speaking)
    event_payload = {**payload, "snapshot": snapshot}
    log.warning(f"[{call_sid}] OpenAI websocket closed: {payload}")
    safe_record_call_event(call_sid, "openai_websocket_closed", event_payload)
    return event_payload


def record_openai_reader_error(
    call_sid: Optional[str],
    stream_sid: Optional[str],
    exc: BaseException,
    lifecycle: dict,
    session: Optional[dict],
    response_active: bool,
    assistant_speaking: bool,
) -> dict:
    payload = {
        "exception_type": type(exc).__name__,
        "exception": str(exc),
    }
    lifecycle["openai_reader_exit_reason"] = "openai_reader_error"
    snapshot = media_stream_session_snapshot(call_sid, stream_sid, session, lifecycle, response_active, assistant_speaking)
    event_payload = {**payload, "snapshot": snapshot}
    log.error(f"[{call_sid}] OpenAI reader error: {exc}", exc_info=(type(exc), exc, exc.__traceback__))
    safe_record_call_event(call_sid, "openai_reader_error", event_payload)
    return event_payload


def can_schedule_hangup(session: dict) -> bool:
    return bool(
        session.get("pending_hangup")
        and session.get("closing_response_started")
        and not session.get("hangup_scheduled")
        and session.get("submit_service_request_seen")
        and session.get("lead_id")
    )


def hangup_scheduled_payload(session: dict) -> dict:
    return {
        "reason": session.get("hangup_reason") or "service_request_complete",
        "lead_id": session.get("lead_id"),
        "closing_response_started": bool(session.get("closing_response_started")),
        "should_hangup": bool(session.get("complete")),
        "submit_service_request_seen": bool(session.get("submit_service_request_seen")),
    }


def record_media_stream_done(
    call_sid: Optional[str],
    stream_sid: Optional[str],
    session: Optional[dict],
    lifecycle: dict,
    response_active: bool,
    assistant_speaking: bool,
) -> dict:
    if (lifecycle.get("exit_reason") or "unknown") == "unknown":
        if (session or {}).get("complete"):
            set_media_stream_exit_reason(lifecycle, "normal_complete")
        elif lifecycle.get("openai_reader_exit_reason") == "openai_closed":
            set_media_stream_exit_reason(lifecycle, "openai_closed")
        elif lifecycle.get("openai_reader_exit_reason") == "openai_reader_error":
            set_media_stream_exit_reason(lifecycle, "exception")
        else:
            set_media_stream_exit_reason(lifecycle, "unknown")
    snapshot = media_stream_session_snapshot(call_sid, stream_sid, session, lifecycle, response_active, assistant_speaking)
    safe_record_call_event(call_sid, "media_stream_done", snapshot)
    log.info(f"[{call_sid}] media_stream done exit_reason={snapshot['media_stream_exit_reason']} snapshot={snapshot}")
    return snapshot

# ---------------------------------------------------------------------------
# Startup / health
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    init_db()
    log.info("=== Plumber Receptionist (OpenAI Realtime) starting ===")
    log.info(f"  OPENAI_API_KEY       : {'SET' if OPENAI_API_KEY       else 'MISSING'}")
    log.info(f"  TWILIO_ACCOUNT_SID   : {'SET' if TWILIO_ACCOUNT_SID   else 'MISSING'}")
    log.info(f"  TWILIO_AUTH_TOKEN    : {'SET' if TWILIO_AUTH_TOKEN     else 'MISSING'}")
    log.info(f"  TWILIO_PHONE_NUMBER  : {TWILIO_PHONE_NUMBER  or 'MISSING'}")
    log.info(f"  PLUMBER_PHONE_NUMBER : {PLUMBER_PHONE_NUMBER or 'MISSING'}")
    log.info(f"  HOST                 : {HOST}")
    log.info(f"  OAI_URL              : {OAI_URL}")
    log.info(f"  DATABASE_URL         : {'SET' if settings.database_url else 'MISSING'}")
    log.info(f"  ADMIN_PASSWORD       : {'SET' if settings.admin_password else 'MISSING'}")
    log.info("======================================================")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/media-stream")
async def media_stream_probe():
    return PlainTextResponse("WebSocket endpoint — WSS upgrade required")

# ---------------------------------------------------------------------------
# Twilio webhook
# ---------------------------------------------------------------------------

def _allowed_test_callers(profile: Optional[dict]) -> set[str]:
    if not profile:
        return set()
    raw_callers = profile.get("allowed_test_callers_json") or "[]"
    try:
        parsed = json.loads(raw_callers)
    except json.JSONDecodeError:
        parsed = []
    if not isinstance(parsed, list):
        return set()
    return {repository.normalize_phone_number(str(value)) for value in parsed if str(value).strip()}


def _telephony_gate(tenant: dict, tenant_phone: dict, profile: Optional[dict], from_number: str) -> tuple[bool, str, str]:
    tenant_status = tenant.get("status") or "onboarding"
    if tenant_status == "live":
        if tenant_phone.get("accepts_live_calls"):
            return True, "allowed", "Tenant is live and this AI forwarding number accepts live calls."
        return False, "not_live", "Tenant is live, but this AI forwarding number is not enabled for live calls."
    if tenant_status == "testing":
        if not profile or not profile.get("test_mode_enabled"):
            return False, "test_mode_disabled", "Tenant is in testing, but test mode is not enabled."
        normalized_from = repository.normalize_phone_number(from_number)
        if normalized_from and normalized_from in _allowed_test_callers(profile):
            return True, "allowed_test_caller", "Tenant is in testing and caller is allowed."
        return False, "test_caller_not_allowed", "Tenant is in testing, but this caller is not on the allowed test callers list."
    if tenant_status == "paused":
        return False, "paused", "Tenant is paused."
    if tenant_status in {"draft", "onboarding"}:
        return False, "onboarding_blocked", "Tenant is not live yet."
    return False, "not_live", f"Tenant status is {tenant_status}, so calls are not accepted."


@app.post("/voice")
async def voice(request: Request):
    form        = await request.form()
    call_sid    = form.get("CallSid", "unknown")
    from_number = form.get("From",    "unknown")
    to_number   = form.get("To",      "unknown")
    log.info(f"[{call_sid}] Incoming call from {from_number}")

    tenant, tenant_phone, tenant_matched = repository.resolve_tenant_phone_for_number(to_number)
    if not tenant_matched or tenant is None:
        log.warning(f"[{call_sid}] Tenant lookup failed for Twilio To={to_number}; rejecting call")
        repository.create_or_update_call(
            call_sid,
            from_number,
            to_number,
            tenant_id=None,
            default_to_tenant=False,
            status="tenant_lookup_failed",
        )
        repository.record_call_event(
            call_sid,
            "tenant_lookup_failed",
            {
                "from_number": from_number,
                "to_number": to_number,
                "normalized_to_number": repository.normalize_phone_number(to_number),
                "reason": "No active tenant phone number matched incoming Twilio To number.",
            },
            default_to_tenant=False,
        )
        twiml = (
            "<Response>"
            "<Say>Sorry, this line is not configured yet.</Say>"
            "<Hangup/>"
            "</Response>"
        )
        return Response(content=twiml, media_type="application/xml")

    telephony_profile = repository.get_telephony_profile(tenant["id"])
    allowed, gate_status, gate_reason = _telephony_gate(tenant, tenant_phone, telephony_profile, from_number)
    if not allowed:
        log.warning(f"[{call_sid}] Call blocked before media stream: {gate_status} ({gate_reason})")
        repository.create_or_update_call(
            call_sid,
            from_number,
            to_number,
            tenant_id=tenant["id"],
            status=gate_status,
        )
        repository.record_call_event(
            call_sid,
            "call_blocked",
            {
                "reason": gate_status,
                "message": gate_reason,
                "tenant_id": tenant["id"],
                "tenant_status": tenant.get("status"),
                "tenant_phone_id": tenant_phone.get("id") if tenant_phone else None,
                "accepts_live_calls": tenant_phone.get("accepts_live_calls") if tenant_phone else None,
                "from_number": from_number,
                "normalized_from_number": repository.normalize_phone_number(from_number),
                "to_number": to_number,
                "normalized_to_number": repository.normalize_phone_number(to_number),
            },
            tenant_id=tenant["id"],
        )
        twiml = (
            "<Response>"
            "<Say>Sorry, this line is not active yet.</Say>"
            "<Hangup/>"
            "</Response>"
        )
        return Response(content=twiml, media_type="application/xml")

    prompt_profile = repository.get_active_prompt_profile(tenant["id"])
    intake_policy = repository.get_intake_policy(tenant["id"])
    repository.create_or_update_call(
        call_sid,
        from_number,
        to_number,
        tenant_id=tenant["id"],
        prompt_version_id=prompt_profile["id"] if prompt_profile else None,
    )
    repository.record_call_event(
        call_sid,
        "voice_received",
        {
            "from_number": from_number,
            "to_number": to_number,
            "tenant_id": tenant["id"],
            "tenant_phone_id": tenant_phone.get("id") if tenant_phone else None,
            "prompt_version_id": prompt_profile["id"] if prompt_profile else None,
            "intake_policy_id": intake_policy["id"] if intake_policy else None,
        },
    )

    sessions[call_sid] = {
        "from_number": from_number,
        "to_number": to_number,
        "tenant_id": tenant["id"],
        "tenant": tenant,
        "tenant_phone": tenant_phone,
        "telephony_profile": telephony_profile,
        "prompt_profile": prompt_profile,
        "intake_policy": intake_policy,
        "prompt_version_id": prompt_profile["id"] if prompt_profile else None,
        "complete": False,
    }

    twiml = (
        "<Response>"
        "<Connect>"
        f'<Stream url="wss://{HOST}/media-stream"/>'
        "</Connect>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")

# ---------------------------------------------------------------------------
# SMS
# ---------------------------------------------------------------------------

def build_sms_body(
    info: dict,
    from_number: str,
    intake_policy: Optional[dict] = None,
    notification_policy: Optional[dict] = None,
) -> str:
    return build_notification_sms_body(info, from_number, intake_policy, notification_policy)


async def send_sms(
    call_sid: str,
    info: dict,
    from_number: str,
    to_number: str,
    intake_policy: Optional[dict] = None,
    notification_policy: Optional[dict] = None,
) -> SmsSendResult:
    body = build_sms_body(info, from_number, intake_policy, notification_policy)
    log.info(f"[{call_sid}] Sending SMS to {to_number}:\n{body}")
    try:
        message = await asyncio.to_thread(
            lambda: twilio.messages.create(
                body=body,
                from_=TWILIO_PHONE_NUMBER,
                to=to_number,
            )
        )
        log.info(f"[{call_sid}] SMS sent")
        return SmsSendResult(success=True, provider_message_sid=getattr(message, "sid", None))
    except Exception as exc:
        log.exception(f"[{call_sid}] SMS failed: {exc}")
        return SmsSendResult(success=False, error=str(exc))


async def hangup_call(
    call_sid: str,
    delay_seconds: float = 5.0,
    reason: str = "service_request_complete",
    lead_id: Optional[int] = None,
):
    await asyncio.sleep(delay_seconds)
    try:
        await asyncio.to_thread(lambda: twilio.calls(call_sid).update(status="completed"))
        safe_record_call_event(
            call_sid,
            "call_ended",
            {
                "ended_by": "app_rest_api",
                "reason": reason,
                "lead_id": lead_id,
            },
        )
        safe_mark_call_ended(call_sid, "app_hangup_completed")
        log.info(f"[{call_sid}] Call ended")
    except Exception as exc:
        log.exception(f"[{call_sid}] Failed to end call: {exc}")

# ---------------------------------------------------------------------------
# Media-stream WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    log.info(">>> /media-stream accepted")

    call_sid  : Optional[str] = None
    stream_sid: Optional[str] = None
    from_number: str          = "unknown"
    oai_ws    : Optional[websockets.WebSocketClientProtocol] = None
    oai_reader_task: Optional[asyncio.Task] = None
    twilio_to_oai_state = None
    oai_to_twilio_state = None
    assistant_speaking = False
    response_active = False
    suppress_input_until = 0.0
    lifecycle = {
        "exit_reason": "unknown",
        "exit_detail": {},
        "openai_reader_exit_reason": None,
        "openai_close_code": None,
        "openai_close_reason": None,
        "closing_oai_from_media_finally": False,
    }

    # -- OpenAI reader (background task) -------------------------------------

    async def clear_twilio_audio():
        if stream_sid:
            await ws.send_text(json.dumps({
                "event":     "clear",
                "streamSid": stream_sid,
            }))

    async def oai_reader():
        nonlocal assistant_speaking, oai_to_twilio_state, response_active, suppress_input_until
        assistant_transcript = ""
        caller_transcript = ""

        async def send_response_create(reason: str, response_create: dict):
            session = sessions.setdefault(call_sid, {})
            payload = response_create_event_payload(reason, response_create, session)
            if reason == "initial_greeting":
                session["initial_greeting_sent_at"] = asyncio.get_running_loop().time()
                session["caller_spoke_since_initial_greeting"] = False
            safe_record_call_event(call_sid, "response_create_sent", payload)
            session["last_response_create_reason"] = reason
            await oai_ws.send(json.dumps(response_create))

        try:
            async for raw in oai_ws:
                evt   = json.loads(raw)
                etype = evt.get("type", "")

                if etype != "response.output_audio.delta":
                    log.info(f"[{call_sid}] [OAI_IN] {etype}")

                # Forward AI audio to Twilio (pcm16 24kHz → mulaw 8kHz)
                if etype == "response.output_audio.delta":
                    delta = evt.get("delta", "")
                    if delta and stream_sid:
                        assistant_speaking = True
                        pcm24k = base64.b64decode(delta)
                        pcm8k, oai_to_twilio_state = audioop.ratecv(
                            pcm24k, 2, 1, 24000, 8000, oai_to_twilio_state
                        )
                        mulaw = audioop.lin2ulaw(pcm8k, 2)
                        await ws.send_text(json.dumps({
                            "event":     "media",
                            "streamSid": stream_sid,
                            "media":     {"payload": base64.b64encode(mulaw).decode()},
                        }))

                elif etype == "response.output_audio_transcript.delta":
                    assistant_transcript += evt.get("delta", "")

                elif etype == "response.output_audio_transcript.done":
                    transcript = evt.get("transcript") or assistant_transcript
                    if transcript.strip():
                        clean_transcript = transcript.strip()
                        log.info(f"[{call_sid}] AI said: {clean_transcript}")
                        session = sessions.get(call_sid, {})
                        session["last_ai_transcript"] = clean_transcript
                        safe_record_call_event(
                            call_sid,
                            "assistant_transcript",
                            {
                                "transcript": clean_transcript,
                                "response_create_reason": session.get("last_response_create_reason"),
                                "caller_spoke_since_initial_greeting": bool(
                                    session.get("caller_spoke_since_initial_greeting")
                                ),
                                "first_caller_speech_started": bool(session.get("first_caller_speech_started")),
                            },
                        )
                    assistant_transcript = ""

                elif etype == "response.output_audio.done":
                    assistant_speaking = False
                    suppress_input_until = asyncio.get_running_loop().time() + 0.25

                elif etype in (
                    "conversation.item.input_audio_transcription.delta",
                    "input_audio_transcription.delta",
                ):
                    caller_transcript += evt.get("delta", "")

                elif etype in (
                    "conversation.item.input_audio_transcription.completed",
                    "conversation.item.input_audio_transcription.done",
                    "input_audio_transcription.completed",
                    "input_audio_transcription.done",
                ):
                    transcript = evt.get("transcript") or caller_transcript
                    if transcript.strip():
                        clean_transcript = transcript.strip()
                        log.info(f"[{call_sid}] Caller said: {clean_transcript}")
                        if call_sid:
                            session = sessions.setdefault(call_sid, {})
                            if session.get("greeting_sent"):
                                session["caller_spoke_since_initial_greeting"] = True
                            caller_text_parts = session.setdefault("caller_text_parts", [])
                            caller_text_parts.append(clean_transcript)
                            intake_state = session.setdefault("intake_state", {})
                            pending_question = intake_state.get("pending_extra_question") or {}
                            if pending_question.get("asked"):
                                pending_question["caller_response_after_pending"] = True
                                pending_question["caller_response_text"] = clean_transcript
                                safe_record_call_event(
                                    call_sid,
                                    "intake_question_caller_response",
                                    {
                                        "pending_extra_question": pending_question,
                                        "transcript": clean_transcript,
                                    },
                                )
                            safe_record_call_event(
                                call_sid,
                                "caller_transcript",
                                {"transcript": clean_transcript},
                            )
                    caller_transcript = ""

                # Function call — GA API delivers complete call in response.output_item.done
                elif etype == "session.updated":
                    session = sessions.get(call_sid, {})
                    if not session.get("greeting_sent"):
                        session["greeting_sent"] = True
                        greeting_text = session.get("greeting") or "Plumbing office, what's going on?"
                        await send_response_create("initial_greeting", build_initial_greeting_response(greeting_text))
                        log.info(f"[{call_sid}] Session updated, greeting sent")

                elif etype == "input_audio_buffer.speech_started":
                    session = sessions.get(call_sid, {})
                    if call_sid and session.get("greeting_sent") and not session.get("first_caller_speech_started"):
                        session["first_caller_speech_started"] = True
                        session["caller_spoke_since_initial_greeting"] = True
                        elapsed = None
                        if session.get("initial_greeting_sent_at"):
                            elapsed = round(asyncio.get_running_loop().time() - session["initial_greeting_sent_at"], 3)
                        safe_record_call_event(
                            call_sid,
                            "first_caller_speech_started_after_initial_greeting",
                            {
                                "elapsed_seconds_after_initial_greeting": elapsed,
                                "response_active": response_active,
                                "assistant_speaking": assistant_speaking,
                                "last_response_create_reason": session.get("last_response_create_reason"),
                            },
                        )
                    intake_state = session.setdefault("intake_state", {})
                    pending_question = intake_state.get("pending_extra_question") or {}
                    if call_sid and pending_question.get("asked") and not pending_question.get("caller_response_after_pending"):
                        pending_question["caller_response_after_pending"] = True
                        safe_record_call_event(
                            call_sid,
                            "intake_question_caller_response_started",
                            {"pending_extra_question": pending_question},
                        )
                    if assistant_speaking or asyncio.get_running_loop().time() < suppress_input_until:
                        log.info(f"[{call_sid}] Ignoring speech_started during AI audio/suppress window")
                    elif not session.get("complete") and response_active:
                        log.info(f"[{call_sid}] Caller speech started; canceling current AI audio")
                        try:
                            await oai_ws.send(json.dumps({"type": "response.cancel"}))
                        except Exception as exc:
                            log.info(f"[{call_sid}] response.cancel ignored: {exc}")
                        await clear_twilio_audio()
                    else:
                        log.info(f"[{call_sid}] Caller speech started")

                elif etype == "response.created":
                    response_active = True
                    session = sessions.get(call_sid, {})
                    if session.get("pending_hangup") and not session.get("closing_response_started"):
                        session["closing_response_started"] = True
                        log.info(f"[{call_sid}] Closing response started")

                elif etype == "response.done":
                    response_active = False
                    assistant_speaking = False
                    suppress_input_until = max(suppress_input_until, asyncio.get_running_loop().time() + 0.25)
                    session = sessions.get(call_sid, {})
                    if (
                        session.get("pending_hangup")
                        and session.get("closing_response_started")
                        and not session.get("hangup_scheduled")
                    ):
                        if can_schedule_hangup(session):
                            session["hangup_scheduled"] = True
                            payload = hangup_scheduled_payload(session)
                            set_media_stream_exit_reason(lifecycle, "app_hangup_scheduled", payload)
                            safe_record_call_event(call_sid, "hangup_scheduled", payload)
                            log.info(f"[{call_sid}] Closing response done; scheduling hangup payload={payload}")
                            asyncio.create_task(
                                hangup_call(
                                    call_sid,
                                    reason=payload["reason"],
                                    lead_id=payload["lead_id"],
                                )
                            )
                        else:
                            payload = hangup_scheduled_payload(session)
                            safe_record_call_event(call_sid, "hangup_schedule_blocked", payload)
                            log.error(f"[{call_sid}] Hangup schedule blocked by lifecycle guard: {payload}")

                elif etype == "response.output_item.done":
                    item = evt.get("item", {})
                    if item.get("type") == "function_call" and item.get("name") == "submit_service_request":
                        call_id = item.get("call_id", "")
                        try:
                            args = json.loads(item.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            log.error(f"[{call_sid}] Bad function args: {item.get('arguments')!r}")
                            args = {}

                        session = sessions.get(call_sid, {})
                        session["submit_service_request_seen"] = True
                        log.info(f"[{call_sid}] submit_service_request: {args}")
                        intake_policy = session.get("intake_policy")
                        notification_policy = repository.get_notification_policy(session.get("tenant_id")) if session.get("tenant_id") else None
                        safe_record_call_event(
                            call_sid,
                            "submit_service_request_attempt",
                            {
                                "args": args,
                                "caller_spoke_since_initial_greeting": bool(
                                    session.get("caller_spoke_since_initial_greeting")
                                ),
                                "last_response_create_reason": session.get("last_response_create_reason"),
                            },
                        )

                        async def send_tenant_sms(send_call_sid: str, send_args: dict, send_from_number: str, send_to_number: str):
                            return await send_sms(
                                send_call_sid,
                                send_args,
                                send_from_number,
                                send_to_number,
                                intake_policy,
                                notification_policy,
                            )

                        result = await process_service_request(
                            call_sid,
                            args,
                            session.get("from_number", "unknown"),
                            session.get("notification_sms_number", ""),
                            send_tenant_sms,
                            caller_text=" ".join(session.get("caller_text_parts", [])),
                            tenant_id=session.get("tenant_id"),
                            intake_state=session.setdefault("intake_state", {}),
                        )
                        session["complete"] = result.should_hangup
                        session["last_service_request_result"] = result.output
                        if result.output.get("lead_id"):
                            session["lead_id"] = result.output.get("lead_id")
                        if result.should_hangup:
                            session["hangup_reason"] = result.output.get("reason") or "service_request_complete"

                        # Return result so AI delivers the closing line
                        await oai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type":    "function_call_output",
                                "call_id": call_id,
                                "output":  build_service_request_output(result.output),
                            },
                        }))
                        if result.should_hangup:
                            session["pending_hangup"] = True
                            session["closing_response_started"] = False
                        response_create = {"type": "response.create"}
                        if result.closing_instructions:
                            response_create["response"] = {"instructions": result.closing_instructions}
                        await send_response_create(
                            response_create_reason_for_service_result(result.output, result.should_hangup),
                            response_create,
                        )
                        pending_question = session.setdefault("intake_state", {}).get("pending_extra_question") or {}
                        if result.output.get("reason") in {
                            "intake_policy_missing_extra_fields",
                            "intake_policy_unanswered_extra_field",
                        } and pending_question:
                            pending_question["asked"] = True
                            safe_record_call_event(
                                call_sid,
                                "intake_question_prompted",
                                {"pending_extra_question": pending_question},
                            )

                elif etype == "error":
                    err = evt.get("error") or {}
                    log.error(f"[{call_sid}] [OAI_IN] error: {err}")
                    safe_record_call_event(call_sid, "openai_realtime_error", {"error": err})
        except websockets.exceptions.ConnectionClosed as exc:
            if not lifecycle.get("closing_oai_from_media_finally"):
                record_openai_websocket_closed(
                    call_sid,
                    stream_sid,
                    exc,
                    lifecycle,
                    sessions.get(call_sid, {}),
                    response_active,
                    assistant_speaking,
                )
        except Exception as exc:
            record_openai_reader_error(
                call_sid,
                stream_sid,
                exc,
                lifecycle,
                sessions.get(call_sid, {}),
                response_active,
                assistant_speaking,
            )
        finally:
            lifecycle["openai_reader_done"] = True

    # -- Twilio reader (main loop) --------------------------------------------

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except WebSocketDisconnect as exc:
                record_twilio_websocket_disconnect(
                    call_sid,
                    stream_sid,
                    exc,
                    lifecycle,
                    sessions.get(call_sid, {}),
                    response_active,
                    assistant_speaking,
                )
                break
            data = json.loads(raw)
            evt  = data.get("event")

            if evt == "connected":
                log.info(f"Twilio connected  protocol={data.get('protocol')}")

            elif evt == "start":
                call_sid    = data["start"]["callSid"]
                stream_sid  = data["start"]["streamSid"]
                session     = sessions.setdefault(call_sid, {})
                tenant      = session.get("tenant") or repository.get_call_tenant(call_sid) or repository.get_default_tenant()
                prompt_profile = (
                    session.get("prompt_profile")
                    or repository.get_call_prompt_profile(call_sid)
                    or repository.get_active_prompt_profile(tenant["id"])
                )
                intake_policy = session.get("intake_policy") or repository.get_intake_policy(tenant["id"])
                session["tenant"] = tenant
                session["tenant_id"] = tenant["id"]
                session["prompt_profile"] = prompt_profile
                session["intake_policy"] = intake_policy
                session["prompt_version_id"] = prompt_profile["id"] if prompt_profile else None
                session["greeting"] = (
                    prompt_profile.get("greeting") if prompt_profile else tenant.get("greeting")
                ) or DEFAULT_GREETING
                session["notification_sms_number"] = tenant.get("notification_sms_number") or (
                    PLUMBER_PHONE_NUMBER if tenant.get("slug") == "default" else ""
                )
                from_number = session.get("from_number", "unknown")
                log.info(f"[{call_sid}] Stream started  sid={stream_sid}  from={from_number}")
                repository.update_call_stream_started(
                    call_sid,
                    stream_sid,
                    prompt_version_id=session.get("prompt_version_id"),
                )
                safe_record_call_event(
                    call_sid,
                    "media_stream_started",
                    {
                        **data.get("start", {}),
                        "prompt_version_id": session.get("prompt_version_id"),
                        "intake_policy_id": intake_policy["id"] if intake_policy else None,
                    },
                )

                log.info(f"[{call_sid}] Connecting to OpenAI: {OAI_URL}")
                oai_ws = await websockets.connect(
                    OAI_URL,
                    extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                )
                log.info(f"[{call_sid}] OpenAI connected")

                session_update = build_session_update(from_number, tenant, prompt_profile, intake_policy)
                raw_su = json.dumps(session_update)
                log.info(f"[{call_sid}] SENDING session.update")
                oai_reader_task = asyncio.create_task(oai_reader())
                await oai_ws.send(raw_su)
                log.info(f"[{call_sid}] Session update sent")

            elif evt == "media":
                if oai_ws and not oai_ws.closed:
                    if assistant_speaking or asyncio.get_running_loop().time() < suppress_input_until:
                        continue

                    # Twilio mulaw 8kHz → pcm16 24kHz → OpenAI
                    mulaw = base64.b64decode(data["media"]["payload"])
                    pcm8k = audioop.ulaw2lin(mulaw, 2)
                    pcm24k, twilio_to_oai_state = audioop.ratecv(
                        pcm8k, 2, 1, 8000, 24000, twilio_to_oai_state
                    )
                    await oai_ws.send(json.dumps({
                        "type":  "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm24k).decode(),
                    }))

            elif evt == "stop":
                record_twilio_stream_stopped(
                    call_sid,
                    stream_sid,
                    data.get("stop", {}),
                    lifecycle,
                    sessions.get(call_sid, {}),
                    response_active,
                    assistant_speaking,
                )
                break

    except Exception as exc:
        set_media_stream_exit_reason(
            lifecycle,
            "exception",
            {
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            },
        )
        safe_record_call_event(
            call_sid,
            "media_stream_exception",
            {
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "snapshot": media_stream_session_snapshot(
                    call_sid,
                    stream_sid,
                    sessions.get(call_sid, {}),
                    lifecycle,
                    response_active,
                    assistant_speaking,
                ),
            },
        )
        safe_mark_call_ended(call_sid, "media_stream_error")
        log.exception(f"[{call_sid}] media_stream error: {exc}")
    finally:
        if oai_ws:
            try:
                lifecycle["closing_oai_from_media_finally"] = True
                await oai_ws.close()
            except Exception:
                pass
        if oai_reader_task:
            try:
                await asyncio.wait_for(oai_reader_task, timeout=1.0)
            except asyncio.TimeoutError:
                oai_reader_task.cancel()
                safe_record_call_event(
                    call_sid,
                    "openai_reader_cancelled",
                    {"reason": "media_stream_finally_timeout"},
                )
            except Exception as exc:
                record_openai_reader_error(
                    call_sid,
                    stream_sid,
                    exc,
                    lifecycle,
                    sessions.get(call_sid, {}),
                    response_active,
                    assistant_speaking,
                )
        if call_sid:
            record_media_stream_done(
                call_sid,
                stream_sid,
                sessions.get(call_sid, {}),
                lifecycle,
                response_active,
                assistant_speaking,
            )
        else:
            log.info("[unknown] media_stream done before call_sid was known")
