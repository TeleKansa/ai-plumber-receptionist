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
from workflow.notifications import SmsSendResult
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

def make_instructions(caller_number: str, tenant: Optional[dict] = None, prompt_profile: Optional[dict] = None) -> str:
    return prompt_builder.build(caller_number, tenant=tenant, profile=prompt_profile)


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
                "callback": {"type": "string", "description": "Callback phone number"},
                "name":     {"type": "string", "description": "Customer name"},
            },
            "required": ["issue", "urgency", "address", "callback", "name"],
        },
    }
]


def build_session_update(caller_number: str, tenant: Optional[dict] = None, prompt_profile: Optional[dict] = None) -> dict:
    session = {
        "type":        "realtime",
        "instructions": make_instructions(caller_number, tenant, prompt_profile),
        "tools":       TOOLS,
        "tool_choice": "auto",
    }
    return {
        "type": "session.update",
        "session": session,
    }

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

@app.post("/voice")
async def voice(request: Request):
    form        = await request.form()
    call_sid    = form.get("CallSid", "unknown")
    from_number = form.get("From",    "unknown")
    to_number   = form.get("To",      "unknown")
    log.info(f"[{call_sid}] Incoming call from {from_number}")

    tenant, tenant_matched = repository.resolve_tenant_for_phone(to_number)
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

    prompt_profile = repository.get_active_prompt_profile(tenant["id"])
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
            "prompt_version_id": prompt_profile["id"] if prompt_profile else None,
        },
    )

    sessions[call_sid] = {
        "from_number": from_number,
        "to_number": to_number,
        "tenant_id": tenant["id"],
        "tenant": tenant,
        "prompt_profile": prompt_profile,
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

async def send_sms(call_sid: str, info: dict, from_number: str, to_number: str) -> SmsSendResult:
    body = (
        f"NEW PLUMBING LEAD\n\n"
        f"Name: {info.get('name',     'N/A')}\n"
        f"Phone: {info.get('callback', from_number)}\n"
        f"Issue: {info.get('issue',   'N/A')}\n"
        f"Urgency: {info.get('urgency', 'N/A')}\n"
        f"Address: {info.get('address', 'N/A')}"
    )
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


async def hangup_call(call_sid: str, delay_seconds: float = 5.0):
    await asyncio.sleep(delay_seconds)
    try:
        await asyncio.to_thread(lambda: twilio.calls(call_sid).update(status="completed"))
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
    twilio_to_oai_state = None
    oai_to_twilio_state = None
    assistant_speaking = False
    response_active = False
    suppress_input_until = 0.0

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
                    log.info(f"[{call_sid}] AI said: {transcript.strip()}")
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
                        caller_text_parts = session.setdefault("caller_text_parts", [])
                        caller_text_parts.append(clean_transcript)
                        repository.record_call_event(
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
                    greeting = json.dumps({
                        "type": "response.create",
                        "response": {
                            "instructions": f'Say only: "{greeting_text}" Then stop.',
                        },
                    })
                    await oai_ws.send(greeting)
                    log.info(f"[{call_sid}] Session updated, greeting sent")

            elif etype == "input_audio_buffer.speech_started":
                session = sessions.get(call_sid, {})
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
                    session["hangup_scheduled"] = True
                    log.info(f"[{call_sid}] Closing response done; scheduling hangup")
                    asyncio.create_task(hangup_call(call_sid))

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
                    log.info(f"[{call_sid}] submit_service_request: {args}")
                    notification_sms_number = session.get("notification_sms_number") or ""

                    async def send_tenant_sms(send_call_sid: str, send_args: dict, send_from_number: str):
                        return await send_sms(send_call_sid, send_args, send_from_number, notification_sms_number)

                    result = await process_service_request(
                        call_sid,
                        args,
                        session.get("from_number", "unknown"),
                        notification_sms_number,
                        send_tenant_sms,
                        caller_text=" ".join(session.get("caller_text_parts", [])),
                        tenant_id=session.get("tenant_id"),
                    )
                    session["complete"] = result.should_hangup

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
                    await oai_ws.send(json.dumps(response_create))

            elif etype == "error":
                err = evt.get("error") or {}
                log.error(f"[{call_sid}] [OAI_IN] error: {err}")

    # -- Twilio reader (main loop) --------------------------------------------

    try:
        async for raw in ws.iter_text():
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
                session["tenant"] = tenant
                session["tenant_id"] = tenant["id"]
                session["prompt_profile"] = prompt_profile
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
                repository.record_call_event(
                    call_sid,
                    "media_stream_started",
                    {**data.get("start", {}), "prompt_version_id": session.get("prompt_version_id")},
                )

                log.info(f"[{call_sid}] Connecting to OpenAI: {OAI_URL}")
                oai_ws = await websockets.connect(
                    OAI_URL,
                    extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                )
                log.info(f"[{call_sid}] OpenAI connected")

                session_update = build_session_update(from_number, tenant, prompt_profile)
                raw_su = json.dumps(session_update)
                log.info(f"[{call_sid}] SENDING session.update")
                asyncio.create_task(oai_reader())
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
                log.info(f"[{call_sid}] Twilio stream stopped")
                if call_sid:
                    repository.record_call_event(call_sid, "media_stream_stopped", data.get("stop", {}))
                    repository.mark_call_ended(call_sid, "stream_stopped")
                break

    except WebSocketDisconnect:
        log.info(f"[{call_sid}] Twilio disconnected")
    except Exception as exc:
        log.exception(f"[{call_sid}] media_stream error: {exc}")
    finally:
        if oai_ws:
            try:
                await oai_ws.close()
            except Exception:
                pass
        if call_sid:
            repository.record_call_event(call_sid, "media_stream_done", {"stream_sid": stream_sid})
        log.info(f"[{call_sid}] media_stream done")
