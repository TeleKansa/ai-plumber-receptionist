"""
Loopline core engine — industry-agnostic AI receptionist
=========================================================
Twilio voice → /media-stream WebSocket → OpenAI Realtime (bidirectional bridge)

Audio: Twilio mulaw 8kHz ↔ pcm16 24kHz ↔ OpenAI (transcoding via audioop)
Function calling: response.output_item.done (GA API event name)

All industry/brand behavior comes from the vertical config (core/vertical.py).
Deployment-specific values come from env vars (set by the tenant entrypoint, main.py):
  VERTICAL     — vertical config name in /verticals/
  PUBLIC_HOST  — public hostname Twilio connects back to

This is a behavior-preserving refactor of the pre-split main.py: the call-handling
logic is intentionally identical (verified against recorded goldens in tests/).
"""

import asyncio
import audioop
import base64
import json
import logging
import os
from typing import Optional

import websockets
import websockets.exceptions
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse
from twilio.rest import Client as TwilioClient

from core.vertical import build_lead_body, load_vertical

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

vertical = load_vertical(os.getenv("VERTICAL", ""))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(f"loopline.{vertical.name}")

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY",     "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_AUTH_TOKEN",  "")
DELIVERY_FROM      = os.getenv(vertical.delivery["from_env"], "")
DELIVERY_TO        = os.getenv(vertical.delivery["to_env"],   "")
HOST               = os.getenv("PUBLIC_HOST", "")

OAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5"

twilio = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
app    = FastAPI()

# call_sid → {from_number, complete}
sessions: dict[str, dict] = {}


def make_instructions(caller_number: str) -> str:
    return vertical.system_prompt.format(caller_number=caller_number)


def build_session_update(caller_number: str) -> dict:
    session = {
        "type":        "realtime",
        "instructions": make_instructions(caller_number),
        "tools":       vertical.tools,
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
    log.info(f"=== Loopline receptionist core starting (vertical: {vertical.name}) ===")
    log.info(f"  OPENAI_API_KEY       : {'SET' if OPENAI_API_KEY     else 'MISSING'}")
    log.info(f"  TWILIO_ACCOUNT_SID   : {'SET' if TWILIO_ACCOUNT_SID else 'MISSING'}")
    log.info(f"  TWILIO_AUTH_TOKEN    : {'SET' if TWILIO_AUTH_TOKEN  else 'MISSING'}")
    log.info(f"  {vertical.delivery['from_env']:<21}: {DELIVERY_FROM or 'MISSING'}")
    log.info(f"  {vertical.delivery['to_env']:<21}: {DELIVERY_TO or 'MISSING'}")
    log.info(f"  HOST                 : {HOST or 'MISSING'}")
    log.info(f"  OAI_URL              : {OAI_URL}")
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
    log.info(f"[{call_sid}] Incoming call from {from_number}")

    sessions[call_sid] = {"from_number": from_number, "complete": False}

    twiml = (
        "<Response>"
        "<Connect>"
        f'<Stream url="wss://{HOST}/media-stream"/>'
        "</Connect>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")

# ---------------------------------------------------------------------------
# Lead delivery
# ---------------------------------------------------------------------------

async def send_sms(call_sid: str, info: dict, from_number: str) -> bool:
    body = build_lead_body(vertical, info, from_number)
    log.info(f"[{call_sid}] Sending SMS to {DELIVERY_TO}:\n{body}")
    try:
        await asyncio.to_thread(
            lambda: twilio.messages.create(
                body=body,
                from_=DELIVERY_FROM,
                to=DELIVERY_TO,
            )
        )
        log.info(f"[{call_sid}] SMS sent")
        return True
    except Exception as exc:
        log.exception(f"[{call_sid}] SMS failed: {exc}")
        return False


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
                    log.info(f"[{call_sid}] Caller said: {transcript.strip()}")
                caller_transcript = ""

            elif etype == "session.updated":
                session = sessions.get(call_sid, {})
                if not session.get("greeting_sent"):
                    session["greeting_sent"] = True
                    greeting = json.dumps({
                        "type": "response.create",
                        "response": {
                            "instructions": vertical.greeting_instruction,
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

            # Function call — GA API delivers complete call in response.output_item.done
            elif etype == "response.output_item.done":
                item = evt.get("item", {})
                if item.get("type") == "function_call" and item.get("name") == vertical.submit_tool_name:
                    call_id = item.get("call_id", "")
                    try:
                        args = json.loads(item.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        log.error(f"[{call_sid}] Bad function args: {item.get('arguments')!r}")
                        args = {}

                    session = sessions.get(call_sid, {})
                    session["complete"] = True
                    log.info(f"[{call_sid}] {vertical.submit_tool_name}: {args}")

                    sms_sent = await send_sms(call_sid, args, session.get("from_number", "unknown"))

                    # Return result so AI delivers the closing line
                    await oai_ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type":    "function_call_output",
                            "call_id": call_id,
                            "output":  json.dumps({"success": True}),
                        },
                    }))
                    if sms_sent:
                        session["pending_hangup"] = True
                        session["closing_response_started"] = False
                    await oai_ws.send(json.dumps({
                        "type": "response.create",
                    }))

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
                session     = sessions.get(call_sid, {})
                from_number = session.get("from_number", "unknown")
                log.info(f"[{call_sid}] Stream started  sid={stream_sid}  from={from_number}")

                log.info(f"[{call_sid}] Connecting to OpenAI: {OAI_URL}")
                oai_ws = await websockets.connect(
                    OAI_URL,
                    extra_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                )
                log.info(f"[{call_sid}] OpenAI connected")

                session_update = build_session_update(from_number)
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
        log.info(f"[{call_sid}] media_stream done")
