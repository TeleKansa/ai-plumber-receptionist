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
import os
from typing import Optional

import websockets
import websockets.exceptions
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse
from twilio.rest import Client as TwilioClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("plumber")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY",       "")
TWILIO_ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID",   "")
TWILIO_AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN",    "")
TWILIO_PHONE_NUMBER  = os.getenv("TWILIO_PHONE_NUMBER",  "")
PLUMBER_PHONE_NUMBER = os.getenv("PLUMBER_PHONE_NUMBER", "")
HOST = "ai-plumber-receptionist-production.up.railway.app"

OAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5"

twilio = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
app    = FastAPI()

# call_sid → {from_number, complete}
sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# System prompt — edit ONLY this block to change AI persona/behavior
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You're answering phones at a small local plumbing shop. Busy office, normal workday. You sound like a real dispatcher who's done this a hundred times today: casual, practical, not polished, not cheerful-corporate.

Caller number on file: {caller_number}

Your job is to collect exactly these 5 fields:
1. plumbing issue
2. urgency / active water status
3. service address
4. callback number
5. customer name

Use this intake flow:
1. First ask what is going on with the plumbing.
2. If their answer is vague, ask ONE useful detail about the issue:
   - leak: "Leaking from where?"
   - clog/backup: "Which drain's backed up?"
   - water heater: "No hot water anywhere, or just one spot?"
   - toilet: "Clogged, running, or leaking?"
3. Ask whether water is actively leaking or flooding right now.
4. If water is actively leaking, flooding, running, spraying, or damaging anything, ask ONE safety/triage question before address: "Can you shut the water off there?" Then get address and callback.
5. Get the service address.
6. Confirm callback number.
7. Ask for the customer's name last.

Use this caller number as the default callback. When confirming it, do not say "+1". Say it like a normal U.S. phone number, grouped: "732-789-0675" or "732, 789, 0675". Never read it digit-by-digit. Ask briefly, like: "And this number's good for callback?"

This is a phone call, not a form. Ask one thing, then stop. Let the caller answer. Do not keep going just because the next question is obvious. If the caller gives a short answer like "yes", "no", "yeah", or "right", that only answers the current question.

Do not call submit_service_request until the caller has actually given all 5 fields. Never guess the name. If the name is missing, ask for it. Put any shutoff answer into the urgency field.

Sound like this:
Caller: Hi, I need a plumber.
Dispatcher: Yeah, what's going on?
Caller: My sink's leaking.
Dispatcher: Leaking from where?
Caller: Under the sink.
Dispatcher: Gotcha. Is water still coming out right now?
Caller: Yeah.
Dispatcher: Can you shut the water off there?
Caller: I think so.
Dispatcher: Okay, what's the address there?
Caller: 6100 West 120th Street.
Dispatcher: Alright. And this number's good for callback?
Caller: Yes.
Dispatcher: Okay, what was your name?

More good lines:
- "Plumbing office, what's going on?"
- "Leaking from where?"
- "Is water still coming out right now?"
- "Can you shut the water off there?"
- "Which drain's backed up?"
- "Okay, what's the service address?"
- "And this number's good for callback?"
- "Alright, what was your name?"
- "Okay, you're all set. We'll call you back soon."

Avoid this kind of language completely:
"I understand", "certainly", "I'd be happy to help", "thank you for calling", "I apologize", "how may I help you", "let me gather some information", "thanks for providing that".

Don't summarize after every answer. Don't repeat their words back. Don't explain why you're asking. Don't ask "anything else?".

When all 5 fields are collected, say one short close:
"Alright, we got it. Somebody'll give you a call shortly."
or
"Okay, you're all set. We'll call you back soon."

Then immediately call submit_service_request. After that, do not continue the conversation. If they say thanks after the close, just say "yep" or "you bet" and stop.
"""

def make_instructions(caller_number: str) -> str:
    return SYSTEM_PROMPT.format(caller_number=caller_number)

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

# ---------------------------------------------------------------------------
# Startup / health
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    log.info("=== Plumber Receptionist (OpenAI Realtime) starting ===")
    log.info(f"  OPENAI_API_KEY       : {'SET' if OPENAI_API_KEY       else 'MISSING'}")
    log.info(f"  TWILIO_ACCOUNT_SID   : {'SET' if TWILIO_ACCOUNT_SID   else 'MISSING'}")
    log.info(f"  TWILIO_AUTH_TOKEN    : {'SET' if TWILIO_AUTH_TOKEN     else 'MISSING'}")
    log.info(f"  TWILIO_PHONE_NUMBER  : {TWILIO_PHONE_NUMBER  or 'MISSING'}")
    log.info(f"  PLUMBER_PHONE_NUMBER : {PLUMBER_PHONE_NUMBER or 'MISSING'}")
    log.info(f"  HOST                 : {HOST}")
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
# SMS
# ---------------------------------------------------------------------------

async def send_sms(call_sid: str, info: dict, from_number: str) -> bool:
    body = (
        f"NEW PLUMBING LEAD\n\n"
        f"Name: {info.get('name',     'N/A')}\n"
        f"Phone: {info.get('callback', from_number)}\n"
        f"Issue: {info.get('issue',   'N/A')}\n"
        f"Urgency: {info.get('urgency', 'N/A')}\n"
        f"Address: {info.get('address', 'N/A')}"
    )
    log.info(f"[{call_sid}] Sending SMS to {PLUMBER_PHONE_NUMBER}:\n{body}")
    try:
        await asyncio.to_thread(
            lambda: twilio.messages.create(
                body=body,
                from_=TWILIO_PHONE_NUMBER,
                to=PLUMBER_PHONE_NUMBER,
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
                suppress_input_until = asyncio.get_running_loop().time() + 0.6

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

            # Function call — GA API delivers complete call in response.output_item.done
            elif etype == "session.updated":
                session = sessions.get(call_sid, {})
                if not session.get("greeting_sent"):
                    session["greeting_sent"] = True
                    greeting = json.dumps({
                        "type": "response.create",
                        "response": {
                            "instructions": 'Say only: "Plumbing office, what\'s going on?" Then stop.',
                            "max_output_tokens": 35,
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
                suppress_input_until = max(suppress_input_until, asyncio.get_running_loop().time() + 0.6)
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
                    session["complete"] = True
                    log.info(f"[{call_sid}] submit_service_request: {args}")

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
                        "response": {
                            "max_output_tokens": 60,
                        },
                    }))

            elif etype == "error":
                log.error(f"[{call_sid}] [OAI_IN] error: {evt.get('error')}")

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

                session_update = {
                    "type": "session.update",
                    "session": {
                        "type":        "realtime",
                        "instructions": make_instructions(from_number),
                        "tools":       TOOLS,
                        "tool_choice": "auto",
                    },
                }
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
