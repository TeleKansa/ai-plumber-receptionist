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

OAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

twilio = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
app    = FastAPI()

# call_sid → {from_number, complete}
sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# System prompt — edit ONLY this block to change AI persona/behavior
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You're a dispatcher at a plumbing company. You answer phones all day. You're busy, slightly distracted, but know your job cold. You do not sound like an AI or a customer service rep.

Caller's number on file: {caller_number}

Collect these 5 things, in order:
1. What's the issue
2. Active leak or flooding right now — yes or no
3. Service address
4. Callback number — say "Still at {caller_number}?" — only ask for different number if they say no
5. Name — ask last

EMERGENCY: if they mention flooding or water actively running, get address and callback number FIRST before anything else.

STYLE RULES — follow these strictly:
- Most replies: 3 to 8 words
- One question per turn, never two
- Do not repeat what the caller said back to them
- Do not say: "I understand", "certainly", "of course", "I can help with that", "thank you for", "let me", "I apologize", "absolutely", "great"
- Do not summarize or confirm what you just heard
- Do not sound empathetic or therapeutic
- Fragment sentences are fine. Real people talk like that.
- If caller gives extra info, skip past it and ask what you need next

TONE EXAMPLES:
- "Yeah, what's going on?"
- "Leaking where exactly?"
- "Active right now?"
- "Got it. Address?"
- "Still at {caller_number}?"
- "And your name?"
- "Alright, somebody'll be in touch."

WHEN DONE: once you have all 5, say "Alright, we got it — someone'll call you back shortly." then immediately call submit_service_request. Do not ask for confirmation. Do not say anything else after.
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

async def send_sms(call_sid: str, info: dict, from_number: str):
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
    except Exception as exc:
        log.exception(f"[{call_sid}] SMS failed: {exc}")

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

    # -- OpenAI reader (background task) -------------------------------------

    async def oai_reader():
        async for raw in oai_ws:
            evt   = json.loads(raw)
            etype = evt.get("type", "")

            if etype != "response.output_audio.delta":
                log.info(f"[{call_sid}] [OAI_IN] {etype}")

            # Forward AI audio to Twilio (pcm16 24kHz → mulaw 8kHz)
            if etype == "response.output_audio.delta":
                delta = evt.get("delta", "")
                if delta and stream_sid:
                    pcm24k = base64.b64decode(delta)
                    pcm8k, _ = audioop.ratecv(pcm24k, 2, 1, 24000, 8000, None)
                    mulaw = audioop.lin2ulaw(pcm8k, 2)
                    await ws.send_text(json.dumps({
                        "event":     "media",
                        "streamSid": stream_sid,
                        "media":     {"payload": base64.b64encode(mulaw).decode()},
                    }))

            # Function call — GA API delivers complete call in response.output_item.done
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

                    await send_sms(call_sid, args, session.get("from_number", "unknown"))

                    # Return result so AI delivers the closing line
                    await oai_ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type":    "function_call_output",
                            "call_id": call_id,
                            "output":  json.dumps({"success": True}),
                        },
                    }))
                    await oai_ws.send(json.dumps({"type": "response.create"}))

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
                await oai_ws.send(raw_su)

                greeting = json.dumps({
                    "type": "response.create",
                    "response": {
                        "instructions": "Greet briefly and ask what the plumbing issue is.",
                    },
                })
                await oai_ws.send(greeting)
                log.info(f"[{call_sid}] Session configured, greeting sent")

                asyncio.create_task(oai_reader())

            elif evt == "media":
                if oai_ws and not oai_ws.closed:
                    # Twilio mulaw 8kHz → pcm16 24kHz → OpenAI
                    mulaw = base64.b64decode(data["media"]["payload"])
                    pcm8k = audioop.ulaw2lin(mulaw, 2)
                    pcm24k, _ = audioop.ratecv(pcm8k, 2, 1, 8000, 24000, None)
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
