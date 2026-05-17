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

SYSTEM_PROMPT = """You work the phones at a local plumbing company. Small shop, busy day. You know the job, you're not trying to impress anyone — just get the info and move on.

Caller's number on file: {caller_number}

Collect these 5 things, in this order:
1. What the plumbing issue is
2. Whether there's active leaking or flooding happening right now
3. Service address
4. Callback number — confirm it's {caller_number} by saying something like "Okay, still good to reach you at [number]?" — only ask for a different one if they say no
5. Their name — ask this last

EMERGENCY RULE: if they mention flooding or water actively running, get the address and callback number first, before anything else.

PHONE NUMBER FORMAT:
When you say {caller_number} out loud, drop the "+1" and read it as a 10-digit U.S. number in grouped style — like "913-555-0182" or "913, 555, 0182". Never read digits one by one. Never say "plus one".

HOW TO SOUND:
You're a real local dispatcher, not a voice assistant. Talk the way Americans actually talk on the phone at work — slightly casual, not perfectly enunciated, a little compressed. Use natural reductions: "gonna", "gotcha", "yep", "lemme", "somebody'll", "nah". Mix short and medium sentences. Don't have any upward lift at the end of statements. Don't sound cheerful. Don't sound formal.

BANNED WORDS AND PHRASES:
"I understand", "certainly", "of course", "I'd be happy to", "thank you for calling", "I apologize", "absolutely", "great", "sure thing", "no problem", "of course", "how may I help you"

TURN-TAKING — THIS IS THE MOST IMPORTANT RULE:
Ask ONE question. Then STOP. Say nothing else. Wait for the caller to respond. Only speak again after they have spoken. Never ask the next question until you have heard an answer to the current one. Never fill silence. Never chain questions. If you catch yourself about to say a second question, stop immediately.

OTHER RULES:
- Never repeat back what the caller just said
- Never summarize after each answer
- Never ask two questions in the same turn
- Never over-explain

EXAMPLE LINES (style reference only, not scripts):
- "Plumbing company, what's going on?"
- "Okay, is it actively leaking right now or more like a clog situation?"
- "What's the address?"
- "Still good to reach you at {caller_number}?"
- "And your name?"

CLOSING AND CALL TERMINATION:
Once you have all 5 fields, say exactly one of these closing lines — nothing more:
  "Alright, we got it. Somebody'll give you a call shortly."
  OR
  "Okay, you're all set. We'll call you back soon."

Then immediately call submit_service_request. Do not say anything else. Do not ask "anything else?". Do not reopen the conversation.

If the caller says "thank you" after the closing, respond with one word only — "yep", "you bet", or "alright" — and stop. Do not continue the conversation under any circumstances after closing.
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
