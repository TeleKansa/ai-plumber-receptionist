"""
AI Plumber Receptionist — OpenAI Realtime API
==============================================
Twilio voice → /media-stream WebSocket → OpenAI Realtime API (bidirectional audio bridge)

Audio path:
  Twilio  →  mu-law 8 kHz  →  PCM16 24 kHz  →  OpenAI Realtime
  OpenAI  →  PCM16 24 kHz  →  mu-law 8 kHz  →  Twilio

Completion:
  OpenAI calls submit_service_request() when all 5 fields collected
  Backend sends SMS, AI gives closing line, call ends naturally
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

TWILIO_ACCOUNT_SID   = os.getenv("TWILIO_ACCOUNT_SID",   "")
TWILIO_AUTH_TOKEN    = os.getenv("TWILIO_AUTH_TOKEN",    "")
TWILIO_PHONE_NUMBER  = os.getenv("TWILIO_PHONE_NUMBER",  "")
PLUMBER_PHONE_NUMBER = os.getenv("PLUMBER_PHONE_NUMBER", "")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY",       "")
HOST                 = "ai-plumber-receptionist-production.up.railway.app"

OAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

twilio = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
app    = FastAPI()

# call_sid → {from_number, complete}
sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def make_instructions(caller_number: str) -> str:
    return f"""You are a dispatcher for a busy plumbing company. Sound like a real human — casual, brief, slightly rushed. Not corporate. Not overly polite.

The caller's number on file is: {caller_number}

Collect these 5 things in this order:
1. What's the plumbing issue?
2. Is it urgent — active leak or flooding right now?
3. Service address (confirm it carefully)
4. Callback number — ask "Is {caller_number} good to reach you?" — only ask for a different number if they say no
5. Customer name — ask last

EMERGENCY RULE: if caller mentions flooding or active water, ask for address and callback FIRST.

STYLE:
- 1-2 short sentences per turn. One question at a time.
- No small talk. No corporate phrases. Stay focused.
- Sound like you're juggling multiple calls.

WHEN DONE: once you have all 5, say "Alright, we got it. Someone will call you back shortly." and immediately call submit_service_request. Do not ask for confirmation first. Do not say "Is there anything else?"
"""

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
# Audio transcoding
# ---------------------------------------------------------------------------

def mulaw8k_to_pcm16_24k(data: bytes) -> bytes:
    pcm8k         = audioop.ulaw2lin(data, 2)
    pcm24k, _     = audioop.ratecv(pcm8k, 2, 1, 8000, 24000, None)
    return pcm24k

def pcm16_24k_to_mulaw8k(data: bytes) -> bytes:
    pcm8k, _  = audioop.ratecv(data, 2, 1, 24000, 8000, None)
    return audioop.lin2ulaw(pcm8k, 2)

# ---------------------------------------------------------------------------
# Startup / health
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    log.info("=== Plumber Receptionist (OpenAI Realtime) starting ===")
    log.info(f"  TWILIO_ACCOUNT_SID   : {'SET' if TWILIO_ACCOUNT_SID   else 'MISSING'}")
    log.info(f"  TWILIO_AUTH_TOKEN    : {'SET' if TWILIO_AUTH_TOKEN     else 'MISSING'}")
    log.info(f"  TWILIO_PHONE_NUMBER  : {TWILIO_PHONE_NUMBER  or 'MISSING'}")
    log.info(f"  OPENAI_API_KEY       : {'SET' if OPENAI_API_KEY        else 'MISSING'}")
    log.info(f"  PLUMBER_PHONE_NUMBER : {PLUMBER_PHONE_NUMBER or 'MISSING'}")
    log.info(f"  HOST                 : {HOST}")
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
    call_sid    = form.get("CallSid",  "unknown")
    from_number = form.get("From",     "unknown")

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
# Media stream WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    log.info(">>> /media-stream connection attempt")
    try:
        await websocket.accept()
    except Exception as exc:
        log.exception(f"accept() failed: {exc}")
        return
    log.info(">>> /media-stream accepted")

    call_sid    : Optional[str] = None
    stream_sid  : Optional[str] = None
    from_number : str           = "unknown"
    oai_ws      : Optional[websockets.WebSocketClientProtocol] = None
    is_responding               = False   # True while OAI is streaming audio out

    # -- helpers --------------------------------------------------------------

    async def send_audio_to_twilio(pcm24k_b64: str):
        if not stream_sid:
            return
        pcm24k = base64.b64decode(pcm24k_b64)
        mulaw  = pcm16_24k_to_mulaw8k(pcm24k)
        await websocket.send_text(json.dumps({
            "event":    "media",
            "streamSid": stream_sid,
            "media":    {"payload": base64.b64encode(mulaw).decode()},
        }))

    async def clear_twilio_audio():
        if stream_sid:
            await websocket.send_text(json.dumps({
                "event":     "clear",
                "streamSid": stream_sid,
            }))

    async def connect_oai() -> Optional[websockets.WebSocketClientProtocol]:
        for attempt in range(1, 4):
            try:
                ws = await websockets.connect(
                    OAI_URL,
                    extra_headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                    },
                )
                log.info(f"[{call_sid}] Connected to OpenAI Realtime (attempt {attempt})")
                return ws
            except Exception as exc:
                log.error(f"[{call_sid}] OpenAI connect attempt {attempt} failed: {exc}")
                if attempt < 3:
                    await asyncio.sleep(2 ** (attempt - 1))
        log.error(f"[{call_sid}] All OpenAI connect attempts failed")
        return None

    async def configure_oai(ws, trigger_greeting: bool = True):
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities":               ["text", "audio"],
                "instructions":             make_instructions(from_number),
                "voice":                    "alloy",
                "input_audio_format":       "pcm16",
                "output_audio_format":      "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type":                 "server_vad",
                    "threshold":            0.5,
                    "prefix_padding_ms":    300,
                    "silence_duration_ms":  600,
                    "create_response":      True,
                },
                "tools":       TOOLS,
                "tool_choice": "auto",
                "temperature": 0.7,
            },
        }))
        if trigger_greeting:
            await ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities":   ["text", "audio"],
                    "instructions": "Greet briefly and ask what the plumbing issue is.",
                },
            }))
            log.info(f"[{call_sid}] OpenAI session configured, greeting triggered")

    # -- OpenAI event handler (runs as background task) -----------------------

    async def oai_reader():
        nonlocal oai_ws, is_responding
        fn_args = ""
        fn_call_id = None
        fn_name    = None

        while True:
            if oai_ws is None:
                break
            try:
                async for raw in oai_ws:
                    evt   = json.loads(raw)
                    etype = evt.get("type", "")

                    if etype == "session.created":
                        log.info(f"[{call_sid}] OpenAI session created")

                    elif etype == "input_audio_buffer.speech_started":
                        # Caller started speaking — cancel current AI response (barge-in)
                        log.info(f"[{call_sid}] Barge-in detected")
                        is_responding = False
                        try:
                            await oai_ws.send(json.dumps({"type": "response.cancel"}))
                        except Exception:
                            pass
                        await clear_twilio_audio()

                    elif etype == "response.audio.delta":
                        is_responding = True
                        delta = evt.get("delta", "")
                        if delta:
                            try:
                                await send_audio_to_twilio(delta)
                            except Exception as exc:
                                log.warning(f"[{call_sid}] Audio send error: {exc}")

                    elif etype == "response.audio.done":
                        is_responding = False

                    elif etype == "response.function_call_arguments.delta":
                        fn_args += evt.get("delta", "")

                    elif etype == "response.function_call_arguments.done":
                        fn_call_id = evt.get("call_id", "")
                        fn_name    = evt.get("name",    "")
                        log.info(f"[{call_sid}] Function call complete: {fn_name}")

                        if fn_name == "submit_service_request":
                            try:
                                args = json.loads(fn_args)
                            except json.JSONDecodeError:
                                log.error(f"[{call_sid}] Bad function args JSON: {fn_args!r}")
                                args = {}

                            session = sessions.get(call_sid, {})
                            session["complete"] = True
                            log.info(f"[{call_sid}] Collected: {args}")

                            await send_sms(call_sid, args, session.get("from_number", "unknown"))

                            # Return result so AI gives the closing line
                            try:
                                await oai_ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type":    "function_call_output",
                                        "call_id": fn_call_id,
                                        "output":  json.dumps({"success": True}),
                                    },
                                }))
                                await oai_ws.send(json.dumps({"type": "response.create"}))
                            except Exception as exc:
                                log.warning(f"[{call_sid}] Could not send function result: {exc}")

                        fn_args    = ""
                        fn_call_id = None
                        fn_name    = None

                    elif etype == "error":
                        log.error(f"[{call_sid}] OpenAI error: {evt.get('error')}")

            except websockets.exceptions.ConnectionClosed as exc:
                log.warning(f"[{call_sid}] OpenAI disconnected: {exc}")

                if sessions.get(call_sid, {}).get("complete"):
                    break  # Call already done — no need to reconnect

                log.info(f"[{call_sid}] Reconnecting to OpenAI...")
                new_ws = await connect_oai()
                if new_ws:
                    oai_ws = new_ws
                    # Re-configure session without re-greeting (caller already heard intro)
                    await configure_oai(oai_ws, trigger_greeting=False)
                    fn_args = ""  # Reset in-flight function state
                else:
                    log.error(f"[{call_sid}] OpenAI reconnect failed")
                    break

            except Exception as exc:
                log.exception(f"[{call_sid}] oai_reader error: {exc}")
                break

    # -- Twilio audio reader (main loop) --------------------------------------

    try:
        async for raw in websocket.iter_text():
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            evt = data.get("event")

            if evt == "start":
                call_sid    = data["start"]["callSid"]
                stream_sid  = data["start"]["streamSid"]
                session     = sessions.get(call_sid, {})
                from_number = session.get("from_number", "unknown")
                log.info(f"[{call_sid}] Stream started  StreamSid={stream_sid}  from={from_number}")

                oai_ws = await connect_oai()
                if not oai_ws:
                    log.error(f"[{call_sid}] Cannot reach OpenAI — closing call")
                    await websocket.close()
                    break

                asyncio.create_task(oai_reader())
                await configure_oai(oai_ws, trigger_greeting=True)

            elif evt == "media":
                if oai_ws and not oai_ws.closed:
                    raw_mulaw = base64.b64decode(data["media"]["payload"])
                    pcm24k    = mulaw8k_to_pcm16_24k(raw_mulaw)
                    try:
                        await oai_ws.send(json.dumps({
                            "type":  "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm24k).decode(),
                        }))
                    except Exception:
                        pass  # Drop frame if OAI is mid-reconnect

            elif evt == "stop":
                log.info(f"[{call_sid}] Twilio stream stopped")
                break

            elif evt == "connected":
                log.info(f"Twilio: connected  protocol={data.get('protocol')}")

    except WebSocketDisconnect:
        log.info(f"[{call_sid}] Twilio WebSocket disconnected")
    except Exception as exc:
        log.exception(f"[{call_sid}] media_stream error: {exc}")
    finally:
        if oai_ws:
            try:
                await oai_ws.close()
            except Exception:
                pass
        log.info(f"[{call_sid}] media_stream handler done")

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
