"""
AI Plumber Receptionist — Minimal Audio Bridge
===============================================
Phase 1: bare Twilio <-> OpenAI Realtime audio bridge.
No tools, no SMS, no state. Goal: AI speaks and hears caller.

Session pattern mirrors the official OpenAI Twilio demo:
  github.com/openai/openai-realtime-twilio-demo
"""

import asyncio
import json
import logging
import os
from typing import Optional

import websockets
import websockets.exceptions
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("plumber")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID",  "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN",   "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
HOST = "ai-plumber-receptionist-production.up.railway.app"

# Exact model version from the official OpenAI Twilio demo
OAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

app = FastAPI()

# ---------------------------------------------------------------------------
# Startup / health
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    log.info("=== Plumber Receptionist (minimal bridge) starting ===")
    log.info(f"  OPENAI_API_KEY  : {'SET' if OPENAI_API_KEY  else 'MISSING'}")
    log.info(f"  TWILIO_ACCOUNT_SID : {'SET' if TWILIO_ACCOUNT_SID else 'MISSING'}")
    log.info(f"  HOST            : {HOST}")
    log.info(f"  OAI_URL         : {OAI_URL}")
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
    form     = await request.form()
    call_sid = form.get("CallSid", "unknown")
    from_num = form.get("From",    "unknown")
    log.info(f"[{call_sid}] Incoming call from {from_num}")

    twiml = (
        "<Response>"
        "<Connect>"
        f'<Stream url="wss://{HOST}/media-stream"/>'
        "</Connect>"
        "</Response>"
    )
    return Response(content=twiml, media_type="application/xml")

# ---------------------------------------------------------------------------
# Media-stream WebSocket — minimal audio bridge
# ---------------------------------------------------------------------------

@app.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    await ws.accept()
    log.info(">>> /media-stream accepted")

    call_sid  : Optional[str] = None
    stream_sid: Optional[str] = None
    oai_ws    : Optional[websockets.WebSocketClientProtocol] = None

    # -- OpenAI reader (background task) -------------------------------------

    async def oai_reader():
        nonlocal oai_ws
        async for raw in oai_ws:
            evt   = json.loads(raw)
            etype = evt.get("type", "")

            # Log every event type received from OpenAI
            if etype != "response.audio.delta":
                log.info(f"[{call_sid}] [OAI_IN] {etype}")

            if etype == "response.audio.delta":
                delta = evt.get("delta", "")
                if delta and stream_sid:
                    await ws.send_text(json.dumps({
                        "event":    "media",
                        "streamSid": stream_sid,
                        "media":    {"payload": delta},
                    }))

            elif etype == "error":
                log.error(f"[{call_sid}] [OAI_IN] error: {evt.get('error')}")

    # -- Twilio reader (main loop) --------------------------------------------

    try:
        async for raw in ws.iter_text():
            data = json.loads(raw)
            evt  = data.get("event")

            if evt == "connected":
                log.info(f"[{call_sid}] Twilio connected  protocol={data.get('protocol')}")

            elif evt == "start":
                call_sid   = data["start"]["callSid"]
                stream_sid = data["start"]["streamSid"]
                log.info(f"[{call_sid}] Stream started  sid={stream_sid}")

                # Connect to OpenAI — exact headers from official demo
                log.info(f"[{call_sid}] Connecting to OpenAI: {OAI_URL}")
                oai_ws = await websockets.connect(
                    OAI_URL,
                    extra_headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                    },
                )
                log.info(f"[{call_sid}] OpenAI connected")

                # Send session.update immediately on open — exact demo pattern
                session_update = {
                    "type": "session.update",
                    "session": {
                        "modalities":          ["text", "audio"],
                        "voice":               "alloy",
                        "input_audio_format":  "g711_ulaw",
                        "output_audio_format": "g711_ulaw",
                        "turn_detection":      {"type": "server_vad"},
                        "input_audio_transcription": {"model": "whisper-1"},
                    },
                }
                raw_su = json.dumps(session_update)
                log.info(f"[{call_sid}] SENDING session.update: {raw_su}")
                await oai_ws.send(raw_su)

                # Trigger greeting
                greeting = json.dumps({
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "You are a dispatcher for a plumbing company. "
                            "Greet the caller briefly and ask what the plumbing issue is."
                        ),
                    },
                })
                log.info(f"[{call_sid}] SENDING response.create (greeting)")
                await oai_ws.send(greeting)

                asyncio.create_task(oai_reader())

            elif evt == "media":
                if oai_ws and not oai_ws.closed:
                    await oai_ws.send(json.dumps({
                        "type":  "input_audio_buffer.append",
                        "audio": data["media"]["payload"],
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
