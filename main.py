"""
AI Plumber Receptionist
=======================
Flow:
  1. Customer calls Twilio number → POST /voice
  2. We respond with TwiML that speaks a greeting and opens a WebSocket media stream
  3. Twilio streams mu-law 8 kHz audio to /media-stream
  4. We forward each audio chunk to Deepgram for live transcription
  5. On speech_final, we send the transcript to Claude (with full conversation history)
  6. Claude either asks the next question or signals COMPLETE with a JSON payload
  7. To speak: we update the live call via Twilio REST API with new TwiML
     (this closes the current WebSocket; Twilio speaks the text, then reconnects)
  8. On COMPLETE: send SMS to the plumber, hang up
"""

import asyncio
import base64
import json
import logging
import os
import re
import xml.sax.saxutils as saxutils
from typing import Optional

from anthropic import AsyncAnthropic
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents, LiveOptions
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse
from twilio.rest import Client as TwilioClient

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("plumber-receptionist")

# Environment variables
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
DEEPGRAM_API_KEY    = os.getenv("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
PLUMBER_PHONE_NUMBER = os.getenv("PLUMBER_PHONE_NUMBER", "")
HOST = "web-production-4b07e.up.railway.app"

# Clients
twilio_client    = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
dg_client        = DeepgramClient(DEEPGRAM_API_KEY, DeepgramClientOptions(options={"keepalive": "true"}))
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI()

# ---------------------------------------------------------------------------
# In-memory session store  (keyed by Twilio call_sid)
# ---------------------------------------------------------------------------
# {
#   "messages":      list[dict]  — Claude conversation history
#   "is_complete":   bool        — True once all 5 pieces are collected
#   "collected_info":dict        — Parsed final data
#   "from_number":   str         — Caller's phone number
#   "is_responding": bool        — True while Twilio is speaking / stream reconnecting
#   "host":          str         — Effective hostname used for this call's WSS URL
# }
sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# System prompt for Claude
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly AI receptionist for a plumbing company.
Your job is to collect exactly 5 pieces of information from the caller, one at a time.
Keep responses SHORT — this is a phone call. Speak naturally, no bullet points or markdown.

The 5 pieces you must collect (in any convenient order):
1. Customer's full name
2. Service address (full address where the plumber should go)
3. Problem type (what plumbing issue they're having)
4. Urgency — determined by asking 3 sub-questions:
   a. Can they still use the affected fixture/plumbing?
   b. When did the problem start?
   c. Is there any active water damage or flooding?
5. Callback phone number (in case we get disconnected)

Guidelines:
- Ask one question at a time.
- Confirm each piece of information back to the caller before moving on.
- Be warm and reassuring — plumbing problems are stressful.
- If they give you multiple pieces at once, acknowledge all of them and ask only for what's still missing.
- Once you have collected all 5 pieces, output EXACTLY this format on its own line (no extra text before the JSON):
  COMPLETE:{"name":"...","address":"...","problem_type":"...","urgency":"...","callback_number":"..."}
  Then on the NEXT line, say a brief closing message to the caller like:
  "Thank you! We'll have a plumber contact you shortly. Have a good day!"

The urgency field should summarize the 3 sub-answers into a concise description,
e.g. "Cannot use toilet. Started this morning. No active flooding."

Important: Do NOT output the COMPLETE line until you truly have all 5 pieces confirmed.
"""

# ---------------------------------------------------------------------------
# TwiML helpers
# ---------------------------------------------------------------------------

def xml_escape(text: str) -> str:
    return saxutils.escape(text, {'"': "&quot;", "'": "&apos;"})


def build_twiml(text: str, reconnect: bool, host: str) -> str:
    """
    Build TwiML to speak `text`.
    reconnect=True  → reconnect the WebSocket stream after speaking (conversation continues)
    reconnect=False → hang up after speaking (conversation complete)
    """
    escaped = xml_escape(text)
    if reconnect:
        return (
            f'<Response>'
            f'<Say voice="Polly.Joanna">{escaped}</Say>'
            f'<Connect><Stream url="wss://{host}/media-stream"/></Connect>'
            f'</Response>'
        )
    return (
        f'<Response>'
        f'<Say voice="Polly.Joanna">{escaped}</Say>'
        f'<Hangup/>'
        f'</Response>'
    )




# ---------------------------------------------------------------------------
# Startup diagnostics
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    log.info("=== AI Plumber Receptionist starting up ===")
    log.info(f"  TWILIO_ACCOUNT_SID   : {'SET' if TWILIO_ACCOUNT_SID  else 'MISSING'}")
    log.info(f"  TWILIO_AUTH_TOKEN    : {'SET' if TWILIO_AUTH_TOKEN    else 'MISSING'}")
    log.info(f"  TWILIO_PHONE_NUMBER  : {TWILIO_PHONE_NUMBER  or 'MISSING'}")
    log.info(f"  DEEPGRAM_API_KEY     : {'SET' if DEEPGRAM_API_KEY     else 'MISSING'}")
    log.info(f"  ANTHROPIC_API_KEY    : {'SET' if ANTHROPIC_API_KEY    else 'MISSING'}")
    log.info(f"  PLUMBER_PHONE_NUMBER : {PLUMBER_PHONE_NUMBER or 'MISSING'}")
    log.info(f"  HOST                 : {HOST}")
    log.info("===========================================")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "host_env": HOST or "(not set)"}


@app.get("/media-stream")
async def media_stream_probe():
    """HTTP probe so we can confirm the /media-stream route is reachable before testing WSS."""
    return PlainTextResponse("media-stream route is reachable (WebSocket upgrade required for normal use)")


@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Twilio webhook — called when a customer dials the plumber's number.
    Responds with TwiML: speak a greeting, then open a WebSocket media stream.
    """
    form_data = await request.form()
    call_sid    = form_data.get("CallSid", "unknown")
    from_number = form_data.get("From",    "unknown")

    log.info(f"[{call_sid}] Incoming call from {from_number}")
    log.info(f"[{call_sid}] Using host: {HOST!r}")

    sessions[call_sid] = {
        "messages":      [],
        "is_complete":   False,
        "collected_info": {},
        "from_number":   from_number,
        "is_responding": False,
        "host":          HOST,
    }

    greeting = (
        "Hello! Thank you for calling. I'm the virtual receptionist for the plumbing company. "
        "I'll help get a plumber out to you as quickly as possible. "
        "Could I start by getting your full name please?"
    )

    twiml = build_twiml(greeting, reconnect=True, host=effective_host)
    log.info(f"[{call_sid}] Returning TwiML:\n{twiml}")
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle Twilio media stream — STT via Deepgram, conversation via Claude."""

    # Log before accept so we know the route was reached even if accept fails
    log.info(">>> /media-stream WebSocket connection attempt received")

    try:
        await websocket.accept()
    except Exception as exc:
        log.exception(f"WebSocket accept() failed: {exc}")
        return

    log.info(">>> /media-stream WebSocket accepted successfully")

    call_sid: Optional[str] = None
    is_processing = False

    # -- Deepgram setup -------------------------------------------------------
    try:
        dg_connection = dg_client.listen.asynclive.v("1")
        log.info("Deepgram asynclive connection object created")
    except Exception as exc:
        log.exception(f"Failed to create Deepgram connection object: {exc}")
        await websocket.close(code=1011)
        return

    async def on_transcript(self, result, **kwargs):
        nonlocal is_processing
        try:
            alternatives = result.channel.alternatives
            if not alternatives:
                return
            transcript = alternatives[0].transcript.strip()

            if not result.speech_final or not transcript:
                return

            log.info(f"[{call_sid}] speech_final transcript: {transcript!r}")

            if is_processing:
                log.info(f"[{call_sid}] Skipping — already processing")
                return

            session = sessions.get(call_sid)
            if not session:
                log.warning(f"[{call_sid}] No session found for transcript")
                return
            if session.get("is_responding"):
                log.info(f"[{call_sid}] Skipping — is_responding")
                return
            if session.get("is_complete"):
                return

            is_processing = True
            try:
                await handle_transcript(call_sid, transcript)
            finally:
                is_processing = False

        except Exception as exc:
            log.exception(f"on_transcript error: {exc}")
            is_processing = False

    async def on_open(self, open, **kwargs):
        log.info("Deepgram WebSocket opened")

    async def on_error(self, error, **kwargs):
        log.error(f"Deepgram error event: {error}")

    async def on_close(self, close, **kwargs):
        log.info(f"Deepgram WebSocket closed: {close}")

    dg_connection.on(LiveTranscriptionEvents.Open,       on_open)
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.Error,      on_error)
    dg_connection.on(LiveTranscriptionEvents.Close,      on_close)

    dg_options = LiveOptions(
        encoding="mulaw",
        sample_rate=8000,
        channels=1,
        model="nova-2",
        endpointing=400,
        smart_format=True,
    )

    log.info("Starting Deepgram live transcription...")
    try:
        started = await asyncio.wait_for(dg_connection.start(dg_options), timeout=10.0)
        if not started:
            raise RuntimeError("dg_connection.start() returned False")
        log.info("Deepgram live transcription started successfully")
    except asyncio.TimeoutError:
        log.error("Deepgram start() timed out after 10 s")
        await websocket.close(code=1011)
        return
    except Exception as exc:
        log.exception(f"Deepgram start() failed: {exc}")
        await websocket.close(code=1011)
        return

    # -- Main Twilio message loop ---------------------------------------------
    log.info("Entering Twilio media stream message loop")
    chunks_received = 0

    try:
        async for raw_message in websocket.iter_text():
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError as exc:
                log.warning(f"JSON decode error: {exc} — raw: {raw_message[:120]}")
                continue

            event = data.get("event")

            if event == "connected":
                log.info(f"Twilio: connected — protocol={data.get('protocol')} version={data.get('version')}")

            elif event == "start":
                call_sid   = data["start"].get("callSid")
                stream_sid = data["start"].get("streamSid")
                log.info(f"Twilio: start — CallSid={call_sid} StreamSid={stream_sid}")
                log.info(f"Twilio: mediaFormat={data['start'].get('mediaFormat')}")
                if call_sid and call_sid in sessions:
                    sessions[call_sid]["is_responding"] = False
                is_processing = False

            elif event == "media":
                chunks_received += 1
                if chunks_received == 1:
                    log.info(f"[{call_sid}] First audio chunk received — streaming to Deepgram")
                elif chunks_received % 500 == 0:
                    log.info(f"[{call_sid}] Audio chunks forwarded: {chunks_received}")

                session = sessions.get(call_sid) if call_sid else None
                if session and not session.get("is_responding"):
                    audio_bytes = base64.b64decode(data["media"]["payload"])
                    await dg_connection.send(audio_bytes)

            elif event == "stop":
                log.info(f"Twilio: stop — CallSid={call_sid} (total chunks: {chunks_received})")
                break

            else:
                log.info(f"Twilio: unknown event {event!r}")

    except WebSocketDisconnect:
        log.info(f"[{call_sid}] WebSocket disconnected (total chunks: {chunks_received})")
    except Exception as exc:
        log.exception(f"[{call_sid}] Unexpected error in media stream loop: {exc}")
    finally:
        log.info(f"[{call_sid}] Cleaning up — closing Deepgram connection")
        try:
            await dg_connection.finish()
        except Exception as exc:
            log.warning(f"dg_connection.finish() error (non-fatal): {exc}")
        log.info(f"[{call_sid}] media_stream handler done")


# ---------------------------------------------------------------------------
# Core conversation logic
# ---------------------------------------------------------------------------

async def handle_transcript(call_sid: str, transcript: str):
    session = sessions.get(call_sid)
    if not session:
        log.warning(f"[{call_sid}] handle_transcript: no session")
        return

    host = session.get("host", HOST)
    session["messages"].append({"role": "user", "content": transcript})

    log.info(f"[{call_sid}] Calling Claude — {len(session['messages'])} messages in history")
    try:
        response = await anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=session["messages"],
        )
    except Exception as exc:
        log.exception(f"[{call_sid}] Claude API error: {exc}")
        await speak_and_update(call_sid, "I'm sorry, I had a technical issue. Could you please repeat that?", reconnect=True, host=host)
        return

    assistant_text = "".join(b.text for b in response.content if b.type == "text").strip()
    log.info(f"[{call_sid}] Claude response: {assistant_text!r}")

    complete_match = re.search(r"COMPLETE:(\{.*?\})", assistant_text, re.DOTALL)

    if complete_match:
        json_str        = complete_match.group(1)
        closing_message = assistant_text[complete_match.end():].strip()
        if not closing_message:
            closing_message = "Thank you! We'll have a plumber contact you shortly. Have a great day!"

        try:
            collected = json.loads(json_str)
        except json.JSONDecodeError:
            log.error(f"[{call_sid}] Failed to parse COMPLETE JSON: {json_str!r}")
            collected = {}

        session["is_complete"]    = True
        session["collected_info"] = collected
        log.info(f"[{call_sid}] Collected info: {collected}")

        await send_sms_to_plumber(call_sid, collected, session["from_number"])
        session["messages"].append({"role": "assistant", "content": closing_message})
        await speak_and_update(call_sid, closing_message, reconnect=False, host=host)
    else:
        session["messages"].append({"role": "assistant", "content": assistant_text})
        await speak_and_update(call_sid, assistant_text, reconnect=True, host=host)


async def speak_and_update(call_sid: str, text: str, reconnect: bool, host: str):
    """Update the live Twilio call to speak `text`, then reconnect or hang up."""
    session = sessions.get(call_sid)
    if session:
        session["is_responding"] = True

    twiml = build_twiml(text, reconnect=reconnect, host=host)
    log.info(f"[{call_sid}] Updating call — reconnect={reconnect} host={host!r}")
    log.info(f"[{call_sid}] TwiML: {twiml}")

    try:
        await asyncio.to_thread(lambda: twilio_client.calls(call_sid).update(twiml=twiml))
        log.info(f"[{call_sid}] Call update successful")
    except Exception as exc:
        log.exception(f"[{call_sid}] Failed to update call: {exc}")
        if session:
            session["is_responding"] = False


# ---------------------------------------------------------------------------
# SMS notification
# ---------------------------------------------------------------------------

async def send_sms_to_plumber(call_sid: str, info: dict, from_number: str):
    body = (
        f"🔧 NEW PLUMBING REQUEST\n\n"
        f"Name: {info.get('name', 'N/A')}\n"
        f"Address: {info.get('address', 'N/A')}\n"
        f"Problem: {info.get('problem_type', 'N/A')}\n"
        f"Urgency: {info.get('urgency', 'N/A')}\n"
        f"Callback: {info.get('callback_number', 'N/A')}\n"
        f"Caller ID: {from_number}"
    )
    log.info(f"[{call_sid}] Sending SMS to plumber:\n{body}")
    try:
        await asyncio.to_thread(
            lambda: twilio_client.messages.create(
                body=body,
                from_=TWILIO_PHONE_NUMBER,
                to=PLUMBER_PHONE_NUMBER,
            )
        )
        log.info(f"[{call_sid}] SMS sent to {PLUMBER_PHONE_NUMBER}")
    except Exception as exc:
        log.exception(f"[{call_sid}] SMS failed: {exc}")
