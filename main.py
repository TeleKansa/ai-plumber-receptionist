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
from fastapi.responses import Response
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
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PLUMBER_PHONE_NUMBER = os.getenv("PLUMBER_PHONE_NUMBER", "")
HOST = os.getenv("HOST", "")  # Railway domain, e.g. myapp.up.railway.app

# Clients
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
dg_client = DeepgramClient(DEEPGRAM_API_KEY, DeepgramClientOptions(options={"keepalive": "true"}))
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

app = FastAPI()

# ---------------------------------------------------------------------------
# In-memory session store  (keyed by Twilio call_sid)
# ---------------------------------------------------------------------------

# Each session:
# {
#   "messages":      list[dict]  — Claude conversation history
#   "is_complete":   bool        — True once all 5 pieces are collected
#   "collected_info":dict        — Parsed final data
#   "from_number":   str         — Caller's phone number
#   "is_responding": bool        — True while we're calling Twilio to speak
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
    """Escape text for safe inclusion in XML/TwiML."""
    return saxutils.escape(text, {'"': "&quot;", "'": "&apos;"})


def build_twiml(text: str, reconnect: bool = True) -> str:
    """
    Build a TwiML response string.

    If reconnect=True, after speaking the text Twilio will reconnect the
    WebSocket media stream so we can continue the conversation.

    If reconnect=False, Twilio speaks the text then hangs up.
    """
    escaped = xml_escape(text)
    if reconnect:
        return (
            f'<Response>'
            f'<Say voice="Polly.Joanna">{escaped}</Say>'
            f'<Connect><Stream url="wss://{HOST}/media-stream"/></Connect>'
            f'</Response>'
        )
    else:
        return (
            f'<Response>'
            f'<Say voice="Polly.Joanna">{escaped}</Say>'
            f'<Hangup/>'
            f'</Response>'
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Simple health check used by Railway and monitoring tools."""
    return {"status": "ok"}


@app.post("/voice")
async def voice_webhook(request: Request):
    """
    Twilio calls this when a customer dials the plumber's number.
    We respond with TwiML that:
      1. Says a greeting
      2. Opens a WebSocket media stream to /media-stream
    """
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    from_number = form_data.get("From", "unknown")

    log.info(f"Incoming call: CallSid={call_sid} From={from_number}")

    # Initialise session for this call
    sessions[call_sid] = {
        "messages": [],
        "is_complete": False,
        "collected_info": {},
        "from_number": from_number,
        "is_responding": False,
    }

    greeting = (
        "Hello! Thank you for calling. I'm the virtual receptionist for the plumbing company. "
        "I'll help get a plumber out to you as quickly as possible. "
        "Could I start by getting your full name please?"
    )

    twiml = build_twiml(greeting, reconnect=True)
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    Twilio streams mu-law 8 kHz audio here.
    We:
      1. Open a Deepgram live-transcription connection
      2. Forward each audio chunk to Deepgram
      3. On speech_final transcripts, run Claude and update the call
    """
    await websocket.accept()
    log.info("WebSocket connection accepted")

    call_sid: Optional[str] = None

    # -- Deepgram connection --------------------------------------------------
    # Use asynclive (stable across deepgram-sdk v3.x)
    dg_connection = dg_client.listen.asynclive.v("1")

    # Flag to prevent double-processing while we're already responding
    is_processing = False

    async def on_transcript(self, result, **kwargs):
        """Called by Deepgram when a transcript is ready."""
        nonlocal is_processing

        try:
            alternatives = result.channel.alternatives
            if not alternatives:
                return
            transcript = alternatives[0].transcript.strip()

            # Only act on speech_final events with actual content
            if not result.speech_final or not transcript:
                return

            # Skip if we're already mid-response or call is done
            if is_processing:
                log.debug(f"Skipping transcript (already processing): {transcript!r}")
                return

            session = sessions.get(call_sid)
            if not session:
                return

            if session.get("is_responding"):
                log.debug(f"Skipping transcript (is_responding): {transcript!r}")
                return

            if session.get("is_complete"):
                return

            is_processing = True
            log.info(f"[{call_sid}] Transcript: {transcript!r}")

            try:
                await handle_transcript(call_sid, transcript)
            finally:
                is_processing = False

        except Exception as exc:
            log.exception(f"Error in on_transcript: {exc}")
            is_processing = False

    async def on_error(self, error, **kwargs):
        log.error(f"Deepgram error: {error}")

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)

    # Deepgram options: mu-law 8 kHz to match Twilio's media stream
    dg_options = LiveOptions(
        encoding="mulaw",
        sample_rate=8000,
        channels=1,
        model="nova-2",
        endpointing=400,       # ms of silence before declaring end-of-utterance
        smart_format=True,
    )

    # Start Deepgram — must be inside try/except; a failure here would otherwise
    # crash the WebSocket handler and cause Twilio to silently hang up the call.
    try:
        started = await asyncio.wait_for(dg_connection.start(dg_options), timeout=10.0)
        if not started:
            raise RuntimeError("dg_connection.start() returned False")
        log.info("Deepgram connection started successfully")
    except Exception as exc:
        log.exception(f"Deepgram failed to start: {exc}")
        # Speak an error instead of silently hanging up
        await asyncio.to_thread(
            lambda: twilio_client.calls(call_sid or "").update(
                twiml="<Response><Say>Sorry, I'm having a technical issue. Please call back in a moment.</Say><Hangup/></Response>"
            ) if call_sid else None
        )
        await websocket.close()
        return

    # -- Main WebSocket loop --------------------------------------------------
    try:
        async for raw_message in websocket.iter_text():
            try:
                data = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            event = data.get("event")

            if event == "connected":
                log.info("Twilio WebSocket: connected")

            elif event == "start":
                # New stream started (including reconnects after we update the call)
                call_sid = data["start"].get("callSid")
                log.info(f"Twilio WebSocket: start — CallSid={call_sid}")

                # Reset responding flag on every reconnect
                if call_sid and call_sid in sessions:
                    sessions[call_sid]["is_responding"] = False

                is_processing = False

            elif event == "media":
                # Raw audio chunk — forward to Deepgram if not mid-response
                session = sessions.get(call_sid) if call_sid else None
                if session and not session.get("is_responding"):
                    audio_bytes = base64.b64decode(data["media"]["payload"])
                    await dg_connection.send(audio_bytes)

            elif event == "stop":
                log.info(f"Twilio WebSocket: stop — CallSid={call_sid}")
                break

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected — CallSid={call_sid}")
    except Exception as exc:
        log.exception(f"Unexpected error in media_stream: {exc}")
    finally:
        try:
            await dg_connection.finish()
        except Exception:
            pass
        log.info(f"Deepgram connection closed — CallSid={call_sid}")


# ---------------------------------------------------------------------------
# Core conversation logic
# ---------------------------------------------------------------------------

async def handle_transcript(call_sid: str, transcript: str):
    """
    Given a user utterance, ask Claude what to say next, then speak it via
    Twilio by updating the live call.
    """
    session = sessions.get(call_sid)
    if not session:
        log.warning(f"No session for CallSid={call_sid}")
        return

    # Add user message to history
    session["messages"].append({"role": "user", "content": transcript})

    # -- Call Claude ----------------------------------------------------------
    try:
        response = await anthropic_client.messages.create(
            model="claude-opus-4-5",   # as specified in the requirements
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=session["messages"],
        )
    except Exception as exc:
        log.exception(f"Claude API error: {exc}")
        # Speak a generic error so the caller isn't left in silence
        await speak_and_update(call_sid, "I'm sorry, I had a technical issue. Could you please repeat that?", reconnect=True)
        return

    assistant_text = ""
    for block in response.content:
        if block.type == "text":
            assistant_text += block.text

    assistant_text = assistant_text.strip()
    log.info(f"[{call_sid}] Claude response: {assistant_text!r}")

    # -- Check for COMPLETE signal --------------------------------------------
    complete_match = re.search(
        r"COMPLETE:(\{.*?\})",
        assistant_text,
        re.DOTALL,
    )

    if complete_match:
        # Extract the JSON payload and everything after it (the closing message)
        json_str = complete_match.group(1)
        closing_message = assistant_text[complete_match.end():].strip()

        if not closing_message:
            closing_message = "Thank you! We'll have a plumber contact you shortly. Have a great day!"

        try:
            collected = json.loads(json_str)
        except json.JSONDecodeError:
            log.error(f"Failed to parse COMPLETE JSON: {json_str!r}")
            collected = {}

        session["is_complete"] = True
        session["collected_info"] = collected

        log.info(f"[{call_sid}] Collected info: {collected}")

        # Send SMS to plumber
        await send_sms_to_plumber(call_sid, collected, session["from_number"])

        # Add Claude's closing to history (just the closing text, not the COMPLETE line)
        session["messages"].append({"role": "assistant", "content": closing_message})

        # Speak closing message and hang up
        await speak_and_update(call_sid, closing_message, reconnect=False)

    else:
        # Normal conversational turn — add to history and speak
        session["messages"].append({"role": "assistant", "content": assistant_text})
        await speak_and_update(call_sid, assistant_text, reconnect=True)


async def speak_and_update(call_sid: str, text: str, reconnect: bool):
    """
    Update the live Twilio call with new TwiML to speak `text`.

    Twilio will close the current WebSocket, speak the text using Polly,
    then (if reconnect=True) reconnect a new WebSocket to /media-stream.

    We set is_responding=True first so that any trailing audio from Deepgram
    doesn't trigger a second response while Twilio is speaking.
    """
    session = sessions.get(call_sid)
    if session:
        session["is_responding"] = True

    twiml = build_twiml(text, reconnect=reconnect)

    try:
        # Twilio REST calls are synchronous — run in a thread to avoid blocking
        await asyncio.to_thread(
            lambda: twilio_client.calls(call_sid).update(twiml=twiml)
        )
        log.info(f"[{call_sid}] Call updated — reconnect={reconnect}")
    except Exception as exc:
        log.exception(f"[{call_sid}] Failed to update call: {exc}")
        if session:
            session["is_responding"] = False


# ---------------------------------------------------------------------------
# SMS notification
# ---------------------------------------------------------------------------

async def send_sms_to_plumber(call_sid: str, info: dict, from_number: str):
    """Send a formatted SMS with all collected information to the plumber."""
    name = info.get("name", "N/A")
    address = info.get("address", "N/A")
    problem_type = info.get("problem_type", "N/A")
    urgency = info.get("urgency", "N/A")
    callback_number = info.get("callback_number", "N/A")

    body = (
        f"🔧 NEW PLUMBING REQUEST\n\n"
        f"Name: {name}\n"
        f"Address: {address}\n"
        f"Problem: {problem_type}\n"
        f"Urgency: {urgency}\n"
        f"Callback: {callback_number}\n"
        f"Caller ID: {from_number}"
    )

    try:
        await asyncio.to_thread(
            lambda: twilio_client.messages.create(
                body=body,
                from_=TWILIO_PHONE_NUMBER,
                to=PLUMBER_PHONE_NUMBER,
            )
        )
        log.info(f"[{call_sid}] SMS sent to plumber at {PLUMBER_PHONE_NUMBER}")
    except Exception as exc:
        log.exception(f"[{call_sid}] Failed to send SMS: {exc}")
