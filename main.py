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

import websockets
import websockets.exceptions
from anthropic import AsyncAnthropic
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
HOST = "ai-plumber-receptionist-production.up.railway.app"  # Railway deployment URL

# Deepgram WebSocket URL — connect directly, no SDK
DEEPGRAM_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=mulaw&sample_rate=8000&channels=1"
    "&model=nova-2&smart_format=true&endpointing=400"
)

# Clients
twilio_client    = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
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
Your job is to collect 5 pieces of information from the caller, one at a time.
Keep responses SHORT — this is a phone call. Speak naturally, no bullet points or markdown.

The 5 pieces you must collect:
1. Customer's full name
2. Service address (full address where the plumber should go)
3. Problem type (what plumbing issue they're having)
4. Urgency — ask all 3 of these sub-questions:
   a. Can they still use the affected fixture/plumbing?
   b. When did the problem start?
   c. Is there any active water damage or flooding?
5. Callback phone number

Guidelines:
- Ask one question at a time.
- Be warm and reassuring — plumbing problems are stressful.
- If they give you multiple pieces at once, acknowledge all of them and ask only for what's still missing.
- Once you have everything, give a brief warm closing like "Thank you, a plumber will call you back shortly!"
"""

CHECKER_PROMPT = """You are reviewing a phone call transcript between an AI receptionist and a plumbing customer.

Determine whether ALL of the following have been clearly provided:
- Full name
- Full service address
- Problem description
- Urgency info (ability to use fixture, when it started, whether there is active water damage)
- Callback phone number

Respond with ONLY a raw JSON object — no markdown, no code fences, no explanation:
{"complete": true, "name": "...", "address": "...", "problem": "...", "urgency": "...", "callback": "..."}

If anything is missing, set complete to false and leave missing fields as empty strings.
Do not include anything outside the JSON object."""

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
    log.info(f"  DEEPGRAM_URL         : {DEEPGRAM_URL}")
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

    twiml = build_twiml(greeting, reconnect=True, host=HOST)
    log.info(f"[{call_sid}] Returning TwiML:\n{twiml}")
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle Twilio media stream — raw WebSocket to Deepgram, conversation via Claude."""

    log.info(">>> /media-stream WebSocket connection attempt received")
    try:
        await websocket.accept()
    except Exception as exc:
        log.exception(f"WebSocket accept() failed: {exc}")
        return
    log.info(">>> /media-stream WebSocket accepted successfully")

    call_sid: Optional[str] = None
    is_processing = False

    # -- Connect directly to Deepgram (no SDK) --------------------------------
    log.info(f"Connecting to Deepgram: {DEEPGRAM_URL}")
    try:
        dg_ws = await websockets.connect(
            DEEPGRAM_URL,
            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        )
        log.info("Connected to Deepgram WebSocket successfully")
    except Exception as exc:
        log.exception(f"Failed to connect to Deepgram: {exc}")
        await websocket.close(code=1011)
        return

    # -- Task 1: receive audio from Twilio, forward to Deepgram ---------------
    async def receive_from_twilio():
        nonlocal call_sid, is_processing
        chunks = 0
        try:
            async for raw in websocket.iter_text():
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event = data.get("event")

                if event == "connected":
                    log.info(f"Twilio: connected — protocol={data.get('protocol')}")

                elif event == "start":
                    call_sid   = data["start"].get("callSid")
                    stream_sid = data["start"].get("streamSid")
                    log.info(f"Twilio: start — CallSid={call_sid} StreamSid={stream_sid}")
                    log.info(f"Twilio: mediaFormat={data['start'].get('mediaFormat')}")
                    if call_sid and call_sid in sessions:
                        sessions[call_sid]["is_responding"] = False
                    is_processing = False

                elif event == "media":
                    session = sessions.get(call_sid) if call_sid else None
                    if session and not session.get("is_responding"):
                        audio_bytes = base64.b64decode(data["media"]["payload"])
                        await dg_ws.send(audio_bytes)
                        chunks += 1
                        if chunks == 1:
                            log.info(f"[{call_sid}] First audio chunk sent to Deepgram")
                        elif chunks % 500 == 0:
                            log.info(f"[{call_sid}] Audio chunks sent to Deepgram: {chunks}")

                elif event == "stop":
                    log.info(f"Twilio: stop — CallSid={call_sid} chunks={chunks}")
                    break

                else:
                    log.debug(f"Twilio: unknown event {event!r}")

        except WebSocketDisconnect:
            log.info(f"[{call_sid}] Twilio WebSocket disconnected")
        except Exception as exc:
            log.exception(f"[{call_sid}] Error in receive_from_twilio: {exc}")
        finally:
            # Close Deepgram when Twilio side ends
            try:
                await dg_ws.close()
            except Exception:
                pass

    # -- Task 2: receive transcripts from Deepgram, call Claude ---------------
    async def receive_from_deepgram():
        nonlocal is_processing
        try:
            async for message in dg_ws:
                try:
                    result = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = result.get("type")

                if msg_type == "Results":
                    alts = result.get("channel", {}).get("alternatives", [])
                    if not alts:
                        continue
                    transcript   = alts[0].get("transcript", "").strip()
                    speech_final = result.get("speech_final", False)

                    if not speech_final or not transcript:
                        continue

                    log.info(f"[{call_sid}] Deepgram speech_final: {transcript!r}")

                    if is_processing:
                        log.info(f"[{call_sid}] Skipping transcript — already processing")
                        continue

                    session = sessions.get(call_sid) if call_sid else None
                    if not session:
                        log.warning(f"[{call_sid}] No session for transcript")
                        continue
                    if session.get("is_responding") or session.get("is_complete"):
                        continue

                    is_processing = True
                    try:
                        await handle_transcript(call_sid, transcript)
                    finally:
                        is_processing = False

                elif msg_type == "Metadata":
                    log.info(f"Deepgram metadata: {result}")

                elif msg_type == "Error":
                    log.error(f"Deepgram error message: {result}")

        except websockets.exceptions.ConnectionClosed as exc:
            log.info(f"[{call_sid}] Deepgram connection closed: {exc}")
        except Exception as exc:
            log.exception(f"[{call_sid}] Error in receive_from_deepgram: {exc}")

    # -- Run both tasks concurrently ------------------------------------------
    try:
        await asyncio.gather(receive_from_twilio(), receive_from_deepgram())
    except Exception as exc:
        log.exception(f"[{call_sid}] Unexpected error in gather: {exc}")
    finally:
        try:
            await dg_ws.close()
        except Exception:
            pass
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

    # -- Call 1: conversational response --------------------------------------
    log.info(f"[{call_sid}] Calling Claude (conversation) — {len(session['messages'])} messages")
    try:
        conv_response = await anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=session["messages"],
        )
    except Exception as exc:
        log.exception(f"[{call_sid}] Claude conversation error: {exc}")
        await speak_and_update(call_sid, "I'm sorry, I had a technical issue. Could you please repeat that?", reconnect=True, host=host)
        return

    assistant_text = "".join(b.text for b in conv_response.content if b.type == "text").strip()
    log.info(f"[{call_sid}] Claude conversation response: {assistant_text!r}")
    session["messages"].append({"role": "assistant", "content": assistant_text})

    # -- Call 2: completion check ---------------------------------------------
    # Build a plain-text summary of the conversation for the checker
    convo_text = "\n".join(
        f"{'Customer' if m['role'] == 'user' else 'Receptionist'}: {m['content']}"
        for m in session["messages"]
    )
    log.info(f"[{call_sid}] Calling Claude (checker) to assess completion")
    try:
        check_response = await anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            system=CHECKER_PROMPT,
            messages=[{"role": "user", "content": convo_text}],
        )
    except Exception as exc:
        log.exception(f"[{call_sid}] Claude checker error: {exc}")
        # Checker failed — just continue the conversation normally
        await speak_and_update(call_sid, assistant_text, reconnect=True, host=host)
        return

    checker_raw = "".join(b.text for b in check_response.content if b.type == "text").strip()
    log.info(f"[{call_sid}] Checker raw response: {checker_raw!r}")

    # Strip markdown code fences if Claude wraps the JSON anyway
    checker_raw = re.sub(r"^```[a-z]*\n?|\n?```$", "", checker_raw.strip())

    try:
        checker_result = json.loads(checker_raw)
    except json.JSONDecodeError as exc:
        log.error(f"[{call_sid}] Checker JSON parse failed: {exc} — raw: {checker_raw!r}")
        checker_result = {"complete": False}

    is_complete = checker_result.get("complete", False)
    log.info(f"[{call_sid}] Completion check: complete={is_complete} data={checker_result}")

    if is_complete:
        session["is_complete"]    = True
        session["collected_info"] = checker_result
        log.info(f"[{call_sid}] All info collected — sending SMS to {PLUMBER_PHONE_NUMBER!r}")
        await send_sms_to_plumber(call_sid, checker_result, session["from_number"])
        # Speak the conversational closing then hang up
        await speak_and_update(call_sid, assistant_text, reconnect=False, host=host)
    else:
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
        f"Problem: {info.get('problem', info.get('problem_type', 'N/A'))}\n"
        f"Urgency: {info.get('urgency', 'N/A')}\n"
        f"Callback: {info.get('callback', info.get('callback_number', 'N/A'))}\n"
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
