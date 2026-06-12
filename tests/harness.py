"""
Scripted media-stream harness (charter Verification Protocol: "scripted media-stream test").

Drives a full fake call through the app: Twilio webhook -> WS start -> caller audio in ->
session.updated -> greeting -> AI audio out -> function call -> SMS -> hangup scheduling -> stop.
Records every frame the engine sends to OpenAI and to Twilio, plus SMS payloads.

Works against any module exposing: app, sessions, media_stream, and a module-global
`twilio` REST client. OpenAI connection is intercepted by patching websockets.connect.
"""

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest import mock

import httpx

CALLER = "+17327890675"
CALL_SID = "CA_scripted_test_001"
STREAM_SID = "MZ_scripted_test_001"

LEAD_ARGS = {
    "issue": "sink leaking under the sink",
    "urgency": "active leak, caller can shut water off",
    "address": "6100 West 120th Street, Overland Park KS",
    "callback": "+17327890675",
    "name": "Dana Smith",
}

# deterministic audio samples
CALLER_MULAW = base64.b64encode(b"\xff" * 160).decode()       # 20ms mulaw silence
AI_PCM24K = base64.b64encode(b"\x00\x01" * 480).decode()      # 20ms pcm16@24k pattern


class FakeOAIWS:
    def __init__(self):
        self.sent = []            # parsed frames engine sent to OpenAI
        self.queue = asyncio.Queue()
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def close(self):
        self.closed = True
        await self.queue.put(None)

    async def feed(self, obj):
        await self.queue.put(json.dumps(obj))


class _FakeMessages:
    def __init__(self, log):
        self._log = log

    def create(self, body, from_, to):
        self._log.append({"body": body, "from_": from_, "to": to})
        return SimpleNamespace(sid="SM_fake")


class FakeTwilioREST:
    def __init__(self):
        self.sms_log = []
        self.call_update_log = []

    @property
    def messages(self):
        return _FakeMessages(self.sms_log)

    def calls(self, sid):
        return SimpleNamespace(
            update=lambda status: self.call_update_log.append({"sid": sid, "status": status})
        )


class FakeTwilioWS:
    """Stands in for the starlette WebSocket: accept / iter_text / send_text."""

    def __init__(self):
        self.sent = []            # parsed frames engine sent toward Twilio
        self.q = asyncio.Queue()

    async def accept(self):
        pass

    async def send_text(self, raw):
        self.sent.append(json.loads(raw))

    async def iter_text(self):
        while True:
            item = await self.q.get()
            if item is None:
                return
            yield item

    async def feed(self, obj):
        await self.q.put(json.dumps(obj))


async def _wait(cond, timeout=5.0, what=""):
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"timed out waiting for: {what}")
        await asyncio.sleep(0.01)


async def run_scripted_call(mod):
    """Run the full scripted call against `mod`. Returns the behavior record dict."""
    fake_oai = FakeOAIWS()
    fake_rest = FakeTwilioREST()
    ws = FakeTwilioWS()

    async def fake_connect(url, **kwargs):
        fake_oai.connect_url = url
        return fake_oai

    with mock.patch("websockets.connect", fake_connect):
        original_twilio = mod.twilio
        mod.twilio = fake_rest
        try:
            # 1. Twilio voice webhook
            transport = httpx.ASGITransport(app=mod.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/voice", data={"CallSid": CALL_SID, "From": CALLER})
            twiml = resp.text

            # 2. media stream
            task = asyncio.create_task(mod.media_stream(ws))
            await ws.feed({"event": "connected", "protocol": "Call"})
            await ws.feed({"event": "start",
                           "start": {"callSid": CALL_SID, "streamSid": STREAM_SID}})
            await _wait(lambda: len(fake_oai.sent) >= 1, what="session.update")

            # 3. caller audio in -> expect input_audio_buffer.append
            await ws.feed({"event": "media", "media": {"payload": CALLER_MULAW}})
            await _wait(lambda: len(fake_oai.sent) >= 2, what="audio append")

            # 4. session.updated -> greeting
            await fake_oai.feed({"type": "session.updated"})
            await _wait(lambda: len(fake_oai.sent) >= 3, what="greeting")

            # 5. AI audio out -> expect Twilio media frame
            await fake_oai.feed({"type": "response.output_audio.delta", "delta": AI_PCM24K})
            await _wait(lambda: len(ws.sent) >= 1, what="twilio media out")
            await fake_oai.feed({"type": "response.output_audio.done"})
            await fake_oai.feed({"type": "response.output_audio_transcript.done",
                                 "transcript": "Plumbing office, what's going on?"})

            # 6. function call -> SMS + function_call_output + response.create
            await fake_oai.feed({
                "type": "response.output_item.done",
                "item": {"type": "function_call", "name": "submit_service_request",
                         "call_id": "call_123", "arguments": json.dumps(LEAD_ARGS)},
            })
            await _wait(lambda: len(fake_rest.sms_log) >= 1 and len(fake_oai.sent) >= 5,
                        what="sms + fc output + response.create")

            # 7. closing response lifecycle -> hangup scheduled
            await fake_oai.feed({"type": "response.created"})
            await fake_oai.feed({"type": "response.done"})
            await _wait(lambda: mod.sessions.get(CALL_SID, {}).get("hangup_scheduled") is True,
                        what="hangup scheduled")

            # 8. stop
            await ws.feed({"event": "stop"})
            await asyncio.wait_for(task, 5)
        finally:
            mod.twilio = original_twilio
            for t in asyncio.all_tasks() - {asyncio.current_task()}:
                t.cancel()

    return {
        "caller": CALLER,
        "twiml": twiml,
        "oai_connect_url": getattr(fake_oai, "connect_url", None),
        "oai_sent": fake_oai.sent,
        "twilio_ws_sent": ws.sent,
        "sms": fake_rest.sms_log,
        "session_state": {k: v for k, v in mod.sessions.get(CALL_SID, {}).items()},
    }
