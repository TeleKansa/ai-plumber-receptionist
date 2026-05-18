# AI Plumber Receptionist Demo

FastAPI demo for a small plumbing company receptionist. Twilio answers an incoming call, streams call audio over WebSocket, bridges the audio to OpenAI Realtime, collects plumbing lead details, and sends the current SMS notification to the plumber.

This repository is currently in Phase 0: repo hygiene and stabilization baseline. The current phone flow, prompt behavior, OpenAI Realtime bridge, Twilio Media Streams route, and SMS behavior are intentionally left unchanged.

## Requirements

- Python 3.12
- Twilio phone number with Voice webhook access
- OpenAI API key with Realtime API access
- Public HTTPS host for Twilio callbacks, such as Railway or a tunnel for local testing

Python is pinned with `.python-version` because the demo currently uses Python's `audioop` module for audio transcoding. `audioop` is removed in Python 3.13, so this baseline should run on Python 3.12.

Dependencies are pinned in `requirements.txt`. The project currently uses `websockets.connect(..., extra_headers=...)`, so `websockets==12.0` is pinned because that version supports `extra_headers`.

## Environment Variables

Create a local `.env` from `.env.example` and fill in real values. Do not commit `.env`.

Required values:

- `OPENAI_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `PLUMBER_PHONE_NUMBER`

Documented deployment values:

- `HOST` or `PUBLIC_HOST`: public hostname Twilio can reach, without `https://`
- `OAI_URL` or `OPENAI_REALTIME_URL`: OpenAI Realtime WebSocket URL

Note: the current `main.py` uses the hard-coded `HOST` and `OAI_URL` constants. The env names above are documented for deployment hygiene and a future cleanup, without changing Phase 0 runtime behavior.

## Local Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with real Twilio and OpenAI values.

Run the app:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Twilio Webhook Setup

In the Twilio Console, configure the phone number's Voice webhook:

- Method: `POST`
- URL: `https://<public-host>/voice`

The `/voice` endpoint returns TwiML that connects the call to:

```text
wss://<public-host>/media-stream
```

Twilio must be able to reach both the HTTPS webhook and the secure WebSocket endpoint from the public internet.

## Railway Deployment Notes

- Use Python 3.12.
- Set environment variables in Railway instead of committing secrets.
- Railway provides `$PORT`; both `Procfile` and `railway.json` start the app with `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Configure the Twilio Voice webhook to the deployed `/voice` URL.
- Confirm the deployed `/health` endpoint before making a phone call.

## Phone Test

1. Deploy the app or expose it through a public HTTPS host.
2. Confirm `https://<public-host>/health` returns `{"status":"ok"}`.
3. Configure Twilio Voice webhook to `POST https://<public-host>/voice`.
4. Call the Twilio number.
5. Confirm the call connects, the AI receptionist speaks, and the plumber receives the existing SMS lead notification after the intake is complete.

## Phase 0 Scope

Phase 0 is limited to making the current demo easier to install, reproduce, deploy, and verify. Business logic, prompt behavior, Twilio/OpenAI call flow, SMS behavior, storage, and multi-tenant architecture are not changed in this phase.
