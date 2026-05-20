# AI Plumber Receptionist Demo

FastAPI demo for a small plumbing company receptionist. Twilio answers an incoming call, streams call audio over WebSocket, bridges the audio to OpenAI Realtime, collects plumbing lead details, and sends the current SMS notification to the plumber.

This repository is currently focused on Milestone 2: multi-tenant lite on top of the reliable receptionist baseline. The current phone flow, prompt behavior, OpenAI Realtime bridge, Twilio Media Streams route, and SMS template are intentionally kept stable while calls, leads, notifications, and call events are scoped by tenant.

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
- `DATABASE_URL` for Railway/Postgres durable lead storage
- `ADMIN_PASSWORD` to enable internal admin pages

Documented deployment values:

- `HOST` or `PUBLIC_HOST`: public hostname Twilio can reach, without `https://`
- `OAI_URL` or `OPENAI_REALTIME_URL`: OpenAI Realtime WebSocket URL
- `OPENAI_REALTIME_MODEL`: optional model selector, defaults to `gpt-realtime-1.5`; set to `gpt-realtime-2` only when testing Realtime 2
- `DEFAULT_TENANT_NAME`, `DEFAULT_TENANT_SLUG`, `DEFAULT_TENANT_GREETING`: optional default tenant bootstrap values

`HOST`/`PUBLIC_HOST` and `OAI_URL`/`OPENAI_REALTIME_URL` are read from the environment with the previous Railway host and OpenAI Realtime URL preserved as fallbacks. The app builds the final Realtime URL from the selected model so `OAI_URL` and `OPENAI_REALTIME_MODEL` do not need conflicting `model=` values.

If `DATABASE_URL` is not set, the app falls back to local SQLite at `./local_dev.db`. That fallback is only for local/dev use. Do not rely on Railway's filesystem for long-term production lead storage.

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

## Internal Admin

Set `ADMIN_PASSWORD` to enable the internal admin pages:

```text
https://<public-host>/admin
https://<public-host>/admin/leads
https://<public-host>/admin/tenants
https://<public-host>/admin/calls/<call-sid>
```

Use Basic Auth with username `admin` and the configured `ADMIN_PASSWORD`. If `ADMIN_PASSWORD` is missing, admin routes return disabled/unauthorized responses.

## Multi-Tenant Lite

At startup the app safely creates or updates a default tenant from environment variables:

- Tenant name: `DEFAULT_TENANT_NAME` or `Default Plumbing`
- Tenant slug: `DEFAULT_TENANT_SLUG` or `default`
- Tenant phone number: `TWILIO_PHONE_NUMBER`
- Tenant notification number: `PLUMBER_PHONE_NUMBER`
- Tenant greeting: `DEFAULT_TENANT_GREETING` or the current greeting

Existing Milestone 1 rows are not deleted. Startup migration adds nullable `tenant_id` columns where missing and backfills existing calls, leads, notifications, and call events to the default tenant.

To add another plumbing company, open `/admin/tenants`, create the tenant, add its Twilio number, and set its notification SMS number. Incoming calls are routed by Twilio's `To` number.

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
- Add a Railway Postgres database and set `DATABASE_URL` for durable lead storage.
- Set `ADMIN_PASSWORD` before using `/admin`.
- Railway provides `$PORT`; both `Procfile` and `railway.json` start the app with `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Configure the Twilio Voice webhook to the deployed `/voice` URL.
- Confirm the deployed `/health` endpoint before making a phone call.

## Phone Test

1. Deploy the app or expose it through a public HTTPS host.
2. Confirm `https://<public-host>/health` returns `{"status":"ok"}`.
3. Configure Twilio Voice webhook to `POST https://<public-host>/voice`.
4. Call the Twilio number.
5. Confirm the call connects, the AI receptionist speaks, and the plumber receives the existing SMS lead notification after the intake is complete.

## Milestone 1 Scope

Milestone 1 added durability around the current demo: call records, validated leads, notification status, call events, and a minimal internal admin.

## Milestone 2 Scope

Milestone 2 adds multi-tenant lite: tenants, tenant phone numbers, tenant settings, tenant-scoped calls/leads/notifications/events, tenant-specific SMS recipients, and simple admin tenant management. It does not add billing, customer login, full CRM, or prompt versioning.
