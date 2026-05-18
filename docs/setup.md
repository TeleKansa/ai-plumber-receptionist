# Setup Notes

This project runs on Python 3.12. Keep Python on 3.12 while the app uses `audioop` for Twilio/OpenAI audio transcoding.

## Dependency Baseline

Install pinned dependencies:

```bash
pip install -r requirements.txt
```

`websockets==12.0` is pinned so the existing `websockets.connect(..., extra_headers=...)` call remains compatible.

## Required Runtime Configuration

Use `.env.example` as the template for local environment variables. Keep real credentials in `.env` or the deployment provider's environment variable settings.

For a deployed environment, confirm:

- The app runs with Python 3.12.
- `DATABASE_URL` points to Postgres for durable lead storage.
- `ADMIN_PASSWORD` is set before using `/admin`.
- The default tenant uses `TWILIO_PHONE_NUMBER` and `PLUMBER_PHONE_NUMBER`.
- Twilio can reach `POST /voice`.
- Twilio can establish `wss://<public-host>/media-stream`.
- `/health` returns `{"status":"ok"}`.

If `DATABASE_URL` is omitted, the app uses local SQLite at `./local_dev.db`. This is acceptable for local/dev checks only and should not be treated as reliable production storage on Railway.

## Multi-Tenant Startup Migration

Startup creates the tenant tables if needed, adds nullable `tenant_id` columns to existing Milestone 1 tables if missing, and backfills existing rows to the default tenant. It does not drop or wipe existing data.

The default tenant is created from:

- `DEFAULT_TENANT_NAME` or `Default Plumbing`
- `DEFAULT_TENANT_SLUG` or `default`
- `DEFAULT_TENANT_GREETING` or `Plumbing office, what's going on?`
- `TWILIO_PHONE_NUMBER`
- `PLUMBER_PHONE_NUMBER`
