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
- Twilio can reach `POST /voice`.
- Twilio can establish `wss://<public-host>/media-stream`.
- `/health` returns `{"status":"ok"}`.

If `DATABASE_URL` is omitted, the app uses local SQLite at `./local_dev.db`. This is acceptable for local/dev checks only and should not be treated as reliable production storage on Railway.
