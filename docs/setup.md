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
- Twilio can reach `POST /voice`.
- Twilio can establish `wss://<public-host>/media-stream`.
- `/health` returns `{"status":"ok"}`.
