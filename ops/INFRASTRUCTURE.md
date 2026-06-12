# INFRASTRUCTURE

Updated: 2026-06-11. Names only — never values.

## Call path
Caller dials Twilio number → Twilio Voice webhook POST `https://ai-plumber-receptionist-production.up.railway.app/voice` → TwiML `<Connect><Stream>` → WSS `/media-stream` → bidirectional bridge to OpenAI Realtime (`wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5`). Audio transcoding in-process: Twilio mulaw 8kHz ↔ pcm16 24kHz (audioop). On qualified lead: `submit_service_request` function call → SMS via Twilio REST → call hung up ~5s after closing line.

## Services
- Railway: one service, project name unknown from repo (owner has dashboard). Builder NIXPACKS, start `uvicorn main:app --host 0.0.0.0 --port $PORT`, restart ON_FAILURE. Auto-deploys from GitHub main (assumed — unproven until A-001).
- GitHub: TeleKansa/ai-plumber-receptionist (private), branch main.
- Twilio: one voice number (the live plumber line); number value not in repo.
- Database: NONE exists (see DECISIONS D-003).

## Environment variables (Railway service settings; names only)
- OPENAI_API_KEY
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_PHONE_NUMBER (the AI line / SMS sender)
- PLUMBER_PHONE_NUMBER (lead SMS recipient)

## Hardcoded values to know about
- Public host is hardcoded in `main.py` (`HOST = ai-plumber-receptionist-production.up.railway.app`) — TwiML stream URL derives from it. Moving/renaming the Railway service breaks calls until updated.
- OpenAI model pinned in code: gpt-realtime-1.5.

## Known gaps
- No deploy-version visibility from outside (no /version endpoint yet — backlog).
- No missed-call alerting; failures visible only in Railway logs.
