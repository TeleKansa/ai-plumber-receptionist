# INFRASTRUCTURE

Updated: 2026-06-11 (session 3 — corrected with owner's Railway dashboard facts, D-007/D-008). Names only — never values.

## ⚠️ Production source of truth
Railway production service deploys GitHub branch `phase-1a-stability-guardrails` (NOT main), auto-deploy ON — any push there goes live on the phone line (LIVE WIRE RULE). Active production deployment dated 2026-05-21; branch tip 77b5537 (2026-05-21 21:32 -0500). The live build is the MULTI-TENANT app: SQLAlchemy + psycopg → Railway Postgres (service Online, with volume); tenant routing by called number; tenant statuses incl. testing + allowed_test_callers; admin UI; workflow/prompt_builder.py builds prompts from tenant profiles stored in Postgres.

## Call path (production build)
Caller dials a Twilio number → Twilio Voice webhook POST `https://ai-plumber-receptionist-production.up.railway.app/voice` → tenant resolved from the To number (Postgres) → telephony gate (live/testing/paused) → TwiML `<Connect><Stream>` → WSS `/media-stream` → bidirectional bridge to OpenAI Realtime. Audio transcoding in-process: Twilio mulaw 8kHz ↔ pcm16 24kHz (audioop). Qualified lead → function call → notification per tenant policy (SMS) → call lifecycle + events recorded in Postgres.

## Services
- Railway: app service (builder NIXPACKS, start `uvicorn main:app --host 0.0.0.0 --port $PORT`, restart ON_FAILURE, source branch phase-1a-stability-guardrails) + Postgres service (Online, volume-backed).
- GitHub: TeleKansa/ai-plumber-receptionist (private). Branches: phase-1a-stability-guardrails (PRODUCTION), main (old single-file app + charter/ops docs — Railway ignores it), chore/phase0-demo-stabilization-baseline.
- Twilio: one voice number live (plumber line); number value not in repo.
- Database: Railway Postgres — tenants, tenant_ai_profiles (prompt profiles), calls, leads, notifications, call_events, call_reviews, call_feedback (per storage/models.py on the production branch).

## Environment variables (Railway service settings; names only — production build per its .env.example/settings.py)
- OPENAI_API_KEY
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_PHONE_NUMBER
- PLUMBER_PHONE_NUMBER
- DATABASE_URL (Postgres)
- ADMIN_PASSWORD (admin pages)
- (full audit of names against config/settings.py scheduled for consolidation Phase A4)

## Known gaps
- No deploy-version visibility from outside (no /version endpoint — first post-cutover change, A-003 C4).
- No missed-call alerting; failures visible only in Railway logs.
- Deployed SHA assumed = branch tip 77b5537 (dates match); not byte-confirmable until /version ships.
