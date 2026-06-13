# INFRASTRUCTURE

Updated: 2026-06-12 (session 5 — post-cutover + Phase C #1 /version live, D-011/D-013). Names only — never values.

## ⚠️ Production source of truth
Railway production service deploys GitHub branch `main` (since cutover 2026-06-12, D-011), auto-deploy ON — any push that touches CODE goes live on the phone line (LIVE WIRE RULE); docs-only pushes (`ops/**`, `*.md`) are deploy-inert via Watch Paths (D-013). Current tip 74e5dbe; deployed SHA confirmable via `GET /version`. `phase-1a-stability-guardrails` @ 77b5537 is FROZEN as rollback until 2026-06-26 — do not push (rollback lever = repoint Railway Source back to it). The live build is the MULTI-TENANT app: SQLAlchemy + psycopg → Railway Postgres (service Online, with volume); tenant routing by called number; tenant statuses incl. testing + allowed_test_callers; admin UI; workflow/prompt_builder.py builds prompts from tenant profiles stored in Postgres.

## Call path (production build)
Caller dials a Twilio number → Twilio Voice webhook POST `https://ai-plumber-receptionist-production.up.railway.app/voice` → tenant resolved from the To number (Postgres) → telephony gate (live/testing/paused) → TwiML `<Connect><Stream>` → WSS `/media-stream` → bidirectional bridge to OpenAI Realtime. Audio transcoding in-process: Twilio mulaw 8kHz ↔ pcm16 24kHz (audioop). Qualified lead → function call → notification per tenant policy (SMS) → call lifecycle + events recorded in Postgres.

## Services
- Railway: app service (builder NIXPACKS, start `uvicorn main:app --host 0.0.0.0 --port $PORT`, restart ON_FAILURE, source branch **main**, Watch Paths exclude `ops/**` + `*.md` so docs don't trigger deploys) + Postgres service (Online, volume-backed).
- GitHub: TeleKansa/ai-plumber-receptionist (private). Branches: **main (PRODUCTION** — multi-tenant code + charter/ops + /version), phase-1a-stability-guardrails (FROZEN rollback @ 77b5537 until 2026-06-26), change/version-endpoint + consolidation (merged), chore/phase0-demo-stabilization-baseline. Tags: archive/single-file-app @ a2f0585 (old single-file app), archive/p1-split-of-single-file-app @ c1aa2fa.
- Twilio: one voice number live (plumber line); number value not in repo.
- Database: Railway Postgres — tenants, tenant_ai_profiles (prompt profiles), calls, leads, notifications, call_events, call_reviews, call_feedback (per storage/models.py on the production branch).

## Environment variables (Railway service settings; names only — full audit vs config/settings.py done in D-010)
- OPENAI_API_KEY
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_PHONE_NUMBER
- PLUMBER_PHONE_NUMBER
- DATABASE_URL (Postgres)
- ADMIN_PASSWORD (admin pages)
- HOST, PUBLIC_HOST
- OAI_URL, OPENAI_REALTIME_URL, OPENAI_REALTIME_MODEL
- DEFAULT_TENANT_NAME, DEFAULT_TENANT_SLUG, DEFAULT_TENANT_GREETING
- (Railway-injected, read by /version: RAILWAY_GIT_COMMIT_SHA, RAILWAY_GIT_BRANCH)

## Known gaps
- Deploy-version visibility RESOLVED 2026-06-12: `GET /version` returns RAILWAY_GIT_COMMIT_SHA + branch (D-013); deployed SHA now externally confirmable (currently 74e5dbe).
- No missed-call alerting; failures visible only in Railway logs (P4 backlog item).
- No Railway token for the operator → can't read deploy status/logs directly; owner pastes/confirms from the dashboard.
