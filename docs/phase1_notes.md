# Phase 1 Notes

Phase 1 is about adding stability guardrails around the existing AI plumber receptionist demo while keeping the current call behavior reviewable and easy to roll back.

## Completed Phase 1 / Milestone 1 Scope

- Add project-level agent rules in `AGENTS.md`.
- Document Phase 1 boundaries and review questions in this file.
- Add lightweight settings loading with the existing hard-coded `HOST` and `OAI_URL` values preserved as fallbacks.
- Add lead validation and wire it into `submit_service_request`.
- Fix the narrow SMS result bug where failed SMS delivery was still reported to OpenAI as `success=true`.
- Add tests that do not call Twilio, OpenAI, or require real `.env` values.
- Add durable single-tenant call, lead, notification, and call event persistence.
- Add minimal internal admin pages protected by `ADMIN_PASSWORD`.

## Current Non-Goals

- No prompt or persona changes.
- No multi-tenant architecture.
- No manual deployment or merge; Railway may auto-deploy the configured dev branch after push.
- No live phone test.
- No large `main.py` refactor.
- No Twilio/OpenAI realtime event loop rewrite.
- No barge-in overhaul.
- No OpenAI `session.update` schema redesign.
- No SMS template or recipient changes.
- No retry worker or SaaS admin UI.

## Future Candidates

- Add safer config validation for missing production environment variables.
- Add a retry worker for failed notifications.
- Add stronger caller transcript provenance if OpenAI input transcription is unavailable or incomplete on real calls.
- Add structured logging around lead completion and notification results.
- Add a small manual test checklist for Twilio Media Streams and OpenAI Realtime.
- Investigate a Python 3.13-safe replacement for `audioop`.

## Manual Review Questions

- Should `HOST` and `OAI_URL` be fully environment-driven in production, with startup failure if missing?
- Should failed SMS delivery trigger a retry worker or a separate alert channel?
- What is the acceptable behavior when OpenAI calls `submit_service_request` with an incomplete or placeholder address?
- Should backend validation require caller transcript support for every submitted name, including full names, after more real-call transcript samples are available?

## Current Hotfix Note

Name validation now uses caller transcript text when OpenAI provides it. If transcript text exists, the submitted name must be supported by what the caller said. If no transcript is available, single-token names are rejected as too easy to invent, while fuller names remain a best-effort fallback to avoid blocking every call on transcript availability.
