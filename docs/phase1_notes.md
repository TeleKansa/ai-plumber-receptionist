# Phase 1 Notes

Phase 1 is about adding stability guardrails around the existing AI plumber receptionist demo while keeping the current call behavior reviewable and easy to roll back.

## Phase 1A Scope

- Add project-level agent rules in `AGENTS.md`.
- Document Phase 1 boundaries and review questions in this file.
- Add lightweight settings loading with the existing hard-coded `HOST` and `OAI_URL` values preserved as fallbacks.
- Add a standalone lead validation module and unit tests without wiring validation into the live call flow.
- Fix the narrow SMS result bug where failed SMS delivery was still reported to OpenAI as `success=true`.
- Add tests that do not call Twilio, OpenAI, or require real `.env` values.

## Phase 1A Non-Goals

- No prompt or persona changes.
- No database.
- No multi-tenant architecture.
- No deployment.
- No live phone test.
- No large `main.py` refactor.
- No Twilio/OpenAI realtime event loop rewrite.
- No barge-in overhaul.
- No OpenAI `session.update` schema redesign.
- No SMS template or recipient changes.

## Phase 1B Candidates

- Decide whether lead validation should block `submit_service_request` before SMS is sent.
- Add safer config validation for missing production environment variables.
- Add smoke tests around `/health` and `/voice` TwiML generation.
- Add structured logging around lead completion and notification results.
- Add a small manual test checklist for Twilio Media Streams and OpenAI Realtime.
- Investigate a Python 3.13-safe replacement for `audioop`.

## Manual Review Questions

- Should `HOST` and `OAI_URL` be fully environment-driven in production, with startup failure if missing?
- Should failed SMS delivery prevent the closing response, trigger a retry, or notify through a fallback channel?
- Should validation be enforced before sending SMS, or only logged first?
- What is the acceptable behavior when OpenAI calls `submit_service_request` with an incomplete or placeholder address?
- Should Phase 1B add minimal route-level tests for `/voice` without touching the live call flow?
