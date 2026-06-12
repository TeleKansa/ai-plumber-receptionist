# Project Agent Rules

This project is an AI receptionist demo for small plumbing companies. It uses FastAPI, Twilio Media Streams, and OpenAI Realtime to answer calls, collect plumbing lead details, and notify the plumber by SMS.

## Operating Principles

- Reliability first. A lead should not be lost because of a cleanup, refactor, or unreviewed behavior change.
- Preserve the Twilio/OpenAI realtime call flow unless a task explicitly asks to change it.
- Do not let prompt or persona edits break the required intake workflow.
- Keep every change small, reviewable, testable, and easy to roll back.
- Do not commit secrets, `.env`, API keys, phone auth tokens, or private customer data.
- Do not modify `main` or `master` directly.
- Do not deploy from agent work unless the user explicitly asks for deployment.
- Prefer documenting uncertain architecture choices in `docs/phase1_notes.md` over making large changes.
- Every completed change should include a concise summary and verification steps.

## Guardrails

- Do not introduce a database or multi-tenant architecture without explicit approval.
- Do not split or rewrite `main.py` during stabilization work.
- Do not rewrite the Twilio WebSocket event loop or OpenAI Realtime event loop casually.
- Do not change SMS recipients, message templates, or notification behavior unless explicitly scoped.
- Do not run live phone tests unless explicitly requested.
