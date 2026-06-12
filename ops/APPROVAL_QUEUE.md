# Approval Queue

## A-001 — OPEN — Confirm Railway deployed a2f0585 (owner, ~30s)
Railway dashboard → project → Deployments → newest entry should show commit `a2f0585` ("chore: install operator charter and ops scaffold"), status Success. Reply "deploy confirmed" (or paste a Railway project token for CLI verification in future sessions). Risk if unconfirmed: none to live calls (docs-only commit), but the auto-deploy pipeline remains unproven.

## A-002 — OPEN (request follows P1 acceptance) — Merge p1/core-vertical-split to main
Merging deploys the refactored engine to the live plumber line. Will be requested with full evidence: test output, golden diffs, branch commit. Do not merge before reviewing.

## Deferred by owner (2026-06-11, D-002) — do not action yet
- Shoreline Twilio number purchase (~$5–15/mo + usage)
- Shoreline consent script + greeting/identity script approval
- Optional test Twilio number (~$1–2/mo) for live regression calls

## Standing flags
- Recording: OFF everywhere; never enable without approval + disclosure line (FL two-party consent relevant).
- Spend alert threshold: flag if projected monthly OpenAI usage > $50.
