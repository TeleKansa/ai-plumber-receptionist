# Approval Queue

## A-003 — APPROVED 2026-06-11 (D-009); Phase A COMPLETE (D-010) — two items pending for Phase B
1. **A5 merge approval:** consolidation @ 85d5294 → main. Gate evidence in D-010 (0-byte code diff vs production tip; 137/137 tests; boot ok). Inert until the Railway repoint.
2. **Cutover day (weekday, after 21:00 CDT, this week):** needs call-pattern data. Operator has no Railway log access. Either (a) owner reads Railway → service → Observability/Logs, filters "Incoming call" lines for the past 7 days and reports the evening pattern, or (b) owner provides a read-scoped Railway project token and operator does it. Then we fix the day; owner clicks the Source change live with operator directing; operator owns deploy-green + verification call + transcript.

## CLOSED / WITHDRAWN
- A-001 — CLOSED 2026-06-11: owner confirmed via dashboard — production deploys `phase-1a-stability-guardrails` (auto-deploy ON), Postgres Online, active deploy May 21; main push triggered nothing.
- A-002 — WITHDRAWN 2026-06-11: merge of p1/core-vertical-split is moot — it refactored the non-production single-file app (D-008). Branch will be archive-tagged in A-003 Phase A1.

## Deferred — batched into ONE Twilio errand (owner, when ready)
- Shoreline tracking number purchase (~$5–15/mo + usage)
- Dedicated test number for the test path (~$1–2/mo) — approved in principle (D-007 ruling 2)
- Shoreline consent script + greeting/identity script approval (queued for owner review before shoreline go-live)

## Standing flags
- LIVE WIRE: no push to phase-1a-stability-guardrails, ever, without protocol + per-change owner approval. After cutover, the same rule applies to main.
- Recording: OFF everywhere; never enable without approval + disclosure line (FL two-party consent relevant).
- Spend alert threshold: flag if projected monthly OpenAI usage > $50.
- Only-a-real-call-can-verify queue (D-007 ruling 2): post-cutover verification call (A-003 B3); eventual plumber-line regression call after the §1.1 refactor redo lands.
