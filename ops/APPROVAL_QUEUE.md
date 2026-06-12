# Approval Queue

## A-003 — OPEN — Consolidation plan (ops/CONSOLIDATION_PLAN.md)
Approve/modify the A→B→C plan: tag archives → push ops-only main → `consolidation` branch from phase-1a tip + merge main into it → byte-identity gate → owner repoints Railway Source to main at a chosen low-call window → verification call → phase-1a frozen as rollback. Decisions needed from owner: (1) approve plan as written or with changes; (2) pick the cutover window; (3) confirm owner executes the Railway Source repoint personally.

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
