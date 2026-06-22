# Approval Queue

## OPEN
- A-006 — Shoreline scripts: **owner decisions made 2026-06-12** (greeting/role → "project assistant"; identity lines approved; recording = yes-with-disclosure; consent current wording OK, phone leads = direct matching, no resale; D-026). Greeting change applied on branch change/shoreline-scripts-a006. **Still OPEN: pending owner's lawyer review of consent + phone wording before go-live.** Coupled gates: recording-capture is unbuilt (disclosure ships with capture); cross-vertical consent rule with Septic recorded (resale change = both verticals + re-approval).

## CLOSED
- A-008 — DEPLOYED 2026-06-12: complete shoreline software merged → main (4a4ac1d); /version verified; plumber byte-identical (158/158); shoreline dormant until provisioning (D-024).
- A-007 — DEPLOYED 2026-06-12: core/vertical split merged → main (ac1f051); /version verified; plumber output byte-identical (golden) + zero plumber strings in core. Follow-up: one plumber-line regression test call (owner) — see only-a-real-call queue (D-019).
- A-005 — CLOSED 2026-06-12: metrics.json merged → main (7a9c3df); deployed; /version externally verified = merge SHA; route admin-auth gated (D-015).
- A-004 — CLOSED 2026-06-12: ops docs + /version pushed (main@74e5dbe); Railway Watch Paths set to exclude ops/** and *.md; /version externally verified returning the deployed SHA (D-013).
- A-003 — COMPLETE 2026-06-12: consolidation executed (D-010), cutover verified end-to-end (D-011). phase-1a frozen until 2026-06-26.
- A-001 — CLOSED (D-007). A-002 — WITHDRAWN (D-008).

## CLOSED / WITHDRAWN
- A-001 — CLOSED 2026-06-11: owner confirmed via dashboard — production deploys `phase-1a-stability-guardrails` (auto-deploy ON), Postgres Online, active deploy May 21; main push triggered nothing.
- A-002 — WITHDRAWN 2026-06-11: merge of p1/core-vertical-split is moot — it refactored the non-production single-file app (D-008). Branch will be archive-tagged in A-003 Phase A1.

## Deferred — batched into ONE Twilio errand (owner, when ready)
- Shoreline tracking number purchase (~$5–15/mo + usage)
- Dedicated test number for the test path (~$1–2/mo) — approved in principle (D-007 ruling 2)
- Shoreline consent script + greeting/identity script approval (queued for owner review before shoreline go-live)

## Standing flags
- LIVE WIRE: no push to phase-1a-stability-guardrails, ever, without protocol + per-change owner approval. The same rule applies to main (production). Docs-only pushes (ops/**, *.md) are now deploy-inert via Railway Watch Paths (D-013); any push touching code still requires full protocol + per-change approval.
- Recording: OFF everywhere; never enable without approval + disclosure line (FL two-party consent relevant).
- Spend alert threshold: flag if projected monthly OpenAI usage > $50.
- Only-a-real-call-can-verify queue (D-007 ruling 2): **owner placing a plumber-line regression test call now** to confirm the live plumber line after the core/vertical split + shoreline-software deploys (A-007/A-008; D-019/D-024); operator reviews the transcript. Future: a shoreline test call at go-live. (Post-cutover verification call done, A-003 B3.)
