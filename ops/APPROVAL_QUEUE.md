# Approval Queue

## OPEN
- A-007 — Deploy Phase C #3 (core/vertical split): merge `change/core-vertical-split` → main. The foundation that makes shoreline "just a config file." Built + verified @ e5125b6: plumber output byte-identical (golden), zero plumber strings in core (leakage guard), full suite 144/144, no schema change (D-018). Effect of approval: I merge to main + one push → Railway redeploys (restructured prompt engine; plumber prompt/tools/greeting unchanged byte-for-byte). REQUIRES per live-line protocol: one plumber-line regression test call after deploy (you place a call to the plumber line; I review the transcript) — byte-identical output means this should confirm no change. After this, adding shoreline = a new config file only.
- A-006 — Approve shoreline scripts + recording decision (legal-adjacent; critical-path step 4). Draft: `config/tenants/shoreline_scripts_DRAFT.md`. Need from owner: (1) greeting Option A or B, (2) identity lines, (3) consent wording (verbatim contract §1.2), (4) recording yes/no. Recording stays OFF until explicit yes + disclosure line live. Recommend counsel review of consent + recording wording before go-live. After this, the only remaining step-4 item is buying the shoreline Twilio number (owner errand).

## CLOSED
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
- Only-a-real-call-can-verify queue (D-007 ruling 2): post-cutover verification call (A-003 B3); eventual plumber-line regression call after the §1.1 refactor redo lands.
