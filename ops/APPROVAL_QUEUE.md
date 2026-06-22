# Approval Queue

## OPEN
- A-008 — Deploy the COMPLETE shoreline software: merge `change/shoreline-vertical` → main @ 45a9dae (verticals/shoreline.json + delivery spec, tenant→vertical selection, lead delivery module, and per-vertical handler routing). Built + verified: plumber byte-identical (golden) + plumbing regression 51 + leakage + selection + delivery, **full suite 158/158** (D-020/D-021/D-022/D-023). Effect: redeploys the vertical-aware engine + handler, but NO tenant uses shoreline yet → live behavior unchanged until a shorelinecost tenant + number + SHORELINE_LEAD_WEBHOOK_URL exist. Requires a plumber-line regression test call after deploy. **Recommend deploying now** to land the big verified change (then shoreline go-live is just provisioning + env + scripts, no more code deploy) — or hold on the branch until provisioning. Your call.
- A-006 — Approve shoreline scripts + recording decision (legal-adjacent; critical-path step 4). Draft: `config/tenants/shoreline_scripts_DRAFT.md`. Need from owner: (1) greeting Option A or B, (2) identity lines, (3) consent wording (verbatim contract §1.2), (4) recording yes/no. Recording stays OFF until explicit yes + disclosure line live. Recommend counsel review of consent + recording wording before go-live. After this, the only remaining step-4 item is buying the shoreline Twilio number (owner errand).

## CLOSED
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
- Only-a-real-call-can-verify queue (D-007 ruling 2): **DUE NOW — plumber-line regression test call after the core/vertical split deploy (A-007/D-019): owner places one call to the plumber line; operator reviews transcript to confirm byte-identical live behavior.** (Post-cutover verification call already done, A-003 B3.)
