# Approval Queue

## A-004 — OPEN — One production push: ops docs + /version endpoint
First deploy under the new protocol. Contents: all local ops commits (docs only) + /version endpoint (reads Railway-injected RAILWAY_GIT_COMMIT_SHA; no new deps). Evidence: full test suite + new endpoint test green in sandbox (see D-012 when logged). Effect of approval: I push main once → Railway redeploys production (code change = /version only). Decide also (optional, recommended): set Railway → Service → Settings → Build → Watch paths to exclude `ops/**` and `*.md` so future docs-only pushes don't trigger deploys.

## CLOSED
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
- LIVE WIRE: no push to phase-1a-stability-guardrails, ever, without protocol + per-change owner approval. After cutover, the same rule applies to main.
- Recording: OFF everywhere; never enable without approval + disclosure line (FL two-party consent relevant).
- Spend alert threshold: flag if projected monthly OpenAI usage > $50.
- Only-a-real-call-can-verify queue (D-007 ruling 2): post-cutover verification call (A-003 B3); eventual plumber-line regression call after the §1.1 refactor redo lands.
