# STATE

Updated: 2026-06-12 (session 5 — Phase C change #1 /version deployed & verified)

## ⚠️ LIVE WIRE = main
Railway production deploys GitHub **main** (since cutover 2026-06-12 ~04:45 UTC). Any push to main that touches CODE = production deploy → requires scripted verification + per-change owner approval. **Watch Paths now set (D-013): a push touching only `ops/**` and `*.md` is deploy-inert** — ops/docs commits can be pushed without disturbing the live line. Code pushes still go through full protocol.

## Phase
CONSOLIDATION COMPLETE. Phase B closed 2026-06-12: cutover verified end-to-end (answered → qualified → lead SMS delivered in seconds; transcript archived in ops/transcripts/). Phase C in effect:
- phase-1a-stability-guardrails FROZEN as rollback until **2026-06-26**; then archive-tag + delete (owner call).
- Change #1 under new protocol: /version endpoint — **DEPLOYED & verified 2026-06-12** (main@74e5dbe; GET /version returns the deployed SHA; D-013). DONE.
- Change #2: call-metrics read path — **BUILT & verified on branch change/metrics-json @ 4e2ccba (suite 142/142); awaiting A-005 to deploy** (D-014). Read-only GET /admin/metrics.json reusing pilot_metrics + charter-derived rates.
- Then: core/vertical refactor of workflow/prompt_builder.py; shoreline vertical is its first client. Twilio number + consent/greeting scripts remain deferred (owner).

## Production truth
- One branch: main @ 74e5dbe = multi-tenant code (byte-identical to old phase-1a tip 77b5537) + charter/ops + /version endpoint (Phase C #1). Deployed SHA now externally confirmable via GET /version.
- Live build facts from verification call: tenant_id=1 (plumber), model gpt-realtime-2 (env/tenant-config driven), reasoning_effort=low, extra intake fields (property_role, additional_notes), leads delivered by SMS, calls/events recorded in Postgres.
- Tags: archive/single-file-app @ a2f0585; archive/p1-split-of-single-file-app @ c1aa2fa.
- legacy-snapshot/: zero unique work; owner to archive .env values then delete folder.

## Access gaps (standing)
No Railway token (can't read logs/deploy status), no ADMIN_PASSWORD, no DATABASE_URL — owner pastes logs or grants read access when needed.

## Working tree note
origin/main @ 74e5dbe is source of truth. Git write-ops run in a /tmp clone (mount blocks unlink → stale .lock files); the mounted folder's local HEAD may lag origin and show ops edits as uncommitted — cosmetic only. Untracked leftovers (core/, tests/goldens etc. from archived p1 branch, legacy-snapshot/) can't be deleted from sandbox; ignore.
