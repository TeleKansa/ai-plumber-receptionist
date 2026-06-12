# STATE

Updated: 2026-06-12 (session 4 — post-cutover)

## ⚠️ LIVE WIRE = main
Railway production deploys GitHub **main** (since cutover 2026-06-12 ~04:45 UTC). Any push to main = production deploy. No push without scripted verification + per-change owner approval. Ops/docs commits stay LOCAL until a push is approved (or owner sets Railway watch-paths to exclude ops/** and *.md — proposed, undecided).

## Phase
CONSOLIDATION COMPLETE. Phase B closed 2026-06-12: cutover verified end-to-end (answered → qualified → lead SMS delivered in seconds; transcript archived in ops/transcripts/). Phase C in effect:
- phase-1a-stability-guardrails FROZEN as rollback until **2026-06-26**; then archive-tag + delete (owner call).
- Change #1 under new protocol: /version endpoint (in progress, branch change/version-endpoint).
- Change #2: call-metrics read path (DB already records calls/leads/events — scope next session).
- Then: core/vertical refactor of workflow/prompt_builder.py; shoreline vertical is its first client.

## Production truth
- One branch: main @ ac687be = multi-tenant code (byte-identical to old phase-1a tip 77b5537) + charter/ops.
- Live build facts from verification call: tenant_id=1 (plumber), model gpt-realtime-2 (env/tenant-config driven), reasoning_effort=low, extra intake fields (property_role, additional_notes), leads delivered by SMS, calls/events recorded in Postgres.
- Tags: archive/single-file-app @ a2f0585; archive/p1-split-of-single-file-app @ c1aa2fa.
- legacy-snapshot/: zero unique work; owner to archive .env values then delete folder.

## Access gaps (standing)
No Railway token (can't read logs/deploy status), no ADMIN_PASSWORD, no DATABASE_URL — owner pastes logs or grants read access when needed.

## Working tree note
Mounted folder on main @ ac687be, tracked tree clean. Untracked leftovers (core/, tests/goldens etc. from archived p1 branch, legacy-snapshot/) can't be deleted from sandbox; ignore.
