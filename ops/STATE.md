# STATE

Updated: 2026-06-11 (session 3, end)

## ⚠️ LIVE WIRE
Production = Railway service deploying GitHub branch `phase-1a-stability-guardrails` (auto-deploy ON). Any push to that branch hits the live phone line. Never push it without full safety protocol + explicit per-change owner approval (D-007). Postgres service Online (tenant config lives there — charter's config-as-code applies post-consolidation).

## Phase
Reconciliation DONE (D-008, read-only). Awaiting owner decision on A-003 (consolidation plan, ops/CONSOLIDATION_PLAN.md). After approval: Phase A prep → cutover → then P0 continuation (/version endpoint, real config snapshot from Postgres) and the §1.1 refactor redo against workflow/prompt_builder.py.

## Production truth (corrected this session)
- Live build: multi-tenant app @ 77b5537 (2026-05-21) — SQLAlchemy/Postgres, tenant routing by called number, telephony gate with testing-tenant + allowed_test_callers path (the charter's test path EXISTS in production), workflow/prompt_builder.py, admin UI.
- GitHub main = old single-file app + charter/ops docs. Railway ignores main. Local main is 2 ops commits ahead of origin (push pending A-003 Phase A2).
- Session-2 P1 split (c1aa2fa) targeted the wrong baseline; A-002 withdrawn; branch to be archive-tagged (A-003 Phase A1). Harness methodology carries forward.
- legacy-snapshot/: zero unique work (D-008); gitignored; owner to archive its .env then delete the folder.

## Owner rulings in force (D-007)
Test path approved in principle (scripted sim now; test number purchased in one batch with shoreline Twilio funding). Metrics logging approved as P1 item (manual Railway/Twilio reads until then). Repo rename deferred. Legacy PAT being revoked — claude-fleet token only.

## Working tree note
Repo checked out on main. Untracked leftovers from session 2 (core/, tests/, verticals/ — committed on the archived p1 branch) can't be deleted from the sandbox (mount blocks file deletion); harmless, not gitignored, do not commit them on main.
