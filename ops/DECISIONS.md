# Decisions (append-only)

## D-001 — 2026-06-11 — Session-1 audit findings
Repo audit: single-tenant app, no Postgres/tenant DB anywhere in code or requirements; no recording; no testing tenant or test-caller path exists. Live service healthy (`/health` ok, fetched 2026-06-11). Commit a2f0585 (docs-only) confirmed in git, local main == origin/main; Railway deploy NOT verifiable from session (no Railway CLI/token; GitHub unreachable from sandbox). Owner action queued (A-001). Live-call risk of a2f0585: none (zero code changed).

## D-002 — 2026-06-11 — Owner direction
Owner (Vincent), in chat: Twilio number and go-live scripts deferred. Proceed P0 then P1. Plumber behavior must be preserved exactly through the split. Return for number purchase + script approvals only after split passes acceptance.

## D-003 — 2026-06-11 — Operating assumption: repo is source of truth
Charter references "tenant config in Postgres" and a "testing tenant"; neither exists in this codebase. Until owner says a DB/service exists elsewhere, repo + Railway env vars are treated as the complete system. The charter's weekly DB-drift check is a no-op until a DB exists. Config-as-code rule applied to code-embedded config (exported to /config/tenants/).

## D-004 — 2026-06-11 — P1 acceptance method
No live test path exists and test-number spend is deferred (owner, D-002). Charter Verification Protocol explicitly permits "scripted media-stream test" as deploy evidence. P1 acceptance therefore = (a) golden equivalence: byte-identical instructions, session.update, greeting, TwiML, SMS body, audio transcoding frames vs pre-refactor recordings; (b) scripted end-to-end media-stream call replay; (c) zero plumber strings in core/. Refactor stays on branch `p1/core-vertical-split`; merge to main (= production deploy) only after owner reviews evidence. Post-merge live call recommended as final confirmation.
