# Backlog

## NOW — consolidation (blocked on A-003 owner decision)
- [ ] Phase A: archive tags, push ops main, consolidation branch, merge docs in, byte-identity gate, run phase-1a test suite in sandbox
- [ ] Phase B: owner repoints Railway → main (cutover window TBD), verification call, transcript logged
- [ ] Phase C: freeze phase-1a; owner archives legacy-snapshot .env then deletes folder

## P0 continuation (after cutover)
- [ ] /version endpoint returning git SHA — first protocol-following change on the new main; scripted-test verified
- [ ] Re-export REAL tenant config from Postgres → /config/tenants/ (charter config-as-code; needs DB access route — likely via admin UI or a read-only script run, TBD with owner)
- [ ] INFRASTRUCTURE.md: add Postgres service details + env var names from config/settings.py

## P1 (after cutover) — redo against production codebase
- [ ] Call-outcome metrics logging (approved, D-007 ruling 3)
- [ ] Contract §1.1 core/vertical split targeting workflow/prompt_builder.py — reuse session-2 methodology: golden recording → scripted media-stream harness → byte-identity regression → leakage tests (branch p1/core-vertical-split @ c1aa2fa kept as reference)
- [ ] Build the real test path: separate Railway test service + test number (number purchase batched — see APPROVAL_QUEUE)

## P2 — shoreline vertical (blocked on owner Twilio errand + scripts)
- [ ] verticals/shoreline config per contract §1.2 (draftable ahead of time)
- [ ] Lead delivery webhook + email/sheet fallback per contract §3–4

## P3/P4 — not started
- Demo tenant, onboarding template, pricing proposal, per-client metrics reports, missed-call alerting

## Hygiene
- [ ] Owner revoking legacy PAT (in progress, D-007 ruling 5); claude-fleet token only
- [ ] Python 3.12/audioop pin — hardening list (revisit before any runtime upgrade)
- [ ] Repo rename — deferred indefinitely (owner ruling 4)
- [ ] Sandbox FS quirk: stale .git/*.lock files need manual rename; mount blocks deletion — workaround documented in STATE
