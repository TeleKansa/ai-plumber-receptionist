# Backlog

## NOW — 🎯 SHORELINE CRITICAL PATH (the priority; STATE.md has the 6-step countdown)
Consolidation + cutover DONE (D-010/D-011); Phase C #1 /version + #2 metrics DEPLOYED (D-013/D-015). Priority now = make the shoreline vertical usable so it can take a real homeowner call. Owner directive 2026-06-12: minimal-scope refactor; defer all nice-to-haves.
- [x] Step 1: core/vertical split ✅ DEPLOYED 2026-06-12 (main@ac1f051; /version verified; byte-identical; D-019). Pending one plumber-line regression test call (owner).
- [~] Step 2: shoreline.json + tenant→vertical selection BUILT on branch change/shoreline-vertical @ dea2f5b (148/148; plumber byte-identical; D-020/D-021). Remaining: A-006 wording + deploy (A-008, recommend batch with step 3).
- [ ] Step 3: shoreline lead delivery (webhook + email/sheet fallback, §3–4)
- [ ] Step 4 (OWNER, gating): shoreline Twilio number + consent/greeting approvals + webhook host
- [ ] Step 5: provision shoreline tenant + number routing; scripted + live test call, transcript logged
- [ ] Step 6: go live → first real homeowner call

## P0 continuation (after cutover)
- [x] /version endpoint returning git SHA — DEPLOYED & verified 2026-06-12 (main@74e5dbe; full suite 139/139; GET /version returns deployed SHA; D-013)
- [ ] Re-export REAL tenant config from Postgres → /config/tenants/ (charter config-as-code; needs DB access route — likely via admin UI or a read-only script run, TBD with owner)
- [ ] INFRASTRUCTURE.md: add Postgres service details + env var names from config/settings.py

## P1 (after cutover) — redo against production codebase
- [x] **Phase C #2 DEPLOYED 2026-06-12:** GET /admin/metrics.json live (main@7a9c3df; 142/142; reuses pilot_metrics + derived rates; D-015)
- [x] **Phase C #3 = shoreline step 1 DEPLOYED 2026-06-12:** core/vertical split live (main@ac1f051) — industry-agnostic core/ engine + verticals/plumbing.json; plumbing byte-identical (golden) + zero plumber strings in core (leakage guard); D-019.
- [ ] Build the real test path: separate Railway test service + test number (number purchase batched — see APPROVAL_QUEUE)

## P2 — shoreline vertical (blocked on owner Twilio errand + scripts)
- [~] verticals/shoreline.json DRAFTED @ 70a467f (D-020); pending A-006 wording + tenant→vertical selection wiring
- [x] tenant→vertical selection wiring — DONE on branch change/shoreline-vertical @ dea2f5b (D-021); ships with A-008
- [ ] Lead delivery webhook + email/sheet fallback per contract §3–4 (step 3)

## P3/P4 — not started
- Demo tenant, onboarding template, pricing proposal, per-client metrics reports, missed-call alerting

## Hygiene
- [ ] Owner revoking legacy PAT (in progress, D-007 ruling 5); claude-fleet token only
- [ ] Python 3.12/audioop pin — hardening list (revisit before any runtime upgrade)
- [ ] Repo rename — deferred indefinitely (owner ruling 4)
- [ ] Sandbox FS quirk: stale .git/*.lock files need manual rename; mount blocks deletion — workaround documented in STATE
