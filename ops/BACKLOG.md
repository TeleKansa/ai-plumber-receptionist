# Backlog

## NOW — 🎯 SHORELINE CRITICAL PATH (the priority; STATE.md has the 6-step countdown)
Consolidation + cutover DONE (D-010/D-011); Phase C #1 /version + #2 metrics DEPLOYED (D-013/D-015). Priority now = make the shoreline vertical usable so it can take a real homeowner call. Owner directive 2026-06-12: minimal-scope refactor; defer all nice-to-haves.
- [~] Step 1: prompt_builder core/vertical refactor (plumbing preserved, regression-clean) — golden baseline locked on branch change/core-vertical-split @ 2040765; next = design + the split (D-017)
- [ ] Step 2: verticals/shoreline.json per contract §1.2
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
- [ ] **PRIORITY — Phase C #3 = shoreline critical-path step 1:** core/vertical split of workflow/prompt_builder.py, MINIMAL scope to enable verticals/shoreline.json; plumbing preserved exactly (golden recording → scripted media-stream harness → byte-identity regression → leakage tests; methodology from c1aa2fa). Defer nice-to-haves per owner directive.
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
