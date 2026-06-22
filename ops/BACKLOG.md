# Backlog

## NOW — 🎯 SHORELINE CRITICAL PATH (the priority; STATE.md has the 6-step countdown)
Consolidation + cutover DONE (D-010/D-011); Phase C #1 /version + #2 metrics DEPLOYED (D-013/D-015). Priority now = make the shoreline vertical usable so it can take a real homeowner call. Owner directive 2026-06-12: minimal-scope refactor; defer all nice-to-haves.
- [x] Step 1: core/vertical split ✅ DEPLOYED 2026-06-12 (main@ac1f051; /version verified; byte-identical; D-019). Pending one plumber-line regression test call (owner).
- [x] Step 2: shoreline.json + tenant→vertical selection ✅ DEPLOYED (main@4a4ac1d; D-021/D-024). Final wording pending A-006.
- [x] Step 3: lead delivery + handler routing ✅ DEPLOYED (main@4a4ac1d; plumbing unchanged; 158/158; D-023/D-024). Webhook URL owner-set (§5.4)
- [ ] Step 4 (OWNER, gate): A-006 wording + SHORELINE_LEAD_WEBHOOK_URL + buy/route Twilio number
- [ ] Step 5: create shorelinecost tenant + route number (operator, once number exists); shoreline test call
- [ ] Step 6: go live → first real homeowner call
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
- [x] Lead delivery DONE @ 45a9dae — §3 payload + §4 webhook + consent gate + per-vertical handler routing (158/158; D-023). Owner sets SHORELINE_LEAD_WEBHOOK_URL (§5.4); email/sheet fallback TBD.
- [ ] Shoreline lead webhook FUNCTION (Netlify on the Shoreline site, mirroring Septic form-lead) — OPERATOR task per owner; **BLOCKED on connecting the Shoreline site/repo + Septic precedent** (D-026). Then URL → SHORELINE_LEAD_WEBHOOK_URL.
- [ ] Call-recording capture (Twilio) + recording_url + recording-disclosure greeting line — owner approved recording (A-006 ②); NOT built; ship disclosure + capture together; spend/storage consideration (D-026).

## P3/P4 — not started
- Demo tenant, onboarding template, pricing proposal, per-client metrics reports, missed-call alerting

## Hygiene
- [ ] Owner revoking legacy PAT (in progress, D-007 ruling 5); claude-fleet token only
- [ ] Python 3.12/audioop pin — hardening list (revisit before any runtime upgrade)
- [ ] Repo rename — deferred indefinitely (owner ruling 4)
- [ ] Sandbox FS quirk: stale .git/*.lock files need manual rename; mount blocks deletion — workaround documented in STATE
