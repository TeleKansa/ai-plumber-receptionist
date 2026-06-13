# STATE

Updated: 2026-06-12 (session 5 — Phase C #1 /version + #2 metrics DEPLOYED; shoreline-first priority pivot)

## ⚠️ LIVE WIRE = main
Railway production deploys GitHub **main** (since cutover 2026-06-12 ~04:45 UTC). Any push to main that touches CODE = production deploy → requires scripted verification + per-change owner approval. **Watch Paths now set (D-013): a push touching only `ops/**` and `*.md` is deploy-inert** — ops/docs commits can be pushed without disturbing the live line. Code pushes still go through full protocol.

## Phase
CONSOLIDATION COMPLETE. Phase B closed 2026-06-12: cutover verified end-to-end (answered → qualified → lead SMS delivered in seconds; transcript archived in ops/transcripts/). Phase C in effect:
- phase-1a-stability-guardrails FROZEN as rollback until **2026-06-26**; then archive-tag + delete (owner call).
- Change #1 under new protocol: /version endpoint — **DEPLOYED & verified 2026-06-12** (main@74e5dbe; GET /version returns the deployed SHA; D-013). DONE.
- Change #2: call-metrics read path — **DEPLOYED & verified 2026-06-12** (main@7a9c3df; /version=7a9c3df; /admin/metrics.json live, admin-auth; D-015). DONE.
- Change #3 (NOW — the priority): core/vertical refactor of workflow/prompt_builder.py aimed STRAIGHT at a usable shoreline vertical (owner directive). Minimal scope — only what shoreline needs; plumbing preserved exactly. Nice-to-have internal work deprioritized.

## 🎯 SHORELINE FIRST LIVE CALL — critical path (THE priority; open every report with this step count)
Goal: a real Cape Coral homeowner calls the shoreline number → answered as "Shoreline Cost" → qualified → lead delivered within SLA. **6 steps remaining:**
1. prompt_builder core/vertical refactor — core engine + vertical config packages; plumbing preserved exactly (regression-clean via golden harness). Scope = shoreline-only needs. [operator] — **IN PROGRESS:** golden baseline locked (branch change/core-vertical-split @ 2040765; suite 143/143; D-017). Next = design + the split.
2. verticals/shoreline.json — author per contract §1.2 (greeting, identity, 8 qualification Qs, urgency, consent verbatim, transfer, disqualify). [operator; consent + greeting wording need owner sign-off = step 4]
3. shoreline lead delivery — webhook primary + email/sheet fallback, lead schema §3, 5-min SLA. [operator]
4. OWNER one-time errand (GATING): buy shoreline Twilio number; approve consent script (legal-adjacent); approve greeting/identity script; confirm webhook hosting. [owner — currently deferred = the real bottleneck]
5. provision + test: register shoreline tenant + route number → shoreline vertical; scripted media-stream test + one live test call on testing/allowed-test-caller path; transcript logged. [operator; live test needs #4]
6. go live: flip shoreline tenant live → first real homeowner call end to end. [owner + real caller]
Operator can do 1–3 and the software of 5 without the number; the FIRST REAL CALL is gated on owner step 4. Do NOT pursue until shoreline is live: more metrics, more self-verification endpoints, weekly-report automation.

## Production truth
- One branch: main @ 7a9c3df = multi-tenant code (byte-identical to old phase-1a tip 77b5537) + charter/ops + /version (#1) + /admin/metrics.json (#2). Deployed SHA externally confirmable via GET /version (=7a9c3df).
- Live build facts from verification call: tenant_id=1 (plumber), model gpt-realtime-2 (env/tenant-config driven), reasoning_effort=low, extra intake fields (property_role, additional_notes), leads delivered by SMS, calls/events recorded in Postgres.
- Tags: archive/single-file-app @ a2f0585; archive/p1-split-of-single-file-app @ c1aa2fa.
- legacy-snapshot/: zero unique work; owner to archive .env values then delete folder.

## Access gaps (standing)
No Railway token (can't read logs/deploy status), no ADMIN_PASSWORD, no DATABASE_URL — owner pastes logs or grants read access when needed.

## Working tree note
origin/main @ 74e5dbe is source of truth. Git write-ops run in a /tmp clone (mount blocks unlink → stale .lock files); the mounted folder's local HEAD may lag origin and show ops edits as uncommitted — cosmetic only. Untracked leftovers (core/, tests/goldens etc. from archived p1 branch, legacy-snapshot/) can't be deleted from sandbox; ignore.
