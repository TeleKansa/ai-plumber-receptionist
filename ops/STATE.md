# STATE

Updated: 2026-06-12 (session 5 — Phase C #1/#2/#3 DEPLOYED; core/vertical split live; shoreline now config-only)

## ⚠️ LIVE WIRE = main
Railway production deploys GitHub **main** (since cutover 2026-06-12 ~04:45 UTC). Any push to main that touches CODE = production deploy → requires scripted verification + per-change owner approval. **Watch Paths now set (D-013): a push touching only `ops/**` and `*.md` is deploy-inert** — ops/docs commits can be pushed without disturbing the live line. Code pushes still go through full protocol.

## Phase
CONSOLIDATION COMPLETE. Phase B closed 2026-06-12: cutover verified end-to-end (answered → qualified → lead SMS delivered in seconds; transcript archived in ops/transcripts/). Phase C in effect:
- phase-1a-stability-guardrails FROZEN as rollback until **2026-06-26**; then archive-tag + delete (owner call).
- Change #1 under new protocol: /version endpoint — **DEPLOYED & verified 2026-06-12** (main@74e5dbe; GET /version returns the deployed SHA; D-013). DONE.
- Change #2: call-metrics read path — **DEPLOYED & verified 2026-06-12** (main@7a9c3df; /version=7a9c3df; /admin/metrics.json live, admin-auth; D-015). DONE.
- Change #3 (NOW — the priority): core/vertical refactor of workflow/prompt_builder.py aimed STRAIGHT at a usable shoreline vertical (owner directive). Minimal scope — only what shoreline needs; plumbing preserved exactly. Nice-to-have internal work deprioritized.

## 🎯 SHORELINE FIRST LIVE CALL — critical path (THE priority; open every report with this step count)
Goal: a real Cape Coral homeowner calls the shoreline number → answered as "Shoreline Cost" → qualified → lead delivered within SLA. **5 steps remaining (step 1 ✅ deployed):**
1. ✅ **DEPLOYED 2026-06-12** — prompt_builder core/vertical split live (main@ac1f051; /version verified; plumber output byte-identical; D-019). Pending: one plumber-line regression test call (owner). Shoreline is now config-only.
2. verticals/shoreline.json + tenant→vertical selection — **BUILT on branch change/shoreline-vertical @ dea2f5b** (renders on engine; shorelinecost→shoreline; plumber byte-identical; 148/148; D-020/D-021). Remaining: A-006 final wording. Deploy = A-008 (recommend batching with step 3). [operator]
3. shoreline lead delivery — **PART 1 BUILT on branch @ 8a18f4e**: workflow/lead_delivery.py (build_shoreline_lead §3 + deliver_lead_webhook §4) + delivery spec; 155/155; D-022. **PART 2 (next, guarded):** wire main.py handler to route per-vertical submit tool (plumbing path untouched). Owner-gated VALUE: SHORELINE_LEAD_WEBHOOK_URL (ShorelineCost endpoint, §5.4). [operator]
4. OWNER one-time errand (GATING): buy shoreline Twilio number; approve consent script (legal-adjacent); approve greeting/identity script; confirm webhook hosting. [owner — currently deferred = the real bottleneck]
5. provision + test: register shoreline tenant + route number → shoreline vertical; scripted media-stream test + one live test call on testing/allowed-test-caller path; transcript logged. [operator; live test needs #4]
6. go live: flip shoreline tenant live → first real homeowner call end to end. [owner + real caller]
Operator can do 1–3 and the software of 5 without the number; the FIRST REAL CALL is gated on owner step 4. Do NOT pursue until shoreline is live: more metrics, more self-verification endpoints, weekly-report automation.

## Production truth
- One branch: main @ ac1f051 = multi-tenant code (plumber behavior byte-identical to old phase-1a tip 77b5537) + charter/ops + /version (#1) + /admin/metrics.json (#2) + core/vertical split (#3). Deployed SHA externally confirmable via GET /version (=ac1f051).
- Live build facts from verification call: tenant_id=1 (plumber), model gpt-realtime-2 (env/tenant-config driven), reasoning_effort=low, extra intake fields (property_role, additional_notes), leads delivered by SMS, calls/events recorded in Postgres.
- Tags: archive/single-file-app @ a2f0585; archive/p1-split-of-single-file-app @ c1aa2fa.
- legacy-snapshot/: zero unique work; owner to archive .env values then delete folder.

## Access gaps (standing)
No Railway token (can't read logs/deploy status), no ADMIN_PASSWORD, no DATABASE_URL — owner pastes logs or grants read access when needed.

## Working tree note
origin/main @ 74e5dbe is source of truth. Git write-ops run in a /tmp clone (mount blocks unlink → stale .lock files); the mounted folder's local HEAD may lag origin and show ops edits as uncommitted — cosmetic only. Untracked leftovers (core/, tests/goldens etc. from archived p1 branch, legacy-snapshot/) can't be deleted from sandbox; ignore.
