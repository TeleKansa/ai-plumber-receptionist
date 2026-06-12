# CONSOLIDATION PLAN (A-003 — awaiting owner approval; no step has been executed)

Goal: ONE branch (main) = multi-tenant production code + charter/ops files; Railway repointed to main at a moment we choose; old single-file app preserved via tag. Standing invariant: Railway watches ONLY `phase-1a-stability-guardrails`, so every step below is deploy-inert except the explicit cutover (B2). I never push to phase-1a, period (LIVE WIRE RULE, D-007).

## Phase A — prepare (zero production effect)
- A1. Tag `archive/single-file-app` at a2f0585 (old app preserved in history, per owner requirement c). Tag `archive/p1-split-of-single-file-app` at c1aa2fa (session-2 refactor kept as methodology reference). Push tags.
- A2. Push local main (2 ops commits) to origin/main. Inert — Railway ignores main.
- A3. Create branch `consolidation` from 77b5537 (phase-1a tip). Merge main INTO it — never the reverse. Conflict policy: all code/config files resolve to the phase-1a side (main.py, requirements.txt, README, .env.example); .gitignore = union; charter/contract/ops/config-tenants arrive from main cleanly. The plumber tenant snapshot (config/tenants/plumber.json) gets rewritten afterward to describe the REAL tenant config (exported from Postgres once we have DB access).
- A4. Verify in sandbox (deploy-inert): phase-1a's own test suite green (DB-dependent tests run against SQLite/fixtures if supported — will report exactly what ran and what couldn't); app boot/import check; and the critical gate: `git diff 77b5537..consolidation -- <every code path>` is EMPTY (docs-only delta). That guarantees the cutover deploys byte-identical code.
- A5. Owner reviews evidence → I merge consolidation → main and push. Still inert.

## Phase B — cutover (the one deliberate risky moment; owner executes the dashboard step)
- B1. Preconditions: low-call window chosen by owner; rollback lever identified = repoint Source back to phase-1a (branch stays frozen and untouched as last-good).
- B2. OWNER ACTION: Railway → service → Settings → Source → branch `phase-1a-stability-guardrails` → `main`. Railway builds and deploys main.
  - Risk 1 — build failure: Railway keeps the last good deployment serving; rollback = repoint. Pre-mitigated by A4 byte-identity.
  - Risk 2 — instance swap may drop calls in flight at that moment. Mitigation: timing (B1).
  - Risk 3 — env vars: service-level, unaffected by branch change; I will pre-verify the env var name list against the branch's config/settings.py in A4.
- B3. Immediately after: deploy green + /health ok + one verification call (preferred: through the build's testing-tenant/allowed_test_callers path with my scripted plan pre-approved by owner; minimum: owner places one live call) — transcript logged in DECISIONS.
- B4. Anything wrong → owner repoints Source back to phase-1a. Instant, last-good build intact.

## Phase C — after stabilization
- C1. main is now the live wire — LIVE WIRE RULE transfers to main; every future change: branch → scripted verification → owner approval → push.
- C2. phase-1a frozen (not deleted) ≥2 weeks; then tag `archive/phase-1a-final` and delete (owner call).
- C3. legacy-snapshot folder: owner archives its .env values privately, then deletes the folder (sandbox cannot delete files on this mount).
- C4. First post-cutover changes, each via full protocol, in order: (1) /version endpoint returning git SHA — makes every future deploy self-verifiable; (2) call-outcome metrics logging (D-007 ruling 3); (3) restart contract §1.1 core/vertical refactor against workflow/prompt_builder.py using the session-2 golden-harness methodology.

## Where the risk lives (summary)
Accidental push to phase-1a (eliminated by rule + my behavior); merge mistakes in A3 (caught by A4 byte-identity gate); the B2 repoint deploy (build failure / call drop — mitigated, reversible); nothing else touches production.
