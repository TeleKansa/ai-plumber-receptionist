# Backlog

## P0 — housekeeping
- [x] Populate /ops/ files (this session)
- [x] INFRASTRUCTURE.md (Twilio routing, Railway, env var names)
- [x] Export plumber tenant snapshot → /config/tenants/plumber.json
- [ ] Owner confirms Railway deploy of a2f0585 (A-001)
- [ ] /version endpoint returning git SHA (tiny code change; ships with P1 merge so deploys are self-verifiable)

## P1 — core/vertical split (branch p1/core-vertical-split)
- [x] Golden behavior snapshots recorded from pre-refactor code
- [x] core/ engine (industry-agnostic) + verticals/plumbing.json
- [x] Regression: golden equivalence + scripted media-stream test + no-vertical-leakage check
- [ ] Owner reviews acceptance evidence → approves merge to main (A-002)
- [ ] Post-merge: Railway deploy green + one live call on plumber line confirming unchanged behavior

## P2 — shoreline vertical (blocked on owner)
- [ ] verticals/shoreline.json per contract §1.2 (config can be drafted ahead)
- [ ] BLOCKED owner: Twilio number purchase; consent + greeting script approval
- [ ] Lead delivery webhook (primary) + email/sheet fallback per contract §3–4

## P3/P4 — not started
- Demo tenant, onboarding template, pricing proposal; metrics persistence (call outcomes currently not stored anywhere — only Railway stdout logs)

## Hygiene (non-urgent)
- GitHub PAT embedded in git remote URL — rotate to credential helper
- audioop removed in Python 3.13 (pinned 3.12 — fine; revisit before upgrading)
- Repo name `ai-plumber-receptionist` predates Loopline branding
