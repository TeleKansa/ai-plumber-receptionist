# STATE

Updated: 2026-06-11 (session 2, end)

## Phase
P0 housekeeping DONE. P1 core/vertical split CODE-COMPLETE on branch `p1/core-vertical-split` @ c1aa2fa with 11/11 regression evidence (D-006). Awaiting owner: A-001 (Railway source check) and A-002 (merge approval). Repo working tree is left checked out on the P1 branch for review.

## Key discovery this session (D-005)
`legacy-snapshot/` at repo root = parked multi-tenant variant (Postgres, admin pages) on unmerged branch `phase-1a-stability-guardrails` of the same GitHub repo. Explains the charter's "tenant config in Postgres". GitHub main = the simple single-file app. Which branch Railway deploys is unknown → A-001. legacy-snapshot contains live secrets; now gitignored, never to be committed.

## Live system (verified facts)
- One live tenant: plumber line. Single-tenant FastAPI app (`main.py`) on Railway at `ai-plumber-receptionist-production.up.railway.app`. `/health` returned ok on 2026-06-11.
- Lead delivery: SMS to `PLUMBER_PHONE_NUMBER`. No call recording anywhere in code.
- No Postgres / tenant DB exists in this repo (no driver in requirements). Charter's "tenant config in Postgres" does not match code — operating assumption: repo is the only source of truth (DECISIONS D-003).
- git main = origin/main = a2f0585 (docs-only). Railway deploy of a2f0585 unverified — owner action queued (APPROVAL_QUEUE A-001).

## Owner direction (2026-06-11, D-002)
Defer Twilio number purchase and go-live scripts. Order: P0 → P1. Plumber behavior must be preserved exactly. Bring acceptance evidence before approvals.

## Branch policy
`main` = production (Railway auto-deploys). P1 refactor lives on `p1/core-vertical-split` and merges only after acceptance evidence is reviewed by owner.

## Blockers
- a2f0585 deploy confirmation (owner, 30s in Railway dashboard).
- Live test call path: no testing tenant exists; P1 acceptance uses scripted media-stream regression (charter-permitted) + golden equivalence. A live test number remains deferred until owner approves spend.
