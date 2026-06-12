# STATE

Updated: 2026-06-11 (session 2)

## Phase
P0 housekeeping (this session) → P1 core/vertical refactor (in progress on branch `p1/core-vertical-split`).

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
