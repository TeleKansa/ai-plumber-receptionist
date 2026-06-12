# Approval Queue

## A-001 — OPEN — Railway dashboard check (owner, ~1 min, answers TWO questions)
Railway dashboard → project → Deployments / Settings → Source. Report back: (1) which GitHub **branch** the service deploys (main? phase-1a-stability-guardrails?), and (2) whether the newest deployment shows commit `a2f0585` with status Success. Question 1 matters because of D-005: a multi-tenant variant exists on an unmerged branch and HTTP probes can't tell which build is live. Optionally paste a Railway project token for CLI verification in future sessions.

## A-002 — READY FOR REVIEW — Merge p1/core-vertical-split @ c1aa2fa to main
P1 split is code-complete with acceptance evidence (D-006): 11/11 regression tests prove byte-identical behavior vs pre-refactor recordings; zero plumber logic in core. Merging deploys the refactored engine to the live plumber line (if Railway tracks main — see A-001). Recommended order: resolve A-001 first → owner approves merge → deploy → one live test call to the plumber line → log transcript. Do not merge before reviewing.

## Deferred by owner (2026-06-11, D-002) — do not action yet
- Shoreline Twilio number purchase (~$5–15/mo + usage)
- Shoreline consent script + greeting/identity script approval
- Optional test Twilio number (~$1–2/mo) for live regression calls

## Standing flags
- Recording: OFF everywhere; never enable without approval + disclosure line (FL two-party consent relevant).
- Spend alert threshold: flag if projected monthly OpenAI usage > $50.
