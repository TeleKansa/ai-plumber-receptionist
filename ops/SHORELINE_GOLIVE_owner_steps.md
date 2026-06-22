# Shoreline go-live — owner action plan

Status (2026-06-12): all shoreline SOFTWARE is built, tested, and deployed to production
(main @ 4a4ac1d), but DORMANT — no tenant/number points to it yet, so every live call still
goes to the plumber line, unchanged. The steps below are the only things left before a real
Cape Coral homeowner can be answered as "Shoreline Cost." Tasks 1–3 are owner-only and
independent (do them in any order). When they're done I do the provisioning + test setup.

---

## Task 1 — Approve the scripts (A-006) — ✅ your decisions received 2026-06-12; pending lawyer review
Decided: greeting/role → "project assistant" (not "scheduling"); identity lines approved; recording =
yes (with disclosure); consent current wording OK (direct matching to ≤3 contractors, no resale).
Applied to the shoreline config on branch change/shoreline-scripts-a006.
REMAINING on you: have your lawyer review the consent + phone wording before go-live (you flagged this);
A-006 stays open until then.
Notes: (a) actually RECORDING calls is a small software feature I still have to build — the recording
disclosure line goes live together with it (so we never claim to record when we don't). (b) Cross-vertical
rule recorded: Shoreline + Septic consent wording stay unified; allowing resale later changes both + re-approval.

## Task 2 — Connect the Shoreline site so I can build the lead webhook (this is MY job now)
You reassigned the lead-receiver to me — good. I'll build a Netlify Function on the Shoreline site that
mirrors your Septic form-lead Function: it receives Loopline's §3 lead POST and appends it to the same
lead log with `source=phone`, then I hand you the URL to relay to Loopline.
What I need from you (one of):
- Connect the Shoreline site's repo/folder to me AND point me at the Septic Function as the pattern to
  copy — then I add the Function and deploy it the way I deploy Loopline; or
- Give me access to the Shoreline Netlify project (a Netlify connector is available).
Why this matters: I currently only have the `loopline` folder, so I literally can't build it until the
Shoreline site is connected. (This is NOT the old git blocker — Loopline deploys are working; production
is live right now at 4a4ac1d.) Once it's built + deployed I set `SHORELINE_LEAD_WEBHOOK_URL` in Railway
and POST a sample lead to confirm end to end.

## Task 3 — Buy + point the Twilio number   ~15 min (owner-only: funding/payment)
What: a phone number for the shoreline line, pointed at our app.
How:
1. Twilio Console → Phone Numbers → Buy a number. Choose a local Voice-capable number in the
   shoreline market (e.g., a 239 / SW-Florida area code for Cape Coral). ~$1–2/mo + usage.
2. Open that number → Voice configuration → "A CALL COMES IN" → Webhook, HTTP POST, URL:
   `https://ai-plumber-receptionist-production.up.railway.app/voice`
3. (Optional, same errand: grab one more cheap number as a dedicated TEST line.)
4. Send me the new number.

---

## Then I do (once Tasks 1–3 are in)
- Set `SHORELINE_LEAD_WEBHOOK_URL` (I walk you through the Railway dashboard).
- Create the `shorelinecost` tenant (status = testing) with your approved scripts, and register
  the Twilio number → shoreline. (See access note below.)
- POST a sample lead to your webhook to confirm delivery end-to-end.
- You place ONE shoreline test call on the testing path; I review the transcript
  (answered as Shoreline Cost → 8 questions → consent → lead delivered within SLA).
- On your OK, flip `shorelinecost` to live → first real homeowner call.

### Access note (for the provisioning I do)
Creating the tenant + registering the number is a write to the production database. I currently
have no `ADMIN_PASSWORD` or `DATABASE_URL`. So either:
- give me the admin password so I can use the existing admin UI, or
- I give you click-by-click to create it in the admin UI yourself.
Either way I'll prepare the exact tenant config first (config-as-code) so it's reviewed before it's applied.

---

## Also pending (small, separate)
- The plumber-line regression test call for the deploy you just approved: just place a normal
  repair-report call to the plumber line and confirm it answers exactly as before. Paste me the
  Railway log excerpt if anything looks off (I have no log access).
