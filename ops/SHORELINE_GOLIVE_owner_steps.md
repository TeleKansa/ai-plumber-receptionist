# Shoreline go-live — owner action plan

Status (2026-06-12): all shoreline SOFTWARE is built, tested, and deployed to production
(main @ 4a4ac1d), but DORMANT — no tenant/number points to it yet, so every live call still
goes to the plumber line, unchanged. The steps below are the only things left before a real
Cape Coral homeowner can be answered as "Shoreline Cost." Tasks 1–3 are owner-only and
independent (do them in any order). When they're done I do the provisioning + test setup.

---

## Task 1 — Approve the scripts (A-006)   ~10 min (+ optional legal review)
What: lock the greeting / identity / consent wording and the record-or-not decision.
How:
1. Open `config/tenants/shoreline_scripts_DRAFT.md`.
2. Decide four things: (a) greeting Option A (no recording) or B (with recording disclosure);
   (b) the two identity lines; (c) the consent wording (contract-verbatim); (d) record: yes / no.
3. Recommended: Option B + recording ON — Florida is two-party-consent (disclosure required if
   recording) and recordings raise the "phone-qualified" lead value.
4. Legal-adjacent: have a lawyer glance at the consent line (TCPA / lead-sharing) and the
   recording decision before go-live.
5. Reply with your choices (or edits). I lock them into the live shoreline config.

## Task 2 — Stand up the lead webhook   ~30 min
What: a URL ShorelineCost exposes that receives each lead as an HTTP POST and saves it.
The exact JSON we send is the schema in `INTEGRATION_loopline_shorelinecost.md` §3.
How (pick the easiest for you):
- No-code: a Zapier / Make "Catch Hook" → append a row to a Google Sheet (and/or email you).
  Copy the hook URL.
- Serverless: a ~20-line Netlify / Vercel / Cloudflare function that appends to a sheet or DB.
- Your own backend: an endpoint on the ShorelineCost site.
Then: send me the URL. It goes into the Railway env var `SHORELINE_LEAD_WEBHOOK_URL` (I'll give you
the exact click path, same as the watch-paths setup). Before go-live I'll POST a sample lead so you
can confirm your endpoint receives it.

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
