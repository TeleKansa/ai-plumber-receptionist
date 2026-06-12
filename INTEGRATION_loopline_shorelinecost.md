# INTEGRATION CONTRACT: Loopline ⇄ ShorelineCost
# Version 1.0 — place an identical copy in BOTH repos. Canonical copy lives in the Loopline repo.
# Both agents: treat this file as binding. Propose changes by drafting v1.1 into APPROVAL_QUEUE, never by silently diverging.

## 0. Relationship

Loopline Solutions provides white-label AI phone reception. ShorelineCost is its first production client (internal). Loopline owns: call answering, qualification, recording, transcription, lead packaging, delivery. ShorelineCost owns: the phone number placement on the website, lead routing to contractor buyers, revenue.

This integration doubles as Loopline's productization test: success = adding a new industry vertical requires ONLY a new vertical config file, zero changes to core call-handling logic.

---

## 1. Loopline-side deliverables

### 1.1 Refactor: core engine vs vertical config (do this FIRST)
Split the current plumber-specific script into two layers:

**Core engine (industry-agnostic):**
- Greeting using `{client.brand_name}` and `{client.greeting_line}` from config
- Caller info capture (name, callback number, ZIP)
- Consent script (verbatim from config, never improvised)
- Qualification loop driven by `questions[]` from config
- Urgency detection → branch per config rules
- Outcome handling: live transfer / callback promise / message taken
- Recording + transcription + structured lead packaging (schema §3)
- Delivery via configured channel (§4)

**Vertical config (per industry/client):** brand identity, greeting, qualification questions, urgency keywords, transfer rules, disqualification rules.

The existing plumber script becomes `verticals/plumbing.json` (or .yaml). Nothing plumber-specific may remain in core after refactor.

### 1.2 New vertical: `verticals/shoreline.json`

```
brand_name: "Shoreline Cost"
greeting_line: "Shoreline Cost estimate line, this is the scheduling assistant — are you calling about a waterfront repair project?"
identity_rules:
  - Never mention Loopline. Answer as Shoreline Cost only.
  - If asked "are you a contractor?": "We're a free estimate-matching service — we collect your project details and connect you with vetted local waterfront contractors."
  - If asked "is this an AI?": answer honestly, then continue.

qualification_questions (in order, conversational, skip any already volunteered):
  1. project_type — one of: seawall repair / seawall replacement / rip rap / dock repair / boat lift repair / bulkhead repair / erosion control / dredging / dock electrical / other
  2. zip_code
  3. water_setting — freshwater lake or canal / coastal or saltwater / tidal or intracoastal / unsure
  4. approx_size — shoreline or wall length in feet, or dock size; "not sure" acceptable
  5. condition — light repair / moderate damage / structural or urgent / active failure
  6. access — road or driveway / limited / likely barge or water access / unsure
  7. timeline — emergency / within 30 days / 1–3 months / planning only
  8. ownership — confirm caller owns the property or is authorized (HOA/property manager OK)

urgency_rules:
  - Keywords: "collapsing", "washing out", "boat trapped", "wall failed", "storm damage", "flooding" → mark urgency=EMERGENCY, attempt live transfer immediately, skip remaining questions except ZIP + callback number.

consent_script (read before ending any qualified call, verbatim):
  "To get you matched bids, Shoreline Cost will share your project details with up to three vetted local contractors, who may contact you by phone, text, or email about this project. Is that okay?"
  - If no → mark consent=false, do NOT deliver to buyers, log only.

transfer_rules:
  - If a signed contractor exists for {zip → market} and urgency in (EMERGENCY, within 30 days) and within contractor's stated hours → warm live transfer: brief contractor with project summary before connecting.
  - Else → "A matched local specialist will reach out within one business day," end call, package lead.

disqualify (politely exit, log as disqualified):
  - Solicitation/sales calls, job seekers, callers outside US, requests for free engineering advice with no project.
```

### 1.3 Operational requirements
- Provision capacity for one dedicated inbound number for ShorelineCost (number purchase itself = owner approval item).
- All calls recorded with state-appropriate recording disclosure in greeting config.
- Uptime expectation: missed/failed calls logged and reported in weekly metrics.

---

## 2. ShorelineCost-side deliverables

- Place the tracking number sitewide: header, sticky mobile call button, every cost-guide page, and a dedicated "Talk to us" block near the estimate form. Emergency CTA ("Emergency or active damage?") routes to PHONE first, form second.
- Ingest leads from Loopline via the delivery channel (§4) into the same lead log as form leads, tagged `source=phone`.
- Phone leads are priced/sold as "phone-qualified" tier (2–3× form lead price); live transfers as top tier. Reflect this in contractor offers.
- Provide Loopline with the current signed-contractor roster per market (sync `CONTRACTORS.md` summary weekly): market, trade, transfer phone, hours, status.

---

## 3. Lead schema (JSON, both sides conform exactly)

```json
{
  "lead_id": "SC-YYYYMMDD-NNN",
  "timestamp": "ISO8601",
  "source": "phone | form",
  "caller_name": "",
  "callback_phone": "",
  "email": "",
  "zip": "",
  "market": "derived from zip",
  "project_type": "",
  "water_setting": "",
  "approx_size_ft": null,
  "condition": "",
  "access": "",
  "timeline": "",
  "urgency": "EMERGENCY | HIGH | NORMAL | PLANNING",
  "ownership_confirmed": true,
  "consent": true,
  "consent_timestamp": "ISO8601",
  "qualification_status": "qualified | partial | disqualified",
  "transfer_outcome": "live_transfer | callback_promised | message | none",
  "transferred_to": "contractor id or null",
  "recording_url": "",
  "transcript_summary": "3-5 sentence plain-language project summary",
  "notes": ""
}
```

Rules: `consent=false` leads are never delivered to buyers. `partial` leads (caller hung up mid-qualification) are delivered only if callback_phone + zip + project_type captured, flagged as partial and priced lower.

---

## 4. Delivery channel

Primary: Loopline POSTs each completed lead JSON to a webhook endpoint exposed by ShorelineCost (agent to implement; a simple serverless function appending to the lead log is sufficient).
Fallback until webhook is live: structured email to the lead-intake inbox + append to a shared Google Sheet.
SLA: lead delivered within 5 minutes of call end. EMERGENCY leads additionally trigger an immediate notification to the matched contractor and to the owner.

---

## 5. Owner (Vincent) one-time setup items — APPROVAL_QUEUE

1. Purchase tracking phone number (~$5–15/mo) and any telephony usage costs.
2. Approve the consent script wording (legal-adjacent).
3. Approve the greeting/identity script.
4. Confirm webhook hosting (if it incurs cost).

Everything else in this contract is autonomous work for the two agents.

---

## 6. Success criteria for this integration

- Adding the shoreline vertical required zero changes to Loopline core after refactor (productization proof).
- ≥95% of inbound calls answered and logged; every qualified lead delivered within SLA.
- Phone-qualified leads command a measurably higher price than form leads in contractor negotiations.
- This contract, with names swapped, is reusable verbatim as Loopline's onboarding template for its first external client.
