# Shoreline Cost — Greeting / Identity / Consent scripts — DRAFT for owner approval

> STATUS: **DRAFT — not live, not in any tenant config.** These are legal-adjacent (identity, consent, recording disclosure). Per the Operator Charter they require explicit owner approval before going live. Once you approve/edit, the wording feeds `verticals/shoreline.json` (critical-path step 2). I am not a lawyer — see the legal flags at the bottom; the consent + recording wording in particular is worth a quick review by counsel before go-live.
> Source of wording: Integration Contract §1.2 (greeting_line, identity_rules, consent_script are given there verbatim) + §1.3 (recording disclosure). I have not silently changed the contract wording; additions are the recording disclosure and spoken phrasings of the identity rules.

---

## 1. Greeting (opening line)

**Option A — no recording** (use only if we decide NOT to record calls):
> "Shoreline Cost estimate line — this is the scheduling assistant. Are you calling about a waterfront repair project?"

**Option B — with recording disclosure (recommended; required in Florida if we record):**
> "Thanks for calling Shoreline Cost — this is the scheduling assistant, and this call is recorded for quality and accuracy. Are you calling about a waterfront repair project?"

Recommendation: **Option B.** The contract assumes recordings (the lead schema has a `recording_url`, and recordings add value to a "phone-qualified" lead), and Florida is a two-party-consent state, so a clear disclosure at the start is the safe path. But enabling recording is YOUR explicit call (see §4).

---

## 2. Identity responses (spoken when asked)

- **"Is this a real person / a robot / an AI?"** (honesty rule — the AI must never claim to be human):
  > "I'm an automated scheduling assistant for Shoreline Cost. I can take your project details and get you matched with local waterfront contractors. What's going on with your project?"

- **"Are you a contractor / do you do the work?"** (contract wording):
  > "We're a free estimate-matching service — we collect your project details and connect you with vetted local waterfront contractors."

- **Internal rule (not spoken):** never mention Loopline; answer only as Shoreline Cost.

---

## 3. Consent script (read before ending any qualified call — verbatim from contract §1.2)

> "To get you matched bids, Shoreline Cost will share your project details with up to three vetted local contractors, who may contact you by phone, text, or email about this project. Is that okay?"

- If the caller says **no** → mark `consent=false`, do **not** deliver the lead to buyers, log only.
- If **yes** → record `consent=true` + timestamp; lead is deliverable.

---

## 4. The one decision I need from you: record or not?

| | If we RECORD | If we DON'T record |
|---|---|---|
| Greeting | Option B (with disclosure) | Option A |
| Florida two-party consent | satisfied by the disclosure line | n/a |
| Lead `recording_url` | populated (adds lead value) | empty |
| Approval needed | recording ON is a separate owner approval | none |

Recording is **OFF by default** and I will not enable it without your explicit yes + the disclosure line live.

---

## 5. What I need from you (this unblocks the "approve scripts" part of critical-path step 4)

1. Greeting: approve **Option A** or **Option B** (or edit).
2. Identity responses: approve or edit the two lines in §2.
3. Consent: approve the §3 wording as-is, or give edits (contract change = I draft a v1.1).
4. Recording: **yes** or **no.**

The only remaining piece of step 4 after this is buying the Shoreline Twilio number (your one-time errand).

---

## Legal flags (I am not a lawyer — factual considerations only)

- **TCPA / contact consent:** the consent line obtains the caller's permission to be contacted by phone, text, and email and to have their info shared with up to three contractors. If contractors (or you) text callers, TCPA rules around prior express consent apply; the §3 wording is designed to capture that, but a lawyer should confirm it's sufficient for your texting/calling practices and that "up to three vetted local contractors" matches how leads are actually sold.
- **Florida recording:** two-party-consent state — recording without disclosure is a legal risk; the Option B greeting is the mitigation. Calls may also come from other states; a disclosure-on-every-call approach is the conservative default.
- **Accuracy:** "free estimate-matching service" and "vetted" should be literally true of how Shoreline Cost operates, or the wording should change.
- Recommend a quick counsel review of §3 (consent) and the recording decision before go-live. This is queued as **A-006**.
