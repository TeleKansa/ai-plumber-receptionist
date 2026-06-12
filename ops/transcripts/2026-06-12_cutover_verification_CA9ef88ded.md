# Cutover verification call — 2026-06-12 04:53–04:54 UTC
Call SID: CA9ef88ded0c3ec6b3b566677d00fe883b · Stream: MZ751ff29e036aabaa43521af2de169341 · tenant_id: 1
Deployment: main @ ac687be (repointed from phase-1a-stability-guardrails @ 77b5537 — byte-identical code, D-010)
Source: Railway log excerpt pasted by owner in chat (operator has no direct log access).

## What the log shows
- Call answered; full multi-tenant pipeline live: tenant routing (tenant_id=1), realtime model gpt-realtime-2, reasoning_effort=low.
- Intake captured all fields: submit_service_request fired with {issue: "Kitchen sink leak from under the sink", urgency: "No active water leaking right now", address: "6100 West 120th Street", callback: "732-789-0675", name: "Sam", extra_fields: {property_role: "renting", additional_notes: "declined"}}.
- AI: "Thanks, I have what I need and I'm sending it over now." → then engine delayed response.create (reason=intake_missing_extra) → AI: "Anything else the plumber should know before I send this over?" → caller hung up (twilio_stop).
- Benign OAI race logged once: response_cancel_not_active (cancellation after response already done) — non-fatal, call continued.
- END-OF-CALL SNAPSHOT FLAGS: submit_service_request_seen=True BUT lead_id=None, complete=False, pending_hangup=False, hangup_reason=None.

## Open question at time of archiving
lead_id=None at stream end → delivery of THIS call's lead unconfirmed from the log alone (owner asked to confirm whether the lead SMS arrived / a lead row exists in admin). NOTE: code is byte-identical to what served production since May 21, so whatever this behavior is, it is pre-existing production behavior, not a cutover regression. If the lead was NOT delivered, that is a pre-existing missed-lead path (caller hangs up during the "anything else" confirmation loop) — priority-1 reliability item for the backlog, separate from cutover acceptance.

## Raw log excerpt (as pasted by owner; function_call_arguments.delta lines condensed)
04:53:57.5–58.0  [OAI_IN] response.function_call_arguments.delta ×34
04:53:58,098 [OAI_IN] response.function_call_arguments.done
04:53:58,112 submit_service_request source=response.function_call_arguments.done: {'issue': 'Kitchen sink leak from under the sink', 'urgency': 'No active water leaking right now', 'address': '6100 West 120th Street', 'callback': '732-789-0675', 'name': 'Sam', 'extra_fields': {'property_role': 'renting', 'additional_notes': 'declined'}}
04:53:58,203 response.create delayed reason=intake_missing_extra delay_reason=response_active
04:53:58,217–58,271 [OAI_IN] conversation.item.done / response.output_item.done / response.done / rate_limits.updated
04:53:58,342–59,327 [OAI_IN] response.created → output_audio_transcript.delta ×14 → output_audio.done
04:53:59,327 AI said: Thanks, I have what I need and I'm sending it over now.
04:53:59,698 [OAI_IN] input_audio_buffer.speech_started → Caller speech started
04:54:00,275 speech_stopped / committed / response.created
04:54:00,303 speech_started → Caller speech started; canceling current AI audio
04:54:00,832 [OAI_IN] error: {'type': 'invalid_request_error', 'code': 'response_cancel_not_active', 'message': 'Cancellation failed: no active response found'}
04:54:01,553–02,626 speech_stopped → response.created → transcript deltas ×12 → output_audio.done
04:54:02,628 AI said: Anything else the plumber should know before I send this over?
04:54:02,867 Twilio stream stopped payload={'accountSid': 'AC024b…', 'callSid': 'CA9ef88…'}
04:54:04,969 media_stream done exit_reason=twilio_stop snapshot={'call_sid': 'CA9ef88ded0c3ec6b3b566677d00fe883b', 'stream_sid': 'MZ751ff29e036aabaa43521af2de169341', 'tenant_id': 1, 'realtime_model': 'gpt-realtime-2', 'realtime_reasoning_effort': 'low', 'media_stream_exit_reason': 'twilio_stop', 'openai_reader_exit_reason': None, 'complete': False, 'pending_hangup': False, 'closing_response_started': False, 'hangup_scheduled': False, 'response_active': False, 'assistant_speaking': False, 'last_ai_transcript': 'Anything else the plumber should know before I send this over?', 'last_response_create_reason': 'intake_missing_extra', 'submit_service_request_seen': True, 'lead_id': None, 'hangup_reason': None, 'caller_audio_overlap_frames': 540, 'caller_audio_overlap_bytes': 86400, 'dropped_audio_frames': 0, 'dropped_audio_bytes': 0}
