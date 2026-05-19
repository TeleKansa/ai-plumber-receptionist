import tempfile
import unittest

from fastapi import WebSocketDisconnect

import main
from storage import repository
from storage.database import configure_database, init_db


class CallLifecycleInstrumentationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        configure_database(f"sqlite:///{self.tmpdir.name}/test.db")
        init_db()
        main.sessions.clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_call(self, call_sid="CALL_LIFECYCLE"):
        tenant = repository.get_default_tenant()
        repository.create_or_update_call(
            call_sid,
            "+19135550123",
            "+15551234567",
            tenant_id=tenant["id"],
        )
        return tenant

    def test_twilio_stop_records_exit_reason_and_call_event(self):
        tenant = self._create_call("CALL_TWILIO_STOP")
        lifecycle = {"exit_reason": "unknown", "exit_detail": {}}
        session = {"tenant_id": tenant["id"], "last_ai_transcript": "Okay, what's the service address?"}

        main.record_twilio_stream_stopped(
            "CALL_TWILIO_STOP",
            "MZ_TEST",
            {"reason": "caller_completed"},
            lifecycle,
            session,
            response_active=False,
            assistant_speaking=False,
        )

        detail = repository.get_call_detail("CALL_TWILIO_STOP")
        event_types = {event["event_type"] for event in detail["events"]}
        self.assertEqual(lifecycle["exit_reason"], "twilio_stop")
        self.assertEqual(detail["call"]["status"], "stream_stopped")
        self.assertIn("twilio_stream_stopped", event_types)
        self.assertIn("media_stream_stopped", event_types)

    def test_websocket_disconnect_is_distinct_from_twilio_stop(self):
        tenant = self._create_call("CALL_WS_DISCONNECT")
        lifecycle = {"exit_reason": "unknown", "exit_detail": {}}
        session = {"tenant_id": tenant["id"]}
        exc = WebSocketDisconnect(code=1006, reason="abnormal closure")

        main.record_twilio_websocket_disconnect(
            "CALL_WS_DISCONNECT",
            "MZ_TEST",
            exc,
            lifecycle,
            session,
            response_active=False,
            assistant_speaking=False,
        )

        detail = repository.get_call_detail("CALL_WS_DISCONNECT")
        event_types = {event["event_type"] for event in detail["events"]}
        self.assertEqual(lifecycle["exit_reason"], "websocket_disconnect")
        self.assertEqual(detail["call"]["status"], "websocket_disconnected")
        self.assertIn("twilio_websocket_disconnected", event_types)
        self.assertNotIn("twilio_stream_stopped", event_types)

    def test_hangup_guard_requires_submit_and_lead(self):
        session = {
            "pending_hangup": True,
            "closing_response_started": True,
            "complete": True,
        }
        self.assertFalse(main.can_schedule_hangup(session))

        session["submit_service_request_seen"] = True
        self.assertFalse(main.can_schedule_hangup(session))

        session["lead_id"] = 123
        self.assertTrue(main.can_schedule_hangup(session))

    def test_missing_address_mid_intake_does_not_look_like_app_hangup(self):
        tenant = self._create_call("CALL_MID_INTAKE")
        lifecycle = {"exit_reason": "unknown", "exit_detail": {}}
        session = {
            "tenant_id": tenant["id"],
            "complete": False,
            "pending_hangup": False,
            "closing_response_started": False,
            "hangup_scheduled": False,
            "last_ai_transcript": "Okay, what's the service address?",
        }

        snapshot = main.record_media_stream_done(
            "CALL_MID_INTAKE",
            "MZ_TEST",
            session,
            lifecycle,
            response_active=False,
            assistant_speaking=False,
        )

        self.assertEqual(snapshot["media_stream_exit_reason"], "unknown")
        self.assertFalse(snapshot["complete"])
        self.assertFalse(snapshot["hangup_scheduled"])
        self.assertFalse(main.can_schedule_hangup(session))

    def test_emergency_leak_flow_before_submit_does_not_schedule_hangup(self):
        session = {
            "complete": False,
            "pending_hangup": False,
            "closing_response_started": False,
            "submit_service_request_seen": False,
            "last_ai_transcript": "Can you shut the water off there?",
        }

        self.assertFalse(main.can_schedule_hangup(session))

    def test_openai_reader_error_is_recorded(self):
        tenant = self._create_call("CALL_OAI_ERROR")
        lifecycle = {"exit_reason": "unknown", "exit_detail": {}}
        session = {"tenant_id": tenant["id"]}

        main.record_openai_reader_error(
            "CALL_OAI_ERROR",
            "MZ_TEST",
            RuntimeError("reader failed"),
            lifecycle,
            session,
            response_active=True,
            assistant_speaking=False,
        )

        detail = repository.get_call_detail("CALL_OAI_ERROR")
        event_types = {event["event_type"] for event in detail["events"]}
        self.assertEqual(lifecycle["openai_reader_exit_reason"], "openai_reader_error")
        self.assertIn("openai_reader_error", event_types)


if __name__ == "__main__":
    unittest.main()
