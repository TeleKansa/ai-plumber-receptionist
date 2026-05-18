import json


def build_service_request_output(sms_sent: bool) -> str:
    return json.dumps({"success": bool(sms_sent)})
