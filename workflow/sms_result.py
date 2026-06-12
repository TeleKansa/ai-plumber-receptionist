import json


def build_service_request_output(output: dict) -> str:
    return json.dumps(output)
