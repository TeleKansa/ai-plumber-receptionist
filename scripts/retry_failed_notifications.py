import argparse
import asyncio
import json

from storage import repository
from storage.database import init_db
from main import send_sms


async def retry_failed_notifications(tenant_id: int | None = None, limit: int = 25):
    init_db()
    notifications = repository.list_failed_notifications(limit=limit, tenant_id=tenant_id)
    retried = 0
    for notification in notifications:
        lead = repository.get_lead(notification["lead_id"])
        if not lead:
            continue
        call_detail = repository.get_call_detail(lead["call_sid"])
        call = call_detail.get("call") or {}
        intake_policy = repository.get_intake_policy(lead["tenant_id"]) if lead.get("tenant_id") else None
        notification_policy = repository.get_notification_policy(lead["tenant_id"]) if lead.get("tenant_id") else None
        try:
            args = json.loads(lead.get("raw_args_json") or "{}")
        except json.JSONDecodeError:
            args = {}
        args.update(
            {
                "priority": lead.get("priority") or "normal",
                "priority_reason": lead.get("priority_reason") or "",
            }
        )
        result = await send_sms(
            lead["call_sid"],
            args,
            call.get("from_number") or lead.get("callback") or "",
            notification["to_number"],
            intake_policy,
            notification_policy,
        )
        if result.success:
            repository.mark_notification_sent(notification["id"], result.provider_message_sid)
        else:
            repository.mark_notification_failed(notification["id"], result.error or "Retry failed")
        retried += 1
    print(f"Retried {retried} failed notification(s).")


def main():
    parser = argparse.ArgumentParser(description="Retry failed SMS notifications.")
    parser.add_argument("--tenant-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    asyncio.run(retry_failed_notifications(args.tenant_id, args.limit))


if __name__ == "__main__":
    main()
