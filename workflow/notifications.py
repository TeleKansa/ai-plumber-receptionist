from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SmsSendResult:
    success: bool
    provider_message_sid: Optional[str] = None
    error: Optional[str] = None
