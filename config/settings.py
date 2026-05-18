import os
from dataclasses import dataclass


DEFAULT_HOST = "ai-plumber-receptionist-production.up.railway.app"
DEFAULT_OAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-1.5"
DEFAULT_DATABASE_URL = "sqlite:///./local_dev.db"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_phone_number: str
    plumber_phone_number: str
    host: str
    oai_url: str
    database_url: str
    admin_password: str


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        twilio_phone_number=os.getenv("TWILIO_PHONE_NUMBER", ""),
        plumber_phone_number=os.getenv("PLUMBER_PHONE_NUMBER", ""),
        host=os.getenv("HOST") or os.getenv("PUBLIC_HOST") or DEFAULT_HOST,
        oai_url=os.getenv("OAI_URL") or os.getenv("OPENAI_REALTIME_URL") or DEFAULT_OAI_URL,
        database_url=os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL,
        admin_password=os.getenv("ADMIN_PASSWORD", ""),
    )
