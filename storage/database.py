from contextlib import contextmanager
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import get_settings
from storage.migrations import run_schema_migrations
from storage.models import Base


log = logging.getLogger("plumber")


def normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return "postgresql+psycopg://" + database_url[len("postgres://") :]
    if database_url.startswith("postgresql://"):
        return "postgresql+psycopg://" + database_url[len("postgresql://") :]
    return database_url


def _connect_args(database_url: str) -> dict:
    if database_url.startswith("sqlite:"):
        return {"check_same_thread": False}
    return {"connect_timeout": 10}


def _make_engine(database_url: str):
    normalized_url = normalize_database_url(database_url)
    return create_engine(
        normalized_url,
        connect_args=_connect_args(normalized_url),
        pool_pre_ping=not normalized_url.startswith("sqlite:"),
        future=True,
    )


engine = _make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def configure_database(database_url: str):
    global engine, SessionLocal
    engine = _make_engine(database_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def init_db(settings=None):
    active_settings = settings or get_settings()
    log.info("DB init starting")
    try:
        log.info("DB init step starting: create_all")
        Base.metadata.create_all(bind=engine)
        log.info("DB init step complete: create_all")
        run_schema_migrations(engine, active_settings)
        log.info("DB init complete")
    except Exception:
        log.exception("DB init failed during startup")
        raise


@contextmanager
def session_scope():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
