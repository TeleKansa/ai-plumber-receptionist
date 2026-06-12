"""Phase C change #1: /version endpoint (A-004). Deploy self-verification."""

from fastapi.testclient import TestClient

import main


def test_version_reports_railway_commit(monkeypatch):
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "deadbeef1234")
    monkeypatch.setenv("RAILWAY_GIT_BRANCH", "main")
    client = TestClient(main.app)  # no context manager: skip startup (DB not needed for /version)
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json() == {"commit": "deadbeef1234", "branch": "main"}


def test_version_unknown_without_env(monkeypatch):
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    monkeypatch.delenv("RAILWAY_GIT_BRANCH", raising=False)
    client = TestClient(main.app)
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json() == {"commit": "unknown", "branch": "unknown"}
