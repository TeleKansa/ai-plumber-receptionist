"""
Tenant entrypoint — plumber line (vertical: plumbing)
=====================================================
All call-handling logic lives in core/engine.py (industry-agnostic).
Industry behavior lives in verticals/plumbing.json.
This file only pins the deployment-specific values for THIS tenant.

Start command is unchanged: uvicorn main:app
"""

import os

os.environ.setdefault("VERTICAL", "plumbing")
os.environ.setdefault("PUBLIC_HOST", "ai-plumber-receptionist-production.up.railway.app")

from core.engine import app  # noqa: E402,F401
