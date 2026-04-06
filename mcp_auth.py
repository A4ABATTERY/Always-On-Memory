"""
mcp_auth.py — API key management for the AOM MCP server.

Keys are loaded from the AOM_API_KEYS environment variable.
Format: "agent-name-1:key1,agent-name-2:key2"

Each key is associated with exactly one named agent/LLM. The key is the
Bearer token sent in the Authorization header. The name is used for logging.
"""

import os
import logging
from typing import Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

log = logging.getLogger("memory-agent.mcp-auth")


def load_api_keys() -> Dict[str, str]:
    """
    Parse AOM_API_KEYS env var and return a dict mapping key → agent-name.

    Format: "agent-name-1:secretkey1,agent-name-2:secretkey2"
    Malformed entries (no colon) are silently skipped with a warning.
    Returns empty dict if AOM_API_KEYS is unset or blank (auth disabled).
    """
    raw = os.getenv("AOM_API_KEYS", "").strip()
    if not raw:
        return {}

    keys: Dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            log.warning(f"mcp_auth: ignoring malformed AOM_API_KEYS entry (no colon): '{entry}'")
            continue
        name, key = entry.split(":", 1)
        name = name.strip()
        key = key.strip()
        if name and key:
            keys[key] = name
        else:
            log.warning(f"mcp_auth: ignoring empty name or key in entry: '{entry}'")

    return keys


def validate_key(api_key: str, keys: Dict[str, str]) -> Optional[str]:
    """
    Return the agent name associated with api_key, or None if invalid.
    If keys is empty, auth is disabled and all requests are permitted.
    """
    if not keys:
        return "anonymous"
    return keys.get(api_key)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that enforces Bearer-token API key authentication
    on every MCP request.

    Requests to '/' and '/health' are always permitted (no auth required).
    All other paths require:
        Authorization: Bearer <key>
    where <key> is a value registered in AOM_API_KEYS.
    """

    _OPEN_PATHS = {"/", "/health"}

    def __init__(self, app, api_keys: Dict[str, str]):
        super().__init__(app)
        self._keys = api_keys
        log.info(f"🔑 MCP auth enabled — {len(api_keys)} registered key(s): {list(api_keys.values())}")

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._OPEN_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            client_host = request.client.host if request.client else "unknown"
            log.warning(f"mcp_auth: missing Bearer token from {client_host}")
            return JSONResponse(
                {"error": "Unauthorized", "detail": "Authorization: Bearer <key> header required"},
                status_code=401,
            )

        key = auth_header[len("Bearer "):]
        agent_name = validate_key(key, self._keys)
        if agent_name is None:
            client_host = request.client.host if request.client else "unknown"
            log.warning(f"mcp_auth: invalid API key from {client_host}")
            return JSONResponse(
                {"error": "Forbidden", "detail": "Invalid API key"},
                status_code=403,
            )

        log.debug(f"mcp_auth: authenticated agent '{agent_name}' for {request.url.path}")
        return await call_next(request)
