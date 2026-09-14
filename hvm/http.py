"""Shared HTTP helpers, with proxies disabled for loopback addresses.

`requests` honours HTTP_PROXY / ALL_PROXY from the environment. If those are set
and NO_PROXY does not happen to list 127.0.0.1, every call to a local model
server is routed through a proxy that cannot reach it — and the failure reads as
"connection refused" or a hang, which looks exactly like a server that never
started. Corporate laptops and container shells both do this routinely.

The model server is on loopback by definition here, so a proxy is never wanted.
"""
from __future__ import annotations

from urllib.parse import urlparse

LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def is_loopback(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() in LOOPBACK


def proxies_for(url: str) -> dict | None:
    """Explicitly bypass any configured proxy for a loopback address."""
    return {"http": None, "https": None} if is_loopback(url) else None
