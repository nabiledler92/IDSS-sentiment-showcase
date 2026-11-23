"""Keeps the public AI endpoints from being used as a free LLM proxy.

1. Signed stats: /api/batch returns its aggregated statistics with an HMAC
   signature. /api/brief and /api/ask only accept statistics carrying a valid
   signature, so visitors can't feed the model arbitrary prompts as "data".
   The key comes from IDSS_SIGNING_SECRET, or failing that is derived from
   HF_TOKEN, so the site needs no extra secret to work.
2. Rate limit: a per-visitor sliding window, kept in memory per function
   instance. Fluid compute reuses instances, so this catches most bursts, but
   it isn't global; pair it with a Vercel Firewall rate-limit rule for /api/brief
   and /api/ask for a hard limit.
"""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import defaultdict, deque

_fallback_key = secrets.token_bytes(32)


def _key() -> bytes:
    explicit = os.environ.get("IDSS_SIGNING_SECRET", "").strip()
    if explicit:
        return explicit.encode()
    hf = os.environ.get("HF_TOKEN", "").strip()
    if hf:
        return hashlib.sha256(b"idss-stats-signing:" + hf.encode()).digest()
    return _fallback_key  # local development without a token


def _canonical(stats: dict) -> bytes:
    return json.dumps(stats, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign(stats: dict) -> str:
    return hmac.new(_key(), _canonical(stats), hashlib.sha256).hexdigest()


def verify(stats: dict, signature: str) -> bool:
    return hmac.compare_digest(sign(stats), signature or "")


AI_REQUESTS_PER_WINDOW = 8
WINDOW_SECONDS = 10 * 60
_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def allow(visitor: str) -> tuple[bool, int]:
    """Records an AI request for `visitor`. Returns (allowed, retry_after_s)."""
    now = time.monotonic()
    with _lock:
        q = _hits[visitor]
        while q and now - q[0] > WINDOW_SECONDS:
            q.popleft()
        if len(q) >= AI_REQUESTS_PER_WINDOW:
            return False, int(WINDOW_SECONDS - (now - q[0])) + 1
        q.append(now)
        if len(_hits) > 10_000:  # bound memory on a long-lived instance
            for k in [k for k, v in _hits.items() if not v]:
                del _hits[k]
        return True, 0


def visitor_id(headers) -> str:
    fwd = headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or headers.get("x-real-ip", "") or "unknown"
    return ip
