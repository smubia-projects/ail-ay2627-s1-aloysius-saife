"""Project adapter over the shared AI Lodge rate-limit engine.

``rate_limit_engine.py`` is vendored unmodified from the club's shared engine. This adapter
supplies the project slug and its buckets; the engine handles X-Forwarded-For, central
limits, event mode, the kill switch (503) and fail-open. Without Upstash keys it is off.

One hit = one run, not one model call: a run makes several calls (split, judges,
resamples, repair), all bounded by the preset's budgets.
"""

from __future__ import annotations

import os

from fastapi import Request

from saife.server.rate_limit_engine import RateLimiter


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


limiter = RateLimiter(
    # Slug = full project folder name (matches the central control sheet rows).
    project="ail-ay2627-s1-aloysius-saife",
    buckets={
        "check": _env_int("RATE_LIMIT_CHECK_MAX", 3),  # check an answer you paste
        "ask": _env_int("RATE_LIMIT_ASK_MAX", 3),  # generate an answer, then check it
    },
    redis_url=os.getenv("UPSTASH_REDIS_REST_URL") or os.getenv("UPSTASH_REDIS_URL") or "",
    redis_token=os.getenv("UPSTASH_REDIS_REST_TOKEN") or os.getenv("UPSTASH_REDIS_TOKEN") or "",
    default_window=_env_int("RATE_LIMIT_WINDOW_SECONDS", 432000),
)


def check_rate_limit(request: Request, bucket: str) -> None:
    limiter.enforce(request, bucket)
