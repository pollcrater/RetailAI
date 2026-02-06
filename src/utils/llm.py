from __future__ import annotations

import os
import ssl
import time
import random
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIStatusError, RateLimitError

from src.config import get_settings


@dataclass(frozen=True)
class LlmConfig:
    """Minimal configuration for calling an LLM via the OpenAI Python SDK."""

    api_key: str
    model: str = "gpt-5-nano"
    base_url: str | None = None


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def build_http_client() -> httpx.Client:
    """HTTP client that plays well with corporate networks.

    - Uses a CA bundle if you set `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE`
    - Otherwise, on Windows, attempts to use the OS trust store via `truststore`
    - Respects `HTTPS_PROXY`/`HTTP_PROXY` when `trust_env=True`

    This avoids insecure workarounds like disabling TLS verification.
    """

    ca_bundle = (
        os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
        or os.getenv("CURL_CA_BUNDLE")
    )

    verify: str | ssl.SSLContext | bool
    if ca_bundle:
        verify = ca_bundle
    else:
        try:
            import truststore  # type: ignore

            verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except Exception:
            verify = True

    return httpx.Client(verify=verify, timeout=60.0, trust_env=True)


def get_llm_config() -> LlmConfig:
    settings = get_settings()
    api_key = settings.openai_api_key
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (set it in .env or environment)")

    model = settings.openai_model
    base_url = settings.openai_base_url

    return LlmConfig(api_key=api_key, model=model, base_url=base_url)


def get_openai_client(config: LlmConfig | None = None) -> OpenAI:
    """Returns an `OpenAI` client wired with the corporate-safe httpx client."""

    cfg = config or get_llm_config()
    settings = get_settings()

    kwargs: dict[str, Any] = {
        "api_key": cfg.api_key,
        "http_client": build_http_client(),
        "max_retries": settings.max_retries,
    }
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url

    return OpenAI(**kwargs)


def generate_text(
    prompt: str,
    *,
    model: str | None = None,
) -> str:
    """Convenience wrapper: returns plain text for a single prompt."""

    cfg = get_llm_config()
    client = get_openai_client(cfg)
    settings = get_settings()

    max_retries = max(0, int(settings.max_retries))
    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model or cfg.model,
                input=prompt,
            )
            return response.output_text
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as e:
            should_retry = True
            if isinstance(e, APIStatusError) and e.status_code is not None:
                should_retry = e.status_code >= 500 or e.status_code == 429
            if not should_retry or attempt >= max_retries:
                raise
            sleep_s = min(8.0, (2 ** attempt)) + random.random() * 0.2
            time.sleep(sleep_s)
        except Exception:
            if attempt >= max_retries:
                raise
            sleep_s = min(8.0, (2 ** attempt)) + random.random() * 0.2
            time.sleep(sleep_s)

    raise RuntimeError("LLM call failed after retries")
