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
from azure.identity import ClientSecretCredential
from langchain_openai import AzureChatOpenAI

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

def _get_api_key() -> str:
    """Resolve API key / bearer token for OpenAI or Azure OpenAI (Cassie).

    Priority:
    1) OPENAI_API_KEY (direct key/token)
    2) AZURE_OPENAI_API_KEY (direct key/token)
    3) Azure AD client credentials (AZURE_TENANT_ID/CLIENT_ID/CLIENT_SECRET) -> bearer token
    """

    settings = get_settings()

    # 1) Plain OpenAI-style key (or pre-generated bearer token)
    if settings.openai_api_key:
        return settings.openai_api_key

    # 2) Explicit Azure OpenAI key/token
    if settings.azure_openai_api_key:
        return settings.azure_openai_api_key

    # 3) Azure AD client credentials: obtain a bearer token
    if settings.azure_tenant_id and settings.azure_client_id and settings.azure_client_secret:
        credential = ClientSecretCredential(
            tenant_id=settings.azure_tenant_id,
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
        )
        # Standard Cognitive Services scope works for Azure OpenAI / Cassie gateway.
        # Allow override via TOKEN_ENDPOINT in .env if your gateway requires a custom scope.
        token_scope = (
            settings.token_endpoint
            or "https://cognitiveservices.azure.com/.default"
        )
        token = credential.get_token(token_scope)
        return token.token

    raise RuntimeError(
        "Missing OPENAI_API_KEY / AZURE_OPENAI_API_KEY or Azure client credentials "
        "(AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET). "
        "Set one of these in .env."
    )

def get_llm_config() -> LlmConfig:
    settings = get_settings()
    api_key = _get_api_key()
    model = settings.openai_model

    # Prefer explicit OpenAI base URL, else Azure endpoint if provided
    base_url = settings.openai_base_url or settings.azure_openai_endpoint

    return LlmConfig(api_key=api_key, model=model, base_url=base_url)


def _is_azure_mode() -> bool:
    settings = get_settings()
    return bool(settings.azure_openai_endpoint)


def _get_azure_client(deployment_override: str | None = None) -> AzureChatOpenAI:
    settings = get_settings()
    api_key = _get_api_key()

    deployment = deployment_override or (
        settings.azure_openai_deployment
        or settings.azure_chat_deployment
        or settings.openai_model
    )
    api_version = settings.azure_openai_api_version or settings.openai_api_version

    kwargs: dict[str, Any] = {
        "azure_endpoint": settings.azure_openai_endpoint,
        "azure_deployment": deployment,
        "model": deployment,
        "api_key": api_key,
        "temperature": 0,
        "http_client": build_http_client(),
    }
    if api_version:
        kwargs["api_version"] = api_version

    return AzureChatOpenAI(**kwargs)


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

    if _is_azure_mode():
        settings = get_settings()
        max_retries = max(0, int(settings.max_retries))
        for attempt in range(max_retries + 1):
            try:
                azure_client = _get_azure_client(model)
                result = azure_client.invoke(prompt)
                content = result.content
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            text = item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                        elif isinstance(item, str):
                            parts.append(item)
                    return "\n".join(parts).strip()
                return str(content)
            except Exception:
                if attempt >= max_retries:
                    raise
                sleep_s = min(8.0, (2 ** attempt)) + random.random() * 0.2
                time.sleep(sleep_s)

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
