from __future__ import annotations

import os
import ssl

import httpx
from dotenv import load_dotenv
from openai import OpenAI


def _build_http_client() -> httpx.Client:
  """Build an HTTP client that works in corporate networks (Zscaler, VPN, etc.).

  Prefers an explicit CA bundle via env vars, otherwise falls back to Windows
  system trust store via `truststore` (if installed).
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

  # trust_env=True lets httpx pick up HTTPS_PROXY / HTTP_PROXY automatically.
  return httpx.Client(verify=verify, timeout=60.0, trust_env=True)


def main() -> int:
  load_dotenv()

  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
    raise SystemExit("Missing OPENAI_API_KEY (set it in .env or environment)")

  client = OpenAI(api_key=api_key, http_client=_build_http_client())

  response = client.responses.create(
    model="gpt-5-nano",
    input="Hi, I'm Praveen",
  )

  print(response.output_text)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
