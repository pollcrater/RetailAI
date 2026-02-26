from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from azure.identity import ClientSecretCredential

# Allow running as: `py scripts/test_azure_openai_langchain.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    load_dotenv()

    # Choose auth method: API key (simple) or Azure AD client credentials.
    tenant_id = (os.getenv("AZURE_TENANT_ID") or "").strip()
    client_id = (os.getenv("AZURE_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("AZURE_CLIENT_SECRET") or "").strip()
    credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)
    os.environ['AZURE_OPENAI_API_KEY']= credential.get_token('https://cognitiveservices.azure.com/.default').token

        
    llm = AzureChatOpenAI(
    azure_deployment = 'gpt4o',
    model='gpt4o'
    )

    res = llm.invoke("hi")
    return res


if __name__ == "__main__":
    raise SystemExit(main())




