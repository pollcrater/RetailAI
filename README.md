# RetailAI
RetailAI is a GenAI-powered retail insights assistant that enables conversational analytics and automated business summaries over structured sales data using a multi-agent architecture.

## Quickstart

1) Create and activate a venv

- PowerShell:
	- `py -m venv venv`
	- `venv\Scripts\Activate.ps1`

2) Install deps

- `pip install -r requirements.txt`

3) Configure environment

- Copy `.env.example` to `.env` and set `OPENAI_API_KEY`
- Optional (behind Zscaler/VPN): set one of `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`

4) Test your LLM connectivity

- `py scripts\test_openai.py`

5) Use the shared LLM helper

- Implementation: [src/utils/llm.py](src/utils/llm.py)
- Example:
	- `py scripts\ask_llm.py "Summarize what this project does"`
