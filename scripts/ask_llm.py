from __future__ import annotations

import sys
from pathlib import Path

# Allow running as: `py scripts/ask_llm.py "your question"`
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.llm import generate_text  # noqa: E402


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "Say hi"
    print(generate_text(prompt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
