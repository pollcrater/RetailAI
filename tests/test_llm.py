import json

import duckdb
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.utils.llm import generate_text

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def test_generate_text(prompt,model):

    mock_prompt = MagicMock()
    mock_model = MagicMock()

    cfg = MagicMock()
    client = MagicMock()
    settings = MagicMock()

    mock_response = client.responses.create(model=mock_model,input = mock_prompt)

    
    print(mock_response.output_text)
    return
