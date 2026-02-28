from __future__ import annotations

import logging

from src.config import Settings


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if settings.log_level.upper() != "DEBUG":
        logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
        logging.getLogger("azure.identity._internal.get_token_mixin").setLevel(logging.WARNING)
