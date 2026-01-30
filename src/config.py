from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    # src/config.py -> src -> repo root
    return Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Project settings loaded from environment variables and `.env`.

    Notes:
    - Uses `.env` by default (no need to call `load_dotenv()` elsewhere).
    - Field aliases keep compatibility with existing env vars like `OPENAI_API_KEY`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="RetailAI", validation_alias="APP_NAME")
    app_env: str = Field(default="dev", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    # Reliability
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES")
    timeout_seconds: float = Field(default=30.0, validation_alias="TIMEOUT_SECONDS")

    # Paths
    project_root: Path = Field(default_factory=_default_project_root)
    data_dir: Optional[Path] = Field(default=None, validation_alias="DATA_DIR")
    output_dir: Optional[Path] = Field(default=None, validation_alias="OUTPUT_DIR")
    db_path: Path = Field(default=Path("data/retail_sales.duckdb"), validation_alias="DB_PATH")

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5-nano", validation_alias="OPENAI_MODEL")
    openai_base_url: Optional[str] = Field(default=None, validation_alias="OPENAI_BASE_URL")

    # LangSmith (optional)
    langchain_tracing_v2: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="retail-insights-assistant", validation_alias="LANGCHAIN_PROJECT"
    )

    def model_post_init(self, __context) -> None:  # type: ignore[override]
        # Fill path defaults relative to project root (but allow env overrides).
        if self.data_dir is None:
            object.__setattr__(self, "data_dir", self.project_root / "data")
        else:
            object.__setattr__(self, "data_dir", Path(self.data_dir))

        if self.output_dir is None:
            object.__setattr__(
                self,
                "output_dir",
                self.project_root / "examples" / "example_outputs",
            )
        else:
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # Normalize db_path to be project-root relative when provided as a relative path
        if not self.db_path.is_absolute():
            object.__setattr__(self, "db_path", (self.project_root / self.db_path).resolve())

        # Normalize dirs
        object.__setattr__(self, "data_dir", Path(self.data_dir).resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).resolve())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached Settings singleton."""

    return Settings()
