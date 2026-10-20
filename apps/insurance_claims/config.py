from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = PACKAGE_DIR / "fixtures"
PROJECT_ROOT = PACKAGE_DIR.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str = ""
    openrouter_model: str = "z-ai/glm-5.3-flashx"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    fixtures_dir: Path = FIXTURES_DIR


@lru_cache
def get_settings() -> Settings:
    return Settings()
