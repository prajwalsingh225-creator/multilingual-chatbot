"""Application configuration.

All settings come from environment variables or ``backend/.env`` and are exposed
through one cached object::

    from app.core.config import settings
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # .../backend


def _split_csv(value: object) -> object:
    """Accept 'en,hi,es', '["en","hi"]' or a real list."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            import json

            return json.loads(text)
        return [item.strip() for item in text.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- application ---
    APP_NAME: str = "Multilingual Context-Aware Chatbot"
    APP_ENV: Literal["development", "testing", "production"] = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    # --- database ---
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'chatbot.db'}"

    # --- model / NLP ---
    MODEL_NAME: str = "xlm-roberta-base"  # base checkpoint used for fine-tuning
    MODEL_DIR: Path = BASE_DIR / "trained_models" / "intent_model"  # fine-tuned model
    SUPPORTED_LANGUAGES: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["en", "hi", "es"]
    )
    DEFAULT_LANGUAGE: str = "en"
    INTENT_CONFIDENCE_THRESHOLD: float = Field(default=0.55, ge=0.0, le=1.0)

    # --- conversation ---
    MAX_CONTEXT_MESSAGES: int = Field(default=10, ge=1)
    SESSION_TIMEOUT_MINUTES: int = Field(default=30, ge=1)

    # --- business info used in replies (placeholders) ---
    SUPPORT_EMAIL: str = "support@example.com"
    SUPPORT_PHONE: str = "+91-1800-000-000"

    # --- data paths ---
    DATA_DIR: Path = BASE_DIR / "data"

    @field_validator("SUPPORTED_LANGUAGES", "CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_lists(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("SUPPORTED_LANGUAGES")
    @classmethod
    def _normalize_languages(cls, value: list[str]) -> list[str]:
        cleaned = [lang.strip().lower() for lang in value if lang.strip()]
        if not cleaned:
            raise ValueError("SUPPORTED_LANGUAGES must contain at least one language code")
        return list(dict.fromkeys(cleaned))  # de-duplicate, keep order

    @field_validator("MODEL_DIR", "DATA_DIR", mode="after")
    @classmethod
    def _resolve_paths(cls, value: Path) -> Path:
        return value if value.is_absolute() else (BASE_DIR / value).resolve()

    @model_validator(mode="after")
    def _check_default_language(self) -> "Settings":
        self.DEFAULT_LANGUAGE = self.DEFAULT_LANGUAGE.strip().lower()
        if self.DEFAULT_LANGUAGE not in self.SUPPORTED_LANGUAGES:
            raise ValueError("DEFAULT_LANGUAGE must be one of SUPPORTED_LANGUAGES")
        return self

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def translations_dir(self) -> Path:
        return self.DATA_DIR / "translations"

    @property
    def intents_file(self) -> Path:
        return self.DATA_DIR / "raw" / "intents.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
