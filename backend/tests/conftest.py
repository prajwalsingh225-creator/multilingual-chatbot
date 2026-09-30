"""Pytest fixtures.

The environment is configured *before* any ``app`` module is imported, because
``app.core.config.settings`` is a module-level singleton built at import time.
That points the app at a throwaway SQLite file and a non-existent MODEL_DIR so
tests never touch the real database and never try to load a trained model.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# --- environment first, app imports second -----------------------------------
_TEST_DIR = tempfile.mkdtemp(prefix="chatbot-tests-")
_TEST_DB = Path(_TEST_DIR) / "test_chatbot.db"
os.environ["APP_ENV"] = "testing"
os.environ["DEBUG"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"
os.environ["MODEL_DIR"] = str(Path(_TEST_DIR) / "no-such-model")
os.environ["DEFAULT_LANGUAGE"] = "en"
os.environ["SUPPORTED_LANGUAGES"] = "en,hi,es"

from fastapi.testclient import TestClient  # noqa: E402

from app.api.schemas.chat import ChatResponse  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.nlp.entity_extractor import EntityExtractor  # noqa: E402
from app.nlp.intent_classifier import IntentClassifier  # noqa: E402
from app.nlp.language_detector import LanguageDetector  # noqa: E402
from app.nlp.preprocessor import Preprocessor  # noqa: E402


@pytest.fixture(scope="session")
def settings_obj() -> settings:
    """The configured Settings singleton."""
    return settings


@pytest.fixture
def app():
    """A fresh app instance (built, but not yet started)."""
    return create_app()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    """TestClient used as a context manager so the FastAPI lifespan runs."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def safe_client(app) -> Iterator[TestClient]:
    """TestClient that returns 500 responses instead of re-raising them."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def classifier() -> IntentClassifier:
    """Rule-based intent classifier (no transformer in tests)."""
    return IntentClassifier.from_intents_file(settings.intents_file, 0.55, None)


@pytest.fixture(scope="session")
def detector() -> LanguageDetector:
    return LanguageDetector(settings.SUPPORTED_LANGUAGES, settings.DEFAULT_LANGUAGE)


@pytest.fixture(scope="session")
def pre() -> Preprocessor:
    return Preprocessor()


@pytest.fixture(scope="session")
def extractor() -> EntityExtractor:
    return EntityExtractor()


@pytest.fixture(scope="session")
def chat_schema() -> type[ChatResponse]:
    return ChatResponse


def _cleanup() -> None:
    import shutil

    shutil.rmtree(_TEST_DIR, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def _delete_test_db() -> Iterator[None]:
    """Remove the temporary SQLite file once the session finishes."""
    yield
    _cleanup()