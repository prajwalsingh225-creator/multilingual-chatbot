"""Loads per-language response templates from data/translations/<lang>.json."""

import json
from pathlib import Path


class TemplateStore:
    def __init__(self, translations_dir: Path) -> None:
        self._templates: dict[str, dict[str, str]] = {}
        for path in sorted(translations_dir.glob("*.json")):
            self._templates[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    @property
    def languages(self) -> list[str]:
        return sorted(self._templates)

    def get(self, language: str, key: str) -> str | None:
        return self._templates.get(language, {}).get(key)
