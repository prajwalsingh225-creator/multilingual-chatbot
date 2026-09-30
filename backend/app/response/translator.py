"""Template-based translation with language fallback.

Replies are authored per language (data/translations). If a key is missing in the
target language we fall back to the default language. Plug a machine-translation
service in ``translate`` later if you need free-form text.
"""

from app.response.templates import TemplateStore


class _SafeDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class Translator:
    def __init__(self, store: TemplateStore, default_language: str = "en") -> None:
        self.store = store
        self.default_language = default_language

    def translate(self, key: str, language: str, **params: str) -> str:
        template = (
            self.store.get(language, key)
            or self.store.get(self.default_language, key)
            or key
        )
        return template.format_map(_SafeDict(params))
