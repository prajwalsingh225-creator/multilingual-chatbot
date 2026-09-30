"""Turns a HandlerResult into user-facing text in the right language."""

from app.intents.handlers import HandlerResult
from app.response.translator import Translator


class ResponseManager:
    def __init__(self, translator: Translator) -> None:
        self.translator = translator

    def build_reply(self, result: HandlerResult, language: str) -> str:
        return self.translator.translate(result.template_key, language, **result.params)

    def build_suggestions(self, result: HandlerResult, language: str) -> list[str]:
        return [self.translator.translate(key, language) for key in result.suggestions]
