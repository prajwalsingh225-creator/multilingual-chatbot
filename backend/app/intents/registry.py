"""Maps intent names to handler functions."""

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with handlers.py
    from app.intents.handlers import HandlerContext, HandlerResult

    Handler = Callable[[HandlerContext], HandlerResult]

_DEFAULT_MODULES = (
    "app.intents.greeting",
    "app.intents.orders",
    "app.intents.payments",
    "app.intents.support",
    "app.intents.fallback",
)


class IntentRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {}
        self._loaded = False

    def register(self, *intents: str) -> Callable[["Handler"], "Handler"]:
        def decorator(func: "Handler") -> "Handler":
            for name in intents:
                self._handlers[name] = func
            return func

        return decorator

    def load_default_handlers(self) -> None:
        if not self._loaded:
            for module in _DEFAULT_MODULES:
                importlib.import_module(module)
            self._loaded = True

    def get(self, intent: str) -> "Handler":
        self.load_default_handlers()
        return self._handlers.get(intent) or self._handlers["fallback"]

    def intents(self) -> list[str]:
        self.load_default_handlers()
        return sorted(self._handlers)


registry = IntentRegistry()
