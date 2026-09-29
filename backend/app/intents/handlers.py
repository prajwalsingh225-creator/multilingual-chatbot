"""Shared handler types and the dispatch entry point.

Handlers return a *template key* plus params; the response layer turns that into text
in the user's language. Business logic below is placeholder/mock - swap in real
order/payment services later.
"""

from dataclasses import dataclass, field

from app.intents.registry import registry


@dataclass(frozen=True)
class HandlerContext:
    intent: str
    entities: dict[str, str]
    language: str
    session_id: str
    email: str = ""
    phone: str = ""


@dataclass
class HandlerResult:
    template_key: str
    params: dict[str, str] = field(default_factory=dict)
    pending_intent: str | None = None  # set when a required slot is missing
    suggestions: list[str] = field(default_factory=list)  # template keys for quick replies


def execute_intent(ctx: HandlerContext) -> HandlerResult:
    return registry.get(ctx.intent)(ctx)
