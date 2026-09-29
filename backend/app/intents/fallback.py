from app.intents.handlers import HandlerContext, HandlerResult
from app.intents.registry import registry


@registry.register("fallback")
def handle_fallback(ctx: HandlerContext) -> HandlerResult:
    return HandlerResult(
        "fallback", suggestions=["suggest_track", "suggest_refund", "suggest_support"]
    )
