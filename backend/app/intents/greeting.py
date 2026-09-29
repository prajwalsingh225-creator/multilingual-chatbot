from app.intents.handlers import HandlerContext, HandlerResult
from app.intents.registry import registry


@registry.register("greeting")
def handle_greeting(ctx: HandlerContext) -> HandlerResult:
    return HandlerResult(
        "greeting", suggestions=["suggest_track", "suggest_refund", "suggest_support"]
    )


@registry.register("goodbye")
def handle_goodbye(ctx: HandlerContext) -> HandlerResult:
    return HandlerResult("goodbye")
