from app.intents.handlers import HandlerContext, HandlerResult
from app.intents.registry import registry


@registry.register("payment_issue")
def handle_payment_issue(ctx: HandlerContext) -> HandlerResult:
    return HandlerResult("payment_issue", suggestions=["suggest_support"])


@registry.register("refund")
def handle_refund(ctx: HandlerContext) -> HandlerResult:
    order_id = ctx.entities.get("order_id")
    if not order_id:
        return HandlerResult("ask_order_id", pending_intent="refund")
    return HandlerResult("refund_info", {"order_id": order_id})
