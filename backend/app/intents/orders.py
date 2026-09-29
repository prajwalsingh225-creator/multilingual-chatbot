"""Order intents. Uses a deterministic MOCK lookup; replace with a real order service."""

from app.intents.handlers import HandlerContext, HandlerResult
from app.intents.registry import registry

_STATUSES = ("processing", "shipped", "delivered")


def _mock_status(order_id: str) -> str:
    digits = "".join(ch for ch in order_id if ch.isdigit()) or "0"
    return _STATUSES[int(digits) % len(_STATUSES)]


@registry.register("track_order")
def handle_track_order(ctx: HandlerContext) -> HandlerResult:
    order_id = ctx.entities.get("order_id")
    if not order_id:
        return HandlerResult("ask_order_id", pending_intent="track_order")
    status = _mock_status(order_id)
    return HandlerResult(f"order_status_{status}", {"order_id": order_id})


@registry.register("cancel_order")
def handle_cancel_order(ctx: HandlerContext) -> HandlerResult:
    order_id = ctx.entities.get("order_id")
    if not order_id:
        return HandlerResult("ask_order_id", pending_intent="cancel_order")
    if _mock_status(order_id) == "processing":
        return HandlerResult("order_cancelled", {"order_id": order_id})
    return HandlerResult(
        "order_cannot_cancel", {"order_id": order_id}, suggestions=["suggest_support"]
    )
