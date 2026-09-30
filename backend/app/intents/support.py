from app.intents.handlers import HandlerContext, HandlerResult
from app.intents.registry import registry


@registry.register("contact_support")
def handle_contact_support(ctx: HandlerContext) -> HandlerResult:
    return HandlerResult("contact_support", {"email": ctx.email, "phone": ctx.phone})
