"""Quality levels and what each Claude call costs.

Two quality levels:
- economy: Claude Haiku 5.5. About 20x cheaper per token, slightly less skilled.
- best:    Claude Sonnet 5.5. Best at reading stylized lettering and handling tone/wordplay.

Both run with up-front thinking turned off: translating a page needs little reasoning,
and thinking tokens are billed as output (the most expensive kind of token).

Prices are list prices in USD per million tokens (platform.claude.com/docs/en/about-claude/pricing).
Update PRICES if Anthropic changes them; the app only uses them to *estimate* cost from the
token counts Claude reports. Your Claude Console's Usage page has the exact bill.
"""
import os

QUALITIES = {
    "economy": {"label": "Economy (Claude Haiku 5.5)",
                "model": os.getenv("ECONOMY_MODEL", "claude-haiku-5-5")},
    "best": {"label": "Best quality (Claude Sonnet 5.5)",
             "model": os.getenv("BEST_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"))},
}
DEFAULT_QUALITY = os.getenv("DEFAULT_QUALITY", "economy")

# USD per million tokens: input, output, cache write (5 min), cache read
PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01,
                         # prompts over 100k tokens cost more on Haiku 5.5
                         "long": {"threshold": 100_000, "input": 0.50, "output": 2.50,
                                  "cache_write": 0.625, "cache_read": 0.05}},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
}


def thinking_off(model: str) -> dict:
    """The request setting that turns off up-front thinking for this model.

    Sonnet 5.5 rejects {"type": "disabled"}; its lowest setting is "between_tools".
    """
    if model.startswith("claude-sonnet-5-5"):
        return {"type": "between_tools"}
    return {"type": "disabled"}


def empty_usage() -> dict:
    return {"model": "", "input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "cost_usd": 0.0}


def cost_usd(model: str, usage: dict) -> float:
    """Estimated cost of one request from the usage numbers Claude returned."""
    price = PRICES.get(model)
    if price is None:
        return 0.0
    prompt = (usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
              + usage.get("cache_read_input_tokens", 0))
    long = price.get("long")
    if long and prompt > long["threshold"]:
        price = long
    return (usage.get("input_tokens", 0) * price["input"]
            + usage.get("output_tokens", 0) * price["output"]
            + usage.get("cache_creation_input_tokens", 0) * price["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * price["cache_read"]) / 1_000_000


def add_usage(total: dict, model: str, usage: dict) -> dict:
    """Add one request's usage (and its cost) into a running total, in place."""
    total["model"] = model
    for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
        total[key] = total.get(key, 0) + (usage.get(key) or 0)
    total["cost_usd"] = total.get("cost_usd", 0.0) + cost_usd(model, usage)
    return total
