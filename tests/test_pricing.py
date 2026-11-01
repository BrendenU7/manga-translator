"""Quality levels, thinking settings and cost estimates."""
import types

import numpy as np
import pytest

from backend.app import config, pricing
from backend.app.pipeline import translate
from backend.app.pipeline.models import TextRegion


def test_cost_math():
    # Haiku 5.5: $0.10 in / $0.50 out per million tokens
    assert pricing.cost_usd("claude-haiku-5-5", {"input_tokens": 3000, "output_tokens": 600}) == pytest.approx(0.0006)
    # Sonnet 5.5: $2 in / $10 out
    assert pricing.cost_usd("claude-sonnet-5-5", {"input_tokens": 3000, "output_tokens": 600}) == pytest.approx(0.012)
    # Haiku prompts over 100k tokens use the higher price
    assert pricing.cost_usd("claude-haiku-5-5", {"input_tokens": 200_000}) == pytest.approx(0.1)
    assert pricing.cost_usd("unknown-model", {"input_tokens": 10}) == 0.0


def test_thinking_off_setting_per_model():
    assert pricing.thinking_off("claude-sonnet-5-5") == {"type": "between_tools"}
    assert pricing.thinking_off("claude-haiku-5-5") == {"type": "disabled"}


class FakeClient:
    """Stands in for anthropic.Anthropic and records the request."""
    calls = []

    def __init__(self, api_key=None):
        self.messages = self

    def create(self, **kwargs):
        FakeClient.calls.append(kwargs)
        ids = [int(x) for x in __import__("re").findall(r"id=(\d+)", str(kwargs["messages"]))]
        text = '{"translations": [' + ",".join(f'{{"id": {i}, "text": "hi"}}' for i in ids) + "]}"
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
            usage=types.SimpleNamespace(input_tokens=3000, output_tokens=600,
                                        cache_creation_input_tokens=0, cache_read_input_tokens=0))


@pytest.fixture
def fake_anthropic(monkeypatch):
    import anthropic
    FakeClient.calls = []
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(config, "TRANSLATOR", "claude")
    monkeypatch.setattr(config, "CLAUDE_VISION", True)
    return FakeClient


@pytest.mark.parametrize("quality,model,thinking,cost", [
    ("economy", "claude-haiku-5-5", "disabled", 0.0006),
    ("best", "claude-sonnet-5-5", "between_tools", 0.012),
])
def test_quality_picks_model_turns_off_thinking_and_counts_cost(fake_anthropic, quality, model, thinking, cost):
    page = np.full((1200, 850, 3), 255, np.uint8)
    regions = [TextRegion(id=0, box=(10, 10, 100, 100), source_text="x")]
    usage = pricing.empty_usage()
    translate.translate_regions(regions, "ja", "en", page=page, quality=quality, usage=usage)
    call = fake_anthropic.calls[0]
    assert call["model"] == model
    assert call["thinking"] == {"type": thinking}
    assert call["max_tokens"] >= 8000
    assert usage["model"] == model and usage["input_tokens"] == 3000
    assert usage["cost_usd"] == pytest.approx(cost)
    assert regions[0].translated_text == "hi"


def test_chunked_strip_adds_up_usage(fake_anthropic):
    regions = [TextRegion(id=i, box=(0, 0, 1, 1), source_text="x") for i in range(65)]
    usage = pricing.empty_usage()
    translate.translate_regions(regions, "ko", "en", quality="economy", usage=usage)
    assert len(fake_anthropic.calls) == 3
    assert usage["input_tokens"] == 9000
    assert usage["cost_usd"] == pytest.approx(3 * 0.0006)


def test_cut_off_reply_is_an_error(fake_anthropic, monkeypatch):
    orig = FakeClient.create
    def truncated(self, **kw):
        r = orig(self, **kw); r.stop_reason = "max_tokens"; return r
    monkeypatch.setattr(FakeClient, "create", truncated)
    with pytest.raises(RuntimeError, match="cut off"):
        translate.translate_regions([TextRegion(id=0, box=(0, 0, 1, 1), source_text="x")], "ja", "en")
