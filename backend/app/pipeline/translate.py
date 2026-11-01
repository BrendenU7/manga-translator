"""Step 3: translate every bubble on the page in one LLM call.

Sending the whole page at once (in reading order) lets the model use context:
who is speaking, running jokes, and OCR mistakes it can correct from surrounding lines.
"""
import json
import re

from .. import config
from .models import TextRegion

SYSTEM_PROMPT = """You are a professional manga translator and typesetter's assistant.
You receive OCR'd text from one comic page as a JSON list of speech bubbles in reading order.
Translate each bubble into {target} so it reads naturally as comic dialogue.

Rules:
- Keep the meaning, tone and personality of each line. Casual speech stays casual.
- The OCR may contain errors. Use context to infer the intended text.
- Keep names in their usual romanized form. Keep honorifics (-san, -kun, -senpai) when translating Japanese to English.
- Translate sound effects briefly (e.g. "BAM", "THUD"), or keep them short if untranslatable.
- Keep translations concise: they must fit back into the same bubble.
- If a bubble is unreadable noise, return an empty string for it.

Return ONLY a JSON object: {{"translations": [{{"id": <id>, "text": "<translation>"}}, ...]}}"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


def build_messages(regions: list[TextRegion], source_lang: str, target_lang: str):
    source = config.SOURCE_LANGS.get(source_lang, source_lang)
    target = config.TARGET_LANGS.get(target_lang, target_lang)
    bubbles = [{"id": r.id, "text": r.source_text} for r in regions]
    user = f"Source language: {source}\nBubbles:\n{json.dumps(bubbles, ensure_ascii=False, indent=1)}"
    return SYSTEM_PROMPT.format(target=target), [{"role": "user", "content": user}]


def translate_claude(regions: list[TextRegion], source_lang: str, target_lang: str) -> dict[int, str]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system, messages = build_messages(regions, source_lang, target_lang)
    resp = client.messages.create(model=config.CLAUDE_MODEL, max_tokens=4000,
                                  system=system, messages=messages)
    text = "".join(block.text for block in resp.content if block.type == "text")
    data = _extract_json(text)
    return {int(t["id"]): str(t.get("text", "")) for t in data.get("translations", [])}


def translate_mock(regions: list[TextRegion], source_lang: str, target_lang: str) -> dict[int, str]:
    """Placeholder so the app runs without an API key."""
    return {r.id: f"Mock translation for bubble {r.id + 1}. Add an API key for real text."
            for r in regions}


def translate_regions(regions: list[TextRegion], source_lang: str, target_lang: str) -> list[TextRegion]:
    if not regions:
        return regions
    fn = translate_claude if config.TRANSLATOR == "claude" else translate_mock
    result = fn(regions, source_lang, target_lang)
    for r in regions:
        r.translated_text = result.get(r.id, "")
    return regions
