"""Step 3: translate every bubble on the page in one LLM call.

Sending the whole page at once (in reading order) lets the model use context:
who is speaking, running jokes, and OCR mistakes it can correct from surrounding lines.

With CLAUDE_VISION on (the default), Claude also gets the page image and an image of
every bubble. OCR on manga lettering is unreliable, so Claude reads the text itself
and uses the OCR output only as a hint. It returns its own reading of the source
text along with the translation, which we show in the editor.
"""
import base64
import io
import json
import re

import numpy as np
from PIL import Image

from .. import config, pricing
from .models import TextRegion

STYLES = {
    "standard": "Natural, localized dialogue that reads like an official English release. "
                "Drop honorifics and adapt idioms.",
    "honorifics": "Natural, localized dialogue, but keep Japanese honorifics "
                  "(-san, -kun, -chan, -senpai, -sama) attached to names.",
    "faithful": "Scanlation style: stay close to the original wording, keep honorifics "
                "and original terms (senpai, onii-chan, ki, etc.), don't condense lines.",
}

SYSTEM_PROMPT = """You are a professional manga translator.
You receive one comic page: the bubbles in reading order, each with an OCR reading of its text{vision_note}.
Translate each bubble into {target} so it reads naturally as comic dialogue.

Style: {style}

Rules:
- Keep the meaning, tone and personality of each line. Casual speech stays casual.
- The OCR is often wrong (misread or missing characters). {read_rule}
- Use the whole page for context: who is speaking, what is happening, the series' terminology.
- Keep translations concise: they must fit back into the same bubble.
- If a bubble is already in {target}, keep it as is (fix obvious OCR errors only).
- Keep stammers and trailing dots ("Th-this is...", "Beast...!").
- Translate sound effects briefly ("BAM", "THUD").
- If a region is not text (a mis-detection), return an empty string for it.
{glossary}{context_rule}
Return ONLY a JSON object:
{{"translations": [{{"id": <id>, "source": "<the original text as you read it>", "text": "<translation>"}}, ...]}}"""


MAX_CONTEXT_LINES = 40      # most recent lines from earlier pages sent as context
MAX_BUBBLES_PER_CALL = 30   # long webtoon strips are translated in chunks of this many bubbles


def _is_strip(page: np.ndarray | None) -> bool:
    return page is not None and page.shape[0] > 2.5 * page.shape[1]


def _context_block(previous: list[dict]) -> str:
    lines = [p for p in previous if p.get("text")][-MAX_CONTEXT_LINES:]
    if not lines:
        return ""
    body = "\n".join(f"- {json.dumps(p.get('source', ''), ensure_ascii=False)} -> "
                     f"{json.dumps(p['text'], ensure_ascii=False)}" for p in lines)
    return f"Dialogue from the previous page(s) of this chapter, already translated (context only):\n{body}\n\n"


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Model did not return JSON: {text[:200]}")
    return json.loads(match.group(0))


def _png_b64(img: np.ndarray, max_side: int) -> str:
    pil = Image.fromarray(img)
    scale = max_side / max(pil.size)
    if scale < 1:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
    elif scale > 1 and max(pil.size) < 96:
        # Upscale tiny crops so the characters are legible
        f = min(3, 96 / max(pil.size))
        pil = pil.resize((int(pil.width * f), int(pil.height * f)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _image_block(img: np.ndarray, max_side: int) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(img, max_side)}}


def _crop(page: np.ndarray, box, pad: int = 8) -> np.ndarray:
    h, w = page.shape[:2]
    x1, y1, x2, y2 = box
    return page[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


def build_messages(regions: list[TextRegion], source_lang: str, target_lang: str,
                   page: np.ndarray | None = None, style: str = "standard", glossary: str = "",
                   previous: list[dict] | None = None):
    """Build (system, messages) for the Claude API."""
    source = config.LANG_NAMES_EN.get(source_lang, source_lang)
    target = config.LANG_NAMES_EN.get(target_lang, target_lang)
    vision = page is not None and config.CLAUDE_VISION
    glossary = glossary.strip()
    system = SYSTEM_PROMPT.format(
        target=target,
        style=STYLES.get(style, STYLES["standard"]),
        vision_note=" and an image of the bubble" if vision else "",
        read_rule=("Read the text from each bubble image yourself; use the OCR only as a hint."
                   if vision else "Use context to infer the intended text."),
        glossary=(f"- Glossary (always use these names/terms):\n{glossary}\n" if glossary else ""),
        context_rule=("- You'll also see the previous pages' dialogue. Stay consistent with it: same "
                      "character names, terms, and each character's way of speaking. Don't re-translate it.\n"
                      if previous else ""),
    )
    context = _context_block(previous or [])

    header = f"{context}Source language: {source}. {len(regions)} bubbles, in reading order."
    if not vision:
        bubbles = [{"id": r.id, "ocr": r.source_text} for r in regions]
        text = f"{header}\n{json.dumps(bubbles, ensure_ascii=False, indent=1)}"
        return system, [{"role": "user", "content": text}]

    if _is_strip(page):
        # Scaled down to fit, a long webtoon strip is an unreadable sliver: send only the bubbles
        content: list[dict] = [{"type": "text", "text": f"{header}\nThis is part of a long vertical "
                                "webtoon strip; the bubbles are in order from top to bottom."}]
    else:
        content = [{"type": "text", "text": f"{header}\nThe full page, for context:"},
                   _image_block(page, 1200)]
    for r in regions:
        hint = f" (OCR reading: {json.dumps(r.source_text, ensure_ascii=False)})" if r.source_text else ""
        content.append({"type": "text", "text": f"Bubble id={r.id}{hint}:"})
        content.append(_image_block(_crop(page, r.box), 512))
    content.append({"type": "text", "text": "Return the JSON now."})
    return system, [{"role": "user", "content": content}]


def translate_claude(regions, source_lang, target_lang, page=None, style="standard", glossary="",
                     previous=None, quality=pricing.DEFAULT_QUALITY, usage=None):
    """One Claude call. `quality` picks the model; `usage` (a dict) collects tokens and cost."""
    import anthropic

    model = pricing.QUALITIES.get(quality, pricing.QUALITIES[pricing.DEFAULT_QUALITY])["model"]
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system, messages = build_messages(regions, source_lang, target_lang, page, style, glossary, previous)
    resp = client.messages.create(
        model=model,
        max_tokens=8000,                       # room for a very wordy page's JSON
        thinking=pricing.thinking_off(model),  # translation needs little reasoning: save the tokens
        system=system,
        messages=messages,
    )
    if usage is not None and resp.usage is not None:
        pricing.add_usage(usage, model, {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_creation_input_tokens": getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        })
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Claude's reply was cut off (too many bubbles on one page)")
    text = "".join(block.text for block in resp.content if block.type == "text")
    return parse_response(text)


def parse_response(text: str) -> dict[int, dict]:
    """{id: {"source": ..., "text": ...}} from the model's JSON reply."""
    data = _extract_json(text)
    return {int(t["id"]): {"source": str(t.get("source", "")), "text": str(t.get("text", ""))}
            for t in data.get("translations", [])}


def translate_mock(regions, source_lang, target_lang, page=None, style="standard", glossary="",
                   previous=None, quality=pricing.DEFAULT_QUALITY, usage=None):
    """Placeholder so the app runs without an API key."""
    return {r.id: {"source": "", "text": f"Mock translation for bubble {r.id + 1}."}
            for r in regions}


def translate_regions(regions: list[TextRegion], source_lang: str, target_lang: str,
                      page: np.ndarray | None = None, style: str = "standard",
                      glossary: str = "", previous: list[dict] | None = None,
                      quality: str = pricing.DEFAULT_QUALITY, usage: dict | None = None) -> list[TextRegion]:
    """previous: dialogue from earlier pages [{"source", "text"}], used as context.
    quality: "economy" or "best" (see pricing.py). usage: dict that collects tokens and cost."""
    if not regions:
        return regions
    fn = translate_claude if config.TRANSLATOR == "claude" else translate_mock
    # Usually one call per page. Very long pages (webtoon strips) go in chunks, each chunk
    # seeing the dialogue of the chunks before it.
    result: dict[int, dict] = {}
    context = list(previous or [])
    for start in range(0, len(regions), MAX_BUBBLES_PER_CALL):
        chunk = regions[start:start + MAX_BUBBLES_PER_CALL]
        part = fn(chunk, source_lang, target_lang, page, style, glossary, context,
                  quality=quality, usage=usage)
        result.update(part)
        context += [part[r.id] for r in chunk if r.id in part]
    for r in regions:
        item = result.get(r.id, {})
        r.translated_text = item.get("text", "")
        if item.get("source"):  # Claude's reading of the image beats the OCR guess
            r.source_text = item["source"]
    return regions
