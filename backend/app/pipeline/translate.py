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

from .. import config
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
{glossary}
Return ONLY a JSON object:
{{"translations": [{{"id": <id>, "source": "<the original text as you read it>", "text": "<translation>"}}, ...]}}"""


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
                   page: np.ndarray | None = None, style: str = "standard", glossary: str = ""):
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
    )

    header = f"Source language: {source}. {len(regions)} bubbles, in reading order."
    if not vision:
        bubbles = [{"id": r.id, "ocr": r.source_text} for r in regions]
        text = f"{header}\n{json.dumps(bubbles, ensure_ascii=False, indent=1)}"
        return system, [{"role": "user", "content": text}]

    content: list[dict] = [{"type": "text", "text": f"{header}\nThe full page, for context:"},
                           _image_block(page, 1200)]
    for r in regions:
        hint = f" (OCR reading: {json.dumps(r.source_text, ensure_ascii=False)})" if r.source_text else ""
        content.append({"type": "text", "text": f"Bubble id={r.id}{hint}:"})
        content.append(_image_block(_crop(page, r.box), 512))
    content.append({"type": "text", "text": "Return the JSON now."})
    return system, [{"role": "user", "content": content}]


def translate_claude(regions, source_lang, target_lang, page=None, style="standard", glossary=""):
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system, messages = build_messages(regions, source_lang, target_lang, page, style, glossary)
    resp = client.messages.create(model=config.CLAUDE_MODEL, max_tokens=4000,
                                  system=system, messages=messages)
    text = "".join(block.text for block in resp.content if block.type == "text")
    return parse_response(text)


def parse_response(text: str) -> dict[int, dict]:
    """{id: {"source": ..., "text": ...}} from the model's JSON reply."""
    data = _extract_json(text)
    return {int(t["id"]): {"source": str(t.get("source", "")), "text": str(t.get("text", ""))}
            for t in data.get("translations", [])}


def translate_mock(regions, source_lang, target_lang, page=None, style="standard", glossary=""):
    """Placeholder so the app runs without an API key."""
    return {r.id: {"source": "", "text": f"Mock translation for bubble {r.id + 1}."}
            for r in regions}


def translate_regions(regions: list[TextRegion], source_lang: str, target_lang: str,
                      page: np.ndarray | None = None, style: str = "standard",
                      glossary: str = "") -> list[TextRegion]:
    if not regions:
        return regions
    fn = translate_claude if config.TRANSLATOR == "claude" else translate_mock
    result = fn(regions, source_lang, target_lang, page, style, glossary)
    for r in regions:
        item = result.get(r.id, {})
        r.translated_text = item.get("text", "")
        if item.get("source"):  # Claude's reading of the image beats the OCR guess
            r.source_text = item["source"]
    return regions
