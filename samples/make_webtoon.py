"""Generate a synthetic long webtoon strip (no copyrighted art needed).

Usage: python samples/make_webtoon.py  -> writes samples/sample_webtoon.png (800 x ~4400)
Korean dialogue in horizontal bubbles, panels separated by white gutters, like a manhwa.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
OUT = Path(__file__).with_name("sample_webtoon.png")

PANELS = [  # (panel height, background shade, [(bubble center x, center y, text lines)])
    (900, 235, [(260, 220, ["오늘 날씨", "정말 좋다!"]), (560, 600, ["그러게,", "산책 갈래?"])]),
    (1100, 90, [(400, 300, ["잠깐만…"]), (300, 800, ["저기 누가", "있어!"])]),
    (1000, 200, [(520, 260, ["설마…", "그 사람이야?"]), (260, 700, ["빨리 가자!"])]),
    (900, 160, [(400, 450, ["끝."])]),
]
GUTTER = 120  # white space between panels


def main():
    W = 800
    H = sum(h for h, _, _ in PANELS) + GUTTER * (len(PANELS) + 1)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 38, index=2)  # index 2 = Korean face of the collection

    y = GUTTER
    for ph, shade, bubbles in PANELS:
        d.rectangle((40, y, W - 40, y + ph), fill=(shade, shade, shade + 10 if shade < 245 else shade),
                    outline="black", width=4)
        # some "artwork": diagonal lines
        for k in range(0, W, 60):
            d.line((40 + k, y + ph - 200, 40 + k + 120, y + ph - 20), fill=(shade // 2,) * 3, width=3)
        for cx, cy, lines in bubbles:
            bw = max(d.textlength(t, font=font) for t in lines) + 90
            bh = 54 * len(lines) + 70
            d.ellipse((cx - bw / 2, y + cy - bh / 2, cx + bw / 2, y + cy + bh / 2),
                      fill="white", outline="black", width=4)
            for i, t in enumerate(lines):
                d.text((cx, y + cy - 27 * (len(lines) - 1) + 54 * i), t, font=font, fill="black", anchor="mm")
        y += ph + GUTTER

    img.save(OUT)
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
