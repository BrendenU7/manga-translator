"""Generate a synthetic manga-style test page (no copyrighted art needed).

Usage: python samples/make_sample.py  -> writes samples/sample_page.png
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
OUT = Path(__file__).with_name("sample_page.png")


def vertical_text(draw, x_right, y_top, columns, font, size):
    """Draw Japanese columns top-to-bottom, starting from the right."""
    for c, col in enumerate(columns):
        x = x_right - c * int(size * 1.3)
        for i, ch in enumerate(col):
            draw.text((x, y_top + i * int(size * 1.1)), ch, font=font, fill="black", anchor="ma")


def main():
    W, H = 900, 1200
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 34, index=0)

    # Panel borders
    for box in [(30, 30, 870, 560), (30, 590, 440, 1170), (470, 590, 870, 1170)]:
        d.rectangle(box, outline="black", width=5)

    # Panel 1: some "artwork" (gradient + shapes) with two bubbles
    for y in range(300, 555):
        shade = 200 - (y - 300) // 3
        d.line([(35, y), (865, y)], fill=(shade, shade, shade))
    d.ellipse((560, 60, 840, 330), fill="white", outline="black", width=4)
    vertical_text(d, 735, 95, ["おい、待て", "よ！"], font, 34)
    d.ellipse((90, 70, 370, 330), fill="white", outline="black", width=4)
    vertical_text(d, 265, 105, ["どこに行く", "つもりだ？"], font, 34)

    # Panel 2: one big bubble
    d.ellipse((90, 650, 390, 1010), fill="white", outline="black", width=4)
    vertical_text(d, 285, 700, ["今日は", "本当に", "暑いね"], font, 34)

    # Panel 3: caption over dark artwork
    d.rectangle((475, 595, 865, 1165), fill=(70, 70, 90))
    cap_font = ImageFont.truetype(FONT, 30, index=0)
    d.rectangle((520, 1040, 820, 1110), fill="white", outline="black", width=3)
    d.text((670, 1075), "三日後…", font=cap_font, fill="black", anchor="mm")

    img.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
