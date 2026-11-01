"""Shared data types for the pipeline."""
from dataclasses import dataclass, field, asdict

Box = tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class TextRegion:
    id: int
    box: Box                                  # union of the line boxes
    line_boxes: list[Box] = field(default_factory=list)
    source_text: str = ""
    translated_text: str = ""
    vertical: bool = False
    on_light_bg: bool = True                  # True = inside a white speech bubble
    render_box: Box | None = None             # where the translation gets drawn

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TextRegion":
        d = dict(d)
        for k in ("box", "render_box"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        d["line_boxes"] = [tuple(b) for b in d.get("line_boxes", [])]
        return cls(**d)


def union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def area(b: Box) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
