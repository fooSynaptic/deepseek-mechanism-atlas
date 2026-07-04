#!/usr/bin/env python3
"""Shared SVG validation: XML, bounds, layout overlap, Markdown embed refs."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

UNSAFE = re.compile(r"[₀₁₂₃₄₅₆₇₈₉Δαβ→–·≈×§]")

# 手工精调、未接生成器的旧图：跳过布局启发式（仍做 XML / Markdown 校验）
LEGACY_SKIP_LAYOUT: frozenset[str] = frozenset({"mla-forward-flow.svg"})

IMG_SVG_RE = re.compile(r'<img[^>]+src="([^"]+\.svg)"', re.I)
LINK_SVG_RE = re.compile(r"\]\(([^)]+\.svg)\)")

FONT: dict[str, tuple[float, float]] = {
    "t": (15.0, 18.0),
    "st": (11.0, 14.0),
    "lb": (11.0, 14.0),
    "dt": (10.0, 13.0),
    "an": (9.0, 12.0),
}

MIN_BASELINE_GAP: dict[tuple[str, str], float] = {
    ("lb", "lb"): 15.0,
    ("lb", "dt"): 13.0,
    ("dt", "lb"): 13.0,
    ("dt", "dt"): 11.0,
    ("an", "lb"): 10.0,
    ("an", "dt"): 10.0,
}


@dataclass(frozen=True)
class TextItem:
    x: float
    y: float
    text: str
    cls: str


def _parse_viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    vb = root.attrib.get("viewBox", "")
    parts = vb.split()
    if len(parts) != 4:
        return 0.0, 0.0, 0.0, 0.0
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _iter_elements(root: ET.Element, local: str) -> list[ET.Element]:
    return [el for el in root.iter() if el.tag.split("}")[-1] == local]


def _text_content(el: ET.Element) -> str:
    return "".join(el.itertext()).strip()


def _approx_text_width(text: str, font_px: float) -> float:
    w = 0.0
    for ch in text:
        w += font_px if ord(ch) > 127 else font_px * 0.55
    return w


def _text_bbox(item: TextItem) -> tuple[float, float, float, float]:
    fp, lh = FONT.get(item.cls, (10.0, 13.0))
    tw = _approx_text_width(item.text, fp)
    return (item.x - tw / 2 - 2, item.y - lh, item.x + tw / 2 + 2, item.y + 4)


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _rects(root: ET.Element, *, for_geometry: bool = False) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for el in _iter_elements(root, "rect"):
        if for_geometry and el.attrib.get("stroke-dasharray"):
            continue
        w, h = float(el.attrib.get("width", 0)), float(el.attrib.get("height", 0))
        if for_geometry and w >= 400 and h >= 280:
            continue
        out.append(
            (
                float(el.attrib.get("x", 0)),
                float(el.attrib.get("y", 0)),
                float(el.attrib.get("width", 0)),
                float(el.attrib.get("height", 0)),
            )
        )
    return out


def _segments(root: ET.Element) -> list[tuple[float, float, float, float]]:
    segs: list[tuple[float, float, float, float]] = []
    for el in _iter_elements(root, "line"):
        segs.append(
            (
                float(el.attrib.get("x1", 0)),
                float(el.attrib.get("y1", 0)),
                float(el.attrib.get("x2", 0)),
                float(el.attrib.get("y2", 0)),
            )
        )
    for el in _iter_elements(root, "polyline"):
        pts = el.attrib.get("points", "").replace(",", " ").split()
        coords = [float(p) for p in pts if p]
        for i in range(0, len(coords) - 2, 2):
            segs.append((coords[i], coords[i + 1], coords[i + 2], coords[i + 3]))
    return segs


def _segment_crosses_rect_interior(
    x1: float, y1: float, x2: float, y2: float,
    rx: float, ry: float, rw: float, rh: float,
    pad: float = 4.0,
) -> bool:
    """线段是否穿过矩形内部（非仅贴边连接）。"""
    ix0, iy0, ix1, iy1 = rx + pad, ry + pad, rx + rw - pad, ry + rh - pad
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    def inside(x: float, y: float) -> bool:
        return ix0 <= x <= ix1 and iy0 <= y <= iy1
    if inside(x1, y1) and inside(x2, y2):
        return False
    if abs(y1 - y2) < 1.0:
        y = (y1 + y2) / 2
        if iy0 < y < iy1:
            xmin, xmax = min(x1, x2), max(x1, x2)
            if xmin < ix1 and xmax > ix0:
                return True
    if abs(x1 - x2) < 1.0:
        x = (x1 + x2) / 2
        if ix0 < x < ix1:
            ymin, ymax = min(y1, y2), max(y1, y2)
            if ymin < iy1 and ymax > iy0:
                return True
    return False


def check_svg_geometry(path: Path) -> list[str]:
    """检测线段穿过图块、文字压在图块线上等几何遮挡。"""
    root = ET.parse(path).getroot()
    rects = _rects(root, for_geometry=True)
    segs = _segments(root)
    texts = [
        TextItem(
            float(el.attrib.get("x", 0)),
            float(el.attrib.get("y", 0)),
            _text_content(el),
            el.attrib.get("class", "lb"),
        )
        for el in _iter_elements(root, "text")
        if _text_content(el)
    ]
    errs: list[str] = []

    for i, (x1, y1, x2, y2) in enumerate(segs):
        for j, (rx, ry, rw, rh) in enumerate(rects):
            if _segment_crosses_rect_interior(x1, y1, x2, y2, rx, ry, rw, rh):
                errs.append(f"line#{i+1} crosses rect#{j+1} interior")

    for t in texts:
        if t.cls not in ("lb", "dt", "an"):
            continue
        tb = _text_bbox(t)
        cx, cy = t.x, t.y
        for j, (rx, ry, rw, rh) in enumerate(rects):
            if rx + 4 <= cx <= rx + rw - 4 and ry + 4 <= cy <= ry + rh - 4:
                break
        else:
            for j, (rx, ry, rw, rh) in enumerate(rects):
                rb = (rx, ry, rx + rw, ry + rh)
                if _bbox_overlap(tb, rb) and not (
                    abs(tb[3] - ry) < 8 or abs(tb[1] - (ry + rh)) < 8
                ):
                    errs.append(f"text {t.text[:16]!r} overlaps rect#{j+1}")
            continue
        # 中心在某 rect 内则跳过与其它 rect 的 bbox 交叉检查

    return errs


def check_svg_structure(path: Path) -> list[str]:
    errs: list[str] = []
    text = path.read_text(encoding="utf-8")
    if b"\x00" in text.encode("utf-8"):
        errs.append("contains null bytes")
    if not text.lstrip().startswith("<?xml"):
        errs.append("missing <?xml declaration")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        errs.append(f"xml: {e}")
        return errs
    tag = root.tag.split("}")[-1]
    if tag != "svg":
        errs.append("root is not svg")
    if not root.attrib.get("viewBox"):
        errs.append("missing viewBox")
    if not root.attrib.get("width") or not root.attrib.get("height"):
        errs.append("missing width/height")
    if path.name not in LEGACY_SKIP_LAYOUT and UNSAFE.search(text):
        errs.append("unsafe preview characters")
    if "<a " in text or "<a>" in text:
        errs.append("contains <a> links")
    return errs


def check_svg_bounds(path: Path) -> list[str]:
    errs: list[str] = []
    root = ET.parse(path).getroot()
    _, _, vw, vh = _parse_viewbox(root)
    if vw <= 0 or vh <= 0:
        return ["invalid viewBox for bounds check"]

    margin = 2.0
    for el in _iter_elements(root, "rect"):
        x = float(el.attrib.get("x", 0))
        y = float(el.attrib.get("y", 0))
        w = float(el.attrib.get("width", 0))
        h = float(el.attrib.get("height", 0))
        if x < -margin or y < -margin or x + w > vw + margin or y + h > vh + margin:
            errs.append(f"rect out of viewBox: ({x:.0f},{y:.0f},{w:.0f},{h:.0f})")

    for el in _iter_elements(root, "text"):
        cls = el.attrib.get("class", "lb")
        if cls not in ("lb", "dt"):
            continue
        x = float(el.attrib.get("x", 0))
        y = float(el.attrib.get("y", 0))
        fp, lh = FONT.get(cls, (10.0, 13.0))
        content = _text_content(el)
        tw = _approx_text_width(content, fp)
        if cls == "lb" and (x - tw / 2 < -margin or x + tw / 2 > vw + margin):
            errs.append(f"text horizontal overflow: {content[:24]!r}")
        if y - lh < -margin or y + 6 > vh + margin:
            errs.append(f"text vertical overflow: {content[:24]!r}")

    return errs


def check_svg_overlaps(path: Path) -> list[str]:
    """Heuristic: flag text blocks that likely visually collide."""
    root = ET.parse(path).getroot()
    texts: list[TextItem] = []
    for el in _iter_elements(root, "text"):
        content = _text_content(el)
        if not content:
            continue
        texts.append(
            TextItem(
                x=float(el.attrib.get("x", 0)),
                y=float(el.attrib.get("y", 0)),
                text=content,
                cls=el.attrib.get("class", "lb"),
            )
        )

    errs: list[str] = []
    for i, a in enumerate(texts):
        if a.cls not in ("lb", "dt"):
            continue
        box_a = _text_bbox(a)
        for b in texts[i + 1 :]:
            if b.cls not in ("lb", "dt"):
                continue
            # 分列排版（双栏图）不判遮挡
            if abs(a.x - b.x) > 150:
                continue
            gap_key = (a.cls, b.cls)
            min_gap = MIN_BASELINE_GAP.get(gap_key, MIN_BASELINE_GAP.get((b.cls, a.cls), 10.0))
            dy = abs(a.y - b.y)
            if dy >= min_gap:
                continue
            box_b = _text_bbox(b)
            if _bbox_overlap(box_a, box_b):
                errs.append(
                    f"text overlap: {a.text[:20]!r} ({a.cls}@{a.y:.0f}) "
                    f"vs {b.text[:20]!r} ({b.cls}@{b.y:.0f}), dy={dy:.0f}px"
                )
    return errs


def _resolve_svg_ref(md: Path, ref: str, root: Path) -> str | None:
    ref = ref.split("#")[0].strip()
    if not ref or ref.startswith(("http://", "https://")):
        return None
    target = (md.parent / ref).resolve()
    try:
        return str(target.relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return None


def check_markdown_svg_refs(repo_root: Path | None = None) -> list[str]:
    """Every local *.svg in Markdown must be <img> embedded; refs must resolve to valid SVG."""
    root = repo_root or REPO_ROOT
    errs: list[str] = []
    seen: set[tuple[str, str]] = set()

    for md in sorted(root.rglob("*.md")):
        if "node_modules" in md.parts:
            continue
        body = md.read_text(encoding="utf-8", errors="replace")
        img_resolved = {
            r
            for ref in IMG_SVG_RE.findall(body)
            if (r := _resolve_svg_ref(md, ref, root))
        }
        refs: list[str] = []
        refs.extend(IMG_SVG_RE.findall(body))
        refs.extend(LINK_SVG_RE.findall(body))
        for ref in refs:
            resolved = _resolve_svg_ref(md, ref, root)
            if not resolved:
                continue
            key = (str(md.relative_to(root)), resolved)
            if key in seen:
                continue
            seen.add(key)
            target = root / resolved
            if not target.is_file():
                errs.append(f"{md.relative_to(root)}: missing svg -> {ref}")
                continue
            struct = check_svg_structure(target)
            if struct:
                errs.append(
                    f"{md.relative_to(root)}: broken embed {ref} ({', '.join(struct)})"
                )
        for ref in LINK_SVG_RE.findall(body):
            resolved = _resolve_svg_ref(md, ref, root)
            if resolved and resolved not in img_resolved:
                errs.append(
                    f"{md.relative_to(root)}: svg link without <img> embed -> {ref}"
                )
    return errs


def validate_svg(path: Path, *, check_overlap: bool = True) -> list[str]:
    errs = check_svg_structure(path)
    if errs:
        return errs
    if path.name not in LEGACY_SKIP_LAYOUT:
        errs.extend(check_svg_bounds(path))
        if check_overlap:
            errs.extend(check_svg_overlaps(path))
            errs.extend(check_svg_geometry(path))
    return errs
