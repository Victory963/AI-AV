"""demo 视频的视觉组件库 —— 纯 Pillow，无 GPU 依赖。

定位说明：本库画的是**系统演示画面**（流程图、路由表、质检读数、成本账本），
以及作为镜头占位的程序化合成画面。它不假装自己是 AI 生成的人物影像 ——
本机没有 GPU 也没有厂商密钥，demo 里的"镜头"由 mock 引擎合成，
这一点会在片头显式声明。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
try:
    from longfilm._fonts import require_cjk_font
    FONT_CJK = str(require_cjk_font())
except ImportError:  # 单独运行本文件、没把 src 放进 sys.path 时
    FONT_CJK = str(ROOT / "assets" / "fonts" / "msyh.ttc")
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

W, H = 1280, 720


# ---------------------------------------------------------------- 配色


@dataclass(frozen=True)
class Palette:
    """暗底配色。选暗底是因为 demo 里大量对比图和读数，暗底上亮色读数对比度最高。"""

    bg: tuple[int, int, int] = (10, 12, 17)
    panel: tuple[int, int, int] = (18, 22, 30)
    panel_hi: tuple[int, int, int] = (26, 32, 43)
    line: tuple[int, int, int] = (48, 58, 76)
    text: tuple[int, int, int] = (232, 238, 247)
    text_dim: tuple[int, int, int] = (146, 160, 182)
    accent: tuple[int, int, int] = (94, 168, 255)     # 官方引擎
    accent2: tuple[int, int, int] = (168, 132, 255)   # 开源引擎
    ok: tuple[int, int, int] = (86, 205, 148)
    warn: tuple[int, int, int] = (232, 180, 90)
    bad: tuple[int, int, int] = (238, 108, 110)
    gold: tuple[int, int, int] = (214, 186, 122)


P = Palette()

_font_cache: dict[tuple[str, int, int], ImageFont.FreeTypeFont] = {}


def font(size: int, *, mono: bool = False, index: int = 0) -> ImageFont.FreeTypeFont:
    key = (FONT_MONO if mono else FONT_CJK, size, index)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(key[0], size, index=index)
    return _font_cache[key]


def canvas(bg: tuple[int, int, int] | None = None) -> Image.Image:
    return Image.new("RGB", (W, H), bg or P.bg)


def text_w(d: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont) -> int:
    return int(d.textlength(s, font=f))


# ---------------------------------------------------------------- 基础件


def vgrad(img: Image.Image, top: tuple[int, int, int], bot: tuple[int, int, int]) -> None:
    d = ImageDraw.Draw(img)
    for y in range(img.height):
        t = y / max(img.height - 1, 1)
        d.line(
            [(0, y), (img.width, y)],
            fill=tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)),
        )


def panel(
    d: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int] | None = None,
    outline: tuple[int, int, int] | None = None,
    radius: int = 10,
) -> None:
    d.rounded_rectangle(box, radius=radius, fill=fill or P.panel, outline=outline or P.line, width=1)


def label(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    s: str,
    *,
    size: int = 24,
    color: tuple[int, int, int] | None = None,
    mono: bool = False,
    anchor: str = "la",
) -> None:
    d.text(xy, s, font=font(size, mono=mono), fill=color or P.text, anchor=anchor)


def chip(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    s: str,
    *,
    color: tuple[int, int, int],
    size: int = 18,
    pad: int = 8,
) -> int:
    """小标签块。返回宽度，便于横排布局。"""
    f = font(size)
    w = int(d.textlength(s, font=f))
    box = (xy[0], xy[1], xy[0] + w + pad * 2, xy[1] + size + pad)
    d.rounded_rectangle(box, radius=5, fill=tuple(int(c * 0.22) for c in color), outline=color, width=1)
    d.text((xy[0] + pad, xy[1] + pad // 2), s, font=f, fill=color)
    return w + pad * 2


def header(img: Image.Image, title: str, subtitle: str = "", *, step: str = "") -> None:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 92], fill=P.panel)
    d.line([(0, 92), (W, 92)], fill=P.line)
    if step:
        f = font(20, mono=True)
        w = int(d.textlength(step, font=f)) + 20
        d.rounded_rectangle([40, 30, 40 + w, 62], radius=5, fill=P.accent)
        d.text((50, 34), step, font=f, fill=(8, 12, 20))
        x0 = 40 + w + 18
    else:
        x0 = 40
    label(d, (x0, 24), title, size=34)
    if subtitle:
        label(d, (x0, 62), subtitle, size=19, color=P.text_dim)


def footer(img: Image.Image, left: str, right: str = "") -> None:
    d = ImageDraw.Draw(img)
    d.line([(0, H - 52), (W, H - 52)], fill=P.line)
    label(d, (40, H - 40), left, size=18, color=P.text_dim)
    if right:
        label(d, (W - 40, H - 40), right, size=18, color=P.text_dim, mono=True, anchor="ra")


# ---------------------------------------------------------------- 组合件


def title_card(
    title: str,
    subtitle: str,
    bullets: list[str],
    *,
    note: str = "",
) -> Image.Image:
    img = canvas()
    vgrad(img, (14, 18, 28), (8, 10, 15))
    d = ImageDraw.Draw(img)
    # 背景装饰：稀疏网格，暗示"分镜格"
    for x in range(0, W, 64):
        d.line([(x, 0), (x, H)], fill=(16, 20, 28))
    for y in range(0, H, 64):
        d.line([(0, y), (W, y)], fill=(16, 20, 28))
    label(d, (W // 2, 176), title, size=64, anchor="ma")
    label(d, (W // 2, 262), subtitle, size=26, color=P.accent, anchor="ma")
    y = 344
    for b in bullets:
        d.ellipse([W // 2 - 300, y + 10, W // 2 - 292, y + 18], fill=P.accent2)
        label(d, (W // 2 - 274, y), b, size=23, color=P.text_dim)
        y += 46
    if note:
        d.rounded_rectangle([W // 2 - 420, H - 128, W // 2 + 420, H - 56], radius=8,
                            fill=(30, 24, 16), outline=P.warn, width=1)
        label(d, (W // 2, H - 112), note, size=19, color=P.warn, anchor="ma")
    return img


def kv_table(
    title: str,
    rows: list[tuple[str, str, tuple[int, int, int]]],
    *,
    subtitle: str = "",
    step: str = "",
    col_w: int = 300,
) -> Image.Image:
    """键值表。第三列是颜色，用来给状态着色（PASS 绿 / 降级黄 / 拒单红）。"""
    img = canvas()
    header(img, title, subtitle, step=step)
    d = ImageDraw.Draw(img)
    y = 136
    for k, v, c in rows:
        panel(d, (56, y, W - 56, y + 52), fill=P.panel if (y // 52) % 2 else P.panel_hi)
        label(d, (76, y + 14), k, size=22, color=P.text_dim)
        label(d, (76 + col_w, y + 14), v, size=22, color=c, mono=not _has_cjk(v))
        y += 58
        if y > H - 90:
            break
    return img


def _has_cjk(s: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in s)


def flow_diagram(
    title: str,
    stages: list[tuple[str, str]],
    *,
    active: int = -1,
    subtitle: str = "",
    step: str = "",
) -> Image.Image:
    """产线流程图。active 高亮当前工位，用于逐段推进的动画。"""
    img = canvas()
    header(img, title, subtitle, step=step)
    d = ImageDraw.Draw(img)
    n = len(stages)
    cols = min(n, 5)
    rows = math.ceil(n / cols)
    bw, bh = (W - 120 - (cols - 1) * 28) // cols, 112
    for i, (name, desc) in enumerate(stages):
        r, c = divmod(i, cols)
        x = 60 + c * (bw + 28)
        y = 160 + r * (bh + 76)
        on = i == active
        fill = P.panel_hi if on else P.panel
        oc = P.accent if on else P.line
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=10, fill=fill, outline=oc, width=2 if on else 1)
        label(d, (x + bw // 2, y + 22), name, size=22, color=P.text if on else P.text_dim, anchor="ma")
        for j, line in enumerate(_wrap(desc, 13)[:2]):
            label(d, (x + bw // 2, y + 58 + j * 24), line, size=16, color=P.text_dim, anchor="ma")
        if c < cols - 1 and i < n - 1:
            ax = x + bw + 6
            ay = y + bh // 2
            d.line([(ax, ay), (ax + 16, ay)], fill=P.line, width=2)
            d.polygon([(ax + 16, ay - 5), (ax + 22, ay), (ax + 16, ay + 5)], fill=P.line)
    return img


def _wrap(s: str, n: int) -> list[str]:
    out, cur = [], ""
    for ch in s:
        cur += ch
        if len(cur) >= n:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def bar_chart(
    title: str,
    items: list[tuple[str, float, tuple[int, int, int]]],
    *,
    unit: str = "",
    subtitle: str = "",
    step: str = "",
    vmax: float | None = None,
) -> Image.Image:
    img = canvas()
    header(img, title, subtitle, step=step)
    d = ImageDraw.Draw(img)
    vmax = vmax or max((v for _, v, _ in items), default=1.0) * 1.15 or 1.0
    y = 150
    bar_x0, bar_x1 = 360, W - 160
    for name, v, c in items:
        label(d, (72, y + 6), name, size=21, color=P.text_dim)
        d.rounded_rectangle([bar_x0, y, bar_x1, y + 34], radius=6, fill=P.panel)
        w = int((bar_x1 - bar_x0) * min(v / vmax, 1.0))
        if w > 6:
            d.rounded_rectangle([bar_x0, y, bar_x0 + w, y + 34], radius=6, fill=c)
        txt = f"{v:,.4f}{unit}" if v < 1 else f"{v:,.2f}{unit}"
        label(d, (bar_x1 + 16, y + 6), txt, size=20, color=c, mono=True)
        y += 52
        if y > H - 90:
            break
    return img


def split_compare(
    title: str,
    left_img: Image.Image,
    right_img: Image.Image,
    left_label: str,
    right_label: str,
    *,
    subtitle: str = "",
    step: str = "",
    verdict: str = "",
) -> Image.Image:
    img = canvas()
    header(img, title, subtitle, step=step)
    d = ImageDraw.Draw(img)
    tw, th = (W - 140) // 2, int(((W - 140) // 2) * 9 / 16)
    top = 150
    for k, (sub, lab) in enumerate(((left_img, left_label), (right_img, right_label))):
        x = 50 + k * (tw + 40)
        img.paste(sub.resize((tw, th), Image.LANCZOS), (x, top))
        d.rectangle([x, top, x + tw, top + th], outline=P.line, width=1)
        d.rectangle([x, top + th, x + tw, top + th + 40], fill=P.panel)
        label(d, (x + tw // 2, top + th + 9), lab, size=20, anchor="ma",
              color=P.bad if k == 0 else P.ok)
    if verdict:
        label(d, (W // 2, top + th + 76), verdict, size=22, color=P.text_dim, anchor="ma")
    return img


def code_panel(
    title: str,
    lines: list[tuple[str, tuple[int, int, int]]],
    *,
    subtitle: str = "",
    step: str = "",
    size: int = 17,
) -> Image.Image:
    """等宽代码/JSON/日志面板。lines 是 (文本, 颜色)。"""
    img = canvas()
    header(img, title, subtitle, step=step)
    d = ImageDraw.Draw(img)
    panel(d, (48, 124, W - 48, H - 68), fill=(13, 16, 22))
    y = 142
    mono = font(size, mono=True)
    cjk = font(size)
    for s, c in lines:
        f = cjk if _has_cjk(s) else mono
        d.text((72, y), s, font=f, fill=c)
        y += size + 8
        if y > H - 88:
            break
    return img


def timeline_strip(
    title: str,
    shots: list[tuple[str, float, tuple[int, int, int]]],
    *,
    tracks: list[tuple[str, list[tuple[float, float, tuple[int, int, int]]]]] | None = None,
    subtitle: str = "",
    step: str = "",
) -> Image.Image:
    """成片时间线：画面轨 + 对白/Foley/音乐轨。分轨工业化的可视化。"""
    img = canvas()
    header(img, title, subtitle, step=step)
    d = ImageDraw.Draw(img)
    total = sum(s[1] for s in shots) or 1.0
    x0, x1 = 60, W - 60
    span = x1 - x0

    y = 170
    label(d, (60, y - 34), f"画面轨 · {len(shots)} 个原子镜 · 总时长 {total:.1f}s", size=19, color=P.text_dim)
    cx = x0
    for name, dur, c in shots:
        w = max(int(span * dur / total), 2)
        d.rectangle([cx, y, cx + w - 2, y + 56], fill=c)
        if w > 58:
            label(d, (cx + w // 2 - 1, y + 18), name, size=15, color=(10, 12, 18), anchor="ma")
        cx += w
    d.rectangle([x0, y, x1, y + 56], outline=P.line, width=1)

    y += 96
    for tname, segs in (tracks or []):
        label(d, (60, y - 26), tname, size=17, color=P.text_dim)
        d.rectangle([x0, y, x1, y + 34], fill=P.panel, outline=P.line, width=1)
        for s, e, c in segs:
            sx = x0 + int(span * s / total)
            ex = x0 + int(span * e / total)
            d.rounded_rectangle([sx, y + 4, max(ex, sx + 3), y + 30], radius=4, fill=c)
        y += 74
        if y > H - 96:
            break

    # 时间刻度
    for i in range(0, int(total) + 1, max(1, int(total // 10))):
        tx = x0 + int(span * i / total)
        d.line([(tx, H - 78), (tx, H - 70)], fill=P.line)
        label(d, (tx, H - 66), f"{i}s", size=14, color=P.text_dim, mono=True, anchor="ma")
    return img


# ---------------------------------------------------------------- 程序化镜头画面


def shot_frame(
    *,
    seed: int,
    t: float,
    shot_id: str,
    shot_size: str,
    camera_move: str,
    engine: str,
    engine_color: tuple[int, int, int],
    duration_s: float,
    lens_mm: int = 35,
    hud: bool = True,
    cost: float | None = None,
    qc: float | None = None,
    degrade: str = "",
    size: tuple[int, int] = (W, H),
) -> Image.Image:
    """程序化合成的"镜头"画面（非真人影像）。

    参数要**看得出生效**：seed 决定配色、t 驱动运动、camera_move 决定运动方式、
    shot_size 决定主体占画比。这样 demo 里能直观看到工单参数真的传到了渲染端。
    """
    w, h = size
    img = Image.new("RGB", (w, h), (8, 10, 14))
    d = ImageDraw.Draw(img)

    hue = (seed * 47) % 360
    base = _hsv(hue, 0.55, 0.30)
    base2 = _hsv((hue + 44) % 360, 0.65, 0.16)
    for y in range(0, h, 2):
        f = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(base2[i] + (base[i] - base2[i]) * f) for i in range(3)))

    # 主体占画比由景别决定
    scale = {
        "extreme close-up": 1.55, "close-up": 1.15, "medium close-up": 0.88,
        "medium shot": 0.66, "medium long shot": 0.50, "full shot": 0.40,
        "long shot": 0.26, "extreme long shot": 0.15,
    }.get(shot_size, 0.6)

    # 运镜驱动：位移/缩放/旋转
    p = t / max(duration_s, 1e-6)
    dx = dy = 0.0
    zoom = 1.0
    if "dollies in" in camera_move or "zoom in" in camera_move:
        zoom = 1.0 + 0.30 * p
    elif "dollies out" in camera_move or "zoom out" in camera_move:
        zoom = 1.30 - 0.30 * p
    elif "pans left" in camera_move:
        dx = -0.22 * p * w
    elif "pans right" in camera_move:
        dx = 0.22 * p * w
    elif "tilts up" in camera_move:
        dy = -0.16 * p * h
    elif "tilts down" in camera_move:
        dy = 0.16 * p * h
    elif "orbits" in camera_move:
        dx = math.sin(p * math.tau) * 0.14 * w
    elif "handheld" in camera_move:
        dx = math.sin(t * 6.1) * 7
        dy = math.cos(t * 4.7) * 5

    cx, cy = w / 2 + dx, h * 0.52 + dy
    r = h * 0.34 * scale * zoom

    # 主体：同心环构成的抽象人形轮廓（明确不是真人影像）
    for ring in range(4):
        rr = r * (1 - ring * 0.18)
        col = _hsv((hue + 180 + ring * 12) % 360, 0.45, 0.55 + ring * 0.08)
        n = 40 + ring * 8
        for i in range(n):
            a = i / n * math.tau + t * (0.35 + ring * 0.12)
            x = cx + math.cos(a) * rr * 0.72
            y = cy + math.sin(a) * rr
            rad = max(1.2, (3.2 - ring * 0.5) * (1 + 0.4 * math.sin(a * 3 + t)))
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col)
    d.ellipse([cx - r * 0.30, cy - r * 0.30, cx + r * 0.30, cy + r * 0.30],
              outline=_hsv((hue + 200) % 360, 0.3, 0.85), width=max(1, int(h / 300)))

    # 景深暗角：焦段越长暗角越强，让 lens_mm 也看得出生效
    if lens_mm >= 50:
        vig = Image.new("L", (w, h), 0)
        vd = ImageDraw.Draw(vig)
        vd.ellipse([-w * 0.25, -h * 0.25, w * 1.25, h * 1.25], fill=255)
        vig = vig.filter(ImageFilter.GaussianBlur(h / 8))
        img = Image.composite(img, Image.new("RGB", (w, h), (4, 5, 8)), vig)
        d = ImageDraw.Draw(img)

    if hud:
        _hud(d, w, h, shot_id, shot_size, camera_move, engine, engine_color,
             lens_mm, t, duration_s, cost, qc, degrade)
    return img


def _hud(d, w, h, shot_id, shot_size, camera_move, engine, ec, lens_mm, t, dur, cost, qc, degrade):
    sc = h / 720
    d.rectangle([0, 0, w, int(46 * sc)], fill=(0, 0, 0))
    d.rectangle([0, h - int(74 * sc), w, h], fill=(0, 0, 0))
    fs = max(11, int(20 * sc))
    label(d, (int(28 * sc), int(12 * sc)), f"● REC  {shot_id}", size=fs, color=P.bad, mono=True)
    label(d, (w - int(28 * sc), int(12 * sc)),
          f"{t:5.2f}s / {dur:.1f}s", size=fs, color=P.text_dim, mono=True, anchor="ra")
    label(d, (int(28 * sc), h - int(64 * sc)),
          f"{shot_size} · {camera_move} · {lens_mm}mm", size=max(11, int(19 * sc)), color=P.text)
    x = int(28 * sc)
    y = h - int(34 * sc)
    x += chip(d, (x, y), engine, color=ec, size=max(10, int(15 * sc))) + 10
    if qc is not None:
        c = P.ok if qc >= 0.7 else (P.warn if qc >= 0.5 else P.bad)
        x += chip(d, (x, y), f"QC {qc:.2f}", color=c, size=max(10, int(15 * sc))) + 10
    if cost is not None:
        x += chip(d, (x, y), f"${cost:.4f}", color=P.gold, size=max(10, int(15 * sc))) + 10
    if degrade:
        chip(d, (x, y), degrade, color=P.warn, size=max(10, int(15 * sc)))


def _hsv(hdeg: float, s: float, v: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((hdeg % 360) / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def _selftest() -> None:
    out = ROOT / "out" / "_visuals_test"
    out.mkdir(parents=True, exist_ok=True)
    title_card("长时间 AI 数字人拍片系统", "分镜驱动的多引擎长片产线",
               ["9 参考位导演包", "双引擎路由", "15s 原子镜拼接"],
               note="本 demo 使用 mock 引擎：本机无 GPU、无厂商密钥").save(out / "01.png")
    kv_table("引擎路由决策", [("sc01_s01", "official / tier5", P.accent),
                              ("sc01_s02", "open / comfy-wan", P.accent2),
                              ("sc01_s03", "降级：不支持尾帧", P.warn)], step="STEP 04").save(out / "02.png")
    flow_diagram("产线工位", [("分镜", "storyboard"), ("参考位", "refpack"), ("路由", "router"),
                             ("渲染", "provider"), ("质检", "qc")], active=2).save(out / "03.png")
    bar_chart("成本账本", [("official", 0.84, P.accent), ("open", 0.12, P.accent2)],
              unit=" USD").save(out / "04.png")
    a = shot_frame(seed=3, t=1.0, shot_id="sc01_s01", shot_size="medium shot",
                   camera_move="camera dollies in", engine="official", engine_color=P.accent,
                   duration_s=8, qc=0.88, cost=0.31)
    b = shot_frame(seed=9, t=1.0, shot_id="sc01_s02", shot_size="close-up",
                   camera_move="handheld camera, subtle organic shake", engine="open",
                   engine_color=P.accent2, duration_s=8, qc=0.42, cost=0.02)
    a.save(out / "05.png")
    split_compare("质检门禁", b, a, "RETAKE 0.42", "PASS 0.88", verdict="色漂超阈值 → 自动重拍").save(out / "06.png")
    code_panel("分镜 JSON", [('{"id": "sc01_s03",', P.text),
                             ('  "shot_size": "medium close-up",', P.accent),
                             ('  "camera_move": "camera dollies in"}', P.accent2)]).save(out / "07.png")
    timeline_strip("成片时间线", [("s01", 8, P.accent), ("s02", 12, P.accent2), ("s03", 6, P.accent)],
                   tracks=[("对白轨", [(1, 6, P.ok)]), ("Foley", [(8, 14, P.warn)])]).save(out / "08.png")
    n = len(list(out.glob("*.png")))
    assert n == 8, f"期望 8 张测试图，实际 {n}"
    print(f"demo_visuals selftest OK -> {out} ({n} images)")


if __name__ == "__main__":
    _selftest()
