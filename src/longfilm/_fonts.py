"""中文字体定位。

仓库不附带字体文件：微软雅黑 / 黑体的授权不允许再分发，公开仓库里只能写取得途径
（见 assets/fonts/README.md）。所以字体路径不能写死，按下面的顺序找第一个存在的：

1. 环境变量 LONGFILM_FONT
2. <repo>/assets/fonts/ —— 先认 msyh.ttc / simhei.ttf，再认目录里任意字体
3. 系统里常见的 CJK 字体（Noto Sans CJK、文泉驿、Windows / WSL 挂载的雅黑、macOS 苹方）

libass 按 family name 匹配字体，文件名对它没有意义；拼错会静默回退到默认字体出豆腐块。
所以 family 一律从字体文件里读，不从文件名猜。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_FONTS = _REPO_ROOT / "assets" / "fonts"

_PREFERRED = ("msyh.ttc", "simhei.ttf")
_SYSTEM = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",   # Debian/Ubuntu fonts-noto-cjk
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",        # Arch / Fedora
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",           # fonts-wqy-microhei
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/mnt/c/Windows/Fonts/msyh.ttc",                            # WSL 读宿主机字体（运行时引用，不复制进仓库）
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)
# 已用 PIL.ImageFont.getname() 核过的 family name；Pillow 读不了时的兜底。
_KNOWN_FAMILIES = {
    "msyh.ttc": "Microsoft YaHei",
    "simhei.ttf": "SimHei",
    "notosanscjk-regular.ttc": "Noto Sans CJK JP",
    "wqy-microhei.ttc": "WenQuanYi Micro Hei",
    "wqy-zenhei.ttc": "WenQuanYi Zen Hei",
}


def _candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("LONGFILM_FONT")
    if env:
        out.append(Path(env))
    out += [ASSET_FONTS / n for n in _PREFERRED]
    if ASSET_FONTS.is_dir():
        out += sorted(p for p in ASSET_FONTS.iterdir()
                      if p.suffix.lower() in {".ttc", ".ttf", ".otf"})
    out += [Path(p) for p in _SYSTEM]
    return out


@lru_cache(maxsize=1)
def cjk_font() -> Path | None:
    """第一个存在的中文字体；一个都没有时返回 None，由调用方决定是报错还是降级。"""
    for p in _candidates():
        if p.is_file():
            return p
    return None


def require_cjk_font() -> Path:
    """必须能渲染中文的场合用它（显式标识、字幕）：找不到直接报错，不接受方框。"""
    p = cjk_font()
    if p is None:
        raise FileNotFoundError(
            "找不到中文字体。任选其一：设置 LONGFILM_FONT 指向 .ttf/.ttc；"
            "把字体放进 assets/fonts/；或安装 fonts-noto-cjk。详见 assets/fonts/README.md"
        )
    return p


def family_name(font: str | Path) -> str:
    """读出字体的 family name（TTC 取第 0 个面）。"""
    font = Path(font)
    try:
        from PIL import ImageFont  # noqa: PLC0415  只为读一个字段，不值得进顶层依赖

        return ImageFont.truetype(str(font), 16).getname()[0]
    except Exception:  # noqa: BLE001  Pillow 缺失或字体格式异常都退到已核对的映射表
        return _KNOWN_FAMILIES.get(font.name.lower(), font.stem)
