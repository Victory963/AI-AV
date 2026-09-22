# Fonts (not included in the repo)

Subtitles, visible disclosure labels, and mock-engine watermarks all need a **Chinese font**. The Microsoft YaHei / SimHei used during development are licensed with Windows and cannot be redistributed, so this directory is empty in the repo.

`longfilm/_fonts.py` uses the first font it finds, in this order:

1. The environment variable `LONGFILM_FONT` (points to a `.ttf` / `.ttc` / `.otf` file)
2. `assets/fonts/msyh.ttc`, `assets/fonts/simhei.ttf`, then any font file in this directory
3. Common CJK fonts on the system:
   - Noto Sans CJK (`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`, etc.)
   - WenQuanYi Micro Hei / Zen Hei
   - Microsoft YaHei mounted under WSL at `/mnt/c/Windows/Fonts/msyh.ttc` (referenced at runtime, not copied)
   - macOS PingFang

The easiest way to get one (the SIL Open Font License allows redistribution):

```bash
sudo apt install fonts-noto-cjk          # Debian / Ubuntu
# or put any CJK font directly into this directory
```

libass matches fonts by **family name**; file names mean nothing to it. The code reads the family from the font file itself (`_fonts.family_name`), so after swapping fonts you do not need to change any style settings.

If no font is found:

- Burning the disclosure label and subtitles raises an error. Rendering Chinese as boxes is not accepted.
- The mock watermark falls back to a bitmap font and logs a warning.
