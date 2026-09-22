"""demo 视频的渲染基础设施：静帧 → 带动效的视频段 → 合成音床。

只用 ffmpeg + Pillow，无 GPU。刻意不引入新依赖：这套东西要能在
任何一台没配环境的机器上重跑出同一个 demo。
"""

from __future__ import annotations

import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FFMPEG = str(ROOT / "bin" / "ffmpeg")
FPS = 24
W, H = 1280, 720


def ff(*args: str, quiet: bool = True) -> None:
    """跑 ffmpeg。失败时把 stderr 尾部原样抛出 —— ffmpeg 的真报错在最后几行。"""
    cmd = [FFMPEG, "-y", "-hide_banner"]
    if quiet:
        cmd += ["-loglevel", "error"]
    cmd += list(args)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = "\n".join((p.stderr or "").strip().splitlines()[-12:])
        raise RuntimeError(f"ffmpeg 失败 (rc={p.returncode})\n命令: {' '.join(cmd[:14])} ...\n{tail}")


def probe_duration(path: str | Path) -> float:
    p = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True
    )
    for line in p.stderr.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"无法解析时长: {path}\n{p.stderr[-400:]}")


# ---------------------------------------------------------------- 静帧 → 视频


def ease(t: float, kind: str = "inout") -> float:
    """缓动。线性运动在 demo 里显得机械，缓动能让镜头有"被人推"的质感。"""
    t = max(0.0, min(1.0, t))
    if kind == "in":
        return t * t
    if kind == "out":
        return 1 - (1 - t) ** 2
    if kind == "inout":
        return 3 * t * t - 2 * t * t * t
    return t


def still_to_video(
    img: Image.Image,
    out: Path,
    seconds: float,
    *,
    fps: int = FPS,
    zoom: tuple[float, float] = (1.0, 1.0),
    pan: tuple[float, float] = (0.0, 0.0),
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> Path:
    """一张静帧铺成一段视频，可选 Ken Burns 推移。

    用 Pillow 逐帧裁切而不是 ffmpeg 的 zoompan：zoompan 在小幅度缓慢推进时
    会因为整数取整产生肉眼可见的阶跃抖动，逐帧裁切没有这个问题。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(round(seconds * fps)))
    tmp = out.parent / f"_frames_{out.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()

    src = img.convert("RGB")
    sw, sh = src.size
    for i in range(n):
        t = ease(i / max(n - 1, 1))
        z = zoom[0] + (zoom[1] - zoom[0]) * t
        px, py = pan[0] * t, pan[1] * t
        cw, ch = sw / z, sh / z
        cx = sw / 2 + px * sw
        cy = sh / 2 + py * sh
        cx = max(cw / 2, min(sw - cw / 2, cx))
        cy = max(ch / 2, min(sh - ch / 2, cy))
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        frame = src.resize((W, H), Image.LANCZOS, box=box) if z != 1.0 or px or py else src.resize((W, H), Image.LANCZOS)
        frame.save(tmp / f"{i:05d}.png")

    args = ["-framerate", str(fps), "-i", str(tmp / "%05d.png")]
    vf = []
    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        vf.append(f"fade=t=out:st={max(seconds - fade_out, 0):.3f}:d={fade_out:.3f}")
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(fps), str(out)]
    ff(*args)
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    return out


def frames_to_video(
    frames: list[Image.Image],
    out: Path,
    *,
    fps: int = FPS,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
) -> Path:
    """帧序列 → 视频。用于真正有逐帧运动的段（镜头渲染、动态图表）。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f"_frames_{out.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    for i, im in enumerate(frames):
        im.convert("RGB").resize((W, H), Image.LANCZOS).save(tmp / f"{i:05d}.png")
    dur = len(frames) / fps
    args = ["-framerate", str(fps), "-i", str(tmp / "%05d.png")]
    vf = []
    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        vf.append(f"fade=t=out:st={max(dur - fade_out, 0):.3f}:d={fade_out:.3f}")
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(fps), str(out)]
    ff(*args)
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    return out


def _render_one(spec: tuple) -> str:
    """多进程渲帧的 worker（必须是模块级函数才能被 pickle）。"""
    fn_mod, fn_name, kwargs, path = spec
    import importlib

    fn = getattr(importlib.import_module(fn_mod), fn_name)
    fn(**kwargs).convert("RGB").save(path)
    return path


def parallel_frames(
    fn_mod: str,
    fn_name: str,
    kwargs_list: list[dict],
    out_dir: Path,
    *,
    workers: int = 6,
) -> list[Path]:
    """多进程渲染帧序列。单核 Pillow 画复杂帧约 60-120ms，30 秒镜头就是分钟级。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (fn_mod, fn_name, kw, str(out_dir / f"{i:05d}.png"))
        for i, kw in enumerate(kwargs_list)
    ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        paths = list(ex.map(_render_one, specs, chunksize=4))
    return [Path(p) for p in paths]


def png_dir_to_video(
    d: Path, out: Path, *, fps: int = FPS, fade_in: float = 0.0, fade_out: float = 0.0
) -> Path:
    n = len(list(d.glob("*.png")))
    dur = n / fps
    args = ["-framerate", str(fps), "-i", str(d / "%05d.png")]
    vf = []
    if fade_in > 0:
        vf.append(f"fade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        vf.append(f"fade=t=out:st={max(dur - fade_out, 0):.3f}:d={fade_out:.3f}")
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(fps), str(out)]
    ff(*args)
    return out


# ---------------------------------------------------------------- 音床


def ambient_bed(seconds: float, out: Path, *, key_hz: float = 110.0, seed: int = 7) -> Path:
    """程序化氛围音床：三个纯音叠一层过滤噪声，做成缓慢起伏的 pad。

    不用现成音乐是为了版权干净 —— demo 要能随便发。
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    f1, f2, f3 = key_hz, key_hz * 1.5, key_hz * 2.0   # 根音 + 五度 + 八度
    filt = (
        f"sine=frequency={f1}:duration={seconds}[a];"
        f"sine=frequency={f2}:duration={seconds}[b];"
        f"sine=frequency={f3}:duration={seconds}[c];"
        f"anoisesrc=d={seconds}:c=pink:r=48000:a=0.06[n];"
        f"[a]volume=0.10,tremolo=f=0.11:d=0.6[a2];"
        f"[b]volume=0.055,tremolo=f=0.13:d=0.5[b2];"
        f"[c]volume=0.03,tremolo=f=0.17:d=0.7[c2];"
        f"[n]lowpass=f=900,highpass=f=80[n2];"
        f"[a2][b2][c2][n2]amix=inputs=4:normalize=0,"
        f"lowpass=f=2400,"
        f"afade=t=in:st=0:d=2.5,afade=t=out:st={max(seconds-3.0,0):.2f}:d=3.0,"
        f"alimiter=limit=0.7[out]"
    )
    ff("-filter_complex", filt, "-map", "[out]", "-t", f"{seconds:.3f}",
       "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", str(out))
    return out


def ui_blip(out: Path, *, hz: float = 880.0, ms: int = 70) -> Path:
    """段落切换的提示音。"""
    d = ms / 1000
    ff("-filter_complex",
       f"sine=frequency={hz}:duration={d},volume=0.18,afade=t=out:st=0:d={d}[o]",
       "-map", "[o]", "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(out))
    return out


def mux(video: Path, audio: Path, out: Path, *, lufs: float = -16.0) -> Path:
    """合轨并做响度归一 —— demo 发到任何平台音量都一致。"""
    ff("-i", str(video), "-i", str(audio),
       "-filter_complex", f"[1:a]loudnorm=I={lufs}:TP=-1.5:LRA=11[a]",
       "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
       "-shortest", str(out))
    return out


# ---------------------------------------------------------------- 段落编排


@dataclass
class Segment:
    name: str
    path: Path
    duration_s: float
    transition: str = "fade"     # fade | cut
    transition_s: float = 0.5


@dataclass
class DemoTimeline:
    segments: list[Segment] = field(default_factory=list)

    def add(self, seg: Segment) -> Segment:
        self.segments.append(seg)
        return seg

    @property
    def total_s(self) -> float:
        """xfade 转场会吃掉重叠时间，总时长要减去转场时长。"""
        t = sum(s.duration_s for s in self.segments)
        t -= sum(s.transition_s for s in self.segments[1:] if s.transition == "fade")
        return t

    def concat(self, out: Path) -> Path:
        """用 xfade 串联所有段。

        xfade 的 offset 是**累积的**，并且每接一次总时长就缩短一个转场时长 ——
        这是最容易写错的地方，这里显式维护 acc 而不是用段起点相加。
        """
        if not self.segments:
            raise ValueError("时间线为空")
        if len(self.segments) == 1:
            ff("-i", str(self.segments[0].path), "-c", "copy", str(out))
            return out

        inputs: list[str] = []
        for s in self.segments:
            inputs += ["-i", str(s.path)]

        parts: list[str] = []
        cur = "0:v"
        acc = self.segments[0].duration_s
        for i, s in enumerate(self.segments[1:], start=1):
            tag = f"v{i}"
            if s.transition == "cut":
                parts.append(f"[{cur}][{i}:v]concat=n=2:v=1:a=0[{tag}]")
                acc += s.duration_s
            else:
                off = max(acc - s.transition_s, 0.0)
                parts.append(
                    f"[{cur}][{i}:v]xfade=transition=fade:duration={s.transition_s:.3f}:offset={off:.3f}[{tag}]"
                )
                acc = off + s.duration_s
            cur = tag

        ff(*inputs, "-filter_complex", ";".join(parts), "-map", f"[{cur}]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-r", str(FPS), str(out))
        return out


def _selftest() -> None:
    from demo_visuals import P, title_card, shot_frame

    out = ROOT / "out" / "_render_test"
    out.mkdir(parents=True, exist_ok=True)

    a = still_to_video(
        title_card("渲染自测", "still_to_video", ["Ken Burns 推进"]),
        out / "a.mp4", 2.0, zoom=(1.0, 1.12), fade_in=0.3,
    )
    frames = [
        shot_frame(seed=5, t=i / FPS, shot_id="test_s01", shot_size="medium shot",
                   camera_move="camera dollies in", engine="mock", engine_color=P.accent2,
                   duration_s=2.0, qc=0.9, cost=0.01)
        for i in range(int(2.0 * FPS))
    ]
    b = frames_to_video(frames, out / "b.mp4", fade_out=0.3)

    tl = DemoTimeline()
    tl.add(Segment("a", a, probe_duration(a)))
    tl.add(Segment("b", b, probe_duration(b), transition="fade", transition_s=0.5))
    v = tl.concat(out / "cat.mp4")

    aud = ambient_bed(tl.total_s, out / "bed.m4a")
    final = mux(v, aud, out / "final.mp4")

    got = probe_duration(final)
    want = tl.total_s
    assert abs(got - want) < 0.35, f"拼接后时长 {got:.2f}s 与预期 {want:.2f}s 不符"
    assert final.stat().st_size > 20_000, "产物过小，可能没真的编码"
    print(f"demo_render selftest OK  段时长 2.0+2.0 转场0.5 -> 预期 {want:.2f}s 实测 {got:.2f}s")
    print(f"  产物 {final} ({final.stat().st_size:,} bytes)")


if __name__ == "__main__":
    _selftest()
