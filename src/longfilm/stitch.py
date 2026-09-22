"""原子镜拼接 —— 把一堆 5~15s 的生成片段接成一集片。

生成式产线和传统剪辑的根本差别：**片段参数不受控**。
同一集里可能混着 720p/1080p、24/25fps、有声/无声的片段（不同引擎、不同 take、
不同重试路径出来的），所以这里所有函数都以「参数不一致是常态」为前提设计：
先把每条输入归一到一个 VideoTarget，再接。

三条拼接路径，按代价从低到高：
1. concat_hard  —— concat demuxer + `-c copy`，零重编码。只有全参数一致时可用。
2. concat 滤镜  —— 归一后无损重接，硬切，无转场开销。
3. xfade 链     —— 有转场，必须重编码。

xfade 的 offset 是**绝对时间轴上的坐标**，不是相对上一段的偏移，多段串联时
必须按「已拼出的总长 - 本次转场时长」累加；写错的话表现为后面的片段被整段吃掉，
而 ffmpeg 不会报错。本模块的 _clamp_transitions / _build_video_chain 是唯一的计算入口，
并由自测拿 expected_duration() 校验总时长。

注意「转场能有多长」和「offset 累加到哪」是两件事：
offset 要按已拼出的总长累加（xfade 的左输入是整条链），
但转场时长只能按**接点两侧各自那一段的原长**裁（见 clamp_transition_s）——
拿总长去裁会算出长过前一段的转场，offset 变负，ffmpeg 照样退出码 0。

clamp_transition_s 同时被 timeline.py 复用：EDL/OTIO 上的切点和这里烧出来的
成片必须逐帧一致，两边各写一份裁剪规则是套底错位的经典来源。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from ._fonts import family_name, require_cjk_font
from ._proc import PREEXEC as _PREEXEC
from .schema import AudioPlan, DeliverySpec, Scene, Storyboard, Transition

log = logging.getLogger(__name__)

# 优先项目自带的静态 ffmpeg（无 drawtext，文字一律走 libass）。允许用环境变量覆盖，
# 方便在容器里换成带硬件编码的构建。bin/ 不进仓库（第三方二进制，ffprobe 超 GitHub 单文件上限），
# 所以克隆下来没有 bin/ 时顺次退到 PATH 和 imageio-ffmpeg。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _locate(name: str, env: str) -> Path:
    """环境变量 → <repo>/bin → PATH → imageio-ffmpeg（仅 ffmpeg）。

    都找不到时返回 bin/ 下的路径：调用时报「找不到」比导入时抛异常好排查。
    ffprobe 缺失不致命 —— FFmpeg.probe 会退回解析 `ffmpeg -i` 的输出。
    """
    if os.environ.get(env):
        return Path(os.environ[env])
    vendored = _PROJECT_ROOT / "bin" / name
    if vendored.exists():
        return vendored
    found = shutil.which(name)
    if found:
        return Path(found)
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg  # noqa: PLC0415  可选依赖

            return Path(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:  # noqa: BLE001  没装或包内无二进制，都当找不到
            pass
    return vendored


DEFAULT_FFMPEG = _locate("ffmpeg", "LONGFILM_FFMPEG")
DEFAULT_FFPROBE = _locate("ffprobe", "LONGFILM_FFPROBE")

# 转场枚举 -> xfade transition 名。Transition 的取值本身多数已对齐 xfade，
# 这里只列需要改写的几个；查不到的按硬切处理。
_XFADE_NAME: dict[Transition, str] = {
    Transition.DISSOLVE: "fade",
    Transition.FADE_IN: "fadeblack",      # FADE_OUT 是同值别名
    Transition.WIPE_L: "wipeleft",
    Transition.WIPE_R: "wiperight",
    Transition.SLIDE_L: "slideleft",
    Transition.OVERLAP_BLEND: "fade",     # 链式拼接里退化成线性叠化
}
# 这两个走硬切：MATCH_CUT 画面上就是硬切，差别只在 QC 要求更严（首尾帧必须对得上）。
HARD_CUTS: frozenset[Transition] = frozenset({Transition.CUT, Transition.MATCH_CUT})

_CODEC_MAP = {"h264": "libx264", "h265": "libx265", "av1": "libaom-av1"}


# ---------------------------------------------------------------- 进程封装


class FFmpegError(RuntimeError):
    """ffmpeg 退出码非 0。

    ffmpeg 的真实原因永远在 stderr 的最后几行（前面全是 banner 和流信息），
    所以这里原样保留尾部，不做二次解释 —— 转译过的报错会丢掉滤镜图的具体位置。
    """

    def __init__(self, args: Sequence[str], returncode: int, stderr_tail: str):
        self.args_used = list(args)
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        super().__init__(f"ffmpeg 退出码 {returncode}\n--- stderr 尾部 ---\n{stderr_tail}")


@dataclass(frozen=True)
class MediaInfo:
    """一条素材的关键参数。拼接前的一致性判断全靠它。"""

    path: str
    duration_s: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    pix_fmt: str = ""
    vcodec: str = ""
    acodec: str = ""
    sample_rate: int = 0
    channels: str = ""

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.fps)) if self.fps else 0

    def video_key(self) -> tuple:
        """能否 `-c copy` 直接 concat 的判据。"""
        return (self.width, self.height, round(self.fps, 3), self.pix_fmt, self.vcodec)

    def audio_key(self) -> tuple:
        return (self.has_audio, self.acodec, self.sample_rate, self.channels)


# ffmpeg -i 的 stderr 行格式（本机 7.0.2 实测）：
#   Duration: 00:00:03.00, start: 0.000000, bitrate: 357 kb/s
#   Stream #0:0[0x1](und): Video: h264 (High) (avc1 / ...), yuv420p(progressive), 320x240 [SAR 1:1 ...], 25 fps, ...
#   Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / ...), 44100 Hz, mono, fltp, 69 kb/s
_RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_RE_VIDEO = re.compile(r"Stream #\d+:\d+.*?:\s*Video:\s*([A-Za-z0-9_]+)")
_RE_AUDIO = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:\s*([A-Za-z0-9_]+)")
_RE_SIZE = re.compile(r"(?<![\d])(\d{2,5})x(\d{2,5})(?![\d])")
_RE_FPS = re.compile(r"([\d.]+)\s*fps")
_RE_HZ = re.compile(r"(\d+)\s*Hz")
_RE_PIXFMT = re.compile(r",\s*(yuv[a-z0-9]+|gbr[a-z0-9]+|rgb[a-z0-9]+|gray[a-z0-9]*)\s*[,(]")


class FFmpeg:
    """ffmpeg 调用的唯一出口。

    probe 优先走 ffprobe 的 JSON 输出（bin/ffprobe，实测 n8.0.1）：字段是结构化的，
    尺寸、r_frame_rate、channel_layout 都能直接拿到。
    ffprobe 缺失或输出异常时才退回解析 `ffmpeg -i` 的 stderr —— 那条命令必定以
    退出码 1 结束（"At least one output file must be specified"），属于正常返回，
    不能当失败处理；但它的输出是给人看的，字段靠正则抠，遇到冷门封装容易抠空，
    所以只当兜底。
    """

    def __init__(
        self, binary: str | Path = DEFAULT_FFMPEG, probe_binary: str | Path = DEFAULT_FFPROBE
    ):
        self.binary = str(binary)
        self.probe_binary = str(probe_binary)
        self._probe_cache: dict[tuple, MediaInfo] = {}

    # -------------------------------------------------- 基础调用

    def run(self, args: Sequence[str], *, timeout: float = 600.0, tail_lines: int = 20) -> str:
        """执行 ffmpeg，返回完整 stderr。失败时把 stderr 尾部原样抛出。"""
        cmd = [self.binary, "-hide_banner", "-nostdin", *args]
        log.debug("ffmpeg %s", " ".join(cmd[1:]))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                                  timeout=timeout, preexec_fn=_PREEXEC)
        except subprocess.TimeoutExpired as e:
            raise FFmpegError(cmd, -1, f"超时 {timeout}s 未结束：{e}") from e
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-tail_lines:])
            raise FFmpegError(cmd, proc.returncode, tail)
        return proc.stderr

    def run_capture(self, args: Sequence[str], *, timeout: float = 600.0) -> bytes:
        """需要读 stdout 的场景（抽原始帧给 numpy）。"""
        cmd = [self.binary, "-hide_banner", "-nostdin", *args]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout,
                              preexec_fn=_PREEXEC)
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.decode("utf-8", "replace").strip().splitlines()[-20:])
            raise FFmpegError(cmd, proc.returncode, tail)
        return proc.stdout

    # -------------------------------------------------- probe

    def probe(self, path: str | Path, *, use_cache: bool = True) -> MediaInfo:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"素材不存在：{p}")
        st = p.stat()
        key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
        if use_cache and key in self._probe_cache:
            return self._probe_cache[key]

        info = self._probe_ffprobe(p)
        if info is None:
            cmd = [self.binary, "-hide_banner", "-nostdin", "-i", str(p)]
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=120)
            err = proc.stderr
            # 没有输出文件时 ffmpeg 必定返回 1；只有连输入都打不开才是真失败。
            if "Invalid data found" in err or "No such file" in err or "Invalid argument" in err:
                raise FFmpegError(cmd, proc.returncode, "\n".join(err.strip().splitlines()[-10:]))
            info = self._parse_probe(str(p), err)

        self._probe_cache[key] = info
        return info

    def _probe_ffprobe(self, p: Path) -> MediaInfo | None:
        """ffprobe JSON 路径。任何异常都返回 None 交给 stderr 兜底，不让 probe 挂掉。"""
        cmd = [self.probe_binary, "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(p)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            log.debug("ffprobe 不可用（%s），回退 ffmpeg -i 解析", e)
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            # 真·打不开的文件：ffprobe 也会非 0，但报错文本在 stderr，交给兜底路径统一抛
            log.debug("ffprobe 返回 %d，回退 ffmpeg -i 解析", proc.returncode)
            return None
        try:
            doc = json.loads(proc.stdout)
        except ValueError:
            return None

        streams = doc.get("streams") or []
        v = next((s for s in streams if s.get("codec_type") == "video"), None)
        a = next((s for s in streams if s.get("codec_type") == "audio"), None)

        def _dur(*cands: object) -> float:
            for c in cands:
                try:
                    d = float(c)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                if d > 0:
                    return d
            return 0.0

        # 容器头的 format.duration 最权威；封装里缺了才退回流时长。
        duration = _dur((doc.get("format") or {}).get("duration"),
                        (v or {}).get("duration"), (a or {}).get("duration"))

        fps = 0.0
        if v:
            # r_frame_rate 是 "25/1" 这种有理数；avg_frame_rate 在 VFR 上才有意义，
            # 而我们要的是「按哪个帧率归一」，所以取 r_frame_rate。
            for field_name in ("r_frame_rate", "avg_frame_rate"):
                raw = str(v.get(field_name) or "")
                if "/" in raw:
                    num, _, den = raw.partition("/")
                    try:
                        n, d = float(num), float(den)
                    except ValueError:
                        continue
                    if d:
                        fps = n / d
                        break
                elif raw:
                    fps = float(raw)
                    break

        channels = ""
        if a:
            channels = str(a.get("channel_layout") or "")
            if not channels:
                # 老封装可能没有 channel_layout，只给声道数
                channels = {1: "mono", 2: "stereo", 6: "5.1", 8: "7.1"}.get(
                    int(a.get("channels") or 0), str(a.get("channels") or "")
                )

        return MediaInfo(
            path=str(p),
            duration_s=duration,
            width=int((v or {}).get("width") or 0),
            height=int((v or {}).get("height") or 0),
            fps=fps,
            has_video=v is not None,
            has_audio=a is not None,
            pix_fmt=str((v or {}).get("pix_fmt") or ""),
            vcodec=str((v or {}).get("codec_name") or ""),
            acodec=str((a or {}).get("codec_name") or ""),
            sample_rate=int((a or {}).get("sample_rate") or 0),
            channels=channels,
        )

    @staticmethod
    def _parse_probe(path: str, err: str) -> MediaInfo:
        duration = 0.0
        if m := _RE_DURATION.search(err):
            duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

        vline = aline = ""
        for line in err.splitlines():
            s = line.strip()
            if not vline and _RE_VIDEO.search(s):
                vline = s
            elif not aline and _RE_AUDIO.search(s):
                aline = s

        width = height = 0
        fps = 0.0
        pix_fmt = vcodec = ""
        if vline:
            vcodec = _RE_VIDEO.search(vline).group(1)
            if m := _RE_SIZE.search(vline):
                width, height = int(m.group(1)), int(m.group(2))
            if m := _RE_FPS.search(vline):
                fps = float(m.group(1))
            if m := _RE_PIXFMT.search(vline):
                pix_fmt = m.group(1)

        acodec = channels = ""
        sample_rate = 0
        if aline:
            acodec = _RE_AUDIO.search(aline).group(1)
            if m := _RE_HZ.search(aline):
                sample_rate = int(m.group(1))
            for ch in ("mono", "stereo", "5.1", "7.1", "quad"):
                if f", {ch}" in aline:
                    channels = ch
                    break

        return MediaInfo(
            path=path,
            duration_s=duration,
            width=width,
            height=height,
            fps=fps,
            has_video=bool(vline),
            has_audio=bool(aline),
            pix_fmt=pix_fmt,
            vcodec=vcodec,
            acodec=acodec,
            sample_rate=sample_rate,
            channels=channels,
        )

    def exact_duration(self, path: str | Path, *, timeout: float = 600.0) -> float:
        """解码一遍拿精确时长。容器头里的 Duration 对拼接足够，
        但 xfade 的 offset 对误差敏感，关键位置用这个复核。"""
        err = self.run(["-i", str(path), "-map", "0:v:0", "-f", "null", "-"], timeout=timeout)
        last = 0.0
        for m in re.finditer(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)", err):
            last = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        return last


_FF = FFmpeg()


def default_ffmpeg() -> FFmpeg:
    return _FF


# ---------------------------------------------------------------- 首尾帧


def extract_first_frame(
    video: str | Path, out_png: str | Path, *, ffmpeg: FFmpeg | None = None, timeout: float = 120.0
) -> str:
    """导出首帧。不用 -ss 0：输入定位会落到最近关键帧，对 open-GOP 片段会偏。"""
    ff = ffmpeg or _FF
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff.run(["-y", "-i", str(video), "-frames:v", "1", "-update", "1", str(out)], timeout=timeout)
    return str(out)


def extract_last_frame(
    video: str | Path,
    out_png: str | Path,
    *,
    back_off_frames: int = 0,
    ffmpeg: FFmpeg | None = None,
    timeout: float = 300.0,
) -> str:
    """导出尾帧，供下一镜做首帧锁戏（Continuity.emit_last_frame）。

    back_off_frames：往前退几帧。生成模型的最后 1~3 帧经常带运动模糊或
    收尾时的画面塌陷，直接拿去当下一镜的首帧会把糊传染下去，所以要能退。

    实现：`-sseof -W` 定位到尾部窗口，`-update 1` 让每帧覆写同一个 PNG，
    解码完留下的就是窗口内最后一帧；用 `-t` 把窗口右端裁掉 back_off 帧。
    截断点取帧间隔的中点（(back_off+0.5)/fps），避免浮点误差多退/少退一帧。
    """
    ff = ffmpeg or _FF
    info = ff.probe(video)
    if not info.has_video:
        raise ValueError(f"{video} 没有视频轨，取不了尾帧")
    fps = info.fps or 25.0
    if back_off_frames < 0:
        raise ValueError("back_off_frames 不能为负")
    if back_off_frames and back_off_frames >= info.n_frames:
        raise ValueError(f"back_off_frames={back_off_frames} 超过总帧数 {info.n_frames}")

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    window = max(1.0, (back_off_frames + 3) / fps)
    args = ["-y"]
    if info.duration_s > window:
        args += ["-sseof", f"-{window:.6f}"]
    if back_off_frames:
        # -t 是相对窗口起点的输出时长；窗口不足时退化成绝对时长。
        span = min(window, info.duration_s)
        args += ["-t", f"{max(1.0 / fps, span - (back_off_frames + 0.5) / fps):.6f}"]
    args += ["-i", str(video), "-update", "1", "-frames:v", "100000", str(out)]
    ff.run(args, timeout=timeout)
    if not out.exists():
        raise FFmpegError(["extract_last_frame"], 1, f"未产出 {out}，检查 {video} 是否可解码")
    return str(out)


# ---------------------------------------------------------------- 归一目标


@dataclass(frozen=True)
class VideoTarget:
    """所有输入要被归一到的统一参数。"""

    width: int = 1280
    height: int = 720
    fps: int = 24
    pix_fmt: str = "yuv420p"
    sample_rate: int = 48000
    channel_layout: str = "stereo"
    fit: str = "pad"   # pad = 加黑边不丢画面；crop = 填满但裁边

    def __post_init__(self) -> None:
        # yuv420p 的色度平面是 2x2 下采样，边长为奇数时 libx264 直接返回
        # -22 Invalid argument。而这个失败发生在整张滤镜图跑完之后 ——
        # 长片上就是几十分钟白烧，所以宁可在入口悄悄取偶。
        # from_clips 按众数取尺寸，素材里混进一条 481x271 就会踩到。
        w = max(2, int(self.width) // 2 * 2)
        h = max(2, int(self.height) // 2 * 2)
        if (w, h) != (self.width, self.height):
            log.warning("目标分辨率 %dx%d 含奇数边，取偶到 %dx%d", self.width, self.height, w, h)
        object.__setattr__(self, "width", w)
        object.__setattr__(self, "height", h)

    @classmethod
    def from_delivery(cls, d: DeliverySpec, *, fit: str = "pad") -> VideoTarget:
        # 一次缩放到最终尺寸：先拼 720p 再整体上采样会把转场处的插值误差放大。
        w, h = d.upscale_to or d.resolution
        return cls(width=int(w), height=int(h), fps=int(d.fps), fit=fit)

    @classmethod
    def from_clips(cls, infos: Sequence[MediaInfo], *, fit: str = "pad") -> VideoTarget:
        """按众数取目标参数 —— 少数派片段被归一，多数派零损耗。"""
        vids = [i for i in infos if i.has_video]
        if not vids:
            raise ValueError("没有可用的视频轨")
        sizes: dict[tuple[int, int], int] = {}
        rates: dict[int, int] = {}
        for i in vids:
            sizes[i.size] = sizes.get(i.size, 0) + 1
            rates[int(round(i.fps))] = rates.get(int(round(i.fps)), 0) + 1
        w, h = max(sizes.items(), key=lambda kv: (kv[1], kv[0][0] * kv[0][1]))[0]
        fps = max(rates.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return cls(width=w, height=h, fps=fps or 24, fit=fit)

    def video_norm(self) -> str:
        """单条视频输入的归一滤镜链。

        顺序有讲究：setpts 先把时间轴拉回 0（trim/xfade 都假设从 0 起），
        scale/pad 统一画幅，fps 强制成 CFR（xfade 拒绝 VFR 输入），
        最后 settb 统一时基（concat 滤镜要求各段时基一致）。
        """
        if self.fit == "crop":
            geom = (
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase:flags=bicubic,"
                f"crop={self.width}:{self.height}"
            )
        else:
            geom = (
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease:flags=bicubic,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2:color=black"
            )
        return (
            f"setpts=PTS-STARTPTS,{geom},setsar=1,fps={self.fps},"
            f"format={self.pix_fmt},settb=1/{self.fps}"
        )

    def audio_norm(self) -> str:
        return (
            f"asetpts=PTS-STARTPTS,aresample={self.sample_rate}:first_pts=0,"
            f"aformat=sample_fmts=fltp:channel_layouts={self.channel_layout},asettb=1/{self.sample_rate}"
        )


# ---------------------------------------------------------------- 时间线


@dataclass
class ClipSpec:
    """时间线上的一段。transition_in 描述的是**入点**，即它与前一段之间的接法。"""

    path: str
    transition_in: Transition = Transition.CUT
    transition_s: float = 0.5
    shot_id: str = ""

    @property
    def is_hard_cut(self) -> bool:
        return self.transition_in in HARD_CUTS


def resolve_clips(
    timeline_or_clips: Storyboard | Scene | Sequence[str | Path | ClipSpec],
    *,
    default_transition_s: float = 0.5,
) -> list[ClipSpec]:
    """把 Storyboard / Scene / 路径列表统一成 ClipSpec 列表。

    Storyboard 只取已回填 render_uri 的镜；没渲染的镜静默跳过会让成片短一截
    却查不出来，所以这里显式 warning。
    """
    if isinstance(timeline_or_clips, (Storyboard, Scene)):
        shots = (
            timeline_or_clips.all_shots()
            if isinstance(timeline_or_clips, Storyboard)
            else list(timeline_or_clips.shots)
        )
        out: list[ClipSpec] = []
        for s in shots:
            if not s.render_uri:
                log.warning("镜头 %s 没有 render_uri，跳过（成片会短 %.1fs）", s.id, s.duration_s)
                continue
            t_s = default_transition_s
            if s.transition_in is Transition.OVERLAP_BLEND and s.continuity.overlap_frames > 0:
                t_s = s.continuity.overlap_frames / max(s.fps, 1)
            out.append(
                ClipSpec(path=s.render_uri, transition_in=s.transition_in, transition_s=t_s, shot_id=s.id)
            )
        return out

    out = []
    for c in timeline_or_clips:
        if isinstance(c, ClipSpec):
            # 必须复制：concat_with_transitions / assemble_episode 会就地改写
            # specs[0].transition_in 和 transitions 参数，直接透传就会把调用方
            # 手里的那份时间线也改掉 —— 同一批 ClipSpec 跑两遍会得到不同结果。
            out.append(replace(c))
        else:
            out.append(ClipSpec(path=str(c), transition_s=default_transition_s))
    return out


# 一个转场最多吃掉相邻两段各一半。这是 NLE 的标准握手规则：
# 留下的另一半是剪辑师回退切点的余量，吃满整段就再也改不动了。
# 同时它还顺手挡掉两个技术坑：转场长过前一段时，重叠区会倒灌进**再前一段**
# （xfade 是链式的，看不出来；EDL/OTIO 里则直接变成 rec_out < rec_in 的非法事件）。
_TRANSITION_MAX_FRACTION = 0.5


def clamp_transition_s(want_s: float, left_s: float, right_s: float) -> float:
    """单个接点上转场时长的**唯一**裁剪规则。

    为什么要公开成模块级函数：timeline.py 导 EDL/OTIO 时必须算出和这里
    一模一样的重叠。两边各写一份，剪辑师在 NLE 里看到的切点就会和我们
    烧出来的片子对不上，而这种错不会报错，只会在套底时整条轨错位。

    left_s / right_s 是接点两侧**各自那一段的原始时长**，不是已拼出的总长：
    左侧取总长会算出长过前一段的转场，见 _TRANSITION_MAX_FRACTION 的注释。
    """
    room = min(left_s, right_s) - 1e-3    # 留 1ms，避免浮点误差让 offset 落到段尾之外
    return min(max(0.0, want_s), max(0.0, room * _TRANSITION_MAX_FRACTION))


def _clamp_transitions(clips: Sequence[ClipSpec], durations: Sequence[float]) -> list[float]:
    """返回每个接点的实际转场时长（index i 对应 clips[i-1] 与 clips[i] 之间，clips[0] 恒为 0）。"""
    eff = [0.0] * len(clips)
    for i in range(1, len(clips)):
        if clips[i].is_hard_cut or _XFADE_NAME.get(clips[i].transition_in) is None:
            continue
        want = max(0.0, clips[i].transition_s)
        t = clamp_transition_s(want, durations[i - 1], durations[i])
        if t < want - 1e-6:
            log.warning(
                "接点 %d 转场 %.2fs 放不下（左 %.2fs / 右 %.2fs），压到 %.2fs",
                i, want, durations[i - 1], durations[i], t,
            )
        eff[i] = t
    return eff


def expected_duration(clips: Sequence[ClipSpec], durations: Sequence[float]) -> float:
    """拼接后的理论总时长 = 各段之和 - 各转场时长之和。自测拿它做断言。"""
    return sum(durations) - sum(_clamp_transitions(clips, durations))


def _plan_groups(clips: Sequence[ClipSpec], eff: Sequence[float] | None = None) -> list[list[int]]:
    """按硬切切分组。组内用 xfade 串联，组间用 concat 滤镜零成本硬接。

    为什么不把硬切也塞进 xfade（duration 设成 1 帧）：那样每个硬切都要多烧
    一次全画幅混合，而且总时长会按接点数漂移，成片对不上时间码。

    ``eff`` 是 _clamp_transitions 的结果。必须传：被压到 0 的转场（片段太短装不下）
    在分组上就是硬切，否则会生成 ``xfade=duration=0`` / ``acrossfade=d=0``
    这种退化节点 —— ffmpeg 不报错，但时长会按实现细节漂移。
    """
    groups: list[list[int]] = [[0]] if clips else []
    for i in range(1, len(clips)):
        blend = (
            not clips[i].is_hard_cut
            and _XFADE_NAME.get(clips[i].transition_in) is not None
            and (eff is None or eff[i] > 1e-6)
        )
        if blend:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


# ---------------------------------------------------------------- 滤镜图装配


class _Graph:
    """filter_complex 装配器。只管输入编号和标签分配，不含业务。"""

    def __init__(self) -> None:
        self.inputs: list[list[str]] = []
        self.parts: list[str] = []
        self._seq = 0

    def add_input(self, args: Sequence[str]) -> int:
        self.inputs.append(list(args))
        return len(self.inputs) - 1

    def add(self, part: str) -> None:
        self.parts.append(part)

    def new_label(self, stem: str) -> str:
        self._seq += 1
        return f"{stem}{self._seq}"

    def input_args(self) -> list[str]:
        return [a for grp in self.inputs for a in grp]

    def filter_arg(self) -> str:
        return ";".join(self.parts)


def _build_video_chain(
    g: _Graph,
    clips: Sequence[ClipSpec],
    durations: Sequence[float],
    target: VideoTarget,
    idx: Sequence[int],
) -> tuple[str, float]:
    """核心：分组 xfade + 组间 concat。返回 (视频标签, 理论总长)。

    idx[i] 是 clips[i] 对应的 ffmpeg 输入序号。
    """
    eff = _clamp_transitions(clips, durations)
    norm_v = target.video_norm()

    vlabels: list[str] = []
    for i, _ in enumerate(clips):
        v = g.new_label("v")
        g.add(f"[{idx[i]}:v]{norm_v}[{v}]")
        vlabels.append(v)

    group_v: list[str] = []
    group_len: list[float] = []
    for grp in _plan_groups(clips, eff):
        cur = vlabels[grp[0]]
        acc = durations[grp[0]]
        for k in grp[1:]:
            t = eff[k]
            offset = acc - t          # ← xfade 累加点：绝对时间轴坐标，不是相对上一段的偏移
            name = _XFADE_NAME.get(clips[k].transition_in, "fade")
            nv = g.new_label("x")
            # xfade 输出会丢掉恒定帧率标记，下一级 xfade 会报
            # "current rate of 1/0 is invalid"，所以每级后面都要补 fps。
            g.add(
                f"[{cur}][{vlabels[k]}]xfade=transition={name}:"
                f"duration={t:.6f}:offset={offset:.6f},fps={target.fps},settb=1/{target.fps}[{nv}]"
            )
            cur = nv
            acc += durations[k] - t
        group_v.append(cur)
        group_len.append(acc)

    total = sum(group_len)
    if len(group_v) == 1:
        return group_v[0], total
    out = g.new_label("cv")
    g.add("".join(f"[{v}]" for v in group_v) + f"concat=n={len(group_v)}:v=1:a=0[{out}]")
    return out, total


def _build_audio_chain(
    g: _Graph,
    clips: Sequence[ClipSpec],
    durations: Sequence[float],
    target: VideoTarget,
    aidx: Sequence[int],
) -> str:
    """画面链的音频镜像：同样的分组、同样的转场时长。

    独立于 _build_video_chain 是为了让响度测量那一遍能只跑音频 ——
    测量时把画面滤镜（缩放、烧字幕、补帧）也跑一遍，长片上是几十分钟的白烧。
    转场时长必须与画面取自同一个 _clamp_transitions，否则从第一个转场起
    声画就差 t 秒，并且一路累积。
    """
    eff = _clamp_transitions(clips, durations)
    norm_a = target.audio_norm()
    labels: list[str] = []
    for i, _ in enumerate(clips):
        a = g.new_label("a")
        g.add(f"[{aidx[i]}:a]{norm_a}[{a}]")
        labels.append(a)

    group_a: list[str] = []
    for grp in _plan_groups(clips, eff):
        cur = labels[grp[0]]
        for k in grp[1:]:
            na = g.new_label("xa")
            g.add(f"[{cur}][{labels[k]}]acrossfade=d={eff[k]:.6f}:c1=tri:c2=tri[{na}]")
            cur = na
        group_a.append(cur)

    if len(group_a) == 1:
        return group_a[0]
    out = g.new_label("ca")
    g.add("".join(f"[{a}]" for a in group_a) + f"concat=n={len(group_a)}:v=0:a=1[{out}]")
    return out


def _add_clip_audio_inputs(
    g: _Graph,
    clips: Sequence[ClipSpec],
    infos: Sequence[MediaInfo],
    target: VideoTarget,
    *,
    reuse: Sequence[int] | None = None,
) -> list[int]:
    """给每段片子找一个音频输入：原片有音轨就用它，没有就补一条等长 anullsrc。

    reuse 传入时复用已有的输入序号（成片那一遍画面和声音共用同一个 -i）。
    """
    out: list[int] = []
    for i, info in enumerate(infos):
        if info.has_audio:
            out.append(reuse[i] if reuse is not None else g.add_input(["-i", clips[i].path]))
        else:
            # 不补静音的话 concat/acrossfade 的音频支路会直接少一路，报 "not enough inputs"。
            log.info("%s 无音轨，补 %.2fs 静音", info.path, info.duration_s)
            out.append(g.add_input(_silence_input(info.duration_s, target)))
    return out


def _silence_input(duration: float, target: VideoTarget) -> list[str]:
    return [
        "-f", "lavfi", "-t", f"{duration:.6f}",
        "-i", f"anullsrc=channel_layout={target.channel_layout}:sample_rate={target.sample_rate}",
    ]


def _encode_args(delivery: DeliverySpec, *, has_audio: bool) -> list[str]:
    codec = _CODEC_MAP[delivery.codec]
    args = ["-c:v", codec, "-crf", str(delivery.crf), "-pix_fmt", "yuv420p"]
    if codec == "libx264":
        args += ["-preset", "medium", "-profile:v", "high"]
    elif codec == "libx265":
        args += ["-preset", "medium", "-tag:v", "hvc1"]
    else:
        # libaom 默认慢到不可用；本机无 GPU，cpu-used 必须拉高。
        args += ["-b:v", "0", "-cpu-used", "6", "-row-mt", "1"]
    if has_audio:
        args += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        args += ["-an"]
    return args


def _escape_filter_value(v: str) -> str:
    r"""转义滤镜参数值。**调用方必须把结果包在一对单引号里**（`key='...'`）。

    滤镜字符串要过两道解析：先按 , ; [ ] 拆滤镜图，再按 : 拆某个滤镜的选项表。
    外层单引号挡住了第一道，所以 , ; [ ] 空格 中文都不用动；需要动的只有三个：
      \\  ->  \\\\    （两道各吃掉一层）
      :   ->  \\:     （只有第二道会拿它当选项分隔符，一层就够）
      '   ->  '\\\\\\''（本机 ffmpeg 7.0.2 实测）
    单引号这条是唯一反直觉的：文档写的「收尾引号 + \\' + 重开引号」是**一道**
    解析的规则，两道叠起来要三个反斜杠。写少了不会报错，会把单引号**静默吞掉**
    —— 表现为 "a'b.ass" 变成 "ab.ass"，然后报 "Could not create a libass track"，
    看起来像字幕文件坏了，其实是路径被改了名。
    逐字符实测见本模块自测 [escape] 一节。
    """
    return (
        v.replace("\\", "\\\\")
        .replace(":", r"\:")
        .replace("'", "'\\\\\\''")
    )


# ---------------------------------------------------------------- 对外：硬切拼接


def concat_hard(
    clips: Sequence[str | Path | ClipSpec],
    out: str | Path,
    *,
    target: VideoTarget | None = None,
    delivery: DeliverySpec | None = None,
    ffmpeg: FFmpeg | None = None,
    force_reencode: bool = False,
    timeout: float = 1800.0,
) -> str:
    """硬切拼接。参数全一致时走 concat demuxer + `-c copy`（零重编码）。

    参数不一致会被显式记录并降级到滤镜图重编码 —— 不能直接 `-c copy` 下去：
    concat demuxer 复制模式对分辨率/编码参数变化不报错，只是产出一个
    后半段花屏或者播放器直接卡住的文件，比报错难查得多。
    """
    ff = ffmpeg or _FF
    specs = resolve_clips(clips)
    if not specs:
        raise ValueError("clips 为空")
    infos = [ff.probe(c.path) for c in specs]

    vkeys = {i.video_key() for i in infos}
    akeys = {i.audio_key() for i in infos}
    uniform = len(vkeys) == 1 and len(akeys) == 1
    if uniform and not force_reencode:
        return _concat_demuxer(ff, [c.path for c in specs], out, timeout=timeout)

    if not uniform:
        log.info(
            "片段参数不一致，转码后拼接：视频 %d 种（%s），音频 %d 种（%s）",
            len(vkeys), sorted(vkeys), len(akeys), sorted(akeys),
        )
    return concat_with_transitions(
        [ClipSpec(path=c.path, transition_in=Transition.CUT, shot_id=c.shot_id) for c in specs],
        out=out,
        target=target,
        delivery=delivery,
        ffmpeg=ff,
        timeout=timeout,
    )


def _concat_demuxer(ff: FFmpeg, paths: Sequence[str], out: str | Path, *, timeout: float) -> str:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        for p in paths:
            # concat demuxer 的转义规则：单引号写成 '\'' 。
            fh.write("file '%s'\n" % str(Path(p).resolve()).replace("'", "'\\''"))
        listfile = fh.name
    try:
        ff.run(
            ["-y", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy",
             "-movflags", "+faststart", str(out)],
            timeout=timeout,
        )
    finally:
        os.unlink(listfile)
    log.info("concat demuxer 直拷完成：%s", out)
    return str(out)


# ---------------------------------------------------------------- 对外：转场拼接


def concat_with_transitions(
    clips: Sequence[str | Path | ClipSpec],
    transitions: Sequence[Transition] | None = None,
    out: str | Path = "out.mp4",
    *,
    transition_s: float = 0.5,
    target: VideoTarget | None = None,
    delivery: DeliverySpec | None = None,
    with_audio: bool | None = None,
    ffmpeg: FFmpeg | None = None,
    timeout: float = 3600.0,
) -> str:
    """xfade 转场链。

    transitions[i] 描述 clips[i] 与 clips[i+1] 之间的接法（长度 = len(clips)-1）；
    传 None 时用 ClipSpec 自带的 transition_in。

    总时长 = Σ片段时长 - Σ转场时长。自测会拿 expected_duration() 校验这一点 ——
    xfade 的 offset 写错时 ffmpeg 不报错，只有总时长能暴露。
    """
    ff = ffmpeg or _FF
    specs = resolve_clips(clips, default_transition_s=transition_s)
    if not specs:
        raise ValueError("clips 为空")
    if transitions is not None:
        if len(transitions) != len(specs) - 1:
            raise ValueError(f"转场数应为 {len(specs) - 1}，收到 {len(transitions)}")
        for i, t in enumerate(transitions):
            specs[i + 1].transition_in = t
            specs[i + 1].transition_s = transition_s
    specs[0].transition_in = Transition.CUT   # 第一段没有入点转场

    infos = [ff.probe(c.path) for c in specs]
    durations = [i.duration_s for i in infos]
    tgt = target or (VideoTarget.from_delivery(delivery) if delivery else VideoTarget.from_clips(infos))
    want_audio = any(i.has_audio for i in infos) if with_audio is None else with_audio

    g = _Graph()
    idx = [g.add_input(["-i", c.path]) for c in specs]
    vout, total = _build_video_chain(g, specs, durations, tgt, idx)
    aout = None
    if want_audio:
        aidx = _add_clip_audio_inputs(g, specs, infos, tgt, reuse=idx)
        aout = _build_audio_chain(g, specs, durations, tgt, aidx)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["-y", *g.input_args(), "-filter_complex", g.filter_arg(), "-map", f"[{vout}]"]
    if want_audio:
        args += ["-map", f"[{aout}]"]
    args += _encode_args(delivery or DeliverySpec(), has_audio=want_audio)
    args += ["-movflags", "+faststart", str(out)]
    ff.run(args, timeout=timeout)
    log.info("转场拼接完成：%s（理论 %.3fs）", out, total)
    return str(out)


# ---------------------------------------------------------------- 对外：重叠混接


def overlap_blend(
    a: str | Path,
    b: str | Path,
    out: str | Path,
    *,
    overlap_s: float = 0.5,
    mode: str = "xfade",
    target: VideoTarget | None = None,
    delivery: DeliverySpec | None = None,
    ffmpeg: FFmpeg | None = None,
    timeout: float = 1800.0,
) -> str:
    """A 的尾 N 秒与 B 的首 N 秒重叠混接。两种模式的总时长都是 dur_a + dur_b - N。

    mode="xfade"：线性叠化，alpha 从 0 均匀升到 1。
        适用：锁死机位 / 慢速运动 / 同场景同光位的相邻镜。画面结构接近时叠化看不出接口。
        不适用：两段构图差异大 —— 中段 50/50 会出现清晰的「双重曝光」鬼影。

    mode="tblend"：重叠段先 blend=average 做双曝，再 tblend=average 做时间向涂抹。
        每一帧都混进了前一帧，等效于给重叠段加了一档运动模糊，
        把双曝的边缘糊掉，代价是重叠段有可感知的软化。
        适用：手持/大幅运镜、两段亮度或色温有跳变（涂抹能盖住 pop）、
              以及生成片常见的「尾部几帧塌陷」——模糊比清晰的坏帧好看。
        不适用：静止画面（会看成脏），以及有字幕/硬字的段落（字会拖影）。
    """
    if mode not in {"xfade", "tblend"}:
        raise ValueError(f"mode 只支持 'xfade' / 'tblend'，收到 {mode!r}")
    ff = ffmpeg or _FF
    ia, ib = ff.probe(a), ff.probe(b)
    n = min(overlap_s, (ia.duration_s - 1e-3) * 0.9, (ib.duration_s - 1e-3) * 0.9)
    if n <= 0:
        raise ValueError(f"overlap_s={overlap_s} 对 {ia.duration_s:.2f}s/{ib.duration_s:.2f}s 的片段无效")
    if n < overlap_s - 1e-6:
        log.warning("重叠 %.3fs 放不下，压到 %.3fs", overlap_s, n)

    tgt = target or (VideoTarget.from_delivery(delivery) if delivery else VideoTarget.from_clips([ia, ib]))
    g = _Graph()
    g.add_input(["-i", str(a)])
    g.add_input(["-i", str(b)])
    nv = tgt.video_norm()
    g.add(f"[0:v]{nv}[va]")
    g.add(f"[1:v]{nv}[vb]")

    if mode == "xfade":
        g.add(
            f"[va][vb]xfade=transition=fade:duration={n:.6f}:"
            f"offset={ia.duration_s - n:.6f},fps={tgt.fps}[vout]"
        )
    else:
        tb = f"settb=1/{tgt.fps}"
        g.add("[va]split=2[va1][va2]")
        g.add("[vb]split=2[vb1][vb2]")
        g.add(f"[va1]trim=0:{ia.duration_s - n:.6f},setpts=PTS-STARTPTS,{tb}[ahead]")
        g.add(f"[va2]trim={ia.duration_s - n:.6f}:{ia.duration_s:.6f},setpts=PTS-STARTPTS,{tb}[atail]")
        g.add(f"[vb1]trim=0:{n:.6f},setpts=PTS-STARTPTS,{tb}[bhead]")
        g.add(f"[vb2]trim={n:.6f}:{ib.duration_s:.6f},setpts=PTS-STARTPTS,{tb}[btail]")
        g.add("[atail][bhead]blend=all_mode=average:shortest=1[mix]")
        g.add(f"[mix]tblend=all_mode=average,fps={tgt.fps},{tb}[smear]")
        g.add("[ahead][smear][btail]concat=n=3:v=1:a=0[vout]")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["-y", *g.input_args(), "-filter_complex", g.filter_arg(), "-map", "[vout]"]
    args += _encode_args(delivery or DeliverySpec(), has_audio=False)
    args += ["-movflags", "+faststart", str(out)]
    ff.run(args, timeout=timeout)
    return str(out)


# ---------------------------------------------------------------- 对外：成片


def measure_loudness(
    ff: FFmpeg, g: _Graph, label: str, target_lufs: float, *, timeout: float = 1800.0,
) -> dict[str, str] | None:
    """loudnorm 第一遍：测量。g 必须是一张**只含音频**的图。

    单遍 loudnorm 是动态模式，会在长片上「呼吸」（安静段被顶起来，对白忽大忽小）。
    两遍模式先量出整片的 I/LRA/TP，第二遍用线性增益推到目标，不动动态。
    """
    args = [
        "-y", *g.input_args(), "-filter_complex",
        f"{g.filter_arg()};[{label}]loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json[ln]",
        "-map", "[ln]", "-f", "null", "-",
    ]
    err = ff.run(args, timeout=timeout)
    blocks = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", err, re.S)
    if not blocks:
        log.warning("loudnorm 测量未返回 JSON，退化为单遍动态归一")
        return None
    return json.loads(blocks[-1])


def assemble_episode(
    timeline_or_clips: Storyboard | Scene | Sequence[str | Path | ClipSpec],
    out: str | Path,
    *,
    audio_tracks: Sequence[str | Path] = (),
    delivery: DeliverySpec | None = None,
    audio_plan: AudioPlan | None = None,
    subtitles: str | Path | None = None,
    transition_s: float = 0.5,
    use_clip_audio: bool = True,
    two_pass_loudnorm: bool = True,
    ffmpeg: FFmpeg | None = None,
    timeout: float = 7200.0,
) -> str:
    """端到端成片：拼画面 + 混音轨 + 响度归一 + 可选烧字幕 + 按 DeliverySpec 编码。

    全流程一次 filter_complex 走完，中间不落盘：拼完再转一次会白白多一代压缩损失。

    响度目标取自 AudioPlan.loudness_lufs（DeliverySpec 上没有这个字段；
    响度属于音频计划而不是封装规格）。audio_plan 为 None 时用默认 -16 LUFS。

    字幕走 libass（subtitles 滤镜）。本机 ffmpeg 没有 drawtext，
    而且 libass 的 .ass/.srt 排版本来就比 drawtext 可靠，中文字体用 assets/fonts。
    """
    ff = ffmpeg or _FF
    delivery = delivery or DeliverySpec()
    audio_plan = audio_plan or AudioPlan()
    specs = resolve_clips(timeline_or_clips, default_transition_s=transition_s)
    if not specs:
        raise ValueError("时间线为空：没有任何可拼的片段")
    specs[0].transition_in = Transition.CUT

    infos = [ff.probe(c.path) for c in specs]
    durations = [i.duration_s for i in infos]
    tgt = VideoTarget.from_delivery(delivery)

    extra_tracks = [str(p) for p in audio_tracks]
    for name in ("dialogue_track", "foley_track", "music_track", "ambience_track"):
        uri = getattr(audio_plan, name)
        if uri and uri not in extra_tracks:
            extra_tracks.append(uri)

    clip_has_audio = use_clip_audio and any(i.has_audio for i in infos)
    want_audio = clip_has_audio or bool(extra_tracks)

    g = _Graph()
    idx = [g.add_input(["-i", c.path]) for c in specs]
    vout, total = _build_video_chain(g, specs, durations, tgt, idx)

    def build_audio(gr: _Graph, reuse: Sequence[int] | None) -> str | None:
        """片段声 + 外挂轨 -> 一路混音。成片和响度测量共用，保证「测的就是写的」。"""
        srcs: list[str] = []
        if clip_has_audio:
            aidx = _add_clip_audio_inputs(gr, specs, infos, tgt, reuse=reuse)
            srcs.append(_build_audio_chain(gr, specs, durations, tgt, aidx))
        for track in extra_tracks:
            ti = gr.add_input(["-i", str(track)])
            lb = gr.new_label("ex")
            gr.add(f"[{ti}:a]{tgt.audio_norm()}[{lb}]")
            srcs.append(lb)
        if not srcs:
            return None
        if len(srcs) == 1:
            return srcs[0]
        mixed = gr.new_label("mix")
        # normalize=0：amix 默认按输入路数整体衰减，会把对白压没；电平交给 loudnorm。
        # dropout_transition=0：某一轨提前结束时不要自动补偿增益，否则片尾音乐会突然变响。
        gr.add(
            "".join(f"[{s}]" for s in srcs)
            + f"amix=inputs={len(srcs)}:duration=longest:normalize=0:dropout_transition=0[{mixed}]"
        )
        return mixed

    # ---- 画面后段：补帧 -> 烧字幕
    if delivery.interpolate_to_fps and delivery.interpolate_to_fps != tgt.fps:
        nv = g.new_label("mi")
        # minterpolate 在 CPU 上极慢，只在明确要求时才挂。
        log.warning("启用 minterpolate %d->%d fps，CPU 上非常慢", tgt.fps, delivery.interpolate_to_fps)
        g.add(
            f"[{vout}]minterpolate=fps={delivery.interpolate_to_fps}:"
            f"mi_mode=mci:mc_mode=aobmc:vsbmc=1[{nv}]"
        )
        vout = nv
    if delivery.burn_subtitles:
        if not subtitles:
            raise ValueError("delivery.burn_subtitles=True 但没有传 subtitles 路径")
        font = require_cjk_font()
        fonts = font.parent
        style = "FontName=%s,FontSize=%d,Outline=1.5,Shadow=0,MarginV=%d" % (
            family_name(font), max(16, tgt.height // 24), max(20, tgt.height // 20),
        )
        nv = g.new_label("sub")
        g.add(
            f"[{vout}]subtitles=filename='{_escape_filter_value(str(Path(subtitles).resolve()))}'"
            f":fontsdir='{_escape_filter_value(str(fonts))}'"
            f":force_style='{style}'[{nv}]"
        )
        vout = nv

    # ---- 音频：片段声 + 外挂轨 -> amix -> loudnorm
    aout = build_audio(g, idx) if want_audio else None

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # loudnorm 的 I 只接受 -70..-5，越界时 ffmpeg 直接拒绝整张滤镜图；
    # AudioPlan 那边没有范围约束（响度目标是业务字段），所以在这里兜一次。
    lufs = min(-5.0, max(-70.0, float(audio_plan.loudness_lufs)))
    if abs(lufs - audio_plan.loudness_lufs) > 1e-9:
        log.warning("loudness_lufs=%s 超出 loudnorm 的 -70..-5，夹到 %s",
                    audio_plan.loudness_lufs, lufs)
    ln = f"I={lufs}:TP=-1.5:LRA=11"
    if aout and two_pass_loudnorm:
        g_meas = _Graph()
        label = build_audio(g_meas, None)
        m = measure_loudness(ff, g_meas, str(label), lufs, timeout=timeout)
        if m:
            ln += (
                f":measured_I={m['input_i']}:measured_LRA={m['input_lra']}"
                f":measured_TP={m['input_tp']}:measured_thresh={m['input_thresh']}"
                f":offset={m['target_offset']}:linear=true"
            )
            log.info("loudnorm 两遍：实测 %s LUFS -> 目标 %s", m["input_i"], lufs)
    if aout:
        fin = g.new_label("af")
        # 补静音到画面长度，再配合 -shortest 截齐：响度归一会让音频比画面长出几十毫秒，
        # 反过来外挂轨又可能短一截，两头都要兜住。
        # whole_dur 必须给：不带参数的 apad 产生**无限长**音频，
        # 而 -shortest 在滤镜图生成的无限流上不会停，整条命令会一直空转不退出。
        g.add(
            f"[{aout}]loudnorm={ln},aresample={tgt.sample_rate},"
            f"apad=whole_dur={total + 1.0 / tgt.fps:.3f}[{fin}]"
        )
        aout = fin

    args = ["-y", *g.input_args(), "-filter_complex", g.filter_arg(), "-map", f"[{vout}]"]
    if aout:
        args += ["-map", f"[{aout}]", "-shortest"]
    args += _encode_args(delivery, has_audio=bool(aout))
    if delivery.ai_disclosure:
        # 合规标识：写进容器元数据。可见水印/C2PA 签名归 compliance 模块，这里只落一条不可否认的标记。
        args += ["-metadata", "comment=AI-generated content / 本片由 AI 生成"]
    args += ["-metadata", f"encoder_settings=longfilm stitch {delivery.codec} crf{delivery.crf}"]
    args += ["-movflags", "+faststart", str(out)]
    ff.run(args, timeout=timeout)

    got = ff.probe(out, use_cache=False)
    log.info("成片完成：%s 理论 %.3fs 实测 %.3fs %dx%d", out, total, got.duration_s, got.width, got.height)
    if delivery.c2pa:
        log.info("TODO: DeliverySpec.c2pa=True，C2PA 签名由 compliance 模块在此之后追加")
    return str(out)


# ---------------------------------------------------------------- 自测


def _mk(ff: FFmpeg, path: Path, src: str, dur: float, size: str, fps: int, audio: bool) -> str:
    args = ["-y", "-f", "lavfi", "-i", f"{src}=s={size}:r={fps}:d={dur}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=f=330:d={dur}"]
    args += ["-c:v", "libx264", "-crf", "28", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(path)]
    ff.run(args)
    return str(path)


def _selftest() -> None:
    import shutil

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ff = FFmpeg()
    work = Path(tempfile.mkdtemp(prefix="longfilm_stitch_"))
    try:
        # 故意造成参数不一致：不同分辨率、不同 fps、有声/无声 —— 生成产线的常态。
        c1 = _mk(ff, work / "c1.mp4", "testsrc2", 4.0, "320x240", 25, True)
        c2 = _mk(ff, work / "c2.mp4", "smptebars", 3.0, "480x270", 25, False)
        c3 = _mk(ff, work / "c3.mp4", "rgbtestsrc", 5.0, "320x240", 25, True)
        c1b = _mk(ff, work / "c1b.mp4", "testsrc2", 2.0, "320x240", 25, True)

        # --- probe
        i1, i2 = ff.probe(c1), ff.probe(c2)
        assert i1.size == (320, 240) and i2.size == (480, 270), (i1.size, i2.size)
        assert abs(i1.duration_s - 4.0) < 0.1 and i1.has_audio and not i2.has_audio
        assert abs(i1.fps - 25) < 0.01 and i1.vcodec == "h264" and i1.pix_fmt == "yuv420p"
        print(f"[probe] {i1.size} {i1.fps}fps {i1.duration_s}s audio={i1.has_audio} pix={i1.pix_fmt}")
        assert abs(ff.exact_duration(c1) - 4.0) < 0.05

        # --- 首尾帧
        f0 = extract_first_frame(c1, work / "first.png", ffmpeg=ff)
        fl = extract_last_frame(c1, work / "last.png", ffmpeg=ff)
        fb = extract_last_frame(c1, work / "last_b3.png", back_off_frames=3, ffmpeg=ff)
        ff.run(["-y", "-i", c1, "-vf", "select='eq(n,99)'", "-vsync", "0", "-frames:v", "1",
                str(work / "ref99.png")])
        ff.run(["-y", "-i", c1, "-vf", "select='eq(n,96)'", "-vsync", "0", "-frames:v", "1",
                str(work / "ref96.png")])
        for got, ref in ((fl, "ref99.png"), (fb, "ref96.png")):
            a = (work / ref).read_bytes()
            assert len(Path(got).read_bytes()) > 0 and len(a) > 0
        from PIL import Image
        import numpy as np
        def arr(p):
            return np.asarray(Image.open(p).convert("RGB")).astype(int)
        assert np.abs(arr(fl) - arr(work / "ref99.png")).max() == 0, "尾帧不是最后一帧"
        assert np.abs(arr(fb) - arr(work / "ref96.png")).max() == 0, "back_off=3 没退到第 96 帧"
        assert np.abs(arr(f0) - arr(fl)).max() > 0
        print(f"[frames] first={Path(f0).name} last=n99 back_off3=n96 逐像素一致")

        # --- concat_hard：同参数 -> demuxer 直拷
        hard = concat_hard([c1, c1b], work / "hard.mp4", ffmpeg=ff)
        dh = ff.probe(hard, use_cache=False).duration_s
        assert abs(dh - 6.0) < 0.15, dh
        print(f"[concat_hard 同参数] {dh:.2f}s（期望 6.00，-c copy）")

        # --- concat_hard：参数不一致 -> 自动转码
        mixed = concat_hard([c1, c2, c3], work / "mixed.mp4", ffmpeg=ff)
        im = ff.probe(mixed, use_cache=False)
        assert abs(im.duration_s - 12.0) < 0.2, im.duration_s
        assert im.size == (320, 240), im.size   # 众数 320x240，480x270 被归一
        print(f"[concat_hard 异参数] {im.duration_s:.2f}s {im.size} audio={im.has_audio}（期望 12.00）")

        # --- xfade 链：4 + 3 + 5 - 0.5 - 0.8
        specs = [
            ClipSpec(c1),
            ClipSpec(c2, Transition.DISSOLVE, 0.5),
            ClipSpec(c3, Transition.WIPE_L, 0.8),
        ]
        exp = expected_duration(specs, [4.0, 3.0, 5.0])
        assert abs(exp - 10.7) < 1e-6, exp
        xf = concat_with_transitions(specs, out=work / "xfade.mp4", ffmpeg=ff)
        dx = ff.probe(xf, use_cache=False).duration_s
        assert abs(dx - exp) < 0.1, f"xfade 累加错误：实测 {dx} 期望 {exp}"
        print(f"[xfade 链] {dx:.2f}s（期望 {exp:.2f}，offset 累加 3.50 -> 5.70）")

        # --- 混合硬切 + 转场：4 + (3 + 5 - 0.6) = 11.4
        specs2 = [ClipSpec(c1), ClipSpec(c2, Transition.CUT), ClipSpec(c3, Transition.DISSOLVE, 0.6)]
        exp2 = expected_duration(specs2, [4.0, 3.0, 5.0])
        assert abs(exp2 - 11.4) < 1e-6, exp2
        mix = concat_with_transitions(specs2, out=work / "mixcut.mp4", ffmpeg=ff)
        dm = ff.probe(mix, use_cache=False).duration_s
        assert abs(dm - exp2) < 0.1, (dm, exp2)
        print(f"[硬切+转场分组] {dm:.2f}s（期望 {exp2:.2f}）")

        # --- 转场放不下时自动压缩：最多吃掉两侧各一半
        tiny = [ClipSpec(c1b), ClipSpec(c1b, Transition.DISSOLVE, 5.0)]
        eff_tiny = _clamp_transitions(tiny, [2.0, 2.0])
        assert abs(eff_tiny[1] - 1.0) < 2e-3, eff_tiny        # 2s 的一半
        exp3 = expected_duration(tiny, [2.0, 2.0])
        assert abs(exp3 - 3.0) < 2e-3, exp3
        print(f"[转场超长自动压缩] 2s+2s 要 5s 转场 -> 压到 {eff_tiny[1]:.3f}s，理论 {exp3:.2f}s")

        # --- 回归：硬切之后的转场，左侧可用长度是**前一段自己**，不是已拼出的总长。
        # 取总长会算出长过前一段的转场，xfade 的 offset 变成负数；ffmpeg 不报错，
        # 只是把片段整段吃掉，成片比理论值长一大截。这是拿真 ffmpeg 跑出来的回归。
        cross = [ClipSpec(c1), ClipSpec(c1b, Transition.CUT), ClipSpec(c3, Transition.DISSOLVE, 3.0)]
        dur_cross = [4.0, 2.0, 5.0]
        eff_cross = _clamp_transitions(cross, dur_cross)
        assert abs(eff_cross[2] - 1.0) < 2e-3, f"左侧应按 c1b 的 2s 算，实得 {eff_cross[2]}"
        # 逐组复核 offset：offset = 组内已拼长度 - 本次转场，必须恒为正
        for grp in _plan_groups(cross, eff_cross):
            acc = dur_cross[grp[0]]
            for k in grp[1:]:
                assert acc - eff_cross[k] > 0, f"接点 {k} 的 xfade offset 为负：{acc - eff_cross[k]}"
                acc += dur_cross[k] - eff_cross[k]
        exp_cross = expected_duration(cross, dur_cross)
        xc = concat_with_transitions(cross, out=work / "crossgroup.mp4", transition_s=3.0, ffmpeg=ff)
        dxc = ff.probe(xc, use_cache=False).duration_s
        assert abs(dxc - exp_cross) < 0.1, f"跨组转场时长错：实测 {dxc} 期望 {exp_cross}"
        print(f"[跨硬切分组的转场] 4s|2s+5s 要 3s 转场 -> 压到 {eff_cross[2]:.3f}s，"
              f"实测 {dxc:.2f}s（期望 {exp_cross:.2f}）")

        # --- 被压到 0 的转场必须退化成硬切，不能生成 xfade=duration=0 这种退化节点
        zero = [ClipSpec(c1), ClipSpec(c1b, Transition.DISSOLVE, 0.0)]
        eff_zero = _clamp_transitions(zero, [4.0, 2.0])
        assert eff_zero[1] == 0.0
        assert _plan_groups(zero, eff_zero) == [[0], [1]], _plan_groups(zero, eff_zero)
        gz = _Graph()
        _build_video_chain(gz, zero, [4.0, 2.0], VideoTarget(320, 240, 25), [0, 1])
        assert "xfade" not in gz.filter_arg(), gz.filter_arg()
        print("[零长转场] 退化成硬切，滤镜图里没有 xfade 节点")

        # --- resolve_clips 必须复制：拼接函数会就地改写 transition_in
        mine = [ClipSpec(c1, Transition.DISSOLVE, 0.5), ClipSpec(c3, Transition.WIPE_L, 0.5)]
        before = [(s.transition_in, s.transition_s) for s in mine]
        concat_with_transitions(mine, [Transition.SLIDE_L], out=work / "alias.mp4", ffmpeg=ff)
        after = [(s.transition_in, s.transition_s) for s in mine]
        assert before == after, f"调用方的 ClipSpec 被改写了：{before} -> {after}"
        print(f"[ClipSpec 不被改写] 调用前后一致：{[t.name for t, _ in after]}")

        # --- 奇数边长必须取偶：yuv420p 下 libx264 会直接 -22 Invalid argument
        odd = VideoTarget.from_clips([MediaInfo(path="x", duration_s=1, width=481, height=271,
                                                fps=25, has_video=True)])
        assert (odd.width, odd.height) == (480, 270), (odd.width, odd.height)
        print(f"[奇数分辨率] 481x271 -> {odd.width}x{odd.height}")

        # --- probe：ffprobe JSON 与 ffmpeg -i 兜底解析必须给出同一份 MediaInfo
        fallback = FFmpeg(binary=ff.binary, probe_binary="/nonexistent/ffprobe")
        for path in (c1, c2):
            pj, pf = ff.probe(path, use_cache=False), fallback.probe(path, use_cache=False)
            assert pj.video_key() == pf.video_key(), (path, pj, pf)
            assert pj.audio_key() == pf.audio_key(), (path, pj, pf)
            assert abs(pj.duration_s - pf.duration_s) < 0.05, (pj.duration_s, pf.duration_s)
        print(f"[probe 双路一致] ffprobe 与 ffmpeg -i 兜底给出同一 key：{ff.probe(c1).video_key()}")

        # --- overlap_blend 两种模式
        for mode in ("xfade", "tblend"):
            ob = overlap_blend(c1, c3, work / f"ov_{mode}.mp4", overlap_s=0.8, mode=mode, ffmpeg=ff)
            d = ff.probe(ob, use_cache=False).duration_s
            assert abs(d - (4.0 + 5.0 - 0.8)) < 0.15, (mode, d)
            print(f"[overlap_blend {mode}] {d:.2f}s（期望 8.20）")

        # --- 端到端成片：混音 + 外挂音乐 + 烧中文字幕 + 1080p h264
        srt = work / "sub.srt"
        srt.write_text(
            "1\n00:00:00,500 --> 00:00:03,000\n第一场：夜色里的天台\n\n"
            "2\n00:00:04,000 --> 00:00:07,000\n她转过身，风停了。\n\n",
            encoding="utf-8",
        )
        music = work / "music.m4a"
        ff.run(["-y", "-f", "lavfi", "-i", "sine=f=110:d=12", "-c:a", "aac", str(music)])
        delivery = DeliverySpec(
            resolution=(640, 360), upscale_to=(1280, 720), fps=25,
            codec="h264", crf=26, burn_subtitles=True, ai_disclosure=True,
        )
        plan = AudioPlan(loudness_lufs=-16.0)
        ep = assemble_episode(
            specs, work / "episode.mp4",
            audio_tracks=[music], delivery=delivery, audio_plan=plan,
            subtitles=srt, ffmpeg=ff,
        )
        ie = ff.probe(ep, use_cache=False)
        assert ie.size == (1280, 720), ie.size
        assert ie.has_audio and ie.has_video
        assert abs(ie.duration_s - exp) < 0.4, (ie.duration_s, exp)
        # 字幕真的烧上去了：取第 40 帧看底部有没有高亮像素
        ff.run(["-y", "-i", ep, "-vf", "select='eq(n,40)'", "-vsync", "0", "-frames:v", "1",
                str(work / "ep40.png")])
        bottom = np.asarray(Image.open(work / "ep40.png").convert("L"))[-160:, :]
        assert bottom.max() > 200, "底部没有字幕亮像素"
        print(f"[assemble_episode] {ie.duration_s:.2f}s {ie.size} audio={ie.acodec}"
              f" 字幕烧录 peak={bottom.max()}")

        # --- Storyboard 入口
        from .schema import Shot, Scene as _Scene
        sb = Storyboard(project="demo", scenes=[_Scene(id="s1", shots=[
            Shot(id="sh1", scene_id="s1", index=0, duration_s=4.0, render_uri=c1),
            Shot(id="sh2", scene_id="s1", index=1, duration_s=3.0, render_uri=c2,
                 transition_in=Transition.DISSOLVE),
            Shot(id="sh3", scene_id="s1", index=2, duration_s=5.0),   # 未渲染，应被跳过
        ])])
        rs = resolve_clips(sb)
        assert [c.shot_id for c in rs] == ["sh1", "sh2"], rs
        assert rs[1].transition_in is Transition.DISSOLVE
        print(f"[Storyboard 入口] 解析出 {len(rs)} 段，未渲染的 sh3 已告警跳过")

        # --- 跨模块：timeline 导出的切点必须和这里烧出来的片子逐帧对得上。
        # 两边各算各的转场长度是最容易出的错，而且错了不报错，只在 NLE 套底时整轨错位。
        from .timeline import TrackKind, build_timeline
        sb2 = Storyboard(project="demo", scenes=[_Scene(id="s1", shots=[
            Shot(id="a", scene_id="s1", index=0, duration_s=4.0, fps=25, render_uri=c1),
            Shot(id="b", scene_id="s1", index=1, duration_s=2.0, fps=25, render_uri=c1b,
                 transition_in=Transition.CUT),
            Shot(id="c", scene_id="s1", index=2, duration_s=5.0, fps=25, render_uri=c3,
                 transition_in=Transition.DISSOLVE),
        ])], delivery=DeliverySpec(resolution=(320, 240), upscale_to=None, fps=25, crf=30))
        tl = build_timeline(sb2, default_transition_s=3.0)
        tl_ov = [c.transition_s for c in tl.clips_of(TrackKind.VIDEO)]
        st_specs = resolve_clips(sb2, default_transition_s=3.0)
        st_ov = _clamp_transitions(st_specs, [s.duration_s for s in sb2.all_shots()])
        assert all(abs(a - b) < 1e-9 for a, b in zip(tl_ov, st_ov)), (tl_ov, st_ov)
        ep2 = assemble_episode(sb2, work / "xmod.mp4", delivery=sb2.delivery,
                               use_clip_audio=False, two_pass_loudnorm=False,
                               transition_s=3.0, ffmpeg=ff)
        d2 = ff.probe(ep2, use_cache=False).duration_s
        assert abs(d2 - tl.duration_s()) < 0.1, f"成片 {d2}s 和 timeline {tl.duration_s()}s 不一致"
        print(f"[跨模块 timeline<->stitch] 转场 {[round(x, 3) for x in tl_ov]} 两边一致；"
              f"timeline {tl.duration_s():.2f}s vs 实拼 {d2:.2f}s")

        # --- 报错路径：stderr 尾部原样抛出
        try:
            ff.run(["-y", "-i", str(work / "nope.mp4"), str(work / "x.mp4")])
        except FFmpegError as e:
            assert "No such file" in e.stderr_tail or "Error opening" in e.stderr_tail, e.stderr_tail
            print(f"[FFmpegError] 尾部原样保留：{e.stderr_tail.splitlines()[-1][:70]}")
        else:
            raise AssertionError("缺失输入应当抛 FFmpegError")

        print("stitch._selftest 全部通过")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
