"""质检门禁 —— 生成产线的废片过滤器。

为什么这个模块是成本核心：扩散视频模型的单次出片可用率经常只有 50%，
如果靠人去看每条 5~15s 的原子镜，一集 300s 的片子光审片就要几百次人工决策。
门禁的目标不是「评出好坏」，而是**把明显的废片在人看到之前挡掉**，
并给出可执行的重拍方向，让重拍不是盲目重摇 seed。

设计要点：

1. **一次解码，喂所有指标**。六个 CPU 指标里有五个只需要逐帧的
   YAVG/UAVG/VAVG/YDIF —— ffmpeg 的 signalstats 一趟解码就能全给出来。
   所以逐帧统计、freeze/black 事件、probe 结果全部挂在 QCContext 上做缓存，
   指标之间共享。指标各自起一个 ffmpeg 进程是这类模块最常见的性能坑。

2. **判据要能区分「废」和「不同」**。闪烁用逐帧亮度的**一阶差分标准差**
   而不是差分均值：匀速变亮（合法的打光变化）一阶差分是常数、标准差近 0，
   而逐帧忽明忽暗的闪烁一阶差分正负交替、标准差极大。这条是本模块
   最关键的一个判别式，自测里用合成片验证过（好片 0.07，闪烁片 142.9）。

3. **重模型指标一律做成可选插件**。本产线的质检要能在没有 GPU 的调度机上跑，
   所以 insightface / CLIP 美学 / SyncNet 这类走延迟导入 + 缺依赖即 SKIP，
   绝不在 import 期崩掉整条产线。

外部指标口径（2026-09-19 查证）：
- VBench temporal_flickering：对相邻帧求 MAE 后归一为 (255 - mean(MAE)) / 255，
  越高越好。本模块在 detail 里同口径给出 `vbench_flicker_score` 便于横向对齐。
  来源 https://github.com/Vchitect/VBench （vbench/temporal_flickering.py）
- VBench 的 aesthetic_quality 用 LAION Aesthetic Predictor（CLIP ViT-L/14 + MLP，
  权重 sac+logos+ava1-l14-linearMSE.pth），输出约 1~10 分。
  来源 https://github.com/christophschuhmann/improved-aesthetic-predictor
- 口型同步 LSE-D（越低越好）/ LSE-C（越高越好）由 syncnet_python 计算；
  Wav2Lip 论文 Table 1 给的真人实拍基准：LRS2 真视频 LSE-D 6.736 / LSE-C 7.838，
  LRS3 真视频 6.956 / 7.592，LRW 真视频 7.012 / 6.931。
  来源 https://arxiv.org/abs/2008.10010 、https://github.com/Rudrabha/Wav2Lip
  （2026-09-20 复核：LSE-D/LSE-C 的定义与量级无误；打分脚本的落地细节见
  LipSyncScore 的 docstring —— 分数写在 all_scores.txt 里，不在 stdout。）
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Sequence

import numpy as np

from ._proc import popen as _sp_popen  # noqa: F401
from ._proc import run as _sp_run
from .schema import Shot, Storyboard, Transition
from .stitch import FFmpeg, FFmpegError, MediaInfo, default_ffmpeg

log = logging.getLogger(__name__)


# ================================================================ 结果类型


class MetricStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"      # 依赖缺失或前提不成立（无对白就不测口型），不计入总分
    ERROR = "error"    # 指标自己炸了 —— 不能当通过，必须人工看


class Verdict(str, Enum):
    PASS = "pass"
    RETAKE = "retake"
    ESCALATE = "escalate"   # 重拍次数用尽或指标失效，交人


@dataclass(frozen=True)
class MetricResult:
    """单项指标的结论。

    value 的量纲由各指标自己定义（见 detail["unit"]），门禁不解释它；
    门禁只用 passed（是否越线）和 score（归一到 0..1 的质量分）做决策。
    约定 score 在恰好压线时 = 0.5，这样「总分阈值」和「单项阈值」语义一致。
    """

    name: str
    value: float
    passed: bool
    threshold: float
    detail: dict[str, Any] = field(default_factory=dict)
    score: float = 1.0
    status: MetricStatus = MetricStatus.PASS
    advice: str = ""

    @property
    def counted(self) -> bool:
        """是否计入加权总分。SKIP / ERROR 不计 —— 让缺失的指标拉低分数没有意义。"""
        return self.status in (MetricStatus.PASS, MetricStatus.FAIL)

    def __str__(self) -> str:
        mark = {"pass": "OK", "fail": "NG", "skip": "--", "error": "!!"}[self.status.value]
        return f"[{mark}] {self.name}={self.value:.4g} (阈值 {self.threshold:.4g}) score={self.score:.2f}"


def _score_lower_better(value: float, threshold: float) -> float:
    """越低越好的指标归一：0 -> 1.0，压线 -> 0.5，两倍阈值及以上 -> 0.0。"""
    if threshold <= 0:
        return 1.0 if value <= 0 else 0.0
    return float(min(1.0, max(0.0, 1.0 - 0.5 * value / threshold)))


def _score_band(value: float, lo: float, hi: float) -> float:
    """区间型指标归一：落在 [lo, hi] 内 = 1.0，两侧线性/反比衰减，连续无跳变。"""
    if value < lo:
        return float(max(0.0, value / lo)) if lo > 0 else 0.0
    if value > hi:
        return float(max(0.0, hi / value)) if value > 0 else 0.0
    return 1.0


# ================================================================ 共享分析层


_RE_FRAME = re.compile(r"^frame:(\d+)\s+pts:(-?\d+)\s+pts_time:([-\d.]+)")
_RE_KV = re.compile(r"^lavfi\.signalstats\.([A-Z]+)=([-\d.eE+]+)")
# freezedetect 把区间写成三条独立的 metadata 日志；结尾那段若延续到 EOF 就只有 start。
_RE_FREEZE = re.compile(r"lavfi\.freezedetect\.freeze_(start|end|duration):\s*([-\d.]+)")
_RE_BLACK = re.compile(r"black_start:([-\d.]+)\s+black_end:([-\d.]+)\s+black_duration:([-\d.]+)")

_STAT_KEYS = ("YAVG", "UAVG", "VAVG", "YDIF", "UDIF", "VDIF", "SATAVG")


@dataclass(frozen=True)
class FrameStats:
    """逐帧信号统计。一次解码的产物，被多个指标共享。

    yavg/uavg/vavg 是 0~255 码值域的每帧均值；
    ydif/udif/vdif 是 signalstats 给的「与前一帧的平均绝对差」，第 0 帧恒为 0。
    """

    pts_time: np.ndarray
    yavg: np.ndarray
    uavg: np.ndarray
    vavg: np.ndarray
    ydif: np.ndarray
    udif: np.ndarray
    vdif: np.ndarray
    satavg: np.ndarray
    analyzed_size: tuple[int, int]

    @property
    def n(self) -> int:
        return int(self.yavg.size)

    def channel(self, name: str) -> np.ndarray:
        return {"y": self.yavg, "u": self.uavg, "v": self.vavg}[name]


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class DetectEvents:
    """freezedetect / blackdetect 的解析结果。"""

    freezes: tuple[Interval, ...] = ()
    blacks: tuple[Interval, ...] = ()

    def frozen_s(self) -> float:
        return sum(i.duration_s for i in self.freezes)

    def black_s(self) -> float:
        return sum(i.duration_s for i in self.blacks)


def _parse_signalstats(text: str) -> FrameStats:
    cols: dict[str, list[float]] = {k: [] for k in _STAT_KEYS}
    times: list[float] = []
    for line in text.splitlines():
        line = line.strip()
        if m := _RE_FRAME.match(line):
            times.append(float(m.group(3)))
            continue
        if m := _RE_KV.match(line):
            key = m.group(1)
            if key in cols:
                cols[key].append(float(m.group(2)))
    n = min([len(times)] + [len(v) for v in cols.values()])
    if n == 0:
        raise FFmpegError(["signalstats"], 1, "signalstats 没有输出任何帧，视频可能不可解码")
    arr = {k: np.asarray(v[:n], dtype=np.float64) for k, v in cols.items()}
    return FrameStats(
        pts_time=np.asarray(times[:n], dtype=np.float64),
        yavg=arr["YAVG"], uavg=arr["UAVG"], vavg=arr["VAVG"],
        ydif=arr["YDIF"], udif=arr["UDIF"], vdif=arr["VDIF"],
        satavg=arr["SATAVG"],
        analyzed_size=(0, 0),
    )


def _parse_events(text: str, total_s: float) -> DetectEvents:
    freezes: list[Interval] = []
    blacks: list[Interval] = []
    pending_start: float | None = None
    for line in text.splitlines():
        if m := _RE_FREEZE.search(line):
            kind, val = m.group(1), float(m.group(2))
            if kind == "start":
                # 上一段还没收尾就来了新的 start，说明上一段一直冻到这里。
                if pending_start is not None:
                    freezes.append(Interval(pending_start, val))
                pending_start = val
            elif kind == "end" and pending_start is not None:
                freezes.append(Interval(pending_start, val))
                pending_start = None
        elif m := _RE_BLACK.search(line):
            blacks.append(Interval(float(m.group(1)), float(m.group(2))))
    if pending_start is not None:
        # 冻结延续到文件结束时 ffmpeg 不打 freeze_end，必须用总时长补齐，
        # 否则「生成到一半卡死直到结尾」这种最典型的废片会被算成 0 秒冻结。
        freezes.append(Interval(pending_start, total_s))
    return DetectEvents(tuple(freezes), tuple(blacks))


@dataclass
class QCContext:
    """一次质检的上下文：工单期望 + 解码结果缓存。

    同一个 context 可以连续喂多条视频（batch 用），缓存按解析后的绝对路径分桶。
    """

    shot: Shot | None = None
    storyboard: Storyboard | None = None

    # 拼接产物的接缝时间点（秒）。SeamCheck 只在给了这个时才有意义。
    seam_times_s: tuple[float, ...] = ()

    # 工单期望。留空则从 shot / storyboard.delivery 推导。
    expect_duration_s: float | None = None
    expect_fps: int | None = None
    expect_resolution: tuple[int, int] | None = None

    ffmpeg: FFmpeg = field(default_factory=default_ffmpeg)
    # 逐帧分析的降采样宽度。质检看的是全局统计量，全分辨率解码纯属浪费；
    # 320 宽在 1080p 上约省一个数量级的时间，对 YAVG/YDIF 的影响在 1% 以内。
    analyze_width: int = 320
    freeze_noise: float = 0.002
    freeze_min_s: float = 0.6
    black_min_s: float = 0.25
    black_pixel_th: float = 0.10

    _stats: dict[str, FrameStats] = field(default_factory=dict, repr=False)
    _events: dict[str, DetectEvents] = field(default_factory=dict, repr=False)
    _media: dict[str, MediaInfo] = field(default_factory=dict, repr=False)

    # -------------------------------------------------- 期望值推导

    def wanted_duration_s(self) -> float | None:
        if self.expect_duration_s is not None:
            return self.expect_duration_s
        return self.shot.duration_s if self.shot else None

    def wanted_fps(self) -> int | None:
        if self.expect_fps is not None:
            return self.expect_fps
        if self.shot:
            return self.shot.fps
        return self.storyboard.delivery.fps if self.storyboard else None

    def wanted_resolution(self) -> tuple[int, int] | None:
        if self.expect_resolution is not None:
            return self.expect_resolution
        return tuple(self.storyboard.delivery.resolution) if self.storyboard else None

    # -------------------------------------------------- 缓存入口

    @staticmethod
    def _key(video: str | Path) -> str:
        return str(Path(video).resolve())

    def media(self, video: str | Path) -> MediaInfo:
        k = self._key(video)
        if k not in self._media:
            self._media[k] = self.ffmpeg.probe(video)
        return self._media[k]

    def stats(self, video: str | Path) -> FrameStats:
        """逐帧 signalstats。整个门禁只解码一次。"""
        k = self._key(video)
        if k in self._stats:
            return self._stats[k]
        info = self.media(video)
        if not info.has_video:
            raise ValueError(f"{video} 没有视频轨，无法质检")
        chain: list[str] = []
        if info.width > self.analyze_width > 0:
            chain.append(f"scale={self.analyze_width}:-2:flags=bilinear")
        # signalstats 只吃 YUV；生成侧偶尔直出 rgb 编码，这里统一一次。
        chain += ["format=yuv420p", "signalstats", "metadata=print:file=-"]
        out = self.ffmpeg.run_capture(
            ["-nostats", "-loglevel", "error", "-an", "-i", str(video),
             "-vf", ",".join(chain), "-f", "null", "-"]
        )
        st = _parse_signalstats(out.decode("utf-8", "replace"))
        scale = min(1.0, self.analyze_width / info.width) if info.width else 1.0
        st = dataclasses.replace(
            st, analyzed_size=(int(info.width * scale), int(info.height * scale))
        )
        self._stats[k] = st
        return st

    def events(self, video: str | Path) -> DetectEvents:
        """freeze / black 区间。两个滤镜串在同一趟解码里。"""
        k = self._key(video)
        if k in self._events:
            return self._events[k]
        info = self.media(video)
        vf = (
            f"freezedetect=n={self.freeze_noise}:d={self.freeze_min_s},"
            f"blackdetect=d={self.black_min_s}:pix_th={self.black_pixel_th}"
        )
        # 这两个滤镜把结论写在 info 级日志里，所以不能压 loglevel。
        err = self.ffmpeg.run(
            ["-nostats", "-loglevel", "info", "-an", "-i", str(video), "-vf", vf, "-f", "null", "-"]
        )
        ev = _parse_events(err, info.duration_s)
        self._events[k] = ev
        return ev

    def frames(self, video: str | Path, *, count: int = 8, width: int = 224) -> np.ndarray:
        """均匀抽 count 帧，返回 (count, h, w, 3) 的 uint8 RGB。

        只给可选插件用（人脸/美学要真实像素）；CPU 指标一律走 signalstats，
        不为了统计量去搬像素。
        """
        info = self.media(video)
        if not info.has_video or info.duration_s <= 0:
            raise ValueError(f"{video} 无法抽帧：has_video={info.has_video} dur={info.duration_s}")
        h = max(2, int(round(width * info.height / max(info.width, 1))) // 2 * 2)
        fps_expr = max(count / info.duration_s, 1e-3)
        raw = self.ffmpeg.run_capture(
            ["-nostats", "-loglevel", "error", "-an", "-i", str(video),
             "-vf", f"fps={fps_expr:.6f},scale={width}:{h}:flags=bilinear,format=rgb24",
             "-frames:v", str(count), "-f", "rawvideo", "-"]
        )
        got = len(raw) // (width * h * 3)
        if got == 0:
            raise FFmpegError(["frames"], 1, f"{video} 抽帧失败，未取到完整帧")
        return np.frombuffer(raw[: got * width * h * 3], dtype=np.uint8).reshape(got, h, width, 3)


# ================================================================ 指标抽象


class QCMetric(abc.ABC):
    """一项质检指标。

    实现约定：evaluate 只读 context 的缓存接口，不自己起 ffmpeg 进程；
    前提不成立时返回 status=SKIP 而不是抛异常 —— 抛异常会被门禁记成 ERROR 并升级人工。
    """

    name: ClassVar[str] = "metric"

    @abc.abstractmethod
    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        ...

    def _skip(self, reason: str) -> MetricResult:
        return MetricResult(
            name=self.name, value=float("nan"), passed=True, threshold=float("nan"),
            detail={"reason": reason}, score=1.0, status=MetricStatus.SKIP,
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


# ---------------------------------------------------------------- 1. 闪烁


class TemporalFlicker(QCMetric):
    """帧间亮度/色度突变。

    判据是逐帧 YAVG 一阶差分的标准差，而不是差分本身的均值：
    合法的渐变打光一阶差分近似常数（标准差≈0），而生成模型典型的「呼吸/闪烁」
    是逐帧正负交替，标准差会大一到三个数量级。色度同理，取三通道最大值。
    """

    name: ClassVar[str] = "temporal_flicker"

    def __init__(self, threshold: float = 2.0, chroma_weight: float = 0.7) -> None:
        self.threshold = threshold
        # 色度闪烁在观感上比亮度闪烁轻一些（色度被压缩、被眼睛低通），故打折计入。
        self.chroma_weight = chroma_weight

    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        st = context.stats(video)
        if st.n < 4:
            return self._skip(f"只有 {st.n} 帧，算不出一阶差分统计")
        dy = np.diff(st.yavg)
        du = np.diff(st.uavg)
        dv = np.diff(st.vavg)
        fy, fu, fv = float(dy.std()), float(du.std()), float(dv.std())
        value = max(fy, self.chroma_weight * max(fu, fv))
        # VBench temporal_flickering 同口径：相邻帧 MAE 归一，越高越好。
        # 这里用 signalstats 的 YDIF（逐帧平均绝对差）近似其灰度分量。
        vbench = float((255.0 - st.ydif[1:].mean()) / 255.0) if st.n > 1 else 1.0
        passed = value <= self.threshold
        return MetricResult(
            name=self.name, value=value, passed=passed, threshold=self.threshold,
            detail={
                "unit": "码值(0-255)一阶差分标准差",
                "luma_std": round(fy, 4), "u_std": round(fu, 4), "v_std": round(fv, 4),
                "abs_diff_p99": round(float(np.percentile(np.abs(dy), 99)), 4),
                "vbench_flicker_score": round(vbench, 5),
                "frames": st.n,
            },
            score=_score_lower_better(value, self.threshold),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                f"亮度/色度逐帧抖动 {value:.2f}（阈值 {self.threshold}）。"
                "建议：①降低引导强度 CFG 1~2 档，过高的 CFG 是闪烁第一诱因；"
                "②固定 seed 后只调 CFG 复现，别整条重摇；"
                "③若引擎支持首帧锁戏，加 first_frame 参考位稳住基调；"
                "④后处理兜底可上 deflicker（治标，慎用于强运动镜）。"
            ),
        )


# ---------------------------------------------------------------- 2. 色漂


class ColorDrift(QCMetric):
    """片段内首尾色彩漂移。

    取前 10% 帧与后 10% 帧的 Y/U/V 均值差。扩散视频越往后越容易整体偏色/掉饱和，
    拼进场景后会和相邻镜头对不上，是调色阶段最贵的返工来源。
    """

    name: ClassVar[str] = "color_drift"

    def __init__(self, threshold: float = 6.0, tail_ratio: float = 0.10) -> None:
        self.threshold = threshold
        self.tail_ratio = tail_ratio

    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        st = context.stats(video)
        k = max(1, int(round(st.n * self.tail_ratio)))
        if st.n < 2 * k + 1:
            return self._skip(f"只有 {st.n} 帧，首尾窗口会重叠")
        deltas = {}
        for ch in ("y", "u", "v"):
            a = st.channel(ch)
            deltas[ch] = float(a[-k:].mean() - a[:k].mean())
        dsat = float(st.satavg[-k:].mean() - st.satavg[:k].mean())
        value = max(abs(v) for v in deltas.values())
        passed = value <= self.threshold
        worst = max(deltas, key=lambda c: abs(deltas[c]))
        kind = {"y": "亮度", "u": "蓝黄轴", "v": "红青轴"}[worst]
        return MetricResult(
            name=self.name, value=value, passed=passed, threshold=self.threshold,
            detail={
                "unit": "码值(0-255)首尾均值差",
                "dY": round(deltas["y"], 3), "dU": round(deltas["u"], 3),
                "dV": round(deltas["v"], 3), "dSAT": round(dsat, 3),
                "window_frames": k, "worst_channel": worst,
            },
            score=_score_lower_better(value, self.threshold),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                f"首尾{kind}漂移 {deltas[worst]:+.1f} 码值（阈值 ±{self.threshold}）。"
                "建议：①给 Continuity 加尾帧锚（上一镜 emit_last_frame + 本镜 inherit_last_frame），"
                "让模型有固定的收尾目标；②降低 CFG / 去掉过强的 style 参考位；"
                "③把 duration_s 拆短（漂移随时长累积，>8s 的镜优先拆）；"
                "④确属轻微时可交给 grade.py 统一到场景 base_grade，不必重拍。"
            ),
        )


# ---------------------------------------------------------------- 3. 冻结 / 黑帧


class FreezeDetect(QCMetric):
    """生成到一半卡住：画面长时间完全不变。

    这是 API 侧超时/截断最典型的表现，单看缩略图查不出来，必须逐帧比。
    阈值用「冻结时长占比」而不是绝对秒数 —— 5s 镜和 15s 镜的容忍度本就不同。
    """

    name: ClassVar[str] = "freeze"

    def __init__(self, max_ratio: float = 0.12) -> None:
        self.max_ratio = max_ratio

    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        info = context.media(video)
        ev = context.events(video)
        total = max(info.duration_s, 1e-6)
        frozen = ev.frozen_s()
        value = frozen / total
        passed = value <= self.max_ratio
        longest = max((i.duration_s for i in ev.freezes), default=0.0)
        tail_frozen = any(i.end_s >= total - 0.05 for i in ev.freezes)
        return MetricResult(
            name=self.name, value=value, passed=passed, threshold=self.max_ratio,
            detail={
                "unit": "冻结时长占比",
                "frozen_s": round(frozen, 3), "total_s": round(total, 3),
                "segments": [(round(i.start_s, 2), round(i.end_s, 2)) for i in ev.freezes],
                "longest_s": round(longest, 3), "frozen_to_eof": tail_frozen,
            },
            score=_score_lower_better(value, self.max_ratio),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                f"冻结 {frozen:.2f}s / {total:.2f}s"
                + ("（且一直冻到片尾，典型的生成截断）" if tail_frozen else "")
                + "。建议：①这类废片重摇 seed 通常能解决，不要改提示词；"
                "②若同一 provider 连续出现，按 FailureKind.SERVER 切备用引擎；"
                "③工单时长若接近 provider 的 max_duration_s，先按 caps.clamp_duration 缩短再发。"
            ),
        )


class BlackDetect(QCMetric):
    """黑帧。生成失败最廉价的信号，也最容易被「有输出文件就算成功」漏掉。

    声明了淡入淡出的镜头本来就该有黑场，所以阈值按 transition_in 放宽 ——
    不区分的话每条 fade 镜都会被误杀。
    """

    name: ClassVar[str] = "black"

    _FADES = frozenset({Transition.FADE_IN, Transition.FADE_OUT})

    def __init__(self, max_ratio: float = 0.02, fade_max_ratio: float = 0.18) -> None:
        self.max_ratio = max_ratio
        self.fade_max_ratio = fade_max_ratio

    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        info = context.media(video)
        ev = context.events(video)
        total = max(info.duration_s, 1e-6)
        is_fade = bool(context.shot and context.shot.transition_in in self._FADES)
        thr = self.fade_max_ratio if is_fade else self.max_ratio
        black = ev.black_s()
        value = black / total
        passed = value <= thr
        mid_black = [i for i in ev.blacks if i.start_s > 0.3 and i.end_s < total - 0.3]
        return MetricResult(
            name=self.name, value=value, passed=passed, threshold=thr,
            detail={
                "unit": "黑帧时长占比",
                "black_s": round(black, 3), "total_s": round(total, 3),
                "segments": [(round(i.start_s, 2), round(i.end_s, 2)) for i in ev.blacks],
                "mid_clip_black": len(mid_black), "fade_allowance": is_fade,
            },
            score=_score_lower_better(value, thr),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                f"黑帧 {black:.2f}s / {total:.2f}s"
                + ("，且有 {} 段出现在片段中段（不是淡入淡出）".format(len(mid_black)) if mid_black else "")
                + "。建议：①直接重摇 seed；②检查 provider 返回的 video_uri 是否被截断"
                "（下载不完整会表现为尾部黑帧）；③若 transition_in 本就是 fadeblack，"
                "请在 Shot 上正确声明，门禁会自动放宽阈值。"
            ),
        )


# ---------------------------------------------------------------- 4. 运动合理性


class MotionSanity(QCMetric):
    """运动幅度是否落在合理区间。

    下界抓「没动」（模型把动词理解成静止），上界抓「崩了」（结构在帧间乱跳）。
    用 signalstats 的 YDIF（逐帧平均绝对差）而不是 mestimate：mestimate 只产出
    供 codecview 可视化的运动矢量 side data，没有可解析的统计输出；帧差在 CPU 上
    便宜一个数量级，而区分「没动」和「崩」本来也不需要真实光流。
    """

    name: ClassVar[str] = "motion_sanity"

    def __init__(self, lo: float = 0.4, hi: float = 35.0, static_frame_eps: float = 0.15,
                 max_static_ratio: float = 0.5) -> None:
        self.lo = lo
        self.hi = hi
        self.static_frame_eps = static_frame_eps
        self.max_static_ratio = max_static_ratio

    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        st = context.stats(video)
        if st.n < 3:
            return self._skip(f"只有 {st.n} 帧")
        d = st.ydif[1:]
        # 用中位数而不是均值：单个硬切/闪帧会把均值拉高，掩盖「整体不动」。
        median = float(np.median(d))
        p95 = float(np.percentile(d, 95))
        static_ratio = float((d < self.static_frame_eps).mean())
        burstiness = p95 / max(median, 1e-3)

        too_static = median < self.lo or static_ratio > self.max_static_ratio
        too_wild = median > self.hi
        passed = not (too_static or too_wild)
        value = median
        score = _score_band(median, self.lo, self.hi)
        if static_ratio > self.max_static_ratio:
            score = min(score, 1.0 - static_ratio)
        if too_static:
            tip = ("画面几乎没动。建议：①提示词里的动作写成可观测的镜头语言"
                   "（'她转身走向窗边' 而不是 '她若有所思'）；②camera_move 从 STATIC 改成"
                   " DOLLY_IN / PAN 之类给模型一个确定的运动源；③提高 motion/dynamic 强度参数；"
                   "④检查是不是 first_frame 权重过高把画面钉死了。")
        elif too_wild:
            tip = ("帧间变化过剧，多半是结构崩坏而非快速运动。建议：①降低 motion 强度与 CFG；"
                   "②缩短 duration_s；③换更高 quality_tier 的引擎重拍（EngineHint.HERO）。")
        else:
            tip = ""
        return MetricResult(
            name=self.name, value=value, passed=passed, threshold=self.hi,
            detail={
                "unit": "逐帧平均绝对差 YDIF 中位数(0-255)",
                "band": [self.lo, self.hi],
                "median": round(median, 4), "p95": round(p95, 4),
                "static_frame_ratio": round(static_ratio, 4),
                "burstiness_p95_over_median": round(burstiness, 3),
                "reason": "too_static" if too_static else ("too_wild" if too_wild else "ok"),
            },
            score=float(max(0.0, min(1.0, score))),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=tip,
        )


# ---------------------------------------------------------------- 5. 规格符合性


class DurationFpsConformance(QCMetric):
    """实际时长 / fps / 分辨率是否符合工单。

    provider 经常悄悄改规格（把 8s 截成 5s、把 24fps 出成 25fps、
    把 1080p 降成 720p），这些偏差单条看不出来，拼起来才发现音画对不齐。
    所以它是硬性否决项：规格不符不是质量问题，是工单没被执行。
    """

    name: ClassVar[str] = "conformance"

    def __init__(self, duration_tol: float = 0.08, fps_tol: float = 0.02) -> None:
        self.duration_tol = duration_tol
        self.fps_tol = fps_tol

    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        info = context.media(video)
        want_d = context.wanted_duration_s()
        want_fps = context.wanted_fps()
        want_res = context.wanted_resolution()
        # 非正的期望值不是「期望 0」，而是上游没填对。原来靠 `if want_d:` 的真值判断
        # 把它们和 None 混为一谈，结果 duration_s=0 的工单会静默跳过时长检查、
        # 拿一个 value=0 的满分。这里显式剔掉并留痕。
        bad_expect: list[str] = []
        if want_d is not None and want_d <= 0:
            bad_expect.append(f"duration_s={want_d}")
            want_d = None
        if want_fps is not None and want_fps <= 0:
            bad_expect.append(f"fps={want_fps}")
            want_fps = None
        if want_res is not None and (want_res[0] <= 0 or want_res[1] <= 0):
            bad_expect.append(f"resolution={tuple(want_res)}")
            want_res = None
        if bad_expect:
            log.warning("工单期望值非法，已忽略：%s", "、".join(bad_expect))

        if want_d is None and want_fps is None and want_res is None:
            return self._skip(
                "没有工单期望值（未提供 shot / storyboard / expect_*）"
                + (f"；另有非法期望值被忽略：{'、'.join(bad_expect)}" if bad_expect else "")
            )

        problems: list[str] = []
        errs: list[float] = []
        detail: dict[str, Any] = {
            "unit": "最大相对偏差",
            "actual": {"duration_s": round(info.duration_s, 3), "fps": round(info.fps, 3),
                       "resolution": list(info.size)},
        }
        if bad_expect:
            detail["ignored_expectations"] = bad_expect
        if want_d is not None:
            e = abs(info.duration_s - want_d) / want_d
            errs.append(e)
            detail["duration_rel_err"] = round(e, 4)
            detail["want_duration_s"] = want_d
            if e > self.duration_tol:
                problems.append(f"时长 {info.duration_s:.2f}s ≠ 工单 {want_d:.2f}s（偏差 {e:.1%}）")
        if want_fps is not None:
            e = abs(info.fps - want_fps) / want_fps
            errs.append(e)
            detail["fps_rel_err"] = round(e, 4)
            detail["want_fps"] = want_fps
            if e > self.fps_tol:
                problems.append(f"帧率 {info.fps:.3f} ≠ 工单 {want_fps}")
        if want_res is not None:
            detail["want_resolution"] = list(want_res)
            if tuple(info.size) != tuple(want_res):
                # 同宽高比只是缩放，交给 upscale 还能救；比例变了说明构图被裁，救不回来。
                a_got = info.width / max(info.height, 1)
                a_want = want_res[0] / max(want_res[1], 1)
                same_aspect = abs(a_got - a_want) < 0.02
                errs.append(0.30 if same_aspect else 1.0)
                problems.append(
                    f"分辨率 {info.width}x{info.height} ≠ 工单 {want_res[0]}x{want_res[1]}"
                    + ("（同比例，可走 upscale）" if same_aspect else "（宽高比都变了，构图已被裁）")
                )
        value = max(errs) if errs else 0.0
        passed = not problems
        detail["problems"] = problems
        return MetricResult(
            name=self.name, value=value, passed=passed, threshold=self.duration_tol,
            detail=detail,
            score=1.0 if passed else _score_lower_better(value, self.duration_tol),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                "；".join(problems) + "。建议：①先核对 router 是否按 caps.clamp_duration / "
                "nearest_resolution 归整过工单，规格不符多半是工单本身越界被 provider 静默改写；"
                "②确认后重提工单，不要用重摇 seed 解决规格问题；"
                "③时长偏短且尾部冻结时，按 extend 链补足而不是整条重拍。"
            ),
        )


# ---------------------------------------------------------------- 6. 接缝


class SeamCheck(QCMetric):
    """拼接接缝处是否有跳变。

    判据是接缝那一帧的通道跳变量除以该片自身的「正常帧间跳变」中位数，
    做成无量纲比值而不是绝对码值：快剪动作戏的正常跳变本来就大，
    用绝对阈值会把所有动作镜误判成接缝崩。
    """

    name: ClassVar[str] = "seam"

    def __init__(self, threshold: float = 4.0, window_s: float = 0.5) -> None:
        self.threshold = threshold
        self.window_s = window_s

    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        if not context.seam_times_s:
            return self._skip("未提供 seam_times_s（单条原子镜没有接缝）")
        st = context.stats(video)
        if st.n < 6:
            return self._skip(f"只有 {st.n} 帧")

        diffs = {c: np.abs(np.diff(st.channel(c))) for c in ("y", "u", "v")}
        # 基线取全片的帧间跳变中位数，+1.0 防止静止片除零把任何微小跳变放大成天文数字。
        base = {c: float(np.median(diffs[c])) + 1.0 for c in diffs}

        # 落在片长之外的接缝时间点一定是调用方算错了偏移（xfade 的 offset 是累积的，
        # 最容易在这里翻车）。原实现把它夹到首/末帧照常打分，等于凭空编造一次测量，
        # 还会让「偏移算错」这种真 bug 以一个漂亮的 pass 蒙混过去。
        t0, t1 = float(st.pts_time[0]), float(st.pts_time[-1])
        in_range = [t for t in context.seam_times_s if t0 <= t <= t1]
        out_range = [round(t, 3) for t in context.seam_times_s if not (t0 <= t <= t1)]
        if out_range:
            log.warning("接缝时间点 %s 落在片长 [%.3f, %.3f] 之外，已忽略", out_range, t0, t1)
        if not in_range:
            r = self._skip(
                f"给定的接缝时间点 {out_range} 全部落在片长 [{t0:.3f}, {t1:.3f}] 之外，无从检查"
            )
            return dataclasses.replace(r, detail={**r.detail, "out_of_range": out_range})

        worst = 0.0
        rows: list[dict[str, Any]] = []
        for t in in_range:
            idx = int(np.searchsorted(st.pts_time, t))
            idx = max(1, min(idx, st.n - 1))
            local = max(
                (abs(float(st.channel(c)[idx] - st.channel(c)[idx - 1])) / base[c] for c in diffs)
            )
            jump = {c: round(float(st.channel(c)[idx] - st.channel(c)[idx - 1]), 2) for c in diffs}
            rows.append({"t": round(t, 3), "frame": idx, "ratio": round(local, 3), "jump": jump})
            worst = max(worst, local)

        passed = worst <= self.threshold
        bad = [r for r in rows if r["ratio"] > self.threshold]
        return MetricResult(
            name=self.name, value=worst, passed=passed, threshold=self.threshold,
            detail={"unit": "接缝跳变 / 本片帧间跳变中位数", "seams": rows, "failed_seams": bad,
                    "out_of_range": out_range},
            score=_score_lower_better(worst, self.threshold),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                f"{len(bad)} 处接缝跳变超阈值（最大 {worst:.1f}×）。"
                "建议：①把该接缝的 transition_in 从 CUT 改成 DISSOLVE / OVERLAP_BLEND，"
                "用 stitch.overlap_blend 做重叠混接；②两侧镜头先过 grade.py 统一到场景 base_grade；"
                "③若本意是 MATCH_CUT，说明首尾帧没对上，需要给后一镜加 inherit_last_frame 重拍。"
            ),
        )


# ================================================================ 可选插件指标


class OptionalMetric(QCMetric):
    """需要重依赖（GPU/大模型）的指标基类。

    约定：依赖检查走 missing_dependency()，返回非空字符串即表示不可用；
    evaluate 在不可用时返回 SKIP。任何重依赖都必须在 _run 内部延迟导入 ——
    本模块在无 GPU 的调度机上也要 import 成功。
    """

    def missing_dependency(self) -> str:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return not self.missing_dependency()

    def evaluate(self, video: str | Path, context: QCContext) -> MetricResult:
        if miss := self.missing_dependency():
            return self._skip(miss)
        return self._run(video, context)

    @abc.abstractmethod
    def _run(self, video: str | Path, context: QCContext) -> MetricResult:
        ...


def _module_missing(*mods: str) -> str:
    """返回缺失依赖的中文说明；全都在就返回空串。只查 spec，不真的 import。"""
    import importlib.util

    absent = [m for m in mods if importlib.util.find_spec(m.split(".")[0]) is None]
    if not absent:
        return ""
    return f"缺少依赖 {'、'.join(absent)}，该指标已跳过（pip install {' '.join(absent)}）"


class IdentityDrift(OptionalMetric):
    """人脸身份跨帧漂移：抽帧做人脸 embedding，算与首个有效帧的余弦距离。

    长片最贵的返工是「同一个角色在第 7 个镜头变了脸」。这个指标是唯一能自动
    抓到它的手段，但需要人脸识别模型，所以做成可选插件。
    """

    name: ClassVar[str] = "identity_drift"

    def __init__(self, threshold: float = 0.35, samples: int = 8) -> None:
        self.threshold = threshold
        self.samples = samples

    def missing_dependency(self) -> str:
        return _module_missing("insightface")

    def _run(self, video: str | Path, context: QCContext) -> MetricResult:
        if context.shot is not None and not context.shot.subject_ids:
            return self._skip("该镜没有出场角色（subject_ids 为空），不测身份一致性")
        from insightface.app import FaceAnalysis  # 延迟导入：模型包 >100MB

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        frames = context.frames(video, count=self.samples, width=640)
        embs: list[np.ndarray] = []
        for fr in frames:
            faces = app.get(fr[:, :, ::-1])  # insightface 吃 BGR
            if not faces:
                continue
            # 取最大脸：主角在画面里通常最大，配角/路人不该决定身份分。
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            e = np.asarray(f.normed_embedding, dtype=np.float64)
            embs.append(e / (np.linalg.norm(e) + 1e-9))
        if len(embs) < 2:
            return self._skip(f"只在 {len(embs)} 帧上检到人脸，样本不足")
        ref = embs[0]
        dists = [float(1.0 - float(ref @ e)) for e in embs[1:]]
        value = max(dists)
        passed = value <= self.threshold
        return MetricResult(
            name=self.name, value=value, passed=passed, threshold=self.threshold,
            detail={"unit": "1 - 余弦相似度（相对首个有效帧）",
                    "max": round(value, 4), "mean": round(float(np.mean(dists)), 4),
                    "faces_found": len(embs), "samples": int(frames.shape[0])},
            score=_score_lower_better(value, self.threshold),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                f"人脸身份漂移 {value:.3f}（阈值 {self.threshold}）。"
                "建议：①提高 identity 参考位权重并补一张同角度定妆照；"
                "②给角色挂 LoRA（CharacterBible.lora）并调高 strength；"
                "③缩短 duration_s，身份漂移随时长单调累积；"
                "④换支持 first_frame 锁戏的引擎重拍。"
            ),
        )


class AestheticScore(OptionalMetric):
    """美学评分。口径对齐 VBench 的 aesthetic_quality：
    LAION Aesthetic Predictor（CLIP ViT-L/14 + 线性 MLP），输出约 1~10 分，越高越好。

    TODO(2026-09-19 查证)：improved-aesthetic-predictor 仓库页没有列出权重文件的
    稳定直链与官方推荐阈值，本实现要求调用方用 weights_path 显式给出本地权重
    （sac+logos+ava1-l14-linearMSE.pth），不去猜下载地址。阈值 5.0 取自
    LAION-Aesthetics V2 "5+" 子集的划分口径，属经验值，上线前应按本片风格重标。
    """

    name: ClassVar[str] = "aesthetic"

    def __init__(self, threshold: float = 5.0, samples: int = 6,
                 weights_path: str | Path | None = None) -> None:
        self.threshold = threshold
        self.samples = samples
        self.weights_path = Path(weights_path) if weights_path else None

    def missing_dependency(self) -> str:
        if miss := _module_missing("torch", "open_clip"):
            return miss
        if not self.weights_path or not self.weights_path.exists():
            return "未提供 LAION 美学预测器权重（weights_path），该指标已跳过"
        return ""

    def _run(self, video: str | Path, context: QCContext) -> MetricResult:
        import torch  # 延迟导入：torch 在无 GPU 的调度机上根本不该被加载
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device="cpu"
        )
        head = torch.nn.Sequential(
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16), torch.nn.Linear(16, 1),
        )
        # 官方权重是从一个把 nn.Sequential 挂在 self.layers 上的 LightningModule 存的，
        # 所以 checkpoint 的键长成 "layers.0.weight"；这里的 head 是裸 Sequential，
        # 键是 "0.weight"。不剥前缀 load_state_dict 会整个报 unexpected keys。
        # （christophschuhmann/improved-aesthetic-predictor，2026-09-20 查证）
        sd = torch.load(str(self.weights_path), map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k[len("layers."):] if k.startswith("layers.") else k: v for k, v in sd.items()}
        head.load_state_dict(sd)
        head.eval()

        from PIL import Image

        frames = context.frames(video, count=self.samples, width=448)
        with torch.no_grad():
            batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
            feat = model.encode_image(batch).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            scores = head(feat).squeeze(-1).tolist()
        value = float(np.mean(scores))
        passed = value >= self.threshold
        return MetricResult(
            name=self.name, value=value, passed=passed, threshold=self.threshold,
            detail={"unit": "LAION Aesthetic v2 分(约1-10，越高越好)",
                    "per_frame": [round(s, 3) for s in scores],
                    "min": round(min(scores), 3)},
            # 越高越好：把「达标」映射到 0.5，10 分映射到 1.0。
            score=float(min(1.0, max(0.0, 0.5 * value / max(self.threshold, 1e-6)))),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                f"美学分 {value:.2f} < {self.threshold}。建议：①补 style / lighting 参考位，"
                "把 style_bible 里的打光与质感词拼进提示词；②换高 quality_tier 引擎；"
                "③本项主观性强，连续两次低分再重拍，单次低分可放行给剪辑挑。"
            ),
        )


class LipSyncScore(OptionalMetric):
    """口型同步。用 syncnet_python 的 LSE-D（越低越好）/ LSE-C（越高越好）。

    真人实拍基准（Wav2Lip 论文 Table 1，2026-09-19 查证）：
    LRS2 真视频 LSE-D 6.736 / LSE-C 7.838；LRS3 6.956 / 7.592；LRW 7.012 / 6.931。
    所以「真人水平」大致是 LSE-D ≈ 7、LSE-C ≈ 7；本模块默认阈值 LSE-D ≤ 8.5
    留了一档余量，因为生成片普遍略差于实拍。

    syncnet_root 指向 joonson/syncnet_python 的 clone。两点坑（2026-09-20 查证）：

    1. **打分脚本不在这个仓库里**。syncnet_python 只提供 run_pipeline.py /
       run_syncnet.py / run_visualise.py / demo_syncnet.py；算 LSE-D、LSE-C 的
       calculate_scores_real_videos.py 是 Wav2Lip 的 evaluation/scores_LSE/ 下的脚本，
       约定把它一起放进 syncnet_root 才能跑。缺哪个就报哪个，不去猜路径。
       来源 https://github.com/joonson/syncnet_python 、
            https://github.com/Rudrabha/Wav2Lip （evaluation/README.md）
    2. **分数写文件不写 stdout**。Wav2Lip 的 evaluation/README 明确说
       "The generated scores will be present in the all_scores.txt"。
       原实现去 stdout 里正则捞浮点数，正常情况下一个也捞不到。

    TODO(2026-09-20)：PyPI 上的 `syncnet-python`（0.2.2，2025-07-07，维护者 nawta）
    是社区重构版、带 SyncNetPipeline 这个 Python API，不是 joonson 官方发布。
    要不要切过去得先在有 GPU 的机器上对齐一遍分值口径，这里不贸然依赖。
    """

    name: ClassVar[str] = "lip_sync"

    _SCORE_FILE: ClassVar[str] = "all_scores.txt"

    def __init__(self, lse_d_threshold: float = 8.5, lse_c_min: float = 5.0,
                 syncnet_root: str | Path | None = None) -> None:
        self.lse_d_threshold = lse_d_threshold
        self.lse_c_min = lse_c_min
        self.syncnet_root = Path(syncnet_root) if syncnet_root else None

    def missing_dependency(self) -> str:
        if miss := _module_missing("torch"):
            return miss
        if not self.syncnet_root:
            return "未提供 syncnet_python 仓库路径（syncnet_root），该指标已跳过"
        absent = [
            n for n in ("run_pipeline.py", "calculate_scores_real_videos.py")
            if not (self.syncnet_root / n).exists()
        ]
        if absent:
            return (
                f"{self.syncnet_root} 下缺少 {'、'.join(absent)}"
                "（run_pipeline.py 来自 joonson/syncnet_python，"
                "calculate_scores_real_videos.py 来自 Wav2Lip 的 evaluation/scores_LSE/），"
                "该指标已跳过"
            )
        return ""

    @staticmethod
    def _parse_scores(text: str) -> tuple[float, float] | None:
        """从 all_scores.txt / stdout 里取最后一行的两个浮点数。

        取最后一行而不是第一个匹配：脚本会把每个候选人脸轨道各打一行，
        正则从头捞会捞到轨道编号之类的整数旁边的噪声。
        """
        for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
            nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", line)]
            if len(nums) >= 2:
                return nums[0], nums[1]
        return None

    def _run(self, video: str | Path, context: QCContext) -> MetricResult:
        if context.shot is not None and not context.shot.dialogue:
            return self._skip("该镜没有对白，不测口型同步")
        if not context.media(video).has_audio:
            return self._skip("视频没有音轨，口型同步无从谈起")

        import sys

        # syncnet_python 只有命令行入口，没有可 import 的稳定 API，所以走子进程。
        # 解释器必须用 sys.executable：产线跑在 venv 里，裸 "python" 在很多机器上
        # 根本不在 PATH 里（本机就是），写死会把「环境没配」伪装成「指标炸了」。
        work = Path(tempfile.mkdtemp(prefix="longfilm_syncnet_"))
        root = self.syncnet_root
        assert root is not None    # missing_dependency 已经挡过
        try:
            ref = "qc"
            common = ["--videofile", str(Path(video).resolve()),
                      "--reference", ref, "--data_dir", str(work)]
            _sp_run([sys.executable, "run_pipeline.py", *common],
                           cwd=str(root), check=True, capture_output=True, text=True)
            proc = _sp_run([sys.executable, "calculate_scores_real_videos.py", *common],
                                  cwd=str(root), check=True, capture_output=True, text=True)
            score_file = root / self._SCORE_FILE
            text = score_file.read_text(encoding="utf-8", errors="replace") \
                if score_file.exists() else ""
        finally:
            shutil.rmtree(work, ignore_errors=True)

        # 先读约定的 all_scores.txt，读不到再退回 stdout（社区分支确实有直接打印的）。
        parsed = self._parse_scores(text) or self._parse_scores(proc.stdout)
        if parsed is None:
            return self._skip(
                f"syncnet 分数无法解析：{self._SCORE_FILE}={text[-200:]!r} "
                f"stdout={proc.stdout[-200:]!r}"
            )
        lse_d, lse_c = parsed
        passed = lse_d <= self.lse_d_threshold and lse_c >= self.lse_c_min
        return MetricResult(
            name=self.name, value=lse_d, passed=passed, threshold=self.lse_d_threshold,
            detail={"unit": "LSE-D(越低越好)", "lse_d": lse_d, "lse_c": lse_c,
                    "lse_c_min": self.lse_c_min,
                    "real_video_reference": {"lrs2": [6.736, 7.838], "lrs3": [6.956, 7.592],
                                             "lrw": [7.012, 6.931]}},
            score=_score_lower_better(lse_d, self.lse_d_threshold),
            status=MetricStatus.PASS if passed else MetricStatus.FAIL,
            advice=(
                "" if passed else
                f"口型 LSE-D {lse_d:.2f} / LSE-C {lse_c:.2f}（真人实拍约 7 / 7）。"
                "建议：①走音频先行（AudioPlan.audio_first=True），用对白轨驱动生成；"
                "②确认 GenRequest.audio_uri 真的带上了这条对白；"
                "③换支持 native audio 的引擎（caps.supports_native_audio）；"
                "④兜底可对口部做一次 lipsync 重绘，不必整条重拍。"
            ),
        )


# ---------------------------------------------------------------- 插件注册表


MetricFactory = Callable[[], QCMetric]
_PLUGINS: dict[str, MetricFactory] = {}


def register_plugin(name: str, factory: MetricFactory) -> None:
    """注册可选指标。名字重复直接覆盖 —— 允许调用方替换成自家实现。"""
    _PLUGINS[name] = factory


def plugin_availability() -> dict[str, str]:
    """返回 {名字: ""(可用) | 不可用原因}。CLI 用它打印降级情况。"""
    out: dict[str, str] = {}
    for name, factory in _PLUGINS.items():
        m = factory()
        out[name] = m.missing_dependency() if isinstance(m, OptionalMetric) else ""
    return out


def available_plugins() -> list[QCMetric]:
    """实例化所有当前可用的插件指标。不可用的直接不进门禁，避免一堆 SKIP 噪音。"""
    out: list[QCMetric] = []
    for name, factory in _PLUGINS.items():
        m = factory()
        if isinstance(m, OptionalMetric) and not m.available:
            log.info("质检插件 %s 不可用：%s", name, m.missing_dependency())
            continue
        out.append(m)
    return out


register_plugin("identity_drift", IdentityDrift)
register_plugin("aesthetic", AestheticScore)
register_plugin("lip_sync", LipSyncScore)


# ================================================================ 门禁


@dataclass
class GateEntry:
    metric: QCMetric
    weight: float = 1.0
    blocking: bool = False   # 硬性否决：这项不过，总分再高也不放行


@dataclass
class QCReport:
    video: str
    shot_id: str
    verdict: Verdict
    score: float
    metrics: list[MetricResult] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    takes: int = 0
    elapsed_s: float = 0.0

    @property
    def failures(self) -> list[MetricResult]:
        return [m for m in self.metrics if m.status is MetricStatus.FAIL]

    @property
    def blocking_failures(self) -> list[str]:
        return [m.name for m in self.metrics if m.detail.get("_blocking") and not m.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video, "shot_id": self.shot_id, "verdict": self.verdict.value,
            "score": round(self.score, 4), "takes": self.takes,
            "elapsed_s": round(self.elapsed_s, 2),
            "metrics": [
                {"name": m.name, "value": None if math.isnan(m.value) else round(m.value, 5),
                 "passed": m.passed, "threshold": None if math.isnan(m.threshold) else m.threshold,
                 "score": round(m.score, 4), "status": m.status.value, "detail": m.detail}
                for m in self.metrics
            ],
            "advice": self.advice,
        }

    def table(self) -> str:
        lines = [
            f"QC {Path(self.video).name}  shot={self.shot_id or '-'}  "
            f"verdict={self.verdict.value.upper()}  score={self.score:.3f}  takes={self.takes}",
            f"  {'指标':<20}{'状态':<7}{'数值':>12}{'阈值':>12}{'分数':>7}",
        ]
        for m in self.metrics:
            v = "n/a" if math.isnan(m.value) else f"{m.value:.4g}"
            t = "n/a" if math.isnan(m.threshold) else f"{m.threshold:.4g}"
            flag = "*" if m.detail.get("_blocking") else " "
            lines.append(f"  {m.name + flag:<20}{m.status.value:<7}{v:>12}{t:>12}{m.score:>7.2f}")
        for a in self.advice:
            lines.append(f"  → {a}")
        return "\n".join(lines)


class QCGate:
    """把多个指标组装成一道门禁。

    verdict 的三态是有成本含义的：
      pass     —— 直接进拼接，零人工；
      retake   —— 自动重拍，成本 = 一次生成；
      escalate —— 人工介入，成本最高，所以只在「重拍次数用尽」或
                  「指标本身失效（ERROR）」时才给，绝不因为分低就叫人。
    """

    def __init__(
        self,
        entries: Sequence[GateEntry],
        *,
        pass_score: float = 0.70,
        max_takes: int = 3,
        single_fail_floor: float = 0.25,
        min_evidence_ratio: float = 0.5,
    ) -> None:
        if not entries:
            raise ValueError("门禁至少要有一项指标")
        self.entries = list(entries)
        self.pass_score = pass_score
        self.max_takes = max_takes
        # 加权平均的天然缺陷：一项彻底崩掉会被其余六项平掉。但观众看的是最烂的那一项，
        # 色漂到两倍阈值的片子不会因为「其他都好」而变得可用。所以任何单项掉到这条
        # 地板线以下（≈ 1.5 倍阈值）都直接否决，不参与平均表决。
        self.single_fail_floor = single_fail_floor
        # SKIP 不计分，于是「几乎全 SKIP」的片子会拿到一个虚高的总分。
        # 最典型的是被截断成一两帧的废片：逐帧类指标全因帧数不足而 SKIP，
        # freeze/black 在只有一帧时也查不出任何事件，结果 1.0 分堂堂放行。
        # 所以要求实际计分的权重占比达到这条线，否则判定「证据不足」升级人工 ——
        # 门禁没看清的东西，绝不能替人签字说它没问题。
        self.min_evidence_ratio = min_evidence_ratio

    @classmethod
    def default(cls, *, include_plugins: bool = True, **kw: Any) -> QCGate:
        """产线默认门禁。

        blocking 的选取原则：**能判定「这条根本不是要的东西」的项才做硬否决**。
        冻结、黑帧、规格不符属于「工单没被执行」，一票否决；
        闪烁/色漂/运动是程度问题，交给加权总分。
        """
        entries = [
            GateEntry(FreezeDetect(), weight=1.5, blocking=True),
            GateEntry(BlackDetect(), weight=1.5, blocking=True),
            GateEntry(DurationFpsConformance(), weight=1.0, blocking=True),
            GateEntry(TemporalFlicker(), weight=1.5),
            GateEntry(ColorDrift(), weight=1.2),
            GateEntry(MotionSanity(), weight=1.2),
            GateEntry(SeamCheck(), weight=1.0),
        ]
        if include_plugins:
            entries += [GateEntry(m, weight=1.0) for m in available_plugins()]
        return cls(entries, **kw)

    def evaluate(
        self,
        video: str | Path,
        shot: Shot | None = None,
        storyboard: Storyboard | None = None,
        *,
        context: QCContext | None = None,
    ) -> QCReport:
        import time

        t0 = time.monotonic()
        ctx = context or QCContext(shot=shot, storyboard=storyboard)
        if shot is not None and ctx.shot is None:
            ctx.shot = shot
        if storyboard is not None and ctx.storyboard is None:
            ctx.storyboard = storyboard

        results: list[MetricResult] = []
        for e in self.entries:
            try:
                r = e.metric.evaluate(video, ctx)
            except (FFmpegError, ValueError, OSError) as exc:
                # 单项指标炸掉不该拖垮整批质检，但也绝不能当成通过 ——
                # 记成 ERROR 并在下面强制升级人工。
                log.warning("指标 %s 在 %s 上失败：%s", e.metric.name, video, exc)
                r = MetricResult(
                    name=e.metric.name, value=float("nan"), passed=False,
                    threshold=float("nan"), detail={"error": str(exc)},
                    score=0.0, status=MetricStatus.ERROR,
                    advice=f"指标 {e.metric.name} 执行失败，需人工确认：{exc}",
                )
            r.detail["_blocking"] = e.blocking
            r.detail["_weight"] = e.weight
            results.append(r)

        counted = [(r, e.weight) for r, e in zip(results, self.entries) if r.counted]
        total_w = sum(w for _, w in counted)
        all_w = sum(e.weight for e in self.entries)
        score = sum(r.score * w for r, w in counted) / total_w if total_w else 0.0
        evidence = total_w / all_w if all_w else 0.0

        blocked = [r.name for r, e in zip(results, self.entries) if e.blocking and not r.passed]
        floored = [r.name for r, _ in counted
                   if not r.passed and r.score <= self.single_fail_floor and r.name not in blocked]
        errored = [r.name for r in results if r.status is MetricStatus.ERROR]
        takes = shot.takes if shot else (ctx.shot.takes if ctx.shot else 0)
        thin = evidence < self.min_evidence_ratio

        if errored or thin:
            verdict = Verdict.ESCALATE
        elif blocked or floored or score < self.pass_score:
            verdict = Verdict.ESCALATE if takes + 1 >= self.max_takes else Verdict.RETAKE
        else:
            verdict = Verdict.PASS

        advice = self._advice(results, verdict, blocked, floored, errored, score, takes,
                              evidence=evidence if thin else None)
        return QCReport(
            video=str(video),
            shot_id=(shot.id if shot else (ctx.shot.id if ctx.shot else "")),
            verdict=verdict, score=score, metrics=results, advice=advice,
            takes=takes, elapsed_s=time.monotonic() - t0,
        )

    def _advice(
        self, results: list[MetricResult], verdict: Verdict, blocked: list[str],
        floored: list[str], errored: list[str], score: float, takes: int,
        *, evidence: float | None = None,
    ) -> list[str]:
        """重拍建议。排序规则：硬否决项在前，其余按失分多少排 ——
        先修掉扣分最狠的那项，避免重拍时到处乱改导致无法归因。"""
        if verdict is Verdict.PASS:
            return []
        out: list[str] = []
        if evidence is not None:
            skipped = [r.name for r in results if r.status is MetricStatus.SKIP]
            out.append(
                f"有效指标权重只占 {evidence:.0%}（低于 {self.min_evidence_ratio:.0%}），"
                f"跳过的是 {'、'.join(skipped) or '（无）'}，自动判级证据不足，请人工抽看。"
                "最常见的成因是片子被截断到只剩几帧、或工单期望值没传进 QCContext。"
            )
        if errored:
            out.append(f"指标 {'、'.join(errored)} 执行失败，自动判级不可信，请人工抽看。")
        if blocked:
            out.append(f"硬性否决项未通过：{'、'.join(blocked)}（这类问题重摇 seed 通常即可解决）。")
        if floored:
            out.append(
                f"单项严重超标：{'、'.join(floored)}（已超阈值 1.5 倍以上，不参与加权表决）。"
            )
        fails = [r for r in results if r.status is MetricStatus.FAIL and r.advice]
        fails.sort(key=lambda r: (not r.detail.get("_blocking"), r.score))
        out += [r.advice for r in fails]
        if verdict is Verdict.ESCALATE and takes + 1 >= self.max_takes:
            out.append(
                f"已重拍 {takes} 次（上限 {self.max_takes}），继续重摇是在烧额度；"
                "建议改工单：拆短时长、换引擎（EngineHint.HERO/OPEN）或改写这条镜头的动作描述。"
            )
        if not out:
            out.append(f"各单项均未越线但加权总分 {score:.3f} < {self.pass_score}，属综合质量偏低。")
        return out


# ================================================================ 批量


def run_batch(
    videos: Sequence[str | Path],
    *,
    gate: QCGate | None = None,
    shots: Sequence[Shot] | None = None,
    storyboard: Storyboard | None = None,
    context_factory: Callable[[int], QCContext] | None = None,
) -> list[QCReport]:
    """逐条跑门禁。顺序执行：质检瓶颈在 ffmpeg 解码（本身已多线程），
    再叠一层进程池在 3GB 内存的机器上只会把自己 OOM 掉。"""
    g = gate or QCGate.default()
    out: list[QCReport] = []
    for i, v in enumerate(videos):
        shot = shots[i] if shots and i < len(shots) else None
        ctx = context_factory(i) if context_factory else QCContext(shot=shot, storyboard=storyboard)
        out.append(g.evaluate(v, shot, storyboard, context=ctx))
    return out


def batch_qc(
    videos: Sequence[str | Path],
    *,
    gate: QCGate | None = None,
    shots: Sequence[Shot] | None = None,
    storyboard: Storyboard | None = None,
    context_factory: Callable[[int], QCContext] | None = None,
    reports_out: list[QCReport] | None = None,
) -> str:
    """批量质检报告：按分数升序（最差的在最上面，这是要先处理的）。

    reports_out 给需要拿结构化结果的调用方回填，免得为此再跑一遍解码。
    """
    reports = run_batch(
        videos, gate=gate, shots=shots, storyboard=storyboard, context_factory=context_factory
    )
    if reports_out is not None:
        reports_out.extend(reports)
    reports_sorted = sorted(reports, key=lambda r: r.score)

    names = [m.name for m in reports[0].metrics] if reports else []
    head = f"{'片段':<22}{'镜头':<10}{'判定':<10}{'总分':>7}  " + "".join(f"{n[:9]:>11}" for n in names)
    lines = ["=" * len(head), head, "-" * len(head)]
    for r in reports_sorted:
        cells = ""
        for m in r.metrics:
            cells += f"{('-' if m.status is MetricStatus.SKIP else f'{m.score:.2f}'):>11}"
        lines.append(
            f"{Path(r.video).name[:21]:<22}{(r.shot_id or '-')[:9]:<10}"
            f"{r.verdict.value:<10}{r.score:>7.3f}  {cells}"
        )
    lines.append("-" * len(head))

    n = len(reports)
    counts = {v: sum(1 for r in reports if r.verdict is v) for v in Verdict}
    scrap = (counts[Verdict.RETAKE] + counts[Verdict.ESCALATE]) / n if n else 0.0
    lines.append(
        f"共 {n} 条：pass {counts[Verdict.PASS]} / retake {counts[Verdict.RETAKE]} / "
        f"escalate {counts[Verdict.ESCALATE]}；废片率 {scrap:.0%}"
    )
    worst = [r for r in reports_sorted if r.verdict is not Verdict.PASS]
    if worst:
        lines.append("")
        lines.append("重拍建议（最差的在前）：")
        for r in worst:
            lines.append(f"  {Path(r.video).name} [{r.verdict.value}]")
            for a in r.advice:
                lines.append(f"     - {a}")
    lines.append("=" * len(head))
    return "\n".join(lines)


# ================================================================ 自测


def _mk(ff: FFmpeg, path: Path, src: str, *, vf: str = "", dur: float = 3.0,
        size: str = "320x180", fps: int = 24) -> str:
    # lavfi 源的第一个参数用 '='，源本身已带参数时后续用 ':' 续写。
    sep = ":" if "=" in src else "="
    args = ["-y", "-v", "error", "-f", "lavfi", "-i", f"{src}{sep}s={size}:r={fps}:d={dur}"]
    if vf:
        args += ["-vf", vf]
    args += ["-pix_fmt", "yuv420p", str(path)]
    ff.run(args)
    return str(path)


def _selftest() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ff = default_ffmpeg()
    work = Path(tempfile.mkdtemp(prefix="longfilm_qc_"))
    try:
        # ---- 合成素材：一条好片 + 四种典型废片 + 一条接缝片
        good = _mk(ff, work / "good.mp4", "testsrc2")
        static = _mk(ff, work / "bad_static.mp4", "color=c=0x606060")
        flicker = _mk(ff, work / "bad_flicker.mp4", "testsrc2",
                      vf="eq=brightness='if(mod(n,2),0.30,-0.30)':eval=frame")
        drift = _mk(ff, work / "bad_drift.mp4", "testsrc2",
                    vf="eq=gamma_r='1+0.8*t/3':gamma_b='1-0.35*t/3':eval=frame")
        # 黑帧片：中段插 1.5s 纯黑，同时也会被 freezedetect 抓到（黑场本身是冻结）
        blackv = str(work / "bad_black.mp4")
        ff.run(["-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=1.5",
                "-f", "lavfi", "-i", "color=c=black:s=320x180:r=24:d=1.5",
                "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=1.5",
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
                "-map", "[v]", "-pix_fmt", "yuv420p", blackv])
        # 接缝片：好片 + 明显偏暖的同源片硬接，接缝在 3.0s
        warm = _mk(ff, work / "warm.mp4", "testsrc2", vf="eq=gamma_r=1.8:gamma_b=0.65")
        seamv = str(work / "seam.mp4")
        ff.run(["-y", "-v", "error", "-i", good, "-i", warm,
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map", "[v]", "-pix_fmt", "yuv420p", seamv])

        ctx = QCContext()

        # ---- 1. 共享分析层
        st = ctx.stats(good)
        assert st.n >= 70, st.n
        assert 100 < st.yavg.mean() < 140, st.yavg.mean()
        assert ctx.stats(good) is st, "stats 没走缓存，会重复解码"
        print(f"[stats] good.mp4 {st.n} 帧 @{st.analyzed_size} YAVG均值={st.yavg.mean():.2f} "
              f"YDIF中位={np.median(st.ydif[1:]):.2f}")

        # ---- 2. 各指标在好/坏片上的区分度（这是自测的核心）
        flick_m, drift_m, motion_m = TemporalFlicker(), ColorDrift(), MotionSanity()
        v_good = flick_m.evaluate(good, ctx).value
        v_flick = flick_m.evaluate(flicker, QCContext()).value
        assert v_good < 1.0 and v_flick > 50.0, (v_good, v_flick)
        assert v_flick / max(v_good, 1e-6) > 100, "闪烁指标区分度不足"
        print(f"[flicker] 好片={v_good:.3f}  闪烁片={v_flick:.1f}  "
              f"区分度={v_flick / v_good:.0f}×")

        d_good = drift_m.evaluate(good, ctx).value
        d_bad = drift_m.evaluate(drift, QCContext()).value
        assert d_good < 2.0 < 6.0 < d_bad, (d_good, d_bad)
        print(f"[color_drift] 好片={d_good:.2f}  色漂片={d_bad:.2f}")

        m_good = motion_m.evaluate(good, ctx)
        m_static = motion_m.evaluate(static, QCContext())
        assert m_good.passed and not m_static.passed
        assert m_static.detail["reason"] == "too_static", m_static.detail
        print(f"[motion] 好片 YDIF中位={m_good.value:.2f}(pass)  "
              f"静止片={m_static.value:.2f}(fail/{m_static.detail['reason']})")

        ev = QCContext().events(blackv)
        assert ev.black_s() > 1.0, ev
        assert ev.frozen_s() > 1.0, ev
        bctx = QCContext()
        assert not BlackDetect().evaluate(blackv, bctx).passed
        assert BlackDetect().evaluate(good, ctx).passed
        assert not FreezeDetect().evaluate(static, QCContext()).passed
        print(f"[black/freeze] 黑帧片 black={ev.black_s():.2f}s freeze={ev.frozen_s():.2f}s；"
              f"好片 black=0.00s")

        # 冻结到片尾必须被补齐（ffmpeg 不打 freeze_end）
        ev_static = QCContext().events(static)
        assert ev_static.frozen_s() > 2.0, ev_static
        assert ev_static.freezes[-1].end_s > 2.5, ev_static.freezes

        # ---- 3. 规格符合性
        shot_ok = Shot(id="s01", scene_id="sc01", index=0, duration_s=3.0, fps=24)
        shot_bad = Shot(id="s02", scene_id="sc01", index=1, duration_s=8.0, fps=30)
        c_ok = DurationFpsConformance().evaluate(good, QCContext(shot=shot_ok))
        c_bad = DurationFpsConformance().evaluate(good, QCContext(shot=shot_bad))
        assert c_ok.passed and not c_bad.passed, (c_ok, c_bad)
        assert len(c_bad.detail["problems"]) == 2, c_bad.detail["problems"]
        assert DurationFpsConformance().evaluate(good, QCContext()).status is MetricStatus.SKIP
        # 非法（非正）期望值不能被真值判断当成「没有期望」而静默放行
        c_zero = DurationFpsConformance().evaluate(
            good, QCContext(expect_duration_s=0.0, expect_fps=0)
        )
        assert c_zero.status is MetricStatus.SKIP, c_zero
        assert "duration_s=0.0" in c_zero.detail["reason"], c_zero.detail
        c_mix = DurationFpsConformance().evaluate(
            good, QCContext(expect_duration_s=0.0, expect_fps=24)
        )
        assert c_mix.passed and c_mix.detail["ignored_expectations"] == ["duration_s=0.0"], c_mix
        assert "duration_rel_err" not in c_mix.detail, c_mix.detail
        print(f"[conformance] 合规 pass；工单 8s/30fps 实得 3s/24fps -> "
              f"{c_bad.detail['problems']}；非法期望值被剔除 {c_mix.detail['ignored_expectations']}")

        # ---- 4. 接缝
        seam_ok = SeamCheck().evaluate(seamv, QCContext(seam_times_s=(1.5,)))
        seam_ng = SeamCheck().evaluate(seamv, QCContext(seam_times_s=(3.0,)))
        assert seam_ok.passed and not seam_ng.passed, (seam_ok.value, seam_ng.value)
        assert SeamCheck().evaluate(good, QCContext()).status is MetricStatus.SKIP
        # 越界的接缝时间点（xfade 的 offset 累积算错时最典型）必须被识破，
        # 不能夹到首/末帧编一次测量出来，更不能因此拿到 pass。
        seam_oob = SeamCheck().evaluate(seamv, QCContext(seam_times_s=(-1.0, 99.0)))
        assert seam_oob.status is MetricStatus.SKIP, seam_oob
        assert seam_oob.detail["out_of_range"] == [-1.0, 99.0], seam_oob.detail
        seam_mix = SeamCheck().evaluate(seamv, QCContext(seam_times_s=(3.0, 99.0)))
        assert not seam_mix.passed and seam_mix.detail["out_of_range"] == [99.0]
        assert len(seam_mix.detail["seams"]) == 1, seam_mix.detail["seams"]
        print(f"[seam] 片内 1.5s 处={seam_ok.value:.2f}×(pass)  真接缝 3.0s 处="
              f"{seam_ng.value:.2f}×(fail)  越界时间点 {seam_oob.detail['out_of_range']} -> skip")

        # ---- 5. 插件降级：无 GPU 环境下必须 skip 而不是崩
        avail = plugin_availability()
        assert set(avail) == {"identity_drift", "aesthetic", "lip_sync"}, avail
        assert all(v for v in avail.values()), "本机不该有这些重依赖，却报告可用"
        idr = IdentityDrift().evaluate(good, QCContext())
        assert idr.status is MetricStatus.SKIP and idr.score == 1.0
        assert available_plugins() == []
        print(f"[plugins] 全部优雅降级：" + "；".join(f"{k}={v[:14]}" for k, v in avail.items()))

        # ---- 6. 门禁整体判定
        gate = QCGate.default(include_plugins=False)
        r_good = gate.evaluate(good, shot_ok)
        assert r_good.verdict is Verdict.PASS, r_good.table()
        assert r_good.score > 0.85, r_good.score
        assert r_good.advice == []

        shot3 = Shot(id="s03", scene_id="sc01", index=2, duration_s=3.0, fps=24)
        r_flick = gate.evaluate(flicker, shot3)
        r_static = gate.evaluate(static, shot3)
        r_black = gate.evaluate(blackv, Shot(id="s04", scene_id="sc01", index=3,
                                             duration_s=4.5, fps=24))
        r_drift = gate.evaluate(drift, shot3)
        for r in (r_flick, r_static, r_black, r_drift):
            assert r.verdict is Verdict.RETAKE, (r.video, r.verdict, r.score)
            assert r.advice, r.video
        assert r_good.score > max(r.score for r in (r_flick, r_static, r_black, r_drift)), \
            "好片总分没有高过所有废片，门禁没区分力"
        assert "black" in r_black.blocking_failures and "freeze" in r_black.blocking_failures
        assert "CFG" in " ".join(r_flick.advice), r_flick.advice
        assert "尾帧锚" in " ".join(r_drift.advice), r_drift.advice
        print()
        print(r_good.table())
        print()
        print(r_flick.table())

        # ---- 7. 重拍次数用尽 -> escalate
        spent = Shot(id="s05", scene_id="sc01", index=4, duration_s=3.0, fps=24, takes=2)
        assert gate.evaluate(static, spent).verdict is Verdict.ESCALATE
        assert "烧额度" in " ".join(gate.evaluate(static, spent).advice)

        # ---- 8. 指标异常 -> escalate 而不是静默放行
        class _Boom(QCMetric):
            name = "boom"

            def evaluate(self, video, context):
                raise ValueError("模拟指标内部错误")

        boom_gate = QCGate([GateEntry(_Boom()), GateEntry(MotionSanity())])
        r_boom = boom_gate.evaluate(good, shot_ok)
        assert r_boom.verdict is Verdict.ESCALATE and r_boom.metrics[0].status is MetricStatus.ERROR
        print(f"[robust] 指标抛异常 -> {r_boom.verdict.value}（不静默放行）")

        # ---- 8b. 证据不足 -> 绝不能给 pass
        # 被截断成一帧的废片：逐帧指标全 SKIP，freeze/black 查不出事件。
        # 门禁必须承认「我没看清」，而不是靠剩下两项凑出 1.0 分放行。
        one = work / "one_frame.mp4"
        ff.run(["-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=24:d=1",
                "-frames:v", "1", "-pix_fmt", "yuv420p", str(one)])
        r_one = gate.evaluate(one)
        assert r_one.verdict is Verdict.ESCALATE, r_one.table()
        assert "证据不足" in " ".join(r_one.advice), r_one.advice
        skipped = [m.name for m in r_one.metrics if m.status is MetricStatus.SKIP]
        assert {"temporal_flicker", "color_drift", "motion_sanity"} <= set(skipped), skipped
        # 反向：正常片即使没有工单（conformance/seam 跳过）也要够证据，不能被误伤
        assert gate.evaluate(good).verdict is Verdict.PASS
        print(f"[证据门槛] 单帧截断片：{len(skipped)} 项 SKIP -> {r_one.verdict.value}；"
              f"无工单的正常片仍 {gate.evaluate(good).verdict.value}")

        # ---- 8c. 纯音频 / 缺文件：指标炸掉必须升级人工，不能当通过
        aud = work / "audio_only.m4a"
        ff.run(["-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=2", "-c:a", "aac", str(aud)])
        r_aud = gate.evaluate(aud)
        assert r_aud.verdict is Verdict.ESCALATE, r_aud.table()
        assert any(m.status is MetricStatus.ERROR for m in r_aud.metrics)
        r_missing = gate.evaluate(work / "根本不存在.mp4")
        assert r_missing.verdict is Verdict.ESCALATE, r_missing.table()
        print(f"[边界] 纯音频 -> {r_aud.verdict.value}；缺文件 -> {r_missing.verdict.value}")

        # ---- 8d. 路径含空格与中文：整条链路不能在引号转义上翻车
        odd_dir = work / "场次 01 镜头"
        odd_dir.mkdir(parents=True, exist_ok=True)
        odd = _mk(ff, odd_dir / "好 片 A.mp4", "testsrc2", dur=3.0)
        r_odd = gate.evaluate(odd, shot_ok)
        assert r_odd.verdict is Verdict.PASS, r_odd.table()
        assert QCContext().frames(odd, count=4, width=64).shape == (4, 36, 64, 3)
        print(f"[怪路径] '场次 01 镜头/好 片 A.mp4' -> {r_odd.verdict.value} "
              f"score={r_odd.score:.3f}，抽帧正常")

        # ---- 8e. 空批量不能崩
        assert "共 0 条" in batch_qc([], gate=gate)

        # ---- 9. 批量报告
        reports: list[QCReport] = []
        table = batch_qc(
            [good, flicker, static, blackv, drift],
            gate=gate,
            shots=[shot_ok, shot3, shot3,
                   Shot(id="s04", scene_id="sc01", index=3, duration_s=4.5, fps=24), shot3],
            reports_out=reports,
        )
        assert len(reports) == 5
        assert "废片率 80%" in table, table
        # 排序：最差的在最上面
        body = [ln for ln in table.splitlines() if ln.startswith(("good", "bad"))]
        assert body[-1].startswith("good"), body
        print()
        print(table)

        # ---- 10. 序列化
        d = r_black.to_dict()
        assert d["verdict"] == "retake" and isinstance(d["metrics"], list)
        assert d["metrics"][0]["detail"]["_blocking"] is True
        print("\n[selftest] qc.py 全部断言通过")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
