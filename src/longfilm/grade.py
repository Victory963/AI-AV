"""跨镜色彩一致 —— 把不同镜头/不同引擎出来的片段拉到同一个基准上。

生成式产线的老问题：同一场戏里，A 镜是 Seedance 出的偏暖，B 镜是自建 Wan 出的偏青，
C 镜重试过一次曝光又高了半档。单看每条都不错，拼起来一眼假 —— 观众对**跨镜跳色**
的敏感度远高于对绝对色准的敏感度。

处理链：
    sample_stats  抽帧统计每条片子的均值/标准差/直方图/色温
    scene_reference  取场景基准（中位数，抗异常镜污染）
    compile_match  Reinhard 均值-标准差匹配 -> 每通道仿射 y = g·x + b
    match_to_reference / generate_lut  把仿射编译成 ffmpeg curves / colorbalance+eq / .cube

统计和匹配都在**伽马域**（8bit rgb24）里做，因为 ffmpeg 的 curves/lut3d/colorbalance
也在这个域里工作。换到线性光域算再套回去，参数就对不上了。

关于 colorbalance 的权重：本文件里的 shadows/midtones/highlights 权重曲线是在
本机 ffmpeg 7.0.2 上**实测**出来的（灰阶梯度扫描，见 _CB_SCALE 处注释），
不是照搬文档 —— 它的 midtones 实际峰值在像素值 0.25 而不是 0.5，照文档写会调歪。
"""

from __future__ import annotations

import logging
import math
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .schema import Grade
from .stitch import FFmpeg, MediaInfo, default_ffmpeg

log = logging.getLogger(__name__)

HIST_BINS = 64

# sRGB(D65) -> CIE XYZ。用于把平均色算成色温，只做相对漂移指示。
_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]]
)

# --- colorbalance 的实测模型（ffmpeg 7.0.2，static build）---------------------
# 扫描方式：对 0..255 的灰阶梯度分别施加 rs/rm/rh = 0.2，测每个亮度点的输出增量。
# 结论：l = max(r,g,b) + min(r,g,b)（归一化后 0..2，灰像素即 2·v），
#       权重 w_s/w_m/w_h 如下，整体再乘 _CB_SCALE = 0.7。
# 实测把 rm=0.2 施加到 v=0.25 的灰上，输出正好 +36/255 = 0.2·0.7 —— 与下式吻合。
# 注意 midtones 的峰值在 v=0.25，不是直觉上的 0.5；v≥0.44 全部归 highlights。
_CB_SCALE = 0.7
_CB_A = 4.0
_CB_B = 1.0 / 3.0


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _ff_quote_path(path: str | Path) -> str:
    """把路径包成能安全嵌进 ffmpeg 滤镜参数的形式。

    实测（ffmpeg 7.0.2）滤镜参数里路径的三种情况：
      - 裸路径：空格和中文都没问题，但 ':' 会被当成下一个选项的分隔符 -> 解析失败；
      - 单引号包住：空格/中文/':' 里只有前两者变安全，':' 仍要额外写成 '\\:'；
      - 单引号内的 "'" 无论怎么转义都还原不回来（'\\'' 那套是 shell 的规矩，
        ffmpeg 的两级解析器不认），所以带单引号的路径只能由调用方改道。

    因此这里统一走「单引号 + 转义反斜杠和冒号」，遇到单引号抛错，
    由 match_to_reference 改用临时副本，绝不生成一条会静默走歪的滤镜链。
    """
    s = str(path)
    if "'" in s:
        raise ValueError(f"ffmpeg 滤镜参数无法表达含单引号的路径：{s}")
    return "'" + s.replace("\\", "\\\\").replace(":", "\\:") + "'"


def _natural_cubic(xc: Sequence[float], yc: Sequence[float], x: np.ndarray) -> np.ndarray:
    """自然三次样条，口径对齐 ffmpeg vf_curves.c 的 interpolate()。

    为什么需要它：curves 滤镜**不是**在控制点之间连直线，而是过点做自然三次样条。
    控制点一旦因为夹取而不共线（仿射顶到 0/1 时必然发生），样条就会在拐点附近
    冲出去。有了这个模型，fit_error("curves") 才能给出真实误差而不是拍脑袋的 0。
    实测与 ffmpeg 真实输出的差异 ≤ 1.9/255（即 8bit 取整噪声量级）。
    """
    xa = np.asarray(xc, dtype=np.float64)
    ya = np.asarray(yc, dtype=np.float64)
    n = xa.size
    if n < 3:
        return np.interp(x, xa, ya)
    h = np.diff(xa)
    alpha = np.zeros(n)
    alpha[1:-1] = 3.0 * ((ya[2:] - ya[1:-1]) / h[1:] - (ya[1:-1] - ya[:-2]) / h[:-1])
    ll = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    for i in range(1, n - 1):
        ll[i] = 2.0 * (xa[i + 1] - xa[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / ll[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / ll[i]
    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(n - 2, -1, -1):
        c[i] = z[i] - mu[i] * c[i + 1]
        b[i] = (ya[i + 1] - ya[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
    xv = np.asarray(x, dtype=np.float64)
    idx = np.clip(np.searchsorted(xa, xv, side="right") - 1, 0, n - 2)
    dx = xv - xa[idx]
    return ya[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3


def _cb_weights(v: float) -> tuple[float, float, float]:
    """中性灰像素值 v（0..1）上，colorbalance 三档的权重。"""
    l = 2.0 * v
    ws = _clip01((_CB_B - l) * _CB_A + 0.5)
    wm = _clip01((l - _CB_B) * _CB_A + 0.5) * _clip01((1.0 - l - _CB_B) * _CB_A + 0.5)
    wh = _clip01((l + _CB_B - 1.0) * _CB_A + 0.5)
    return ws, wm, wh


# 三个解耦锚点：在这三个亮度上，三档权重恰好是 (1,0,0) / (0,1,0) / (0,0,1)，
# 所以可以直接除出各档的系数，不用解最小二乘。
_CB_ANCHORS = (0.05, 0.25, 0.50)


# ---------------------------------------------------------------- 统计量


@dataclass(frozen=True)
class ColorStats:
    """一条片子（或一个场景基准）的色彩指纹。"""

    source: str = ""
    n_frames: int = 0
    width: int = 0
    height: int = 0
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)    # RGB，0..255，伽马域
    std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    luma_mean: float = 0.0
    luma_std: float = 0.0
    sat_mean: float = 0.0                                  # (max-min)/max，0..1
    cct_k: float = 0.0                                     # 平均色的等效色温
    hist: tuple[tuple[float, ...], ...] = ()               # 3 × HIST_BINS，各自归一化

    def drift_from(self, ref: ColorStats) -> dict[str, float]:
        """相对基准的漂移量。score 是给 QC 门禁用的单一标量。"""
        dm = tuple(self.mean[c] - ref.mean[c] for c in range(3))
        ds = tuple(self.std[c] - ref.std[c] for c in range(3))
        mean_l2 = math.sqrt(sum(x * x for x in dm) / 3)
        std_l2 = math.sqrt(sum(x * x for x in ds) / 3)
        # 色偏（去掉整体明暗后的通道差）才是「一眼假」的主因，单列出来。
        avg = sum(dm) / 3
        cast = math.sqrt(sum((x - avg) ** 2 for x in dm) / 3)
        return {
            "d_r": dm[0], "d_g": dm[1], "d_b": dm[2],
            "d_luma": self.luma_mean - ref.luma_mean,
            "mean_l2": mean_l2,
            "std_l2": std_l2,
            "cast": cast,
            "d_cct": self.cct_k - ref.cct_k,
            "d_sat": self.sat_mean - ref.sat_mean,
            # 色偏权重给高：同样 5/255 的偏差，纯亮度差看不出来，色偏一眼就出。
            "score": mean_l2 + 2.0 * cast + 0.5 * std_l2,
        }

    def to_grade(self, ref: ColorStats) -> Grade:
        """折算成 schema.Grade，回填进 Shot.grade 供复现。

        三个字段都是**修正量**（把 self 拉到 ref 要做什么），不是本片现状：
        contrast / saturation 是乘性系数，temperature 是 -1 冷 .. +1 暖 的推杆。

        temperature 的符号：色温 K 值越高画面越**冷**（蓝），所以「本片比基准冷」
        （self.cct_k > ref.cct_k）需要的修正是**加暖**，即正值。
        写成 (self - ref) 而不是 (ref - self)，方向才和 Grade 的语义一致。
        """
        return Grade(
            contrast=round(ref.luma_std / self.luma_std, 4) if self.luma_std > 1e-6 else 1.0,
            saturation=round(ref.sat_mean / self.sat_mean, 4) if self.sat_mean > 1e-6 else 1.0,
            temperature=round(max(-1.0, min(1.0, (self.cct_k - ref.cct_k) / 3000.0)), 4),
        )

    def summary(self) -> str:
        return (
            f"RGB({self.mean[0]:.1f},{self.mean[1]:.1f},{self.mean[2]:.1f}) "
            f"σ({self.std[0]:.1f},{self.std[1]:.1f},{self.std[2]:.1f}) "
            f"Y={self.luma_mean:.1f} sat={self.sat_mean:.3f} CCT≈{self.cct_k:.0f}K"
        )


def _cct_from_rgb(mean_rgb: Sequence[float]) -> float:
    """McCamy(1992) 近似：由色度坐标反推相关色温。

    先把伽马域的平均 RGB 反伽马到线性光再转 XYZ —— 直接拿伽马值转会系统性偏冷。
    这个值只用来**比较**镜头之间的冷暖差，不是色度学意义上的白点测量。
    """
    v = np.clip(np.asarray(mean_rgb, dtype=float) / 255.0, 1e-6, 1.0)
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    xyz = _RGB2XYZ @ lin
    s = float(xyz.sum())
    if s <= 1e-9:
        return 0.0
    x, y = float(xyz[0]) / s, float(xyz[1]) / s
    if abs(0.1858 - y) < 1e-6:
        return 0.0
    n = (x - 0.3320) / (0.1858 - y)
    return 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33


def stats_from_frames(frames: np.ndarray, *, source: str = "") -> ColorStats:
    """frames: (N, H, W, 3) uint8 / float，RGB 伽马域。"""
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.size == 0:
        # numpy 对空数组求 mean 只给一堆 nan + RuntimeWarning，随后一路污染到
        # compile_match 的增益里；在入口就拦掉，报错信息才指向真正的源头。
        raise ValueError(f"frames 形状非法或为空：{frames.shape}，需要 (N,H,W,3) 且 N>0")
    a = frames.astype(np.float32)
    n, h, w = a.shape[0], a.shape[1], a.shape[2]
    flat = a.reshape(-1, 3)
    mean = tuple(float(x) for x in flat.mean(axis=0))
    std = tuple(float(x) for x in flat.std(axis=0))
    luma = flat @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mx, mn = flat.max(axis=1), flat.min(axis=1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    hist = tuple(
        tuple(
            (np.histogram(flat[:, c], bins=HIST_BINS, range=(0, 255))[0] / flat.shape[0]).tolist()
        )
        for c in range(3)
    )
    return ColorStats(
        source=source, n_frames=n, width=w, height=h,
        mean=mean, std=std,
        luma_mean=float(luma.mean()), luma_std=float(luma.std()),
        sat_mean=float(sat.mean()), cct_k=_cct_from_rgb(mean), hist=hist,
    )


def sample_stats(
    video: str | Path,
    *,
    frames: int = 9,
    sample_size: tuple[int, int] = (160, 90),
    edge_skip_s: float = 0.0,
    ffmpeg: FFmpeg | None = None,
    timeout: float = 300.0,
) -> ColorStats:
    """抽 N 帧算统计量。

    为什么不用 signalstats 滤镜：它只给 YUV 的 min/max/avg，拿不到每通道直方图，
    也拿不到我们要的 RGB 域标准差，还得再解析一遍 metadata 日志。
    抽帧降采样到 160x90 后交给 numpy，代价是几十毫秒，拿到的信息完整得多。

    降采样是有意的：色彩统计只关心低频，缩图还顺带把生成片的高频噪点平均掉，
    统计量比全分辨率更稳。

    edge_skip_s：跳过首尾若干秒。生成片的第一帧常有一次曝光跳变，
    尾帧常塌，算进基准会把整条片子拉偏。
    """
    ff = ffmpeg or default_ffmpeg()
    info: MediaInfo = ff.probe(video)
    if not info.has_video:
        raise ValueError(f"{video} 没有视频轨，无法统计色彩")
    frames = max(1, frames)
    w, h = sample_size
    if w < 1 or h < 1:
        raise ValueError(f"sample_size 非法：{sample_size}")

    # 掐头去尾不能把整条片子掐没了。短镜（生成产线里 2~3s 很常见）套上默认
    # edge_skip_s 会直接变成空窗口，原实现在那里硬抛「一帧都没抽到」——
    # 对调用方来说这是个假故障，真实情况只是「这条太短，跳不了边」。
    skip = max(0.0, edge_skip_s)
    if info.duration_s - 2 * skip < 0.2:
        if skip > 0:
            log.info("%s 只有 %.2fs，放弃掐头去尾 %.2fs（否则无帧可抽）",
                     Path(video).name, info.duration_s, skip)
        skip = 0.0
    usable = max(info.duration_s - 2 * skip, 1e-3)
    rate = max(frames / usable, 1e-3)
    args: list[str] = []
    if skip > 0:
        args += ["-ss", f"{skip:.3f}", "-t", f"{usable:.3f}"]
    args += [
        "-i", str(video),
        "-vf", f"fps={rate:.6f},scale={w}:{h}:flags=area,format=rgb24",
        # -fps_mode passthrough 是 -vsync 0 在 ffmpeg 5+ 的正式写法；
        # 旧名在 7.0.2 上还能用但每次都打 deprecated 警告，污染 stderr。
        "-frames:v", str(frames), "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = ff.run_capture(args, timeout=timeout)
    px = w * h * 3
    got = len(raw) // px
    if got == 0:
        raise ValueError(f"{video} 一帧都没抽到（duration={info.duration_s}s）")
    arr = np.frombuffer(raw[: got * px], dtype=np.uint8).reshape(got, h, w, 3)
    return stats_from_frames(arr, source=str(video))


def scene_reference(
    clips: Sequence[str | Path] | Sequence[ColorStats],
    *,
    frames: int = 9,
    ffmpeg: FFmpeg | None = None,
) -> ColorStats:
    """一个场景的色彩基准。逐项取**中位数**，不是均值。

    理由：一场戏里只要有一条镜出错（引擎把夜戏渲成了白天、某次重试曝光高了一档、
    或者本来就有一个爆闪镜），均值会被它整体拽走，然后所有正常镜都被"匹配"到
    错误基准上 —— 一条坏镜污染全场。中位数的崩溃点是 50%：只要过半镜头正常，
    基准就完全不受异常值影响。

    另一个考量：按镜取中位数而不是按帧，等于每条镜一票。否则一条 15s 的
    空镜头会比三条 5s 的对白镜权重还大，基准会被最长的那条定义。
    """
    stats = [
        c if isinstance(c, ColorStats) else sample_stats(c, frames=frames, ffmpeg=ffmpeg)
        for c in clips
    ]
    if not stats:
        raise ValueError("clips 为空，求不出场景基准")

    med = statistics.median
    mean = tuple(med([s.mean[c] for s in stats]) for c in range(3))
    std = tuple(med([s.std[c] for s in stats]) for c in range(3))
    hist: tuple[tuple[float, ...], ...] = ()
    if all(s.hist for s in stats):
        hist = tuple(
            tuple(med([s.hist[c][b] for s in stats]) for b in range(HIST_BINS)) for c in range(3)
        )
    return ColorStats(
        source=f"<scene_reference of {len(stats)} clips>",
        n_frames=sum(s.n_frames for s in stats),
        width=stats[0].width, height=stats[0].height,
        mean=mean, std=std,
        luma_mean=med([s.luma_mean for s in stats]),
        luma_std=med([s.luma_std for s in stats]),
        sat_mean=med([s.sat_mean for s in stats]),
        cct_k=_cct_from_rgb(mean), hist=hist,
    )


# ---------------------------------------------------------------- 匹配参数


@dataclass(frozen=True)
class MatchParams:
    """Reinhard 匹配编译出的每通道仿射：y = gain·x + offset（x, y 都在 0..1）。

    推导
    ----
    目标是让被调片的每通道一阶、二阶矩等于基准：
        y_c = (x_c - μ_src,c) · (σ_ref,c / σ_src,c) + μ_ref,c
    展开就是仿射
        gain_c   = σ_ref,c / σ_src,c
        offset_c = (μ_ref,c - gain_c · μ_src,c) / 255      （μ 以 0..255 计）
    strength s ∈ [0,1] 做部分匹配（整场一次调到位容易把风格抹平）：
        gain'_c   = 1 + s·(gain_c - 1)
        μ'_c      = μ_src,c + s·(μ_ref,c - μ_src,c)
        offset'_c = (μ'_c - gain'_c·μ_src,c) / 255
    """

    gain: tuple[float, float, float]
    offset: tuple[float, float, float]
    src_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    strength: float = 1.0

    # ---- 数值应用（生成 LUT / 自检用）

    def apply01(self, x: np.ndarray) -> np.ndarray:
        """x: (..., 3) 0..1。返回夹到 0..1 的结果。"""
        g = np.asarray(self.gain, dtype=np.float64)
        b = np.asarray(self.offset, dtype=np.float64)
        return np.clip(x * g + b, 0.0, 1.0)

    # ---- 编译到 ffmpeg

    # 控制点数默认值。ffmpeg 的 curves 过控制点做**自然三次样条**而不是折线，
    # 所以点越少、仿射越强（被 0/1 夹过），样条冲得越离谱。本机 ffmpeg 7.0.2
    # 实测（灰阶梯度扫描，5 组典型仿射取最坏值，误差相对精确仿射）：
    #     5 点 max 24.0/255   17 点 max 6.0/255
    #    33 点 max  4.0/255   65 点 max 2.0/255   129 点 max 1.5/255
    # 8bit 量化本身就有 ±0.5/255 的底噪，65 点已经贴着底噪，再加点只是徒增
    # 命令行长度（257 点会把滤镜串撑到 ffmpeg 拒绝解析）。
    DEFAULT_POINTS: int = 65

    def curves_filter(self, *, points: int | None = None) -> str:
        """curves：把仿射编译成曲线，是三条路里最准的一条。

        注意它**不是**逐点精确：curves 在控制点之间走自然三次样条，
        端点被夹取后控制点不再共线，样条就会在拐点附近过冲。
        默认 65 点把最坏误差压到 2/255（≈ 8bit 量化底噪），
        真实误差随时可以用 fit_error("curves") 查。
        """
        n = max(2, int(points if points is not None else self.DEFAULT_POINTS))
        chans = "rgb"
        parts = []
        for c in range(3):
            pts = []
            for i in range(n):
                x = i / (n - 1)
                y = _clip01(x * self.gain[c] + self.offset[c])
                pts.append(f"{x:.4f}/{y:.4f}")
            parts.append(f"{chans[c]}='{' '.join(pts)}'")
        return "curves=" + ":".join(parts)

    def _cb_coeffs(self) -> tuple[float, float, list[float], list[float], list[float]]:
        """eq + colorbalance 的系数，eq_colorbalance_filter 与 fit_error 共用。

        抽出来是因为两边各写一份时改了一处忘另一处，报出来的误差就不是
        实际要跑的那条滤镜的误差 —— 那比没有误差估计更糟。
        """
        contrast = sum(self.gain) / 3.0
        brightness = sum(self.offset) / 3.0 - 0.5 * (1.0 - contrast)
        s, m, h = [0.0] * 3, [0.0] * 3, [0.0] * 3
        for c in range(3):
            for k, v in enumerate(_CB_ANCHORS):
                want = _clip01(v * self.gain[c] + self.offset[c])
                after_eq = _clip01((v - 0.5) * contrast + 0.5 + brightness)
                delta = (want - after_eq) / _CB_SCALE
                (s, m, h)[k][c] = max(-1.0, min(1.0, delta))
        return contrast, brightness, s, m, h

    def eq_colorbalance_filter(self) -> str:
        """eq + colorbalance：给「要在 DaVinci / OBS 里手动复刻」的场合。

        eq 只能整体调（out = (x-0.5)·contrast + 0.5 + brightness），
        所以先用三通道的平均增益做全局部分：
            contrast   = mean_c(gain_c)
            brightness = mean_c(offset_c) - 0.5·(1 - contrast)
        剩下的每通道残差交给 colorbalance。用三个解耦锚点（v=0.05/0.25/0.50，
        对应实测权重恰为 (1,0,0)/(0,1,0)/(0,0,1)）直接除出系数：
            shadow_c = Δ_c(0.05)/0.7,  mid_c = Δ_c(0.25)/0.7,  high_c = Δ_c(0.50)/0.7
        其中 Δ_c(v) = 目标输出 - eq 之后的输出。

        这条路是**近似**：colorbalance 在 v>0.5 的区间全部落进 highlights 一档，
        表达不出那里的斜率差异。精确匹配请用 curves 或 LUT。
        """
        contrast, brightness, s, m, h = self._cb_coeffs()
        return (
            f"eq=contrast={contrast:.4f}:brightness={brightness:.4f}:saturation=1.0,"
            f"colorbalance=rs={s[0]:.4f}:gs={s[1]:.4f}:bs={s[2]:.4f}"
            f":rm={m[0]:.4f}:gm={m[1]:.4f}:bm={m[2]:.4f}"
            f":rh={h[0]:.4f}:gh={h[1]:.4f}:bh={h[2]:.4f}"
        )

    def fit_error(self, kind: str = "eq_colorbalance", *, points: int | None = None) -> float:
        """该编译方式相对精确仿射的 RMS 误差（0..255 刻度），供调用方选路。

        三条路都按各自滤镜的**真实数学模型**重算一遍，不给任何一条开后门：
        curves 走自然三次样条（见 _natural_cubic），eq+colorbalance 走实测权重模型，
        lut3d 逐点就是仿射本身，所以只剩 LUT 网格的线性插值误差（size≥33 时可忽略）。
        """
        xs = np.linspace(0.0, 1.0, 256)
        want = np.stack([np.clip(xs * self.gain[c] + self.offset[c], 0, 1) for c in range(3)], 1)

        if kind == "lut3d":
            return 0.0
        if kind == "curves":
            n = max(2, int(points if points is not None else self.DEFAULT_POINTS))
            xc = [i / (n - 1) for i in range(n)]
            got = np.stack([
                np.clip(
                    _natural_cubic(xc, [_clip01(x * self.gain[c] + self.offset[c]) for x in xc], xs),
                    0.0, 1.0,
                )
                for c in range(3)
            ], axis=1)
        elif kind == "eq_colorbalance":
            contrast, brightness, s, m, h = self._cb_coeffs()
            got = np.zeros_like(want)
            for i, x in enumerate(xs):
                base = _clip01((x - 0.5) * contrast + 0.5 + brightness)
                ws, wm, wh = _cb_weights(base)
                for c in range(3):
                    got[i, c] = _clip01(base + _CB_SCALE * (s[c] * ws + m[c] * wm + h[c] * wh))
        else:
            raise ValueError(f"kind 只支持 curves / eq_colorbalance / lut3d，收到 {kind!r}")
        return float(np.sqrt(((want - got) ** 2).mean()) * 255.0)

    def describe(self) -> str:
        return " ".join(
            f"{ch}:×{self.gain[i]:.3f}{self.offset[i] * 255:+.1f}"
            for i, ch in enumerate("RGB")
        )


def compile_match(
    src: ColorStats, ref: ColorStats, *, strength: float = 1.0, max_gain: float = 2.5
) -> MatchParams:
    """Reinhard 均值/标准差匹配 -> 每通道仿射。公式见 MatchParams 的 docstring。

    max_gain 限幅是必须的：生成片偶尔会出一条几乎纯色的镜（σ≈0），
    σ_ref/σ_src 会炸到几十倍，套上去整条片子变成噪点墙。
    """
    strength = max(0.0, min(1.0, strength))
    gain: list[float] = []
    offset: list[float] = []
    for c in range(3):
        ss = src.std[c]
        if ss < 1e-3:
            g = 1.0
            log.warning("%s 通道 %s 标准差近似 0（σ=%.4f），该通道只做平移不做增益",
                        src.source, "RGB"[c], ss)
        else:
            g = max(1.0 / max_gain, min(max_gain, ref.std[c] / ss))
        g = 1.0 + strength * (g - 1.0)
        mu = src.mean[c] + strength * (ref.mean[c] - src.mean[c])
        gain.append(g)
        offset.append((mu - g * src.mean[c]) / 255.0)
    return MatchParams(
        gain=(gain[0], gain[1], gain[2]),
        offset=(offset[0], offset[1], offset[2]),
        src_mean=src.mean,
        strength=strength,
    )


# ---------------------------------------------------------------- 应用


def match_to_reference(
    video: str | Path,
    ref_stats: ColorStats,
    out: str | Path,
    *,
    src_stats: ColorStats | None = None,
    strength: float = 1.0,
    method: str = "curves",
    crf: int = 18,
    frames: int = 9,
    ffmpeg: FFmpeg | None = None,
    timeout: float = 1800.0,
) -> MatchParams:
    """把一条片子匹配到基准，写出新文件，返回用到的仿射参数。

    method:
      "curves"          精确仿射，默认。
      "eq_colorbalance" 近似，但参数能一对一搬进 DaVinci 的 Lift/Gamma/Gain 思路里。
      "lut3d"           先生成 .cube 再套；一条片子只调一次时比 curves 慢，
                        但同一组参数要套几十条片时，LUT 只算一次表。

    RGB 域调色意味着 yuv420p -> gbrp -> yuv420p 的往返，色度会有一次重采样。
    这是所有 RGB 调色滤镜的固有代价，换 yuv 域滤镜就没法做精确的每通道仿射。
    """
    ff = ffmpeg or default_ffmpeg()
    src = src_stats or sample_stats(video, frames=frames, ffmpeg=ff)
    params = compile_match(src, ref_stats, strength=strength)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir: str | None = None
    try:
        if method == "curves":
            vf = params.curves_filter()
            log.info("curves 曲线拟合，RMS 误差 %.2f/255", params.fit_error("curves"))
        elif method == "eq_colorbalance":
            vf = params.eq_colorbalance_filter()
            log.info("eq+colorbalance 近似，RMS 误差 %.2f/255", params.fit_error())
        elif method == "lut3d":
            cube = out.with_suffix(".cube")
            generate_lut(src, ref_stats, cube, strength=strength)
            try:
                ref_path = _ff_quote_path(cube)
            except ValueError:
                # 交付路径里带单引号时 ffmpeg 滤镜参数表达不了，改喂一份临时副本；
                # 交付用的 .cube 仍然留在原位，不牺牲产物。
                import shutil as _sh
                import tempfile as _tf

                tmp_dir = _tf.mkdtemp(prefix="longfilm_lut_")
                staged = Path(tmp_dir) / "match.cube"
                _sh.copyfile(cube, staged)
                log.info("LUT 路径含单引号，改用临时副本 %s 喂给 ffmpeg", staged)
                ref_path = _ff_quote_path(staged)
            vf = f"lut3d=file={ref_path}:interp=tetrahedral"
        else:
            raise ValueError(f"method 只支持 curves / eq_colorbalance / lut3d，收到 {method!r}")

        info = ff.probe(video)
        args = ["-y", "-i", str(video), "-vf", f"format=gbrp,{vf},format=yuv420p",
                "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"]
        args += (["-c:a", "copy"] if info.has_audio else ["-an"])
        args += [str(out)]
        ff.run(args, timeout=timeout)
    finally:
        if tmp_dir:
            import shutil as _sh

            _sh.rmtree(tmp_dir, ignore_errors=True)
    log.info("匹配完成 %s -> %s  %s", Path(video).name, out.name, params.describe())
    return params


def generate_lut(
    src_stats: ColorStats,
    ref_stats: ColorStats,
    out_cube: str | Path,
    *,
    size: int = 33,
    strength: float = 1.0,
    soft_knee: float = 0.0,
    title: str = "longfilm scene match",
) -> str:
    """把匹配参数烘成 .cube 3D LUT，供 lut3d 滤镜 / DaVinci / OBS 使用。

    CUBE 格式（Adobe 规范）：头部 TITLE / LUT_3D_SIZE / DOMAIN_MIN / DOMAIN_MAX，
    随后 size³ 行 "r g b"，**红色分量变化最快**，蓝色最慢。写反了不会报错，
    只会得到一个颜色被转置的 LUT，所以这个顺序是本函数唯一容易错的地方。

    soft_knee > 0 时对两端做 tanh 软压，避免匹配后大面积死白/死黑；
    默认 0（硬夹），这样 LUT 的结果与 curves 路径逐点一致，便于交叉验证。
    """
    params = compile_match(src_stats, ref_stats, strength=strength)
    size = max(2, min(64, size))
    axis = np.linspace(0.0, 1.0, size)
    g = np.asarray(params.gain)
    b = np.asarray(params.offset)

    # 红最快变化 -> 用 indexing="ij" 构造 (B, G, R) 网格再展平，行序自然正确。
    bb, gg, rr = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
    vals = grid * g + b
    if soft_knee > 0:
        k = min(0.45, soft_knee)
        hi = vals > (1 - k)
        vals[hi] = (1 - k) + k * np.tanh((vals[hi] - (1 - k)) / k)
        lo = vals < k
        vals[lo] = k * np.tanh((vals[lo] - k) / k) + k
    vals = np.clip(vals, 0.0, 1.0)

    out = Path(out_cube)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    lines.extend(f"{r:.6f} {gv:.6f} {bv:.6f}" for r, gv, bv in vals)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("写出 LUT %s（%d³ = %d 格，%s）", out, size, size**3, params.describe())
    return str(out)


# ---------------------------------------------------------------- 报告


def _disp_width(s: str) -> int:
    """终端显示宽度：中日韩全角字符占 2 列。

    用 len() 排版的表格在中文表头上必然错位 —— 报告是给人看的，这点必须算对。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    """width > 0 右对齐，< 0 左对齐（按显示宽度截断超长的镜头名）。"""
    w = max(2, abs(width))   # w<=1 时下面的循环对空串是 0 > -1 恒真，会死循环
    while s and _disp_width(s) > w - 1:
        s = s[:-1]
    fill = " " * max(0, w - _disp_width(s))
    return (s + fill) if width < 0 else (fill + s)


def grade_report(
    clips: Sequence[str | Path],
    graded: Sequence[str | Path] | None = None,
    *,
    ref: ColorStats | None = None,
    frames: int = 9,
    ffmpeg: FFmpeg | None = None,
) -> str:
    """调色前（可选：与调色后对比）的跨镜漂移报告。

    graded 传入时按位置一一对应，额外给出收敛率 —— 产线里这是判断
    「这场戏还要不要人工介入」的唯一数字。
    """
    if not clips:
        raise ValueError("clips 为空，没有可报告的镜头")
    # 数量校验放在抽帧之前：先解码十几条片子再告诉调用方参数不对，
    # 白烧的是产线里最贵的那部分时间。
    if graded is not None and len(graded) != len(clips):
        raise ValueError(f"graded 数量 {len(graded)} 与 clips 数量 {len(clips)} 不一致")

    ff = ffmpeg or default_ffmpeg()
    before = [sample_stats(c, frames=frames, ffmpeg=ff) for c in clips]
    base = ref or scene_reference(before)
    after = (
        [sample_stats(c, frames=frames, ffmpeg=ff) for c in graded] if graded else None
    )

    cols: list[tuple[str, int]] = [
        ("镜头", -24), ("ΔR", 7), ("ΔG", 7), ("ΔB", 7),
        ("Δσ", 7), ("色偏", 7), ("ΔCCT", 8), ("漂移", 8),
    ]
    if after:
        cols += [("调后漂移", 10), ("收敛", 8)]
    width = sum(abs(w) for _, w in cols)

    rows: list[str] = []
    rows.append(f"场景基准（{len(before)} 条镜的中位数）：{base.summary()}")
    rows.append("")
    rows.append("".join(_pad(h, w) for h, w in cols))
    rows.append("-" * width)

    # drift_from 要跑一串 sqrt，下面表格 + 合计 + 最差三处都要用，算一次存起来。
    d_before = [s.drift_from(base) for s in before]
    d_after = [s.drift_from(base) for s in after] if after else None

    tot_b = tot_a = 0.0
    for i, s in enumerate(before):
        d = d_before[i]
        cells = [
            Path(s.source).name, f"{d['d_r']:.1f}", f"{d['d_g']:.1f}", f"{d['d_b']:.1f}",
            f"{d['std_l2']:.1f}", f"{d['cast']:.2f}", f"{d['d_cct']:.0f}", f"{d['score']:.2f}",
        ]
        tot_b += d["score"]
        if d_after:
            da = d_after[i]
            tot_a += da["score"]
            drop = (1 - da["score"] / d["score"]) * 100 if d["score"] > 1e-6 else 0.0
            cells += [f"{da['score']:.2f}", f"{drop:.0f}%"]
        rows.append("".join(_pad(c, w) for c, (_, w) in zip(cells, cols)))

    rows.append("-" * width)
    if d_after:
        drop = (1 - tot_a / tot_b) * 100 if tot_b > 1e-6 else 0.0
        rows.append(f"合计漂移 {tot_b:.2f} -> {tot_a:.2f}，收敛 {drop:.0f}%")
        worst = max(range(len(before)), key=lambda i: d_after[i]["score"])
        rows.append(
            f"调后最差：{Path(before[worst].source).name}"
            f"（漂移 {d_after[worst]['score']:.2f}）"
        )
    else:
        rows.append(f"合计漂移 {tot_b:.2f}（未调色）")
        worst = max(range(len(before)), key=lambda i: d_before[i]["score"])
        rows.append(f"最需要处理：{Path(before[worst].source).name}")
    # 只删末尾多余的空行：中间那条空行是表头与正文的分隔，原来被一刀切掉了。
    while rows and rows[-1] == "":
        rows.pop()
    return "\n".join(rows)


# ---------------------------------------------------------------- 自测


def _make_base_png(path: Path, *, w: int = 320, h: int = 240) -> Path:
    """自测底图：中间调、三通道各自有结构、两端都不贴边。

    不用 lavfi 的 gradients/noise：它们带随机种子，每次跑出来的统计量都不一样，
    断言「匹配后收敛到基准」就变成了在测随机数。这里用 numpy 直接画，完全可复现。
    值域压在 40..210，留出余量 —— 贴边的话匹配结果会被 clip 主导，
    验证的就不是 Reinhard 而是夹取逻辑了。
    """
    from PIL import Image

    x = np.linspace(0.0, 1.0, w)[None, :]
    y = np.linspace(0.0, 1.0, h)[:, None]
    r = 75 + 95 * x + 22 * np.sin(6 * np.pi * x) + 0 * y
    g = 90 + 75 * y + 18 * np.cos(5 * np.pi * y) + 10 * x
    b = 100 + 60 * (x * y) + 16 * np.sin(8 * np.pi * x * y) + 12 * y
    img = np.clip(np.stack(np.broadcast_arrays(r, g, b), axis=-1), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    return path


# 0..1 的 256 级梯度，用来把滤镜的传递函数逐级量出来。
_RAMP_X = np.arange(256, dtype=np.float64) / 255.0


def _make_ramp_png(path: Path) -> Path:
    """256 级中性灰梯度图：横轴就是输入码值，一张图量完整条传递曲线。"""
    from PIL import Image

    ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (8, 1, 3))
    Image.fromarray(ramp).save(path)
    return path


def _apply_ramp(ff: FFmpeg, ramp_png: Path, vf: str, work: Path) -> np.ndarray:
    """把滤镜套到梯度图上，返回 (256, 3) 的实测输出码值。

    在 gbrp 里做，和 match_to_reference 走的是同一条色彩路径；
    否则量到的是 yuv 往返误差，不是滤镜本身的误差。
    """
    from PIL import Image

    out = work / "_ramp_out.png"
    ff.run(["-y", "-v", "error", "-i", str(ramp_png),
            "-vf", f"format=gbrp,{vf},format=rgb24", "-frames:v", "1", str(out)])
    return np.asarray(Image.open(out).convert("RGB"), dtype=np.float64)[4]


def _mk_drifted(ff: FFmpeg, out: Path, drift_vf: str, base_png: Path, *, dur: float = 2.0) -> str:
    """把底图套上一组色彩漂移，编码成一条「同场戏但颜色飘了」的测试片。"""
    vf = f"format=gbrp,{drift_vf},format=yuv420p" if drift_vf else "format=yuv420p"
    ff.run([
        "-y", "-loop", "1", "-framerate", "25", "-t", str(dur), "-i", str(base_png),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out),
    ])
    return str(out)


def _selftest() -> None:
    import shutil
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ff = FFmpeg()
    work = Path(tempfile.mkdtemp(prefix="longfilm_grade_"))
    try:
        # 同一底图 + 三组漂移 = 三条「同场戏、不同引擎」的镜：偏暖 / 偏冷 / 偏暗发灰。
        base = _make_base_png(work / "base.png")
        clips = [
            _mk_drifted(ff, work / "warm.mp4", "colorbalance=rh=0.10:bh=-0.08", base),
            _mk_drifted(ff, work / "cold.mp4", "eq=contrast=0.85,colorbalance=bh=0.10:rh=-0.05", base),
            _mk_drifted(ff, work / "dark.mp4", "eq=contrast=1.15:brightness=-0.05", base),
        ]

        # --- sample_stats
        st = [sample_stats(c, frames=9, ffmpeg=ff) for c in clips]
        for s in st:
            assert s.n_frames == 9, s.n_frames
            assert all(x > 0 for x in s.std), s.std
            assert 1000 < s.cct_k < 30000, s.cct_k
            print(f"[sample_stats] {Path(s.source).name:<11} {s.summary()}")
        assert st[0].cct_k < st[1].cct_k, "偏暖镜的 CCT 应低于偏冷镜"

        # --- scene_reference：中位数抗污染
        ref = scene_reference(st)
        for c in range(3):
            assert min(s.mean[c] for s in st) <= ref.mean[c] <= max(s.mean[c] for s in st)
        print(f"[scene_reference] {ref.summary()}")

        # 抗污染：5 条正常镜（同一场戏，彼此只差几个码值）+ 2 条爆闪坏镜。
        # 这段只验 scene_reference 的统计学性质，用合成 ColorStats 才能把变量控死。
        normal = [
            ColorStats(source=f"ok{i}.mp4", n_frames=9, width=160, height=90,
                       mean=(118.0 + i, 124.0 - i, 131.0 + i), std=(30.0, 27.0, 24.0),
                       luma_mean=123.0, luma_std=27.0, sat_mean=0.25, cct_k=6100.0)
            for i in range(5)
        ]
        rogue = ColorStats(source="flash.mp4", n_frames=9, width=160, height=90,
                           mean=(250.0, 250.0, 250.0), std=(2.0, 2.0, 2.0),
                           luma_mean=250.0, luma_std=2.0, sat_mean=0.01, cct_k=6500.0)
        clean_ref = scene_reference(normal)
        poisoned = scene_reference([*normal, rogue, rogue])          # 7 条里 2 条是坏镜
        med_shift = max(abs(poisoned.mean[c] - clean_ref.mean[c]) for c in range(3))
        mean_based = [sum(x) / 7 for x in zip(*[s.mean for s in (*normal, rogue, rogue)])]
        mean_shift = max(abs(mean_based[c] - clean_ref.mean[c]) for c in range(3))
        assert med_shift <= 1.0 and mean_shift > 30.0, (med_shift, mean_shift)
        print(f"[抗污染] 7 条里掺 2 条爆闪坏镜：中位数基准偏移 {med_shift:.1f} 码值，"
              f"若改用均值则偏移 {mean_shift:.1f} 码值")

        # --- compile_match 数值自洽
        p = compile_match(st[0], ref)
        x = np.asarray(st[0].mean) / 255.0
        y = p.apply01(x) * 255.0
        assert np.allclose(y, np.asarray(ref.mean), atol=0.6), (y, ref.mean)
        print(f"[compile_match] {Path(st[0].source).name} {p.describe()}"
              f"  eq+colorbalance 近似误差 {p.fit_error():.2f}/255")

        # --- match_to_reference（curves）：统计量必须收敛
        graded = []
        for i, c in enumerate(clips):
            o = work / f"g_{Path(c).stem}.mp4"
            match_to_reference(c, ref, o, src_stats=st[i], ffmpeg=ff)
            graded.append(str(o))
        gst = [sample_stats(g, frames=9, ffmpeg=ff) for g in graded]
        for i in range(3):
            before = st[i].drift_from(ref)["score"]
            after = gst[i].drift_from(ref)["score"]
            assert after < before * 0.35, (Path(clips[i]).name, before, after)
            for c in range(3):
                assert abs(gst[i].mean[c] - ref.mean[c]) < 4.0, (i, c, gst[i].mean, ref.mean)
                assert abs(gst[i].std[c] - ref.std[c]) < 4.0, (i, c, gst[i].std, ref.std)
            print(f"[match curves] {Path(clips[i]).name:<11} 漂移 {before:6.2f} -> {after:5.2f}"
                  f"  均值 {gst[i].mean[0]:.1f},{gst[i].mean[1]:.1f},{gst[i].mean[2]:.1f}"
                  f" vs 基准 {ref.mean[0]:.1f},{ref.mean[1]:.1f},{ref.mean[2]:.1f}")

        # --- LUT 路径必须与 curves 路径等价
        cube = generate_lut(st[0], ref, work / "match.cube", size=33)
        text = Path(cube).read_text(encoding="utf-8").splitlines()
        assert text[1] == "LUT_3D_SIZE 33"
        assert len([t for t in text if t[:1].isdigit()]) == 33**3
        lut_out = work / "lut_warm.mp4"
        match_to_reference(clips[0], ref, lut_out, src_stats=st[0], method="lut3d", ffmpeg=ff)
        s_lut = sample_stats(lut_out, frames=9, ffmpeg=ff)
        d = max(abs(s_lut.mean[c] - gst[0].mean[c]) for c in range(3))
        assert d < 1.5, (s_lut.mean, gst[0].mean)
        print(f"[generate_lut] 33³ CUBE，lut3d 结果与 curves 差 {d:.2f}/255")

        # --- eq+colorbalance 近似路径也要有明显收敛
        eqo = work / "eq_warm.mp4"
        match_to_reference(clips[0], ref, eqo, src_stats=st[0], method="eq_colorbalance", ffmpeg=ff)
        s_eq = sample_stats(eqo, frames=9, ffmpeg=ff)
        assert s_eq.drift_from(ref)["score"] < st[0].drift_from(ref)["score"] * 0.8
        print(f"[eq+colorbalance] 漂移 {st[0].drift_from(ref)['score']:.2f}"
              f" -> {s_eq.drift_from(ref)['score']:.2f}（近似路径）")

        # --- curves 的样条模型必须对得上 ffmpeg 真实输出
        # 这条是 fit_error("curves") 可信度的全部依据：模型错了，选路建议就是瞎猜。
        ramp = work / "ramp.png"
        _make_ramp_png(ramp)
        worst_model = 0.0
        worst_5pt = 0.0
        worst_default = 0.0
        for gain, off in (((1.2, 0.9, 1.05), (-0.06, 0.05, -0.01)),
                          ((2.0, 2.0, 2.0), (-0.2, -0.2, -0.2)),
                          ((0.5, 0.6, 0.7), (0.2, 0.15, 0.1))):
            mp = MatchParams(gain=gain, offset=off)
            want = np.stack([np.clip(_RAMP_X * gain[c] + off[c], 0, 1) for c in range(3)], 1) * 255
            for pts, bucket in ((5, "5"), (None, "d")):
                got = _apply_ramp(ff, ramp, mp.curves_filter(points=pts), work)
                err = float(np.abs(got - want).max())
                if bucket == "5":
                    worst_5pt = max(worst_5pt, err)
                else:
                    worst_default = max(worst_default, err)
            # 模型（自然三次样条）与真实输出的差
            n = MatchParams.DEFAULT_POINTS
            xc = [i / (n - 1) for i in range(n)]
            model = np.stack([
                np.clip(_natural_cubic(xc, [_clip01(x * gain[c] + off[c]) for x in xc], _RAMP_X),
                        0, 1)
                for c in range(3)
            ], 1) * 255
            got = _apply_ramp(ff, ramp, mp.curves_filter(), work)
            worst_model = max(worst_model, float(np.abs(got - model).max()))
        assert worst_model < 3.0, f"_natural_cubic 与 ffmpeg curves 对不上：{worst_model:.2f}/255"
        assert worst_default < 3.0, f"默认点数的 curves 误差过大：{worst_default:.2f}/255"
        assert worst_5pt > 3 * worst_default, (worst_5pt, worst_default)
        print(f"[curves 精度] 5 点最坏 {worst_5pt:.1f}/255 -> 默认 "
              f"{MatchParams.DEFAULT_POINTS} 点最坏 {worst_default:.1f}/255；"
              f"样条模型 vs ffmpeg 实测差 {worst_model:.1f}/255")

        # fit_error 必须是真算出来的，不能对 curves 开后门返回 0
        mp = MatchParams(gain=(2.0, 2.0, 2.0), offset=(-0.2, -0.2, -0.2))
        e5, ed = mp.fit_error("curves", points=5), mp.fit_error("curves")
        assert e5 > ed > 0.0, (e5, ed)
        assert mp.fit_error("lut3d") == 0.0
        try:
            mp.fit_error("nope")
            raise AssertionError("未知 kind 应报错")
        except ValueError:
            pass
        print(f"[fit_error] 同一组仿射：curves5={e5:.2f} curves{MatchParams.DEFAULT_POINTS}={ed:.2f}"
              f" eq+cb={mp.fit_error():.2f} lut3d=0.00 (/255 RMS)")

        # --- Grade 回填：三个字段都是「把本片拉到基准」的修正量，符号不能反
        gr = st[2].to_grade(ref)
        assert isinstance(gr, Grade) and gr.contrast > 0
        warm_gr, cold_gr = st[0].to_grade(ref), st[1].to_grade(ref)
        assert st[0].cct_k < ref.cct_k < st[1].cct_k, (st[0].cct_k, ref.cct_k, st[1].cct_k)
        # 偏暖镜（CCT 低）要往冷里拉 -> temperature 为负；偏冷镜反之。
        assert warm_gr.temperature < 0 < cold_gr.temperature, (warm_gr, cold_gr)
        # contrast 是乘性修正：dark.mp4 被拉高了对比，修正应当 < 1 把它压回去。
        assert st[2].luma_std > ref.luma_std and gr.contrast < 1.0, (st[2].luma_std, gr)
        print(f"[to_grade] warm temp={warm_gr.temperature:+.3f}（拉冷）"
              f"  cold temp={cold_gr.temperature:+.3f}（拉暖）"
              f"  dark contrast={gr.contrast}（压对比）")

        # --- 边界：空输入 / 数量不符 / 非法帧数组 / 掐头去尾超过时长
        for bad, exc in ((lambda: scene_reference([]), ValueError),
                         (lambda: grade_report([], ffmpeg=ff), ValueError),
                         (lambda: grade_report(clips, clips[:1], ffmpeg=ff), ValueError),
                         (lambda: stats_from_frames(np.zeros((0, 4, 4, 3), np.uint8)), ValueError),
                         (lambda: _ff_quote_path("/tmp/it's.cube"), ValueError)):
            try:
                bad()
                raise AssertionError(f"{bad} 该抛 {exc.__name__} 却通过了")
            except exc:
                pass
        # 2s 的片子要求掐掉首尾各 5s：应当自动放弃掐边而不是报「一帧都没抽到」
        s_skip = sample_stats(clips[0], frames=9, edge_skip_s=5.0, ffmpeg=ff)
        assert s_skip.n_frames == 9, s_skip.n_frames
        assert abs(s_skip.mean[0] - st[0].mean[0]) < 2.0, (s_skip.mean, st[0].mean)
        assert _ff_quote_path("/a b/场次:01/x.cube") == "'/a b/场次\\:01/x.cube'"
        print(f"[边界] 空输入/数量不符/空帧数组/含单引号路径 均已拦截；"
              f"edge_skip 超时长自动退化（仍抽到 {s_skip.n_frames} 帧）")

        # --- 路径含冒号与空格：三条编译路都要能真跑通（lut3d 曾在这里静默炸掉）
        odd = work / "场次 01:镜头 A"
        odd.mkdir(parents=True, exist_ok=True)
        for method in ("curves", "eq_colorbalance", "lut3d"):
            match_to_reference(clips[0], ref, odd / f"out_{method}.mp4",
                               src_stats=st[0], method=method, ffmpeg=ff)
        print("[怪路径] '场次 01:镜头 A/' 下 curves / eq_colorbalance / lut3d 均跑通")

        # --- 报告
        rep = grade_report(clips, graded, ffmpeg=ff)
        assert "收敛" in rep and "场景基准" in rep
        head = [ln for ln in rep.splitlines() if ln.lstrip().startswith("镜头")][0]
        assert _disp_width(head) == sum(abs(w) for w in (-24, 7, 7, 7, 7, 7, 8, 8, 10, 8)), head
        assert "" in rep.splitlines(), "表头与正文之间的空行被吃掉了"
        print("\n" + rep + "\n")

        print("grade._selftest 全部通过")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
