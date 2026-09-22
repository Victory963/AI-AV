"""长片续写链 —— 把一堆 5~10s 的原子片段接成一条不崩的长镜。

四种续接策略，统一成 ChainStrategy 接口：
  OfficialExtendChain  官方延长接口逐段延长（最稳，但只有声明 supports_extend 的家能用）
  JobContinueChain     用上一段的 job_id 续写（同一家、同一条隐变量血脉）
  LastFrameChain       A 的尾帧当 B 的首帧（跨 provider 通用，也最容易漂）
  OverlapChain         段间留 N 帧重叠，交给 stitch 做混接

真正决定长片崩不崩的不是哪个策略更好，而是 **漂移预算**（DriftBudget）：
链式续写的误差会累积。可查证的公开依据只有两条（查证日期 2026-09-20）：
  · 可灵官方的视频延长说明只写了「每次延长 4~5s、可多次延长、总长不超过 3 分钟」，
    并提醒延长提示词必须贴合原片主体，否则会出现镜头切换；它**没有**给出
    「连续 N 次后开始漂移」的数字。
    https://kling.ai/quickstart/ai-video-extension
  · AdaState（arXiv:2605.30349，2026-05-28）实测自回归视频生成超出训练时长后
    color drift 会「很早出现并在后续序列里持续累积」，这是「误差随段数累积」
    这一前提的学术依据，但同样不是某个具体段数。
所以下面 threshold=1.0 对应的「最稳策略恰好连 4 段」是**本产线自己标定的工程常量**，
不是任何厂商或论文的结论；换引擎必须按 DriftBudget 文档里的方法重标定。
把「漂移」当成一种额度来花：每续一段按策略/时长/是否有身份锚扣额度，
花超了就强制插一个**重锚镜** —— 不再接着续，而是用 identity 参考图重新起一段。

尾帧提取与尾帧质检是 LastFrame/Overlap 两条链的地基：
一张带运动模糊或拖影的尾帧会毒化后面**每一段**，所以宁可回退取倒数第 N 帧。
"""

from __future__ import annotations

import abc
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .providers.base import Capabilities, GenResult
from .schema import ImageRef, Scene, Shot, Storyboard, Transition
# ffmpeg 的定位、调用与 probe 只有一套实现（stitch.FFmpeg）。
# 本模块曾自带一份 subprocess 调用，结果是错误信息、超时口径、二进制查找规则
# 和拼接器各说各话；能力重叠的封装必须收敛到一处，否则换 ffmpeg 时会漏改。
from .stitch import FFmpeg, default_ffmpeg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- 尾帧工具


def ffmpeg_bin(ffmpeg: FFmpeg | None = None) -> str:
    """ffmpeg 可执行文件路径。定位规则完全由 stitch 决定，这里只是取个值。"""
    return (ffmpeg or default_ffmpeg()).binary


def local_media_path(uri: str | None) -> Path | None:
    """把 provider 返回的 uri 还原成本地可读文件；远端资源返回 None。

    GenResult.video_uri 既可能是本地产物路径，也可能是对象存储的 https 链接。
    直接 Path(uri) 会把 "https://host/a.mp4" 压成 "https:/host/a.mp4" 这种
    既不存在又看不出问题的路径，报错信息会把人带偏，所以在入口就分流。
    """
    if not uri:
        return None
    if "://" in uri and not uri.startswith("file://"):
        return None
    p = Path(uri[7:] if uri.startswith("file://") else uri)
    return p if p.is_file() else None


@dataclass(frozen=True)
class FrameStat:
    path: Path
    index_from_end: int    # 1 = 最后一帧
    sharpness: float       # 拉普拉斯方差，量纲随内容变，只能同片内横比


@dataclass(frozen=True)
class TailFrame:
    """选定的尾帧。fell_back=True 说明最后一帧被判为不可用。"""

    path: Path
    index_from_end: int
    sharpness: float
    reference_sharpness: float
    fell_back: bool
    reason: str
    fps: float = 0.0          # 实测帧率，用来把「丢了几帧」换算成秒

    @property
    def ratio(self) -> float:
        return self.sharpness / self.reference_sharpness if self.reference_sharpness else 0.0

    @property
    def dropped_s(self) -> float:
        """被丢弃的尾段时长。拼接器要按这个偏移对齐，算错就会串音画。"""
        return (self.index_from_end - 1) / self.fps if self.fps > 0 else 0.0


def frame_sharpness(path: str | Path) -> float:
    """拉普拉斯方差。越低越糊 —— 运动模糊、拖影、编码末帧糊都会砸低它。"""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # 延迟导入：让没装图像栈的机器也能 import 本模块
        raise RuntimeError(f"尾帧质检需要 numpy 与 Pillow：{exc}") from exc

    g = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def tail_fps(video: str | Path, *, hint: float = 24.0, ffmpeg: FFmpeg | None = None) -> float:
    """取视频的实测帧率。

    调用方手上只有 Shot.fps（**目标**帧率），而上一段的产物很可能是引擎按自己的
    档位出的（router 会把 24 降到 16 再交付时补帧）。用目标帧率去换算
    「丢了几帧 = 几秒」会算错，所以一律以实测为准，探不到才退回 hint。
    """
    try:
        got = (ffmpeg or default_ffmpeg()).probe(video).fps
    except Exception as exc:  # noqa: BLE001  探帧率失败不该让整条链断掉
        log.debug("探测 %s 帧率失败，退回 %.3f：%s", video, hint, exc)
        return max(hint, 1.0)
    return got if got > 0 else max(hint, 1.0)


def extract_tail_frames(
    video: str | Path,
    out_dir: str | Path,
    *,
    count: int = 6,
    fps: int | float = 24,
    ffmpeg: FFmpeg | None = None,
) -> list[FrameStat]:
    """抽片尾最后 count 帧。用 -sseof 从文件尾回退定位，不必解全片。

    fps 只用来估「往回倒多少秒才够 count 帧」，估小了会抽不满，
    所以抽不到就翻三倍再试一次；真正的帧率以 tail_fps() 实测为准。
    """
    src = local_media_path(str(video))
    if src is None:
        raise FileNotFoundError(
            f"尾帧抽取只能处理本地文件，拿到的是 {video!r}；"
            "远端产物请先下载到 workdir 再进链"
        )
    ff = ffmpeg or default_ffmpeg()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("tail_*.png"):
        stale.unlink()

    back_s = (count + 1) / max(float(fps), 1.0) + 0.05
    files: list[Path] = []
    for attempt in (back_s, back_s * 3):
        # -fps_mode passthrough 而不是已废弃的 -vsync 0：ffmpeg 7 对后者只告警不报错，
        # 但告警会被 loglevel 吞掉，等它哪天变成硬错误就查不出来了。
        ff.run(
            [
                "-loglevel", "error", "-y",
                "-sseof", f"-{attempt:.3f}", "-i", str(src),
                "-fps_mode", "passthrough", str(out / "tail_%04d.png"),
            ],
            timeout=300.0,
        )
        files = sorted(out.glob("tail_*.png"))
        if files:
            break
        log.debug("ffmpeg 回退 %.3fs 未抽到帧，加大回退窗口重试", attempt)
    else:
        raise RuntimeError(f"无法从 {src} 抽出尾帧（片长可能短于一帧）")

    files = files[-count:]
    n = len(files)
    return [
        FrameStat(path=p, index_from_end=n - i, sharpness=frame_sharpness(p))
        for i, p in enumerate(files)
    ]


def pick_tail_frame(
    video: str | Path,
    out_dir: str | Path,
    *,
    max_back: int = 6,
    fps: int | float = 24,
    min_ratio: float = 0.7,
    ffmpeg: FFmpeg | None = None,
) -> TailFrame:
    """挑一张能当下一段首帧的尾帧。

    判据是**片内相对**清晰度：绝对阈值在暗场/雾景里必然误判。
    参照系取窗口内的最清晰帧而不是中位数 —— 尾部大半都糊时，
    中位数自己就是糊的，会把毒帧当合格帧放过去。
    从最后一帧往前找，第一张达标的就用，这样丢掉的时长最少。
    """
    ff = ffmpeg or default_ffmpeg()
    real_fps = tail_fps(video, hint=float(fps), ffmpeg=ff)
    stats = extract_tail_frames(video, out_dir, count=max_back, fps=real_fps, ffmpeg=ff)
    ref = max(s.sharpness for s in stats)
    threshold = ref * min_ratio
    for s in reversed(stats):  # stats 按时间序，从最后一帧往前找
        if s.sharpness >= threshold:
            fell = s.index_from_end > 1
            reason = (
                f"最后 {s.index_from_end - 1} 帧清晰度低于窗口最佳的 {min_ratio:.0%}"
                f"（疑似运动模糊/拖影），回退到倒数第 {s.index_from_end} 帧"
                if fell else "最后一帧清晰度达标"
            )
            if fell:
                log.warning("%s 尾帧回退到倒数第 %d 帧", Path(video).name, s.index_from_end)
            return TailFrame(s.path, s.index_from_end, s.sharpness, ref, fell, reason, real_fps)

    best = max(stats, key=lambda s: s.sharpness)
    return TailFrame(
        best.path, best.index_from_end, best.sharpness, ref, True,
        "窗口内没有一帧达标，取最清晰的一帧并建议这一段重拍", real_fps,
    )


# ---------------------------------------------------------------- 漂移预算


@dataclass
class DriftBudget:
    """链式续写的误差额度。

    阈值怎么定（threshold=1.0）：
      1.0 不是画质分，是「一次重锚之间允许累积的相对漂移」这个抽象额度，
      标定方式是让**最稳的策略恰好能连续续 4 段**。
      官方延长基准 0.18，8s 段按 per_second 上浮 16%、无 LoRA 上浮 25%、
      带身份锚打 8 折 → 单段约 0.21，4 段 0.83 未超，第 5 段 1.04 触发重锚。
      「4 段」是本产线的初始标定值，**不是**任何厂商文档或论文给出的结论
      （见模块 docstring 里的查证记录：可灵只说总长 3 分钟上限，AdaState 只说
      drift 会累积，都没给段数）。所以它必须当成待重标定的常量对待。
      换引擎或换风格后重标定的做法：让美术盯着连续续写，记下他判定「不是同一个人了」
      的段数 N，把 threshold 设成 单段成本 × (N-1)。
      尾帧链基准 0.30（多一次解码-重编码、语义上下文全丢），同样口径下只能连 2~3 段。
    """

    threshold: float = 1.0
    per_second: float = 0.02            # 段越长，段内自身的漂移越大
    no_lora_multiplier: float = 1.25    # 没 LoRA，身份只能靠参考图撑
    identity_ref_discount: float = 0.8  # 每段都喂身份锚能压住一部分
    residual_after_reanchor: float = 0.15
    accumulated: float = 0.0
    history: list[tuple[str, float, float]] = field(default_factory=list)

    def estimate(
        self, base_cost: float, duration_s: float, *,
        has_identity_ref: bool, has_lora: bool,
    ) -> float:
        cost = base_cost * (1.0 + self.per_second * duration_s)
        if not has_lora:
            cost *= self.no_lora_multiplier
        if has_identity_ref:
            cost *= self.identity_ref_discount
        return round(cost, 6)

    def would_exceed(self, cost: float) -> bool:
        return self.accumulated + cost > self.threshold + 1e-9

    def spend(self, label: str, cost: float) -> float:
        self.accumulated = round(self.accumulated + cost, 6)
        self.history.append((label, cost, self.accumulated))
        return self.accumulated

    def reanchor(self, label: str) -> float:
        """重锚：用 identity 参考重新起一段。

        残留 0.15 而不是清零 —— 重锚镜自己也不可能和原始定妆一模一样，
        清零会让「重锚 → 续 4 段 → 重锚」这个循环越跑越偏而账面上永远健康。
        """
        self.accumulated = self.residual_after_reanchor
        self.history.append((f"{label}(重锚)", 0.0, self.accumulated))
        return self.accumulated

    def reset(self) -> None:
        """清零。history 也一起清 —— 它是「本次重锚周期花了哪些额度」的账本，
        留着上一个场景/上一段周期的条目会让 explain() 的累计值看起来倒退，
        排查漂移时第一眼就被带偏。"""
        self.accumulated = 0.0
        self.history.clear()

    def explain(self) -> str:
        lines = [
            f"漂移预算：阈值 {self.threshold:.2f}，当前 {self.accumulated:.3f}"
            f"（重锚后残留 {self.residual_after_reanchor:.2f}）"
        ]
        for label, cost, acc in self.history:
            lines.append(f"  {label}: +{cost:.3f} → {acc:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 策略


@dataclass
class LinkContext:
    """续接一段所需的全部上下文。"""

    shot: Shot
    storyboard: Storyboard
    caps: Capabilities
    prev_shot: Shot | None = None
    prev_result: GenResult | None = None
    prev_provider: str | None = None
    workdir: Path = field(default_factory=lambda: Path("./out/chain"))

    @property
    def same_provider(self) -> bool:
        return self.prev_provider is not None and self.prev_provider == self.caps.name


@dataclass
class LinkPlan:
    """策略落地的结果：改了 shot 的哪些连续性字段、喂了什么输入。"""

    strategy: str
    inputs: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


class ChainStrategy(abc.ABC):
    """续接策略统一接口。

    supported_by 只看 Capabilities（和 router 同一条纪律：不认名字只认能力），
    applicable 再叠加「上一段的产物是否够用」这类运行时条件。
    """

    name: str = ""
    drift_base: float = 0.3
    needs_prev_job: bool = False

    @abc.abstractmethod
    def supported_by(self, caps: Capabilities) -> bool: ...

    def applicable(self, ctx: LinkContext) -> bool:
        if not self.supported_by(ctx.caps):
            return False
        if self.needs_prev_job:
            # 官方续写只在同一家、同一条 job 血脉里成立，跨家没有可接的隐状态。
            return bool(ctx.prev_result and ctx.prev_result.job_id) and ctx.same_provider
        return True

    @abc.abstractmethod
    def prepare(self, ctx: LinkContext) -> LinkPlan:
        """把续接输入写进 shot.continuity / shot.refs。会修改 ctx.shot。"""


def _put_first_frame(shot: Shot, uri: str, note: str) -> None:
    """把首帧参考塞进 refpack，必要时腾位子。

    参考位有硬上限（RefPack 的 validator 会拒绝超限），首帧又是这条链的命门，
    所以宁可挤掉一个优先级最低的参考位，也不能静默丢掉首帧。
    """
    images = [i for i in shot.refs.images if i.role != "first_frame"]
    ref = ImageRef(role="first_frame", uri=uri, weight=1.0, note=note)
    if len(images) >= shot.refs.MAX_IMAGES:
        drop_order = ["composition", "prop", "lighting", "style", "environment", "wardrobe"]
        for role in drop_order:
            victim = next((i for i in images if i.role == role), None)
            if victim is not None:
                images.remove(victim)
                log.info("镜头 %s 参考位已满，挤掉 %s 参考 %s 给首帧腾位",
                         shot.id, role, victim.uri)
                break
        else:
            # 剩下的全是 identity/last_frame 这类高价值位，挤谁都疼。
            # 仍然要挤，但必须吼一声：掉的是身份锚，一致性可能因此变差。
            victim = images.pop()
            log.warning("镜头 %s 参考位全是高价值位，被迫挤掉 %s 参考 %s 给首帧腾位",
                        shot.id, victim.role, victim.uri)
    shot.refs.images = [ref] + images


class OfficialExtendChain(ChainStrategy):
    """官方延长接口：同一条隐变量血脉往下长，漂移最小。"""

    name = "official_extend"
    drift_base = 0.18
    needs_prev_job = True

    def supported_by(self, caps: Capabilities) -> bool:
        return caps.supports_extend

    def prepare(self, ctx: LinkContext) -> LinkPlan:
        assert ctx.prev_result is not None
        job = ctx.prev_result.job_id
        ctx.shot.continuity.extend_from_job = job
        ctx.shot.continuity.prev_shot_id = ctx.prev_shot.id if ctx.prev_shot else None
        ctx.shot.continuity.inherit_last_frame = False
        return LinkPlan(
            self.name,
            {"extend_from_job": job, "provider": ctx.caps.name},
            (f"用 {ctx.caps.name} 的官方延长接口接 job {job}",),
        )


class JobContinueChain(ChainStrategy):
    """官方「接上一段」：语义上等价于 extend，但走的是续写而非延长端点。"""

    name = "job_continue"
    drift_base = 0.20
    needs_prev_job = True

    def supported_by(self, caps: Capabilities) -> bool:
        return caps.supports_job_continue

    def prepare(self, ctx: LinkContext) -> LinkPlan:
        assert ctx.prev_result is not None
        job = ctx.prev_result.job_id
        ctx.shot.continuity.extend_from_job = job
        ctx.shot.continuity.prev_shot_id = ctx.prev_shot.id if ctx.prev_shot else None
        notes = [f"以 job {job} 为上文继续生成"]
        if ctx.prev_result.video_uri:
            notes.append(f"上一段产物 {ctx.prev_result.video_uri} 仅作对账，不重新上传")
        return LinkPlan(self.name, {"extend_from_job": job}, tuple(notes))


class LastFrameChain(ChainStrategy):
    """尾帧锁戏：A 的尾帧当 B 的首帧。跨 provider 唯一通用的接法。

    代价是每接一次都要「解码 → 选帧 → 重新编码进生成器」，
    语义上下文全丢，只剩一张图，所以漂移基准最高。
    """

    name = "last_frame"
    drift_base = 0.30

    def supported_by(self, caps: Capabilities) -> bool:
        return caps.supports_first_frame

    def applicable(self, ctx: LinkContext) -> bool:
        if not self.supported_by(ctx.caps):
            return False
        if ctx.prev_result is None:
            return False
        # 直出尾帧不需要本地可读（远端 URI 也能直接喂给下一家），
        # 但走「抽尾帧」这条路必须有本地文件：远端 mp4 抽不了帧。
        # 早前这里只判 uri 非空，结果远端产物一路走到 prepare() 才炸 FileNotFoundError，
        # 而那时降级链已经走完，没有别的策略可退了。
        return bool(
            ctx.prev_result.last_frame_uri
            or local_media_path(ctx.prev_result.video_uri) is not None
        )

    def _tail(self, ctx: LinkContext) -> tuple[str, list[str]]:
        assert ctx.prev_result is not None
        notes: list[str] = []
        direct = ctx.prev_result.last_frame_uri
        if direct:
            local = local_media_path(direct)
            if local is not None:
                sharp = frame_sharpness(local)
                notes.append(f"provider 直出尾帧 {local.name}（清晰度 {sharp:.0f}）")
                return str(local), notes
            # 远端 URI 本地看不到，没法质检；如实记下来而不是假装检过。
            notes.append(f"provider 直出尾帧 {direct}（远端资源，跳过本地质检）")
            return direct, notes

        video = local_media_path(ctx.prev_result.video_uri)
        if video is None:
            raise RuntimeError(
                f"镜头 {ctx.shot.id}: 上一段既没有直出尾帧，产物 "
                f"{ctx.prev_result.video_uri!r} 又不是本地文件，尾帧链无法成立"
            )
        tail = pick_tail_frame(
            video, ctx.workdir / f"{ctx.shot.id}_tail", fps=ctx.shot.fps
        )
        notes.append(f"从 {video.name} 抽尾帧：{tail.reason}（相对清晰度 {tail.ratio:.2f}）")
        if tail.fell_back:
            notes.append(
                f"丢弃末尾 {tail.index_from_end - 1} 帧"
                f"（{tail.dropped_s:.3f}s @ 实测 {tail.fps:g}fps），"
                "拼接时按这个偏移对齐"
            )
        return str(tail.path), notes

    def prepare(self, ctx: LinkContext) -> LinkPlan:
        uri, notes = self._tail(ctx)
        _put_first_frame(ctx.shot, uri, "上一镜尾帧")
        ctx.shot.continuity.inherit_last_frame = True
        ctx.shot.continuity.extend_from_job = None
        ctx.shot.continuity.prev_shot_id = ctx.prev_shot.id if ctx.prev_shot else None
        if ctx.prev_shot is not None:
            ctx.prev_shot.continuity.emit_last_frame = True
            ctx.prev_shot.continuity.next_shot_id = ctx.shot.id
        return LinkPlan(self.name, {"first_frame_uri": uri}, tuple(notes))


class OverlapChain(ChainStrategy):
    """重叠帧混接：B 从 A 的倒数第 N 帧起画，两段共有 N 帧，交给 stitch 混。

    比纯尾帧接多一层保险：混接区能把「接缝那一帧的突变」摊平成 N 帧的渐变，
    代价是每段要多烧 N 帧的时长，而且 stitch 必须知道重叠量。
    """

    name = "overlap"
    drift_base = 0.26
    default_overlap_frames = 6

    def supported_by(self, caps: Capabilities) -> bool:
        return caps.supports_first_frame

    def applicable(self, ctx: LinkContext) -> bool:
        if not self.supported_by(ctx.caps):
            return False
        # 重叠链必须真的去抽帧，所以远端产物一律不适用（判据与尾帧链同源）。
        return bool(ctx.prev_result and local_media_path(ctx.prev_result.video_uri))

    def prepare(self, ctx: LinkContext) -> LinkPlan:
        assert ctx.prev_result is not None
        want = ctx.shot.continuity.overlap_frames or self.default_overlap_frames
        video = local_media_path(ctx.prev_result.video_uri)
        if video is None:
            raise RuntimeError(
                f"镜头 {ctx.shot.id}: 上一段产物 {ctx.prev_result.video_uri!r} "
                "不是本地文件，重叠混接无法抽帧"
            )
        real_fps = tail_fps(video, hint=float(ctx.shot.fps))
        stats = extract_tail_frames(
            video, ctx.workdir / f"{ctx.shot.id}_ovl", count=want + 2, fps=real_fps
        )
        notes: list[str] = []
        # 片尾可能短于想要的重叠量（引擎出的片比工单短、或最后一段被裁过）。
        # 这时必须把重叠量压到真实可用的帧数：stitch 与 timeline 都按
        # overlap_frames / fps 算混接时长，虚报会让它们去混一段根本不存在的画面。
        avail = max(s.index_from_end for s in stats)
        n = want
        if n > avail:
            notes.append(f"上一段尾部只抽到 {avail} 帧，重叠量从 {want} 压到 {avail}")
            log.warning("镜头 %s 重叠量 %d 帧放不下，压到 %d 帧", ctx.shot.id, want, avail)
            n = avail
        # 起画点取倒数第 n 帧；若它比窗口最佳糊太多，就往前再挪一帧避开毒帧。
        ref = max(s.sharpness for s in stats)
        pick = next(s for s in stats if s.index_from_end == n)
        notes.insert(0, f"重叠 {n} 帧，起画点取倒数第 {pick.index_from_end} 帧")
        if pick.sharpness < ref * 0.7:
            better = max(
                (s for s in stats if s.index_from_end >= n), key=lambda s: s.sharpness
            )
            notes.append(
                f"倒数第 {n} 帧清晰度 {pick.sharpness:.0f} 远低于窗口最佳 {ref:.0f}，"
                f"改用倒数第 {better.index_from_end} 帧"
            )
            pick = better
        # 重叠量始终等于起画点的倒数序号：这两个数一旦脱钩，
        # stitch 混接的时长就和实际共有的画面对不上，接缝处会闪一下。
        n = pick.index_from_end
        _put_first_frame(ctx.shot, str(pick.path), f"上一镜倒数第 {n} 帧（重叠混接）")
        ctx.shot.continuity.overlap_frames = n
        ctx.shot.continuity.inherit_last_frame = True
        ctx.shot.continuity.prev_shot_id = ctx.prev_shot.id if ctx.prev_shot else None
        ctx.shot.transition_in = Transition.OVERLAP_BLEND
        if ctx.prev_shot is not None:
            ctx.prev_shot.continuity.next_shot_id = ctx.shot.id
            ctx.prev_shot.continuity.emit_last_frame = True
        return LinkPlan(
            self.name,
            {"first_frame_uri": str(pick.path), "overlap_frames": str(n)},
            tuple(notes),
        )


ANCHOR = "anchor"       # 起始镜：用 identity 参考重新起一段
REANCHOR = "reanchor"   # 漂移超预算时的强制重锚
BLOCKED = "blocked"     # 门禁/路由拒单

DEFAULT_STRATEGIES: tuple[ChainStrategy, ...] = (
    OfficialExtendChain(),
    JobContinueChain(),
    OverlapChain(),
    LastFrameChain(),
)


# ---------------------------------------------------------------- 规划


@dataclass
class ChainStep:
    index: int
    shot_id: str
    strategy: str
    provider: str
    duration_s: float
    est_cost_usd: float
    drift_before: float
    drift_cost: float
    drift_after: float
    reanchor: bool
    reason: str
    notes: tuple[str, ...] = ()

    @property
    def is_continuation(self) -> bool:
        return self.strategy not in {ANCHOR, REANCHOR, BLOCKED}

    def explain(self) -> str:
        head = (
            f"[{self.index:02d}] {self.shot_id} {self.strategy} @{self.provider} "
            f"{self.duration_s:g}s ${self.est_cost_usd:.4f} "
            f"漂移 {self.drift_before:.3f}+{self.drift_cost:.3f}={self.drift_after:.3f}"
        )
        if self.reanchor:
            head += "  << 强制重锚"
        lines = [head, f"      理由：{self.reason}"]
        lines += [f"      · {n}" for n in self.notes]
        return "\n".join(lines)


class ChainPlanner:
    """给一个场景排续接方案。

    只做规划：算清楚每一镜用哪种策略、漂移花到哪、什么时候必须重锚。
    真正把 job_id / 尾帧塞进 Shot 的动作在 prepare()，那要等上一段真的出片。
    """

    def __init__(
        self,
        *,
        strategies: tuple[ChainStrategy, ...] = DEFAULT_STRATEGIES,
        budget: DriftBudget | None = None,
        workdir: str | Path = "./out/chain",
    ) -> None:
        self.strategies = strategies
        self.budget = budget or DriftBudget()
        self.workdir = Path(workdir)

    def strategy(self, name: str) -> ChainStrategy:
        for s in self.strategies:
            if s.name == name:
                return s
        raise KeyError(f"未知续接策略 {name}；已注册 {[s.name for s in self.strategies]}")

    # 规划期还没有上一段的产物，所以这里只按 caps 与声明选策略；
    # 运行期再用 applicable() 复核一次，缺 job_id 的自然退回尾帧链。
    def select(
        self, shot: Shot, caps: Capabilities, *, same_provider: bool
    ) -> ChainStrategy | None:
        declared_overlap = (
            shot.continuity.overlap_frames > 0
            or shot.transition_in is Transition.OVERLAP_BLEND
        )
        order: list[ChainStrategy] = []
        by_name = {s.name: s for s in self.strategies}
        if declared_overlap and "overlap" in by_name:
            order.append(by_name["overlap"])   # 导演明确要混接，优先满足
        if same_provider:
            order += [s for s in self.strategies if s.needs_prev_job]
        # 重叠链要多烧 N 帧、还要求 stitch 配合混接，没明确声明就不自作主张，
        # 默认的跨家兜底是尾帧链。
        order += [
            s for s in self.strategies
            if s not in order and not s.needs_prev_job and s.name != "overlap"
        ]
        for s in order:
            if s.supported_by(caps):
                return s
        return None

    @staticmethod
    def wants_continuation(prev: Shot | None, shot: Shot) -> bool:
        """只有**显式声明**的连续性才续接。

        不按「同场景相邻」猜：一个硬切的正反打接上去反而会把上一镜的构图带进来。
        """
        if prev is None:
            return False
        c = shot.continuity
        return bool(
            c.inherit_last_frame
            or c.extend_from_job
            or c.overlap_frames > 0
            or c.prev_shot_id == prev.id
            or prev.continuity.next_shot_id == shot.id
            or shot.transition_in is Transition.OVERLAP_BLEND
        )

    def _has_identity_ref(self, shot: Shot, storyboard: Storyboard) -> bool:
        if any(i.role == "identity" for i in shot.refs.images):
            return True
        return any(
            (c := storyboard.character(sid)) is not None and bool(c.portraits)
            for sid in shot.subject_ids
        )

    def _has_lora(self, shot: Shot, storyboard: Storyboard) -> bool:
        return any(
            (c := storyboard.character(sid)) is not None and c.lora is not None
            for sid in shot.subject_ids
        )

    def plan(self, scene: Scene, storyboard: Storyboard, router) -> list[ChainStep]:
        """排完整条链。router 用来决定每一镜走哪家（只调 plan，不发请求）。"""
        steps: list[ChainStep] = []
        self.budget.reset()
        prev_shot: Shot | None = None
        prev_provider: str | None = None

        for i, shot in enumerate(scene.shots):
            rp = router.plan(shot, storyboard)
            if not rp.routable:
                steps.append(
                    ChainStep(
                        index=i, shot_id=shot.id, strategy=BLOCKED, provider="-",
                        duration_s=shot.duration_s, est_cost_usd=0.0,
                        drift_before=self.budget.accumulated, drift_cost=0.0,
                        drift_after=self.budget.accumulated, reanchor=False,
                        reason="路由拒单：" + "；".join(
                            rp.gate.blockers or [w for _, w in rp.rejected]),
                    )
                )
                prev_shot, prev_provider = None, None   # 断链，下一镜必须重起
                continue

            caps = router.registry.get(rp.primary).caps
            before = self.budget.accumulated
            want = self.wants_continuation(prev_shot, shot)
            strat = (
                self.select(shot, caps, same_provider=prev_provider == rp.primary)
                if want else None
            )

            if not want:
                self.budget.reset()
                reason = (
                    "场景首镜，用 identity 参考起片"
                    if prev_shot is None
                    else "未声明与上一镜连续（硬切/换机位），独立起片，漂移清零"
                )
                strategy, cost, reanchor = ANCHOR, 0.0, False
            elif strat is None:
                self.budget.reset()
                reason = (
                    f"{caps.name} 既不支持官方续写也不支持首帧锁戏 → 只能重锚起片，"
                    "跨镜连续性交给拼接与调色对齐"
                )
                strategy, cost, reanchor = REANCHOR, 0.0, True
            else:
                cost = self.budget.estimate(
                    strat.drift_base, shot.duration_s,
                    has_identity_ref=self._has_identity_ref(shot, storyboard),
                    has_lora=self._has_lora(shot, storyboard),
                )
                if self.budget.would_exceed(cost):
                    self.budget.reanchor(shot.id)
                    reason = (
                        f"续接 {strat.name} 预计再花 {cost:.3f} 漂移额度，"
                        f"累计 {before:.3f}+{cost:.3f} 超过阈值 {self.budget.threshold:.2f}"
                        " → 强制重锚：改用 identity 参考重新起一段，"
                        "宁可接缝硬一点，也不能让人物漂成另一个人"
                    )
                    strategy, cost, reanchor = REANCHOR, 0.0, True
                else:
                    self.budget.spend(f"{shot.id}/{strat.name}", cost)
                    reason = (
                        f"{caps.name} 支持 {strat.name}"
                        + ("（与上一镜同一家，可接 job 血脉）" if strat.needs_prev_job else "")
                        + f"，漂移额度够用（剩 {self.budget.threshold - self.budget.accumulated:.3f}）"
                    )
                    strategy, reanchor = strat.name, False

            steps.append(
                ChainStep(
                    index=i, shot_id=shot.id, strategy=strategy, provider=rp.primary,
                    duration_s=rp.request.duration_s if rp.request else shot.duration_s,
                    est_cost_usd=rp.estimated_cost_usd, drift_before=before,
                    drift_cost=cost, drift_after=self.budget.accumulated,
                    reanchor=reanchor, reason=reason,
                    notes=tuple(rp.degradations),
                )
            )
            prev_shot, prev_provider = shot, rp.primary
        return steps

    def _fallback_order(self, ctx: LinkContext) -> list[ChainStrategy]:
        """降级顺序。与 select() 同一条规矩：没声明重叠就不塞重叠链。"""
        declared_overlap = (
            ctx.shot.continuity.overlap_frames > 0
            or ctx.shot.transition_in is Transition.OVERLAP_BLEND
        )
        return [s for s in self.strategies if declared_overlap or s.name != "overlap"]

    def prepare(self, step: ChainStep, ctx: LinkContext) -> LinkPlan:
        """运行期落地：把上一段的 job/尾帧写进这一镜。

        规划期挑的策略可能因为上一段的实际产物而失效（比如换了家、没拿到 job_id），
        这里按 applicable() 复核，不成立就顺着策略表往下退，而不是硬报错。
        """
        if not step.is_continuation:
            return LinkPlan(step.strategy, {}, (step.reason,))
        chosen = self.strategy(step.strategy)
        if chosen.applicable(ctx):
            return chosen.prepare(ctx)
        for s in self._fallback_order(ctx):
            if s is not chosen and s.applicable(ctx):
                log.warning("镜头 %s 策略 %s 运行期不成立，退到 %s",
                            ctx.shot.id, chosen.name, s.name)
                plan = s.prepare(ctx)
                return LinkPlan(
                    s.name, plan.inputs,
                    (f"规划的 {chosen.name} 运行期不成立，已降级",) + plan.notes,
                )
        return LinkPlan(REANCHOR, {}, ("无任何可用续接策略，退回重锚起片",))


def explain_chain(steps: list[ChainStep], budget: DriftBudget | None = None) -> str:
    lines = ["=== 续写链 ==="]
    lines += [s.explain() for s in steps]
    total = sum(s.est_cost_usd for s in steps)
    lines.append(
        f"合计 {len(steps)} 段 / {sum(s.duration_s for s in steps):g}s / ${total:.4f}，"
        f"重锚 {sum(1 for s in steps if s.reanchor)} 次"
    )
    if budget is not None:
        lines.append(budget.explain())
    return "\n".join(lines)


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    import io
    import tempfile

    logging.basicConfig(level=logging.ERROR, stream=io.StringIO())
    from .providers.base import (
        GenRequest, GenResult, JobStatus, ProviderRegistry, VideoProvider,
    )
    from .router import Router, RoutingPolicy
    from .schema import (
        Appearance, CharacterBible, Continuity, DeliverySpec, Scene as SC,
        Storyboard as SB,
    )

    class _Fake(VideoProvider):
        """自测用的桩 provider：只回放事先备好的 mp4。

        不用 providers.mock.MockProvider —— 它会真的渲染画面（秒级/段），
        而本模块要断言的是「选了哪条链、漂移花到哪」，与画面内容无关；
        桩只实现 VideoProvider 的两个抽象方法，没有另造一套 provider 契约。
        """

        def __init__(self, caps: Capabilities, video: str = "") -> None:
            super().__init__(caps)
            self.video = video
            self._n = 0

        def submit(self, req: GenRequest) -> str:
            self._n += 1
            return f"{self.name}-job-{self._n}"

        def poll(self, job_id: str) -> GenResult:
            return GenResult(job_id=job_id, status=JobStatus.SUCCEEDED, provider=self.name,
                             video_uri=self.video, duration_s=8.0, cost_usd=0.1)

    tmp = Path(tempfile.mkdtemp(prefix="longfilm_chain_"))
    ff = default_ffmpeg()

    def mkvideo(
        path: Path, blur_from: float | None = None, *, duration: float = 2.0, rate: int = 24
    ) -> Path:
        vf = ["-vf", f"gblur=sigma=14:enable='gte(t,{blur_from})'"] if blur_from else []
        ff.run(
            ["-loglevel", "error", "-y", "-f", "lavfi",
             "-i", f"testsrc2=size=320x180:rate={rate}:duration={duration}", *vf,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
            timeout=300.0,
        )
        return path

    sharp = mkvideo(tmp / "sharp.mp4")
    blurry = mkvideo(tmp / "blurry.mp4", blur_from=1.80)

    # 1) 尾帧抽取与质检：干净片取最后一帧
    t1 = pick_tail_frame(sharp, tmp / "t1", fps=24)
    assert t1.index_from_end == 1 and not t1.fell_back, t1
    assert t1.path.exists() and t1.sharpness > 0

    # 2) 尾部糊掉的片：必须回退到倒数第 N 帧，而不是拿毒帧去喂下一段
    t2 = pick_tail_frame(blurry, tmp / "t2", fps=24)
    assert t2.fell_back and t2.index_from_end > 1, t2
    assert t2.sharpness > 10 * frame_sharpness(
        sorted((tmp / "t2").glob("tail_*.png"))[-1]
    ), "回退帧必须明显比末帧清晰"
    print(f"尾帧质检：干净片取倒数第 {t1.index_from_end} 帧；"
          f"糊片回退到倒数第 {t2.index_from_end} 帧（{t2.reason}）")

    # 3) 漂移预算：最稳策略恰好能连 4 段，第 5 段触发重锚
    b = DriftBudget()
    per = b.estimate(OfficialExtendChain.drift_base, 8.0,
                     has_identity_ref=True, has_lora=False)
    n = 0
    while not b.would_exceed(per):
        b.spend(f"seg{n}", per)
        n += 1
    assert n == 4, f"官方延长链应能连 4 段，实得 {n}（单段 {per:.3f}）"
    lf = b.estimate(LastFrameChain.drift_base, 8.0, has_identity_ref=True, has_lora=False)
    assert lf > per, "尾帧链的漂移必须比官方延长贵"
    b2 = DriftBudget()
    m = 0
    while not b2.would_exceed(lf):
        b2.spend(f"lf{m}", lf)
        m += 1
    assert m == 2, f"尾帧链在同一阈值下只能连 {m} 段"
    b.reanchor("sh05")
    assert abs(b.accumulated - 0.15) < 1e-9
    print(f"漂移预算：官方延长单段 {per:.3f} → 连 {n} 段；尾帧链单段 {lf:.3f} → 连 {m} 段")

    # 4) 场景规划：同一家 + supports_extend → 走官方延长，第 5 段强制重锚
    caps_ext = Capabilities(
        name="alpha", kind="official", min_duration_s=4.0, max_duration_s=10.0,
        resolutions=((1280, 720),), max_ref_images=4,
        supports_first_frame=True, supports_extend=True, supports_job_continue=True,
        cost_per_second_usd=0.2, quality_tier=5,
    )
    caps_lf = Capabilities(
        name="beta", kind="open", min_duration_s=4.0, max_duration_s=10.0,
        resolutions=((1280, 720),), max_ref_images=9, supports_first_frame=True,
        supports_last_frame=True, supports_lora=True, moderated=False,
        cost_per_second_usd=0.02, quality_tier=3,
    )
    caps_none = Capabilities(
        name="gamma", kind="open", min_duration_s=4.0, max_duration_s=10.0,
        resolutions=((1280, 720),), moderated=False, cost_per_second_usd=0.01,
        quality_tier=2,
    )

    hero = CharacterBible(
        id="ch_lin", name="林默", age_statement="虚构角色，设定年龄 29 岁",
        appearance=Appearance(face="瘦削", hair="黑色短发"),
        # 有定妆照 → 每段都能喂身份锚，漂移预算按折扣算
        portraits=[ImageRef(role="identity", uri="/id.png", subject_id="ch_lin")],
    )

    def mkshot(i: int, **kw) -> Shot:
        base = dict(
            id=f"sh{i:03d}", scene_id="sc01", index=i, duration_s=8.0,
            subject_ids=["ch_lin"], action=f"第 {i} 段动作",
            continuity=Continuity(inherit_last_frame=i > 0),
        )
        base.update(kw)
        return Shot(**base)

    scene = SC(id="sc01", shots=[mkshot(i) for i in range(7)])
    sb = SB(project="demo", characters=[hero], scenes=[scene],
            delivery=DeliverySpec(resolution=(1280, 720)))

    reg = ProviderRegistry()
    reg.register(_Fake(caps_ext, video=str(sharp)))
    router = Router(reg, RoutingPolicy(episode_budget_usd=100.0), sleeper=lambda _: None)
    planner = ChainPlanner(workdir=tmp / "work")
    steps = planner.plan(scene, sb, router)

    assert steps[0].strategy == ANCHOR, steps[0]
    assert [s.strategy for s in steps[1:5]] == ["official_extend"] * 4, \
        [s.strategy for s in steps]
    assert steps[5].strategy == REANCHOR and steps[5].reanchor, steps[5].explain()
    assert steps[6].strategy == "official_extend"
    assert steps[5].drift_after == 0.15
    print()
    print(explain_chain(steps, planner.budget))

    # 5) 跨 provider：只剩尾帧链，且重锚来得更早
    reg2 = ProviderRegistry()
    reg2.register(_Fake(caps_lf, video=str(sharp)))
    router2 = Router(reg2, RoutingPolicy(episode_budget_usd=100.0), sleeper=lambda _: None)
    scene2 = SC(id="sc01", shots=[mkshot(i) for i in range(6)])
    sb2 = SB(project="demo", characters=[hero], scenes=[scene2])
    planner2 = ChainPlanner(workdir=tmp / "work2")
    steps2 = planner2.plan(scene2, sb2, router2)
    assert steps2[1].strategy == "last_frame", steps2[1]
    reanchors = [s.index for s in steps2 if s.reanchor]
    assert reanchors and reanchors[0] < 5, f"尾帧链应更早重锚：{reanchors}"

    # 6) provider 毫无续接能力 → 每一镜都重锚，并说明原因
    reg3 = ProviderRegistry()
    reg3.register(_Fake(caps_none, video=str(sharp)))
    router3 = Router(reg3, sleeper=lambda _: None)
    scene3 = SC(id="sc01", shots=[mkshot(i) for i in range(3)])
    sb3 = SB(project="demo", characters=[hero], scenes=[scene3])
    steps3 = ChainPlanner(workdir=tmp / "work3").plan(scene3, sb3, router3)
    assert [s.strategy for s in steps3] == [ANCHOR, REANCHOR, REANCHOR], steps3
    assert "既不支持" in steps3[1].reason

    # 7) 硬切（未声明连续）不续接，且漂移清零
    scene4 = SC(id="sc01", shots=[
        mkshot(0), mkshot(1), mkshot(2, continuity=Continuity())])
    sb4 = SB(project="demo", characters=[hero], scenes=[scene4])
    steps4 = ChainPlanner(workdir=tmp / "work4").plan(scene4, sb4, router)
    assert steps4[2].strategy == ANCHOR and steps4[2].drift_after == 0.0

    # 8) 策略落地：官方延长写 job_id
    shot_b = mkshot(1)
    ctx = LinkContext(
        shot=shot_b, storyboard=sb, caps=caps_ext, prev_shot=scene.shots[0],
        prev_result=GenResult(job_id="alpha-job-1", status=JobStatus.SUCCEEDED,
                              provider="alpha", video_uri=str(sharp)),
        prev_provider="alpha", workdir=tmp / "work",
    )
    lp = OfficialExtendChain().prepare(ctx)
    assert shot_b.continuity.extend_from_job == "alpha-job-1" and lp.strategy == "official_extend"

    # 9) 策略落地：尾帧链真的抽帧、写进 refpack，糊尾帧会回退
    shot_c = mkshot(2)
    ctx2 = LinkContext(
        shot=shot_c, storyboard=sb, caps=caps_lf, prev_shot=scene.shots[1],
        prev_result=GenResult(job_id="beta-1", status=JobStatus.SUCCEEDED,
                              provider="beta", video_uri=str(blurry)),
        prev_provider="beta", workdir=tmp / "work",
    )
    lp2 = LastFrameChain().prepare(ctx2)
    ff_ref = [i for i in shot_c.refs.images if i.role == "first_frame"]
    assert len(ff_ref) == 1 and Path(ff_ref[0].uri).exists()
    assert shot_c.continuity.inherit_last_frame
    assert any("回退" in n for n in lp2.notes), lp2.notes
    print()
    print("尾帧链落地：" + "；".join(lp2.notes))

    # 10) 重叠链：写 overlap_frames 与 OVERLAP_BLEND 转场
    shot_d = mkshot(3, continuity=Continuity(inherit_last_frame=True, overlap_frames=6))
    ctx3 = LinkContext(
        shot=shot_d, storyboard=sb, caps=caps_lf, prev_shot=scene.shots[2],
        prev_result=GenResult(job_id="beta-2", status=JobStatus.SUCCEEDED,
                              provider="beta", video_uri=str(sharp)),
        prev_provider="beta", workdir=tmp / "work",
    )
    lp3 = OverlapChain().prepare(ctx3)
    assert shot_d.continuity.overlap_frames == 6
    assert shot_d.transition_in is Transition.OVERLAP_BLEND
    assert lp3.inputs["overlap_frames"] == "6"

    # 11) 参考位满了也必须给首帧腾位子
    full = mkshot(4)
    full.refs.images = [ImageRef(role="prop", uri=f"/p{i}.png") for i in range(9)]
    _put_first_frame(full, "/ff.png", "上一镜尾帧")
    assert len(full.refs.images) == 9 and full.refs.images[0].role == "first_frame"

    # 12) 运行期降级：规划走官方延长，但实际没拿到 job_id → 退到尾帧链
    shot_e = mkshot(5)
    step = ChainStep(index=5, shot_id=shot_e.id, strategy="official_extend",
                     provider="alpha", duration_s=8.0, est_cost_usd=1.6,
                     drift_before=0.0, drift_cost=0.2, drift_after=0.2,
                     reanchor=False, reason="规划期选择")
    ctx4 = LinkContext(
        shot=shot_e, storyboard=sb, caps=caps_ext, prev_shot=scene.shots[4],
        prev_result=GenResult(job_id="", status=JobStatus.SUCCEEDED, provider="beta",
                              video_uri=str(sharp)),
        prev_provider="beta", workdir=tmp / "work",
    )
    lp4 = ChainPlanner(workdir=tmp / "work").prepare(step, ctx4)
    assert lp4.strategy == "last_frame" and "降级" in lp4.notes[0], lp4

    # 13) 远端产物：抽不了帧的两条链必须在 applicable() 就说不，而不是在 prepare() 里炸
    remote = GenResult(job_id="", status=JobStatus.SUCCEEDED, provider="beta",
                       video_uri="https://cdn.example.com/seg.mp4")
    ctx5 = LinkContext(shot=mkshot(6), storyboard=sb, caps=caps_lf,
                       prev_shot=scene.shots[5], prev_result=remote,
                       prev_provider="beta", workdir=tmp / "work")
    assert not LastFrameChain().applicable(ctx5), "远端 mp4 抽不了尾帧，不该判为可用"
    assert not OverlapChain().applicable(ctx5)
    assert local_media_path("https://cdn.example.com/seg.mp4") is None
    assert local_media_path(str(sharp)) == sharp
    assert local_media_path(f"file://{sharp}") == sharp
    assert local_media_path(None) is None and local_media_path("") is None
    # 但 provider 直出的远端尾帧仍然可用：那是一张图，直接喂下一家即可
    ctx6 = LinkContext(shot=mkshot(6), storyboard=sb, caps=caps_lf,
                       prev_shot=scene.shots[5], workdir=tmp / "work",
                       prev_result=GenResult(job_id="", status=JobStatus.SUCCEEDED,
                                             provider="beta", video_uri="https://x/seg.mp4",
                                             last_frame_uri="https://x/last.png"),
                       prev_provider="beta")
    assert LastFrameChain().applicable(ctx6)
    lp5 = LastFrameChain().prepare(ctx6)
    assert lp5.inputs["first_frame_uri"] == "https://x/last.png"
    assert any("跳过本地质检" in n for n in lp5.notes), lp5.notes
    # 全链降级：远端产物 + 无 job 血脉 → 退到重锚，不抛异常
    step5 = ChainStep(index=6, shot_id="sh006", strategy="last_frame", provider="beta",
                      duration_s=8.0, est_cost_usd=0.1, drift_before=0.0, drift_cost=0.3,
                      drift_after=0.3, reanchor=False, reason="规划期选择")
    assert ChainPlanner(workdir=tmp / "work").prepare(step5, ctx5).strategy == REANCHOR

    # 14) 重叠链：片尾帧数不够时必须把 overlap_frames 压到真实可用值。
    #     stitch/timeline 都按 overlap_frames / fps 算混接时长，虚报会去混不存在的画面。
    tiny = mkvideo(tmp / "tiny.mp4", duration=0.25)   # 24fps ≈ 6 帧
    shot_f = mkshot(7, continuity=Continuity(inherit_last_frame=True, overlap_frames=12))
    ctx7 = LinkContext(
        shot=shot_f, storyboard=sb, caps=caps_lf, prev_shot=scene.shots[6],
        prev_result=GenResult(job_id="beta-9", status=JobStatus.SUCCEEDED,
                              provider="beta", video_uri=str(tiny)),
        prev_provider="beta", workdir=tmp / "work",
    )
    lp6 = OverlapChain().prepare(ctx7)
    n_real = shot_f.continuity.overlap_frames
    assert 0 < n_real < 12, f"重叠量应被压到实际可用帧数，实得 {n_real}"
    assert lp6.inputs["overlap_frames"] == str(n_real), (lp6.inputs, n_real)
    assert Path(lp6.inputs["first_frame_uri"]).exists()
    assert any("压到" in n for n in lp6.notes), lp6.notes
    print(f"重叠链短片保护：请求 12 帧 → 实际 {n_real} 帧（{'；'.join(lp6.notes)}）")

    # 15) 尾帧秒数按**实测**帧率换算，而不是按工单里的目标帧率
    slow = mkvideo(tmp / "slow.mp4", blur_from=1.80, duration=2.0, rate=12)
    tf = pick_tail_frame(slow, tmp / "t3", fps=24)   # 故意传错的 hint
    assert abs(tf.fps - 12.0) < 0.01, f"应按实测 12fps，实得 {tf.fps}"
    assert abs(tf.dropped_s - (tf.index_from_end - 1) / 12.0) < 1e-6
    print(f"实测帧率：hint=24 实际 {tf.fps:g}fps，丢弃 {tf.index_from_end - 1} 帧 = {tf.dropped_s:.3f}s")

    # 16) 预算 reset 必须连 history 一起清，否则跨场景复用 planner 时账本会串
    shared = ChainPlanner(workdir=tmp / "work5")
    shared.plan(scene4, sb4, router)
    first_len = len(shared.budget.history)
    shared.plan(scene4, sb4, router)
    assert len(shared.budget.history) == first_len, (
        f"同一个 planner 连排两个场景后 history 应等长，实得 {first_len} -> "
        f"{len(shared.budget.history)}"
    )
    assert all(a <= shared.budget.threshold for _, _, a in shared.budget.history)

    # 17) 空场景：不崩、不产生任何步骤、说明文本仍然成立
    empty = ChainPlanner(workdir=tmp / "work6").plan(SC(id="sc_empty"), sb, router)
    assert empty == []
    assert "合计 0 段" in explain_chain(empty)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nchain 自测通过：尾帧质检/实测帧率/漂移预算/四种策略/重锚/运行期降级/"
          "远端产物/短片重叠 全部符合预期")


if __name__ == "__main__":
    _selftest()
