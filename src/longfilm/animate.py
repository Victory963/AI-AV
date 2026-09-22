"""动作迁移 —— 用表演视频驱动虚构角色。

为什么需要这一工位：文生视频引擎给不了「精确的、可复用的、跨镜一致的表演」。
同一个角色在第 3 镜和第 47 镜要做同一个招牌动作，靠提示词描述是抽奖；
靠一段自录的表演视频做 pose 驱动，才是可复现的。

产线位置::

    prompt_os → [animate] → router/comfy_local → chain → qc

本模块**只产工单**，不跑推理：本机无 GPU，推理在 ComfyUI 侧。
输出的 ``AnimateJob.comfy_values`` 直接喂给
``providers.comfy_local.WorkflowTemplate.render()``。

内容与素材边界（硬约束，不可绕过）：

1. 驱动素材必须是自录或有明确授权的，来源类型写进 :class:`MotionSource.source_kind`，
   未标注来源的素材直接拒单 —— 不做「先跑起来再补授权」。
2. **不做真人换脸**。身份永远来自角色圣经的定妆图（reference_image），
   驱动视频只提供身体姿态。面部驱动通道（face_video）默认关闭，
   开启需要同时满足「自录/三维合成来源」+「有 consent_ref」两个条件，
   且此时被驱动的脸仍是虚构角色的脸，不是把真人的脸贴到别人身上。
3. 只做虚构成年角色 —— 由 :class:`~longfilm.schema.CharacterBible` 的
   ``is_fictional`` / ``age_statement`` 保证，这里只复核不重造。

外部规格查证（2026-09-19）：

* Wan2.2-Animate-14B —— https://github.com/Wan-Video/Wan2.2 README：
  animation / replacement 两种模式；replacement 需要 pose/face/background/mask 四路视频 +
  ``--replace_flag --use_relighting_lora``；预处理脚本
  ``wan/modules/animate/preprocess/preprocess_data.py``（animation 用
  ``--retarget_flag --use_flux``）；``segment_frame_length`` 77 帧，
  ``--resolution_area`` 默认 1280x720。
* ComfyUI ``WanAnimateToVideo`` —— https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy_extras/nodes_wan.py
  required: positive/negative/vae/width(832,step16)/height(480,step16)/length(77,min1,step4)/
  batch_size/continue_motion_max_frames(5)/video_frame_offset(0)；
  optional: reference_image/clip_vision_output/face_video/pose_video/background_video/
  character_mask/continue_motion；返回
  (positive, negative, latent, trim_latent, trim_image, video_frame_offset)。
  官方教程（docs.comfy.org）说每个 Video Extend 模块约 4.8125s = 77 帧 → 推得 **16 fps**。
* UniAnimate-DiT —— https://github.com/ali-vilab/UniAnimate-DiT README：
  底模 Wan2.1-14B-I2V，pose 走 DWPose 存 ``.pkl``；单段上限 81 帧（约 5s）；
  480p(832x480) ~23GB VRAM，720p(1280x720) ~36GB。
* MimicMotion —— https://github.com/Tencent/MimicMotion README：
  底模 SVD img2vid-xt-1-1，576x1024，v1.1 单段 72 帧，DWPose 需
  ``yolox_l.onnx`` + ``dw-ll_ucoco_384.onnx``，72 帧模型约 16GB VRAM。
* Animate-X —— https://github.com/antgroup/animate-x README：
  768x512，默认 32 帧 @ 8fps（``seq_len = max_frames + 1``），
  支持拟人化非人角色，对 pose 对齐宽容（不要求严格骨架对齐）。
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from .schema import CharacterBible, Shot, VideoRef

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- 素材来源与许可


class SourceKind(str, Enum):
    """驱动素材的来源类型。**没有默认值** —— 必须由采集方显式标注。

    这个字段不是元数据装饰，它是产线的准入闸门：``retarget()`` 只放行
    白名单里的来源，其余一律拒单。把来源塞进枚举而不是自由文本，
    是为了让「这段素材能不能用」成为一次 set 查表而不是一次人工判断。
    """

    SELF_RECORDED = "self_recorded"      # 自录：团队成员出镜，有内部授权书
    LICENSED = "licensed"                # 商用授权动捕/表演库，有合同编号
    SYNTHETIC_3D = "synthetic_3d"        # 三维动画/动捕重定向生成，无真人出镜
    PUBLIC_DOMAIN = "public_domain"      # 明确进入公有领域
    UNVERIFIED = "unverified"            # 来源不明 —— 只能进素材池，不能进产线


#: 允许进入产线的来源。UNVERIFIED 不在其中，这是有意的。
ALLOWED_SOURCE_KINDS: frozenset[SourceKind] = frozenset(
    {SourceKind.SELF_RECORDED, SourceKind.LICENSED, SourceKind.SYNTHETIC_3D,
     SourceKind.PUBLIC_DOMAIN}
)

#: 允许开启面部驱动通道的来源。授权库素材不在其中 ——
#: 买到「使用这段视频」的授权，不等于买到「用出镜者的面部表演去驱动另一张脸」的授权。
FACE_DRIVING_SOURCE_KINDS: frozenset[SourceKind] = frozenset(
    {SourceKind.SELF_RECORDED, SourceKind.SYNTHETIC_3D}
)


class MotionSourceError(ValueError):
    """驱动素材不满足产线的来源/许可/格式要求。

    单独一个异常类型而不是复用 ValueError，是为了让 CLI 能把
    「素材合规问题」和「参数写错」分开报给不同的人。
    """


# ---------------------------------------------------------------- 姿态数据


class PoseFormat(str, Enum):
    DWPOSE = "dwpose_coco_wholebody"     # 133 点：17 身体 + 6 脚 + 68 脸 + 42 手
    OPENPOSE = "openpose_body25"         # 25 点
    SMPL_2D = "smpl_2d_proj"             # 三维动捕投影下来的 2D 骨架


#: 每种格式里**躯干与四肢**关键点的下标区间（左闭右开）。
#: 切点算法只看这段：脸部和手指的关键点是高频抖动源，
#: 把它们算进运动能量会让「身体静止但在说话」的帧看起来很"忙"，
#: 从而错过真正适合下刀的低速段。
_BODY_SLICE: dict[PoseFormat, tuple[int, int]] = {
    PoseFormat.DWPOSE: (0, 17),
    PoseFormat.OPENPOSE: (0, 15),
    PoseFormat.SMPL_2D: (0, 24),
}

_EXPECTED_KEYPOINTS: dict[PoseFormat, int] = {
    PoseFormat.DWPOSE: 133,
    PoseFormat.OPENPOSE: 25,
    PoseFormat.SMPL_2D: 24,
}


@dataclass(frozen=True)
class PoseFrame:
    """单帧骨架。坐标归一化到 [0,1]（除以画幅宽高），与分辨率解耦。

    归一化是刻意的：同一段动作要能同时喂给 832x480 的 Wan 和 576x1024 的
    MimicMotion，存像素坐标就得在每个方法里重算一遍。
    """

    index: int
    points: tuple[tuple[float, float, float], ...]   # (x, y, confidence)

    def body(self, fmt: PoseFormat) -> tuple[tuple[float, float, float], ...]:
        lo, hi = _BODY_SLICE[fmt]
        return self.points[lo:hi]

    def centroid(self, fmt: PoseFormat, min_conf: float = 0.3) -> tuple[float, float] | None:
        pts = [(x, y) for x, y, c in self.body(fmt) if c >= min_conf]
        if not pts:
            return None
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


@dataclass
class MotionSource:
    """一段可用于驱动的动作素材：视频 + 已提取的 pose 序列 + 许可信息。

    ``pose`` 可以为空（还没抽），但进 :func:`retarget` 之前必须抽好 ——
    抽 pose 要 GPU，产线把它放在更前面的工位批量做，不在编排时临时触发。
    """

    id: str
    video_uri: str
    fps: float
    source_kind: SourceKind
    pose_format: PoseFormat = PoseFormat.DWPOSE
    pose: list[PoseFrame] = field(default_factory=list)

    # 许可与溯源。license_ref 是合同号/授权书编号，consent_ref 是出镜者知情同意书编号。
    license_ref: str = ""
    consent_ref: str = ""
    performer_note: str = ""        # 出镜者说明，例如"团队成员 A，仅授权身体动作驱动"
    contains_real_face: bool = True  # 画面里有没有真人面孔；三维合成素材才可为 False

    duration_s: float = 0.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise MotionSourceError(f"{self.id}: fps 必须为正，收到 {self.fps}")
        if self.pose and not self.duration_s:
            self.duration_s = len(self.pose) / self.fps

    @property
    def frame_count(self) -> int:
        return len(self.pose)

    def validate(self) -> None:
        """进产线前的硬校验。不通过就抛，不返回布尔 —— 调用方没有"忽略"这个选项。"""
        if self.source_kind not in ALLOWED_SOURCE_KINDS:
            raise MotionSourceError(
                f"{self.id}: 来源类型 {self.source_kind.value} 不在产线白名单；"
                "请先补齐授权并重新标注 source_kind"
            )
        if self.source_kind is SourceKind.LICENSED and not self.license_ref:
            raise MotionSourceError(f"{self.id}: 声明为授权素材但 license_ref 为空")
        if self.source_kind is SourceKind.SELF_RECORDED and not self.consent_ref:
            raise MotionSourceError(f"{self.id}: 自录素材必须填 consent_ref（出镜者知情同意书编号）")
        if not self.pose:
            raise MotionSourceError(
                f"{self.id}: pose 序列为空；请先用 PoseExtractor 抽帧，"
                "或用 PrerenderedPoseSource 载入已有的序列"
            )
        want = _EXPECTED_KEYPOINTS[self.pose_format]
        bad = next((f for f in self.pose if len(f.points) != want), None)
        if bad is not None:
            raise MotionSourceError(
                f"{self.id}: 第 {bad.index} 帧有 {len(bad.points)} 个关键点，"
                f"{self.pose_format.value} 应为 {want} 个"
            )

    def to_video_ref(self) -> VideoRef:
        return VideoRef(
            role="motion",
            uri=self.video_uri,
            note=f"{self.source_kind.value}; license={self.license_ref or '-'}; "
                 f"consent={self.consent_ref or '-'}",
        )


# ---------------------------------------------------------------- 姿态提取


class PoseExtractor:
    """姿态提取器抽象。

    三个实现里只有 :class:`PrerenderedPoseSource` 能在无 GPU 的机器上跑通；
    另外两个在 :meth:`extract` **里面**才 import 重依赖，
    所以本模块在任何机器上都能 import 成功 —— 编排逻辑不该被推理环境绑架。
    """

    format: PoseFormat = PoseFormat.DWPOSE

    def extract(self, video_uri: str, *, fps: float, max_frames: int | None = None) -> list[PoseFrame]:
        raise NotImplementedError

    def to_source(
        self,
        video_uri: str,
        *,
        fps: float,
        source_kind: SourceKind,
        source_id: str = "",
        **meta: Any,
    ) -> MotionSource:
        frames = self.extract(video_uri, fps=fps)
        return MotionSource(
            id=source_id or f"motion_{uuid.uuid4().hex[:8]}",
            video_uri=video_uri,
            fps=fps,
            source_kind=source_kind,
            pose_format=self.format,
            pose=frames,
            **meta,
        )


def _missing_runtime(what: str, how: str) -> MotionSourceError:
    return MotionSourceError(
        f"{what} 不可用：{how}\n"
        "本机（无 GPU）请改用 PrerenderedPoseSource 载入离线抽好的 pose 序列，"
        "或把抽帧任务提交到带 GPU 的 ComfyUI 节点。"
    )


class DWPoseExtractor(PoseExtractor):
    """DWPose（COCO-WholeBody 133 点）。Wan-Animate / UniAnimate-DiT / MimicMotion 都吃这个。

    依赖两个 ONNX 权重：``yolox_l.onnx``（人体检测）+ ``dw-ll_ucoco_384.onnx``（关键点），
    见 MimicMotion README（2026-09-19 查证）。
    """

    format = PoseFormat.DWPOSE

    def __init__(self, det_onnx: str = "yolox_l.onnx", pose_onnx: str = "dw-ll_ucoco_384.onnx"):
        self.det_onnx = det_onnx
        self.pose_onnx = pose_onnx

    def extract(self, video_uri: str, *, fps: float, max_frames: int | None = None) -> list[PoseFrame]:
        try:
            import onnxruntime  # noqa: F401, PLC0415  延迟导入：推理依赖，编排机上没有
        except ImportError as exc:
            raise _missing_runtime("DWPose", f"缺少 onnxruntime（{exc}）") from exc
        for w in (self.det_onnx, self.pose_onnx):
            if not Path(w).exists():
                raise _missing_runtime("DWPose", f"找不到权重文件 {w}")
        # TODO(2026-09-19): 真正的推理循环留给带 GPU 的工位实现。
        # 这里不写假的 ONNX 推理代码 —— 写了只会在真机上被推翻一次。
        raise _missing_runtime("DWPose", "本模块只做编排，未内置推理实现")


class OpenPoseExtractor(PoseExtractor):
    """OpenPose BODY_25。老 workflow（AnimateDiff 系）还在用，保留一条兼容路径。"""

    format = PoseFormat.OPENPOSE

    def __init__(self, model_dir: str = "models/openpose"):
        self.model_dir = model_dir

    def extract(self, video_uri: str, *, fps: float, max_frames: int | None = None) -> list[PoseFrame]:
        try:
            import cv2  # noqa: F401, PLC0415  延迟导入
        except ImportError as exc:
            raise _missing_runtime("OpenPose", f"缺少 opencv-python（{exc}）") from exc
        if not Path(self.model_dir).exists():
            raise _missing_runtime("OpenPose", f"找不到模型目录 {self.model_dir}")
        raise _missing_runtime("OpenPose", "本模块只做编排，未内置推理实现")


class PrerenderedPoseSource(PoseExtractor):
    """直接吃已有的 pose 序列 —— 无 GPU 机器上唯一能跑通的路径。

    接受两种输入：内存里的 list[PoseFrame]，或磁盘上的 JSON
    （``{"format": "...", "fps": 16, "frames": [[[x,y,c], ...], ...]}``）。
    JSON 而不是 ``.pkl``：UniAnimate-DiT 官方存 pkl，但 pkl 是可执行的反序列化格式，
    产线要跨机器传素材，不接受这种攻击面。转换在抽帧工位做一次。
    """

    def __init__(self, frames: Sequence[PoseFrame] | None = None, *, fmt: PoseFormat = PoseFormat.DWPOSE):
        self.format = fmt
        self._frames = list(frames or ())

    @classmethod
    def from_json(cls, path: str | Path) -> PrerenderedPoseSource:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        fmt = PoseFormat(data.get("format", PoseFormat.DWPOSE.value))
        frames = [
            PoseFrame(index=i, points=tuple((float(p[0]), float(p[1]), float(p[2])) for p in row))
            for i, row in enumerate(data["frames"])
        ]
        return cls(frames, fmt=fmt)

    def extract(self, video_uri: str, *, fps: float, max_frames: int | None = None) -> list[PoseFrame]:
        if not self._frames:
            raise MotionSourceError("PrerenderedPoseSource 没有装载任何帧")
        return list(self._frames[:max_frames] if max_frames else self._frames)


# ---------------------------------------------------------------- 方法与规格


class AnimateMethod(str, Enum):
    WAN_ANIMATE = "wan_animate"                  # Wan2.2-Animate animation 模式
    WAN_ANIMATE_REPLACE = "wan_animate_replace"  # replacement 模式（换进实拍背景）
    UNIANIMATE_DIT = "unianimate_dit"
    MIMICMOTION = "mimicmotion"
    ANIMATE_X = "animate_x"


@dataclass(frozen=True)
class MethodSpec:
    """一个动作迁移方法的硬规格。切片长度和分辨率都从这里取，不在调用处写死。

    ``max_frames_per_clip`` 是**模型的单段上限**，不是产线的镜长上限 ——
    二者取小，见 :func:`atomic_seconds`。
    """

    method: AnimateMethod
    base_model: str
    native_fps: float
    max_frames_per_clip: int
    default_resolution: tuple[int, int]
    length_multiple: int = 1          # 帧数必须满足 length % m == 1（Wan 系 latent 对齐）
    needs_face_video: bool = False
    needs_background_and_mask: bool = False
    supports_relight_lora: bool = False
    vram_gb: float = 0.0
    comfy_node: str = ""
    source_url: str = ""
    verified_on: str = "2026-09-19"
    notes: str = ""

    @property
    def max_clip_s(self) -> float:
        return self.max_frames_per_clip / self.native_fps

    def snap_length(self, frames: int) -> int:
        """把帧数吸附到模型允许的取值。

        Wan 的 latent 时间轴按 4 压 1，帧数必须是 ``4n+1``；
        ComfyUI 的 WanAnimateToVideo 也把 length 的 step 声明为 4（min=1）。
        吸附方向取「不超过上限的最近值」，宁可短一帧也不要提交一个会被拒的长度。
        """
        frames = max(1, min(frames, self.max_frames_per_clip))
        m = self.length_multiple
        if m <= 1:
            return frames
        n = (frames - 1) // m
        snapped = m * n + 1
        return max(1, snapped)


#: 各方法规格。数值全部来自 2026-09-19 查证的官方 README / 源码，见模块 docstring。
METHODS: dict[AnimateMethod, MethodSpec] = {
    AnimateMethod.WAN_ANIMATE: MethodSpec(
        method=AnimateMethod.WAN_ANIMATE,
        base_model="Wan2.2-Animate-14B",
        native_fps=16.0,          # 77 帧 ≈ 4.8125s（docs.comfy.org）→ 16fps
        max_frames_per_clip=77,
        default_resolution=(832, 480),
        length_multiple=4,
        supports_relight_lora=True,
        vram_gb=24.0,
        comfy_node="WanAnimateToVideo",
        source_url="https://github.com/Wan-Video/Wan2.2",
        notes="animation 模式：参考图 + 驱动视频；预处理 --retarget_flag --use_flux",
    ),
    AnimateMethod.WAN_ANIMATE_REPLACE: MethodSpec(
        method=AnimateMethod.WAN_ANIMATE_REPLACE,
        base_model="Wan2.2-Animate-14B",
        native_fps=16.0,
        max_frames_per_clip=77,
        default_resolution=(832, 480),
        length_multiple=4,
        needs_face_video=True,
        needs_background_and_mask=True,
        supports_relight_lora=True,
        vram_gb=28.0,
        comfy_node="WanAnimateToVideo",
        source_url="https://github.com/Wan-Video/Wan2.2",
        notes="replacement 模式：pose/face/background/mask 四路 + --use_relighting_lora",
    ),
    AnimateMethod.UNIANIMATE_DIT: MethodSpec(
        method=AnimateMethod.UNIANIMATE_DIT,
        base_model="Wan2.1-14B-I2V + UniAnimate LoRA(rank 64-128)",
        native_fps=16.0,          # 81 帧 ≈ 5s（README 原话 "~5 seconds"）→ 16fps
        max_frames_per_clip=81,
        default_resolution=(832, 480),
        length_multiple=4,
        vram_gb=23.0,
        source_url="https://github.com/ali-vilab/UniAnimate-DiT",
        notes="pose 以 DWPose .pkl 输入；720p 需 ~36GB 显存",
    ),
    AnimateMethod.MIMICMOTION: MethodSpec(
        method=AnimateMethod.MIMICMOTION,
        base_model="stable-video-diffusion-img2vid-xt-1-1",
        native_fps=15.0,          # TODO(2026-09-19): README 未写 fps，按 SVD 默认 15 取，待真机核对
        max_frames_per_clip=72,
        default_resolution=(576, 1024),
        vram_gb=16.0,
        source_url="https://github.com/Tencent/MimicMotion",
        notes="竖屏原生；长视频靠 progressive latent fusion 拼，单段 72 帧",
    ),
    AnimateMethod.ANIMATE_X: MethodSpec(
        method=AnimateMethod.ANIMATE_X,
        base_model="animate-x.pth (LDM)",
        native_fps=8.0,
        max_frames_per_clip=32,
        default_resolution=(768, 512),
        vram_gb=16.0,
        source_url="https://github.com/antgroup/animate-x",
        notes="对 pose 对齐宽容，支持拟人化非人角色；seq_len = max_frames + 1",
    ),
}


def atomic_seconds(method: AnimateMethod, *, pipeline_max_s: float = 8.0) -> float:
    """产线原子镜长 = min(模型单段上限, 产线约定上限)。

    两个上限都会变（模型升级 / 产线调节奏），所以这里取小而不是二选一写死。
    """
    return min(METHODS[method].max_clip_s, pipeline_max_s)


# ---------------------------------------------------------------- 运动能量与切点


def motion_energy(
    frames: Sequence[PoseFrame],
    fmt: PoseFormat,
    *,
    min_conf: float = 0.3,
) -> list[float]:
    """逐帧运动能量：相邻帧之间躯干四肢关键点的平均位移（归一化坐标）。

    有两处刻意的取舍：

    * **减去整体平移**。角色整体走位（centroid 移动）是"镜头/走位"层面的运动，
      在这里切一刀并不难接；真正接不上的是肢体处在半途的姿态。
      所以能量只统计相对质心的肢体位移。
    * **只算两帧都置信的点**。DWPose 在遮挡帧会给出低置信度的乱跳坐标，
      不过滤会凭空造出一个能量尖峰，把切点推到错误的位置。
    """
    if len(frames) < 2:
        return [0.0] * len(frames)
    out = [0.0]
    for prev, cur in zip(frames, frames[1:]):
        pc = prev.centroid(fmt, min_conf)
        cc = cur.centroid(fmt, min_conf)
        dx, dy = (cc[0] - pc[0], cc[1] - pc[1]) if (pc and cc) else (0.0, 0.0)
        acc, n = 0.0, 0
        for (x0, y0, c0), (x1, y1, c1) in zip(prev.body(fmt), cur.body(fmt)):
            if c0 < min_conf or c1 < min_conf:
                continue
            acc += math.hypot((x1 - x0) - dx, (y1 - y0) - dy)
            n += 1
        out.append(acc / n if n else 0.0)
    return out


def smooth(values: Sequence[float], window: int) -> list[float]:
    """居中滑动平均。窗口取奇数，边界用可用区间平均（不补零 —— 补零会把两端压成假的低速段）。"""
    if window <= 1 or len(values) < 2:
        return list(values)
    if window % 2 == 0:
        window += 1
    half = window // 2
    n = len(values)
    return [
        sum(values[max(0, i - half): min(n, i + half + 1)])
        / len(values[max(0, i - half): min(n, i + half + 1)])
        for i in range(n)
    ]


@dataclass
class MotionSlice:
    """一个原子镜长度的动作片段。半开区间 ``[start_frame, end_frame)``。"""

    index: int
    start_frame: int
    end_frame: int
    fps: float
    cut_energy: float = 0.0        # 切入点处的平滑运动能量
    median_energy: float = 0.0     # 整段素材的能量中位数，作为"快/慢"的尺子
    merged_tail: bool = False      # 末段太短被并进来了

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps

    @property
    def cut_quality(self) -> str:
        """切点质量。``good`` 表示落在明显低速段，接起来基本无跳。"""
        if self.median_energy <= 0:
            return "unknown"
        r = self.cut_energy / self.median_energy
        if r <= 0.5:
            return "good"
        if r <= 1.0:
            return "fair"
        return "risky"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "frames": self.frames,
            "duration_s": round(self.duration_s, 4),
            "cut_quality": self.cut_quality,
            "cut_energy_ratio": round(self.cut_energy / self.median_energy, 4)
            if self.median_energy > 0 else None,
        }


def slice_motion(
    motion: MotionSource,
    max_s: float,
    *,
    min_ratio: float = 0.45,
    search_ratio: float = 0.35,
    smooth_window: int | None = None,
) -> list[MotionSlice]:
    """把长动作切成原子镜长度，切点落在动作低速段。

    朴素做法是按 ``max_s`` 等分，但那样有很大概率切在「手举到一半」这种中途姿态上：
    下一段要从这个姿态续起，而生成模型只拿到一张参考图 + 新的 pose 起点，
    接缝处必然出现一次姿态跳变。所以这里做的是**受约束的低速点搜索**：

    1. 算逐帧运动能量并平滑（窗口默认 1/4 秒，滤掉抽帧噪声但保留真实的起止）。
    2. 从当前切点出发，在 ``[min_ratio*max, max]`` 这个窗口里找能量最小的帧下刀。
       下界保证不会切出一堆碎片，上界保证不超过模型单段上限。
    3. 能量并列时取靠后的帧 —— 同样质量下段数越少越省 GPU。
    4. 尾巴短于下界时并进上一段（并完仍不能超上限，否则宁可留一个短段并标记）。

    返回的每段都带 ``cut_quality``，QC 工位据此决定要不要在接缝加重叠混接。
    """
    motion.validate()
    if max_s <= 0:
        raise ValueError(f"max_s 必须为正，收到 {max_s}")

    fps = motion.fps
    n = motion.frame_count
    max_f = max(1, int(round(max_s * fps)))
    if n <= max_f:
        return [MotionSlice(0, 0, n, fps, median_energy=0.0)]

    min_f = max(1, int(round(max_f * min_ratio)))
    win = smooth_window if smooth_window is not None else max(3, int(round(fps / 4)))
    energy = smooth(motion_energy(motion.pose, motion.pose_format), win)
    ordered = sorted(energy)
    median = ordered[len(ordered) // 2]

    # 搜索窗口：在 max_f 之前回看这么多帧。给足回看空间，切点才有得挑。
    look_back = max(1, int(round(max_f * search_ratio)))

    cuts: list[int] = [0]
    pos = 0
    while n - pos > max_f:
        hi = pos + max_f
        lo = max(pos + min_f, hi - look_back)
        # 并列取靠后：key 里对下标取负，min 在能量相等时选下标大的。
        best = min(range(lo, hi + 1), key=lambda i: (energy[i], -i))
        cuts.append(best)
        pos = best
    cuts.append(n)

    # 尾段过短就并进上一段 —— 但并完不能超模型上限。
    merged = False
    if len(cuts) >= 3 and cuts[-1] - cuts[-2] < min_f:
        if cuts[-1] - cuts[-3] <= max_f:
            cuts.pop(-2)
            merged = True
        else:
            log.warning(
                "%s: 尾段仅 %d 帧且无法并入上一段（会超 %d 帧上限），保留短段",
                motion.id, cuts[-1] - cuts[-2], max_f,
            )

    slices: list[MotionSlice] = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
        slices.append(
            MotionSlice(
                index=i,
                start_frame=a,
                end_frame=b,
                fps=fps,
                cut_energy=energy[a],
                median_energy=median,
                merged_tail=merged and i == len(cuts) - 2,
            )
        )
    return slices


# ---------------------------------------------------------------- 工单


@dataclass
class AnimateClip:
    """一段的 ComfyUI 注入值。语义名对齐 ``WanAnimateToVideo`` 的输入端口名，
    由 ``WorkflowTemplate.bind`` 映射到具体节点路径 —— 换 workflow 不改这里。"""

    slice: MotionSlice
    values: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.values["length"])


@dataclass
class AnimateJob:
    """一次动作迁移工单：把一段动作迁到一个虚构角色身上。

    工单是**自描述**的：溯源字段（来源类型、授权编号、是否启用面部驱动）
    跟着工单走到产物 manifest，事后审计不需要回头翻代码或聊天记录。
    """

    id: str
    character_id: str
    method: AnimateMethod
    motion_id: str
    reference_image_uri: str
    resolution: tuple[int, int]
    fps: float
    clips: list[AnimateClip] = field(default_factory=list)

    prompt: str = ""
    negative_prompt: str = ""
    seed: int | None = None
    shot_id: str = ""

    # 溯源与合规
    source_kind: SourceKind = SourceKind.UNVERIFIED
    license_ref: str = ""
    consent_ref: str = ""
    face_driving: bool = False
    identity_policy: str = (
        "身份锚定于角色圣经定妆图；驱动视频仅提供身体姿态；不做真人换脸"
    )
    warnings: list[str] = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(c.length for c in self.clips)

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps

    def comfy_values(self, clip_index: int = 0) -> dict[str, Any]:
        """取某一段的注入值，直接喂 ``WorkflowTemplate.render()``。"""
        return dict(self.clips[clip_index].values)

    def manifest(self) -> dict[str, Any]:
        spec = METHODS[self.method]
        return {
            "job_id": self.id,
            "shot_id": self.shot_id,
            "character_id": self.character_id,
            "method": self.method.value,
            "base_model": spec.base_model,
            "comfy_node": spec.comfy_node,
            "motion_id": self.motion_id,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "clips": [c.slice.to_dict() for c in self.clips],
            "total_frames": self.total_frames,
            "duration_s": round(self.duration_s, 3),
            "provenance": {
                "source_kind": self.source_kind.value,
                "license_ref": self.license_ref or None,
                "consent_ref": self.consent_ref or None,
                "face_driving": self.face_driving,
                "identity_policy": self.identity_policy,
            },
            "warnings": list(self.warnings),
        }


def _identity_ref(character: CharacterBible) -> str:
    """取身份锚图。优先 identity role 的定妆图，其次转身图第一张。"""
    for img in character.portraits:
        if img.role == "identity":
            return img.uri
    if character.portraits:
        return character.portraits[0].uri
    if character.turnaround:
        return character.turnaround[0].uri
    raise MotionSourceError(
        f"角色 {character.id} 没有任何定妆图；动作迁移的身份必须来自定妆图，"
        "不能从驱动视频里取"
    )


def retarget(
    motion: MotionSource,
    character: CharacterBible,
    *,
    method: AnimateMethod = AnimateMethod.WAN_ANIMATE,
    shot: Shot | None = None,
    reference_image_uri: str | None = None,
    resolution: tuple[int, int] | None = None,
    pipeline_max_s: float = 8.0,
    allow_face_driving: bool = False,
    background_video_uri: str | None = None,
    character_mask_uri: str | None = None,
    prompt: str = "",
    seed: int | None = None,
) -> AnimateJob:
    """把一段动作迁移到虚构角色，产出可直接注入 ComfyUI 的工单。

    面部驱动通道的闸门在这里，而不是在调用方：调用方传 ``allow_face_driving=True``
    只是**请求**，是否放行由素材来源决定。这样"谁能开面部驱动"是一处可审计的规则，
    不是散落在各处的 if。
    """
    motion.validate()
    spec = METHODS[method]

    if not character.age_statement.strip():
        raise MotionSourceError(f"角色 {character.id} 未声明 age_statement，产线拒单")

    ref_uri = reference_image_uri or _identity_ref(character)
    res = resolution or spec.default_resolution
    warnings: list[str] = []

    # 面部驱动闸门。两个条件都满足才放行，且 replacement 模式必须开。
    face_driving = False
    if allow_face_driving or spec.needs_face_video:
        if motion.source_kind not in FACE_DRIVING_SOURCE_KINDS:
            raise MotionSourceError(
                f"{motion.id}: 来源 {motion.source_kind.value} 不允许启用面部驱动通道。"
                "本产线不做真人换脸：面部驱动仅限自录（含知情同意）或三维合成素材。"
            )
        if motion.contains_real_face and not motion.consent_ref:
            raise MotionSourceError(
                f"{motion.id}: 画面含真人面孔且无 consent_ref，拒绝启用面部驱动通道"
            )
        face_driving = True

    if spec.needs_background_and_mask and not (background_video_uri and character_mask_uri):
        raise MotionSourceError(
            f"{method.value} 是 replacement 模式，必须同时提供 background_video_uri "
            "和 character_mask_uri（见 Wan2.2 README 的 --replace_flag 四路输入）"
        )

    # 分辨率对齐：ComfyUI 的 WanAnimateToVideo 把 width/height 的 step 声明为 16。
    if method in (AnimateMethod.WAN_ANIMATE, AnimateMethod.WAN_ANIMATE_REPLACE):
        snapped = (res[0] // 16 * 16, res[1] // 16 * 16)
        if snapped != res:
            warnings.append(f"分辨率 {res} 已吸附到 16 的倍数 {snapped}")
            res = snapped

    max_s = atomic_seconds(method, pipeline_max_s=pipeline_max_s)
    if shot is not None and shot.duration_s < max_s:
        max_s = shot.duration_s
    slices = slice_motion(motion, max_s)

    # 驱动素材 fps 与模型原生 fps 不一致时只做提示，不在这里重采样：
    # 重采样属于抽帧工位的职责，在编排里偷偷改帧率会让 pose 文件与工单对不上。
    if abs(motion.fps - spec.native_fps) > 0.51:
        warnings.append(
            f"素材 {motion.fps}fps 与 {spec.base_model} 原生 {spec.native_fps}fps 不一致，"
            "请在抽帧工位重采样 pose 序列，否则动作会变快/变慢"
        )

    clips: list[AnimateClip] = []
    offset = 0
    for i, sl in enumerate(slices):
        length = spec.snap_length(sl.frames)
        if length != sl.frames:
            warnings.append(f"第 {i} 段 {sl.frames} 帧吸附到模型允许的 {length} 帧")
        values: dict[str, Any] = {
            "reference_image": ref_uri,
            "pose_video": f"{motion.video_uri}#pose[{sl.start_frame}:{sl.start_frame + length}]",
            "width": res[0],
            "height": res[1],
            "length": length,
            "batch_size": 1,
            "video_frame_offset": offset,
            "continue_motion_max_frames": 5,
            "positive": prompt or _default_prompt(character, shot),
            "negative": (shot.negative_prompt if shot else "") or character.negative_prompt,
            "seed": seed if seed is not None else (shot.seed if shot else None),
        }
        if i > 0:
            # 段间靠 continue_motion 续接：上一段尾部若干帧作为运动上下文，
            # 比只给一张尾帧图更能保住动作的速度与朝向。
            values["continue_motion"] = f"{{clip_{i - 1}_output}}"
        if face_driving:
            values["face_video"] = f"{motion.video_uri}#face[{sl.start_frame}:{sl.start_frame + length}]"
        if spec.needs_background_and_mask:
            values["background_video"] = background_video_uri
            values["character_mask"] = character_mask_uri
        if spec.supports_relight_lora and spec.needs_background_and_mask:
            values["relight_lora"] = "relighting_lora"
        clips.append(AnimateClip(slice=sl, values=values))
        offset += length

    job = AnimateJob(
        id=f"anim_{uuid.uuid4().hex[:10]}",
        character_id=character.id,
        method=method,
        motion_id=motion.id,
        reference_image_uri=ref_uri,
        resolution=res,
        fps=spec.native_fps,
        clips=clips,
        prompt=prompt or _default_prompt(character, shot),
        negative_prompt=(shot.negative_prompt if shot else "") or character.negative_prompt,
        seed=seed if seed is not None else (shot.seed if shot else None),
        shot_id=shot.id if shot else "",
        source_kind=motion.source_kind,
        license_ref=motion.license_ref,
        consent_ref=motion.consent_ref,
        face_driving=face_driving,
        warnings=warnings,
    )
    risky = [c.slice.index for c in clips if c.slice.cut_quality == "risky"]
    if risky:
        job.warnings.append(
            f"第 {risky} 段的切入点落在高速运动段，建议在拼接时开重叠混接"
            "（Continuity.overlap_frames > 0）"
        )
    log.info(
        "retarget: %s -> %s via %s, %d 段 / %.2fs, face_driving=%s",
        motion.id, character.id, method.value, len(clips), job.duration_s, face_driving,
    )
    return job


def _default_prompt(character: CharacterBible, shot: Shot | None) -> str:
    """默认提示词：身份锚 + 镜头动作。动作迁移里提示词是辅助，主导权在 pose。"""
    bits = [character.identity_prompt()]
    if shot:
        bits += [shot.action, shot.environment, shot.lighting, shot.style]
    return ", ".join(b.strip() for b in bits if b and b.strip())


# ---------------------------------------------------------------- 自测


_PHASE = 20


def _fake_motion(n_frames: int = 200, fps: float = 16.0) -> MotionSource:
    """造一段有明确「快—慢—快—慢」节奏的假 pose 序列。

    慢段做成真正的静止（能量≈0），这样切点算法若正确，
    切点必然落在静止段里 —— 这是可断言的不变量，不是眼看着像。
    交替周期取 20 帧：小于搜索窗口，保证每个候选窗口里一定存在静止帧，
    否则测的就是「窗口里没静止段时算法怎么退化」，那是另一个问题。
    """
    frames: list[PoseFrame] = []
    for i in range(n_frames):
        phase = (i // _PHASE) % 2        # 每 _PHASE 帧切换一次快/慢
        amp = 0.05 if phase == 0 else 0.0
        pts = []
        for k in range(133):
            if k < 17:
                # 躯干四肢：慢段完全静止，快段正弦摆动
                x = 0.5 + amp * math.sin(i * 0.5 + k)
                y = 0.5 + amp * math.cos(i * 0.5 + k)
                pts.append((x, y, 0.9))
            else:
                # 脸和手一直在抖：验证能量函数确实把它们排除在外
                pts.append((0.5 + 0.02 * math.sin(i * 3.1 + k), 0.5, 0.9))
        frames.append(PoseFrame(index=i, points=tuple(pts)))
    return MotionSource(
        id="mo_test",
        video_uri="file:///tmp/perf.mp4",
        fps=fps,
        source_kind=SourceKind.SELF_RECORDED,
        pose=frames,
        consent_ref="CONSENT-2026-001",
        performer_note="团队成员 A，仅授权身体动作驱动",
    )


def _fake_character() -> CharacterBible:
    from .schema import Appearance, ImageRef

    return CharacterBible(
        id="lin",
        name="林舟",
        age_statement="虚构角色，设定年龄 29 岁，成年",
        appearance=Appearance(face="棱角分明", hair="短发", wardrobe="深色风衣"),
        portraits=[ImageRef(role="identity", uri="file:///assets/lin_id.png", subject_id="lin")],
        negative_prompt="blurry, extra limbs",
    )


def _selftest() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # --- 1. 来源闸门：未标注来源必须拒单 ---
    bad = _fake_motion(60)
    bad.source_kind = SourceKind.UNVERIFIED
    try:
        bad.validate()
        raise AssertionError("UNVERIFIED 素材竟然通过了校验")
    except MotionSourceError as e:
        print(f"[1] 来源闸门生效: {str(e).splitlines()[0]}")

    # --- 2. 运动能量：脸手抖动不该污染能量 ---
    motion = _fake_motion(200)
    e = smooth(motion_energy(motion.pose, motion.pose_format), 5)
    fast = [e[i] for i in range(5, 15)]       # 第 0.._PHASE-1 帧是快段
    slow = [e[i] for i in range(25, 35)]      # 第 _PHASE..2*_PHASE-1 帧是静止段
    assert max(slow) < min(fast), f"静止段能量({max(slow):.5f})未低于运动段({min(fast):.5f})"
    print(f"[2] 运动能量: 快段均值 {sum(fast)/len(fast):.5f} / 静止段均值 {sum(slow)/len(slow):.5f}")

    # --- 3. 切点必须落在静止段 ---
    slices = slice_motion(motion, 4.0)          # 4s @16fps = 64 帧
    assert sum(s.frames for s in slices) == motion.frame_count, "切片帧数总和与素材不符"
    assert all(s.frames <= 64 for s in slices), "有切片超过 max_s"
    inner_cuts = [s.start_frame for s in slices[1:]]
    for c in inner_cuts:
        assert (c // _PHASE) % 2 == 1, f"切点 {c} 落在运动段（应落在静止段）"
    print(f"[3] 切片 {[(s.start_frame, s.end_frame, s.cut_quality) for s in slices]}")
    assert all(s.cut_quality in ("good", "fair") for s in slices[1:]), "存在 risky 切点"

    # --- 4. 对比朴素等分：证明低速搜索确实更优 ---
    naive_cuts = list(range(64, motion.frame_count, 64))
    naive_e = sum(e[c] for c in naive_cuts) / len(naive_cuts)
    smart_e = sum(e[c] for c in inner_cuts) / len(inner_cuts)
    assert smart_e < naive_e, f"低速搜索({smart_e:.5f})未优于等分({naive_e:.5f})"
    print(f"[4] 切点能量: 等分 {naive_e:.5f} -> 低速搜索 {smart_e:.5f}")

    # --- 5. retarget：工单与 ComfyUI 注入值 ---
    char = _fake_character()
    shot = Shot(id="s01_01", scene_id="s01", index=0, duration_s=8.0, subject_ids=["lin"],
                action="转身推开门", environment="雨夜巷口", negative_prompt="lowres")
    job = retarget(motion, char, method=AnimateMethod.WAN_ANIMATE, shot=shot)
    assert job.face_driving is False, "默认不该开面部驱动"
    assert job.reference_image_uri == "file:///assets/lin_id.png"
    v0 = job.comfy_values(0)
    assert v0["length"] % 4 == 1, f"length {v0['length']} 不满足 Wan 的 4n+1"
    assert v0["width"] % 16 == 0 and v0["height"] % 16 == 0
    assert "continue_motion" not in v0 and "continue_motion" in job.comfy_values(1)
    assert job.comfy_values(1)["video_frame_offset"] == v0["length"]
    print(f"[5] 工单 {job.id}: {len(job.clips)} 段 / {job.duration_s:.2f}s, "
          f"length={[c.length for c in job.clips]}")

    # --- 6. 面部驱动闸门：授权库素材必须被拒 ---
    licensed = _fake_motion(80)
    licensed.source_kind = SourceKind.LICENSED
    licensed.license_ref = "CONTRACT-77"
    licensed.consent_ref = ""
    try:
        retarget(licensed, char, allow_face_driving=True)
        raise AssertionError("授权库素材竟然获准开启面部驱动")
    except MotionSourceError as e:
        print(f"[6] 换脸闸门生效: {str(e).splitlines()[0]}")

    # --- 7. replacement 模式缺四路输入必须拒 ---
    try:
        retarget(motion, char, method=AnimateMethod.WAN_ANIMATE_REPLACE)
        raise AssertionError("replacement 模式缺 background/mask 竟然通过")
    except MotionSourceError as e:
        print(f"[7] replacement 校验生效: {str(e).splitlines()[0]}")
    job_r = retarget(motion, char, method=AnimateMethod.WAN_ANIMATE_REPLACE,
                     background_video_uri="file:///bg.mp4", character_mask_uri="file:///mask.mp4")
    assert job_r.face_driving is True and "face_video" in job_r.comfy_values(0)
    assert job_r.comfy_values(0)["relight_lora"] == "relighting_lora"
    print(f"[7] replacement 工单 OK: {sorted(job_r.comfy_values(0))}")

    # --- 8. 各方法的帧数吸附都不超上限 ---
    for m, spec in METHODS.items():
        assert spec.snap_length(10_000) <= spec.max_frames_per_clip, m
        assert spec.snap_length(1) >= 1, m
        if spec.length_multiple > 1:
            assert spec.snap_length(10_000) % spec.length_multiple == 1, m
    print("[8] 方法规格: " + ", ".join(
        f"{m.value}={s.max_frames_per_clip}f@{s.native_fps}fps" for m, s in METHODS.items()))

    # --- 9. 无 GPU 时抽帧器报错清晰而不是 import 崩 ---
    try:
        DWPoseExtractor().extract("x.mp4", fps=16)
        raise AssertionError("DWPose 在本机竟然跑通了")
    except MotionSourceError as e:
        assert "PrerenderedPoseSource" in str(e), "报错未给出本机可行的替代路径"
        print(f"[9] DWPose 报错含替代方案: {str(e).splitlines()[0]}")

    # --- 10. manifest 可序列化且带溯源 ---
    mf = job.manifest()
    json.dumps(mf, ensure_ascii=False)
    assert mf["provenance"]["consent_ref"] == "CONSENT-2026-001"
    assert mf["provenance"]["face_driving"] is False
    assert mf["total_frames"] == job.total_frames
    print(f"[10] manifest 溯源: {mf['provenance']}")

    print("animate.py 自测通过")


if __name__ == "__main__":
    _selftest()
