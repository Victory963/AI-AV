"""3D / 动捕底稿（previz）—— 用灰模序列给难镜头定死几何。

为什么要这一工位：扩散模型不懂空间。一个「绕着两个人走半圈的 12 秒长镜」，
纯提示词生成十次会得到十种不同的空间关系；而一旦有了灰模底稿，
depth/normal/pose 三路 ControlNet 把相机轨迹和主体位置钉死，
剩下交给模型的只有材质与光影 —— 这才是可重复的。

但底稿不是免费的：布景 + 调机位 + 渲染的人机成本，只有在
「重抽成本 × 预期重抽次数 > 底稿成本」时才划算。所以本模块的第一件事
不是渲染，而是 :class:`PrevizPlan` 的**判定**：哪些镜头值得上底稿。

产线位置::

    schema/storyboard → [previz3d] → refpack(previz/control 位) → router → qc

本机没装 Blender。所以模块拆成两半：

* **纯 Python 可验证的部分** —— :func:`solve_camera` 把 ShotSize/CameraMove/lens_mm
  换算成相机距离、视场角和关键帧曲线。这部分自测能完整断言。
* **需要 Blender 的部分** —— :func:`render_previz` 只负责拼命令行并给出清晰提示，
  检测不到 blender 就返回一个 ``available=False`` 的结果，不抛不崩。

外部规格查证（2026-09-19）：

* Blender CLI —— https://github.com/blender/blender source/creator/creator_args.cc：
  ``--background|-b``、``--render-anim|-a``、``--render-frame|-f``、
  ``--render-output|-o``、``--frame-start|-s``、``--frame-end|-e``、
  ``--engine|-E``、``--render-format|-F``、``--use-extension|-x``、
  ``--threads|-t``、``--python``、``--python-expr``、``--factory-startup``、``-noaudio``；
  源码原话："Arguments are executed in the order they are given" ——
  所以 ``-b`` 必须在 ``--python`` 之前，脚本自己的参数放在 ``--`` 之后。
* Blender 相机数据块 —— https://docs.blender.org/api/4.2/bpy.types.Camera.html：
  ``lens``(mm)、``sensor_width``/``sensor_height``(mm)、``sensor_fit``、
  ``angle``(弧度)、``clip_start``/``clip_end``、``dof``。
* GVHMR —— https://github.com/zju3dv/GVHMR：单目视频 → SMPL/SMPL-X；
  ``python tools/demo/demo.py --video=xx.mp4 -s``（``-s`` = 相机静止时跳过视觉里程计），
  批量用 ``tools/demo/demo_folder.py -f <in> -d <out> -s``；依赖 torch + ViTPose，
  官方只给了 4090 训练配置。
  TODO(2026-09-19): README 未写明输出文件格式与键名，真机跑通后回填。
* WHAM —— https://github.com/yohanshin/WHAM：``python demo.py --video xx.mov --visualize``，
  ``--calib fx fy cx cy`` 传内参，``--estimate_local_only`` 跳过 SLAM 只出相机坐标系动作。
* Wan / ControlNet 视频侧接法 —— 见 :mod:`longfilm.animate` 里查证的
  ``WanAnimateToVideo``：pose_video / background_video 都吃 IMAGE 序列，
  所以 previz 渲出的 PNG 序列可以直接当控制图喂进去。
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._proc import popen as _sp_popen  # noqa: F401
from ._proc import run as _sp_run
from .schema import (
    CameraMove,
    EngineHint,
    Scene,
    Shot,
    ShotSize,
    Storyboard,
    VideoRef,
)

log = logging.getLogger(__name__)

#: 全画幅传感器宽度（mm）。所有镜头语法换算的基准 ——
#: 只有固定传感器尺寸，"35mm 焦段"才对应到一个确定的视场角。
SENSOR_WIDTH_MM = 36.0

#: 站立成人代理的身高（m）。底稿的绝对尺度基准：机位距离、地面尺寸、
#: depth 归一化区间全部相对它成立，换了这个数所有换算一起变。
PROXY_HEIGHT_M = 1.75


# ---------------------------------------------------------------- 景别 → 取景高度


@dataclass(frozen=True)
class Framing:
    """一个景别对应的取景高度与视觉中心高度（单位：米）。

    ``frame_height_m`` 是**画幅垂直方向要装下的实际高度**，
    不是主体身高 —— 全景要留头顶和脚下的余量，所以 2.05 > 1.75 的身高。
    ``look_at_z`` 是相机瞄准的高度，把主体放在构图中心偏上的位置。
    """

    frame_height_m: float
    look_at_z: float
    note: str = ""


FRAMING: dict[ShotSize, Framing] = {
    ShotSize.ECU: Framing(0.20, 1.62, "眼—唇区域"),
    ShotSize.CU: Framing(0.45, 1.58, "头肩"),
    ShotSize.MCU: Framing(0.70, 1.50, "胸以上"),
    ShotSize.MS: Framing(1.05, 1.30, "腰以上"),
    ShotSize.MLS: Framing(1.45, 1.15, "膝以上"),
    ShotSize.FS: Framing(2.05, 1.00, "全身含余量"),
    ShotSize.LS: Framing(4.50, 1.20, "人物占画面约 1/3"),
    ShotSize.ELS: Framing(14.0, 2.50, "人物成为环境中的点"),
    ShotSize.OTS: Framing(1.10, 1.50, "过肩：前景肩膀 + 对话者头肩"),
    ShotSize.POV: Framing(1.60, 1.60, "主观视点，相机置于人眼高度"),
    ShotSize.INSERT: Framing(0.30, 1.00, "手部/道具特写"),
    ShotSize.TWO: Framing(1.60, 1.30, "双人同框，横向留白更重要"),
}

#: 低于这个焦段拍近景会出现明显的面部透视畸变（鼻子放大）。
#: 数字来自人像摄影的通行经验：等效 50mm 以下拍头肩就开始变形。
_PORTRAIT_MIN_LENS_MM = 50
_TIGHT_SIZES = frozenset({ShotSize.ECU, ShotSize.CU, ShotSize.MCU})

#: 相机离主体的物理下限（m）。比这更近在实拍里机身会撞到人，
#: 在底稿里则意味着换算参数不合理，应该改焦段而不是硬贴上去。
_MIN_DISTANCE_M = 0.35


# ---------------------------------------------------------------- 相机解算


@dataclass(frozen=True)
class CameraKey:
    """一个相机关键帧。位置与朝向都在世界坐标系，朝向已解算成 Blender 的 XYZ 欧拉角。"""

    frame: int
    location: tuple[float, float, float]
    rotation_euler: tuple[float, float, float]
    lens_mm: float


@dataclass
class CameraSolve:
    """一镜的相机解算结果。这是 previz 最有价值的产物 —— 它把镜头语法变成了数字。"""

    shot_id: str
    shot_size: ShotSize
    camera_move: CameraMove
    lens_mm: float
    distance_m: float
    sensor_width_mm: float
    sensor_height_mm: float
    hfov_deg: float
    vfov_deg: float
    frame_height_m: float
    look_at: tuple[float, float, float]
    fps: int
    frame_start: int
    frame_end: int
    keys: list[CameraKey] = field(default_factory=list)
    noise_scale: float = 0.0        # >0 时给旋转曲线挂 NOISE modifier（手持/斯坦尼康）
    warnings: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start + 1

    @property
    def depth_range_m(self) -> tuple[float, float]:
        """depth pass 的归一化区间。

        **固定区间而不是逐帧 Normalize** —— 逐帧归一化会让同一堵墙在
        每帧被映射到不同灰度，ControlNet 读进去就是一路闪烁。
        区间按相机到主体的距离展开，覆盖前景到背景。
        """
        near = max(0.1, self.distance_m * 0.35)
        far = max(near + 1.0, self.distance_m * 3.0 + self.frame_height_m * 2)
        return (round(near, 4), round(far, 4))

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "shot_size": self.shot_size.name,
            "camera_move": self.camera_move.name,
            "lens_mm": self.lens_mm,
            "distance_m": round(self.distance_m, 4),
            "sensor": [self.sensor_width_mm, round(self.sensor_height_mm, 4)],
            "fov_deg": [round(self.hfov_deg, 3), round(self.vfov_deg, 3)],
            "look_at": [round(v, 4) for v in self.look_at],
            "fps": self.fps,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "depth_range_m": list(self.depth_range_m),
            "noise_scale": self.noise_scale,
            "keys": [
                {
                    "frame": k.frame,
                    "location": [round(v, 5) for v in k.location],
                    "rotation_euler": [round(v, 6) for v in k.rotation_euler],
                    "lens_mm": round(k.lens_mm, 4),
                }
                for k in self.keys
            ],
            "warnings": list(self.warnings),
        }


def look_at_euler(
    loc: tuple[float, float, float], target: tuple[float, float, float]
) -> tuple[float, float, float]:
    """求让相机从 ``loc`` 看向 ``target`` 的 Blender XYZ 欧拉角。

    Blender 相机默认沿 **-Z** 看、**+Y** 为上。把 XYZ 欧拉（Blender 的应用顺序是
    ``Rz·Ry·Rx``，roll 留 0）作用到 (0,0,-1) 得到::

        f = (-sin(rz)·sin(rx),  cos(rz)·sin(rx),  -cos(rx))

    对照目标方向 d 分量即可反解。自己算而不是挂 TRACK_TO 约束，
    是因为 pan/tilt 这类**不跟随主体**的运镜必须能独立 K 旋转曲线；
    约束一挂上，相机就永远盯着目标，摇不出去。
    """
    dx, dy, dz = (target[0] - loc[0], target[1] - loc[1], target[2] - loc[2])
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        raise ValueError("相机位置与目标重合，无法求朝向")
    dx, dy, dz = dx / n, dy / n, dz / n
    rx = math.atan2(math.hypot(dx, dy), -dz)
    rz = math.atan2(-dx, dy)
    return (rx, 0.0, rz)


def _orbit_position(radius: float, azimuth_rad: float, z: float) -> tuple[float, float, float]:
    """绕主体（原点）的圆周位置。方位角 0 = 正前方（-Y 侧），正方向为逆时针俯视。"""
    return (radius * math.sin(azimuth_rad), -radius * math.cos(azimuth_rad), z)


def solve_camera(
    shot: Shot,
    *,
    aspect: float = 16 / 9,
    start_frame: int = 1,
) -> CameraSolve:
    """把 Shot 的镜头语法换算成相机参数与关键帧曲线。

    核心换算是薄透镜的相似三角形：画幅上像高 / 实物高 = 焦距 / 物距，即::

        distance = lens_mm × frame_height_m / sensor_height_mm

    三个量单位不同（mm / m / mm）但比值无量纲，结果直接是米。
    例：35mm 镜头拍中景（取景高 1.05m）、16:9 全画幅（sensor_h = 36/(16/9) = 20.25mm）
    → distance = 35 × 1.05 / 20.25 ≈ 1.81m，和实拍中景的机位距离一致。
    """
    fr = FRAMING[shot.shot_size]
    sensor_h = SENSOR_WIDTH_MM / aspect
    lens = float(shot.lens_mm)
    warnings: list[str] = []

    distance = lens * fr.frame_height_m / sensor_h
    if distance < _MIN_DISTANCE_M:
        warnings.append(
            f"换算机位距离 {distance:.2f}m 小于物理下限 {_MIN_DISTANCE_M}m，"
            f"已钳制；{shot.shot_size.name} 建议改用更长焦段"
        )
        distance = _MIN_DISTANCE_M
    if shot.shot_size in _TIGHT_SIZES and lens < _PORTRAIT_MIN_LENS_MM:
        warnings.append(
            f"{shot.shot_size.name} 用 {lens:.0f}mm 会有面部透视畸变，"
            f"建议 {_PORTRAIT_MIN_LENS_MM}mm 以上；若是有意的压迫感请在 style 里写明"
        )

    hfov = 2 * math.degrees(math.atan(SENSOR_WIDTH_MM / (2 * lens)))
    vfov = 2 * math.degrees(math.atan(sensor_h / (2 * lens)))

    # 轴线：l2r 表示主体在画面里向右运动/朝向，机位就该偏到主体的左后方，
    # 让主体的运动方向在画面里保持向右。neutral 走正前方。
    azimuth = {"l2r": math.radians(-22), "r2l": math.radians(22)}.get(
        shot.continuity.screen_direction, 0.0
    )
    if shot.shot_size is ShotSize.OTS:
        # 过肩必须偏轴，否则前景肩膀会正好挡住对话者。
        azimuth += math.radians(28 if shot.continuity.screen_direction != "r2l" else -28)

    look_at = (0.0, 0.0, fr.look_at_z)
    cam_z = fr.look_at_z
    if shot.shot_size in (ShotSize.LS, ShotSize.ELS):
        # 大远景抬机位，否则地平线压在画面正中，空间层次全丢。
        cam_z = fr.look_at_z + distance * 0.12

    n_frames = max(2, int(round(shot.duration_s * shot.fps)))
    f0, f1 = start_frame, start_frame + n_frames - 1

    solve = CameraSolve(
        shot_id=shot.id,
        shot_size=shot.shot_size,
        camera_move=shot.camera_move,
        lens_mm=lens,
        distance_m=distance,
        sensor_width_mm=SENSOR_WIDTH_MM,
        sensor_height_mm=sensor_h,
        hfov_deg=hfov,
        vfov_deg=vfov,
        frame_height_m=fr.frame_height_m,
        look_at=look_at,
        fps=shot.fps,
        frame_start=f0,
        frame_end=f1,
        warnings=warnings,
    )
    solve.keys = _keys_for_move(shot.camera_move, solve, azimuth, cam_z)
    if shot.camera_move in (CameraMove.HANDHELD, CameraMove.STEADICAM):
        # 手持抖动不K成关键帧而是挂 NOISE modifier：抖动是随机的，
        # 写死成关键帧等于把一次随机结果固化，重渲就没法换一种抖法。
        solve.noise_scale = 0.012 if shot.camera_move is CameraMove.HANDHELD else 0.004
    return solve


def _keys_for_move(
    move: CameraMove, s: CameraSolve, azimuth: float, cam_z: float
) -> list[CameraKey]:
    """把运镜枚举展开成关键帧。每种运镜的幅度都按机位距离缩放，
    这样同一个 DOLLY_IN 在特写和全景里都是"推进约 20%"，而不是固定米数。"""
    f0, f1 = s.frame_start, s.frame_end
    d, lens = s.distance_m, s.lens_mm
    tgt = s.look_at
    base = _orbit_position(d, azimuth, cam_z)

    def key(f: int, loc: tuple[float, float, float], look: tuple[float, float, float],
            lens_mm: float = lens) -> CameraKey:
        return CameraKey(f, loc, look_at_euler(loc, look), lens_mm)

    def shifted(dx: float, dy: float, dz: float) -> tuple[float, float, float]:
        return (base[0] + dx, base[1] + dy, base[2] + dz)

    # 相机的"右"方向（水平面内，垂直于视线）。truck 用它平移。
    right = (math.cos(azimuth), math.sin(azimuth), 0.0)

    if move is CameraMove.STATIC:
        return [key(f0, base, tgt), key(f1, base, tgt)]

    if move in (CameraMove.PAN_L, CameraMove.PAN_R):
        # 摇：机位不动，视线在水平面内扫过 ±12°。做成**对称扫过主体**而不是
        # 从主体摇开：这样镜头中段主体正好在画面中心，剪进片子时首尾都可用。
        sweep = math.radians(24) * (1 if move is CameraMove.PAN_R else -1)
        far = d * 4

        def aim_at(a: float) -> tuple[float, float, float]:
            return (base[0] + far * math.sin(a), base[1] - far * math.cos(a), tgt[2])

        return [key(f0, base, aim_at(azimuth - sweep / 2)),
                key(f1, base, aim_at(azimuth + sweep / 2))]

    if move in (CameraMove.TILT_U, CameraMove.TILT_D):
        rise = d * math.tan(math.radians(10)) * (1 if move is CameraMove.TILT_U else -1)
        return [key(f0, base, tgt), key(f1, base, (tgt[0], tgt[1], tgt[2] + rise))]

    if move in (CameraMove.DOLLY_IN, CameraMove.DOLLY_OUT):
        k = 0.80 if move is CameraMove.DOLLY_IN else 1.25
        return [key(f0, base, tgt), key(f1, _orbit_position(d * k, azimuth, cam_z), tgt)]

    if move in (CameraMove.TRUCK_L, CameraMove.TRUCK_R):
        amt = d * 0.30 * (1 if move is CameraMove.TRUCK_R else -1)
        a = shifted(-right[0] * amt, -right[1] * amt, 0.0)
        b = shifted(right[0] * amt, right[1] * amt, 0.0)
        # 平移机位时视线保持平行（不盯主体），否则平移就变成了弧形跟拍。
        return [key(f0, a, (a[0] + (tgt[0] - base[0]), a[1] + (tgt[1] - base[1]), tgt[2])),
                key(f1, b, (b[0] + (tgt[0] - base[0]), b[1] + (tgt[1] - base[1]), tgt[2]))]

    if move in (CameraMove.PEDESTAL_U, CameraMove.PEDESTAL_D):
        dz = s.frame_height_m * 0.35 * (1 if move is CameraMove.PEDESTAL_U else -1)
        return [key(f0, base, tgt), key(f1, shifted(0, 0, dz), tgt)]

    if move is CameraMove.CRANE:
        top = shifted(0, 0, s.frame_height_m * 1.2 + d * 0.25)
        return [key(f0, base, tgt), key(f1, top, tgt)]

    if move in (CameraMove.ORBIT_L, CameraMove.ORBIT_R):
        # 环绕要多打几个关键帧：两点之间的贝塞尔插值走的是直线，
        # 只给首尾会让相机从弦上穿过去，半径缩水，构图在中段明显变紧。
        span = math.radians(45) * (1 if move is CameraMove.ORBIT_R else -1)
        n = 7
        return [
            key(f0 + round((f1 - f0) * i / (n - 1)),
                _orbit_position(d, azimuth - span / 2 + span * i / (n - 1), cam_z), tgt)
            for i in range(n)
        ]

    if move in (CameraMove.ZOOM_IN, CameraMove.ZOOM_OUT):
        k = 1.6 if move is CameraMove.ZOOM_IN else 1 / 1.6
        return [key(f0, base, tgt, lens), key(f1, base, tgt, lens * k)]

    if move is CameraMove.PUSH_PULL:
        # 移轴变焦（vertigo）：焦距变而主体在画面里大小不变。
        # 由 distance = lens × H / sensor_h 可知，要保持 H 不变，
        # 距离必须与焦距同比变化 —— 这正是这个效果成立的数学条件。
        k = 2.0
        return [
            key(f0, base, tgt, lens),
            key(f1, _orbit_position(d * k, azimuth, cam_z), tgt, lens * k),
        ]

    if move is CameraMove.WHIP:
        # 甩镜：中间停一拍再爆发，否则匀速扫过看着像慢摇。
        far = d * 4
        mid = f0 + round((f1 - f0) * 0.35)
        def aim(a: float) -> tuple[float, float, float]:
            return (base[0] + far * math.sin(a), base[1] - far * math.cos(a), tgt[2])
        return [key(f0, base, aim(azimuth)), key(mid, base, aim(azimuth)),
                key(f1, base, aim(azimuth + math.radians(75)))]

    if move in (CameraMove.HANDHELD, CameraMove.STEADICAM):
        # 基础轨迹是缓慢跟进，抖动交给 NOISE modifier。
        return [key(f0, base, tgt), key(f1, _orbit_position(d * 0.93, azimuth, cam_z), tgt)]

    raise ValueError(f"未覆盖的运镜类型：{move}")


# ---------------------------------------------------------------- 值不值得上底稿


#: 空间关系复杂、纯提示词生成不稳的运镜。
_COMPLEX_MOVES = frozenset({
    CameraMove.ORBIT_L, CameraMove.ORBIT_R, CameraMove.CRANE,
    CameraMove.PUSH_PULL, CameraMove.STEADICAM, CameraMove.DOLLY_IN,
    CameraMove.DOLLY_OUT, CameraMove.TRUCK_L, CameraMove.TRUCK_R,
})

#: 几何简单、上底稿纯属浪费的景别。
_TRIVIAL_SIZES = frozenset({ShotSize.ECU, ShotSize.INSERT, ShotSize.CU})


@dataclass
class PrevizVerdict:
    """单镜的底稿判定。``score`` 是加权分，``reasons`` 解释分是怎么来的 ——
    运营要能看懂为什么这一镜被判了要/不要，而不是只看到一个数。"""

    shot_id: str
    score: float
    worth_it: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "score": round(self.score, 2),
            "worth_it": self.worth_it,
            "reasons": list(self.reasons),
        }


#: 判定阈值。调这个数就是调"底稿产能 vs 重抽成本"的天平。
PREVIZ_THRESHOLD = 3.0


def judge(shot: Shot, *, threshold: float = PREVIZ_THRESHOLD) -> PrevizVerdict:
    """单镜判定：这一镜值不值得上 3D 底稿。

    判据全部指向同一个问题 ——「模型在这一镜上会不会把空间搞错」：
    长镜（错误累积久）、复杂运镜（视差关系必须自洽）、多主体（遮挡与站位）、
    大景别（环境几何占画面主体）、强连续性约束（首尾帧必须对得上）。
    反向判据是特写/插入镜：画面里几乎没有空间信息，灰模帮不上忙。
    """
    score = 0.0
    reasons: list[str] = []

    if shot.duration_s >= 10:
        score += 2.0
        reasons.append(f"长镜 {shot.duration_s:.0f}s：空间错误会累积整镜")
    elif shot.duration_s >= 6:
        score += 1.0
        reasons.append(f"中长镜 {shot.duration_s:.0f}s")

    if shot.camera_move in _COMPLEX_MOVES:
        score += 2.0
        reasons.append(f"复杂运镜 {shot.camera_move.name}：视差关系必须自洽")
    elif shot.camera_move is not CameraMove.STATIC:
        score += 0.5
        reasons.append(f"有运镜 {shot.camera_move.name}")

    if len(shot.subject_ids) >= 3:
        score += 2.0
        reasons.append(f"{len(shot.subject_ids)} 个主体：站位与遮挡关系需要预先钉死")
    elif len(shot.subject_ids) == 2:
        score += 1.5
        reasons.append("双主体交互：轴线与遮挡易错")

    if shot.shot_size in (ShotSize.LS, ShotSize.ELS, ShotSize.FS, ShotSize.MLS):
        score += 1.0
        reasons.append(f"{shot.shot_size.name}：环境几何占画面主要部分")

    if shot.continuity.inherit_last_frame or shot.continuity.match_on:
        score += 1.5
        reasons.append("强连续性约束：首尾帧几何必须与邻镜对齐")

    if shot.engine_hint is EngineHint.HERO:
        score += 1.5
        reasons.append("英雄镜：不计成本保下限")

    if shot.shot_size in _TRIVIAL_SIZES and shot.camera_move is CameraMove.STATIC:
        score -= 3.0
        reasons.append(f"{shot.shot_size.name} + 固定机位：画面几乎不含空间信息，灰模无用")
    if shot.engine_hint is EngineHint.DRAFT:
        score -= 2.0
        reasons.append("预演镜：本就不求画质")

    return PrevizVerdict(shot.id, score, score >= threshold, reasons)


@dataclass
class PrevizPlan:
    """全片底稿计划。产出的是**排期**，不是渲染任务 ——
    哪些镜要上、按什么顺序、预计花多少工时，由运营拿去排人。"""

    project: str
    episode: str
    verdicts: list[PrevizVerdict] = field(default_factory=list)
    threshold: float = PREVIZ_THRESHOLD

    @classmethod
    def build(cls, sb: Storyboard, *, threshold: float = PREVIZ_THRESHOLD) -> PrevizPlan:
        return cls(
            project=sb.project,
            episode=sb.episode,
            verdicts=[judge(s, threshold=threshold) for s in sb.all_shots()],
            threshold=threshold,
        )

    @property
    def selected(self) -> list[PrevizVerdict]:
        # 高分优先：底稿工位排期永远排不完，先做收益最大的。
        return sorted((v for v in self.verdicts if v.worth_it), key=lambda v: -v.score)

    def shot_ids(self) -> list[str]:
        return [v.shot_id for v in self.selected]

    def summary(self) -> str:
        sel = self.selected
        lines = [
            f"{self.project}/{self.episode} 3D 底稿计划："
            f"{len(sel)}/{len(self.verdicts)} 镜入选（阈值 {self.threshold}）",
        ]
        for v in sel:
            lines.append(f"  [{v.score:4.1f}] {v.shot_id}  ← {'；'.join(v.reasons[:2])}")
        if not sel:
            lines.append("  （无镜头达到阈值：全片几何简单，直接走生成即可）")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "episode": self.episode,
            "threshold": self.threshold,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "selected": self.shot_ids(),
        }


# ---------------------------------------------------------------- 成本收益


#: 经验系数。改这几个数就是把本产线的真实工时/单价喂进决策，
#: 不要在 cost_benefit 正文里散落魔数。
PREVIZ_SETUP_MIN = 12.0          # 单镜布景 + 调机位的人工分钟数
PREVIZ_RENDER_S_PER_FRAME = 1.8  # EEVEE 灰模单帧渲染秒数（CPU-only 机器会更慢）
SHOT_GEN_COST_USD = 0.55         # 单次生成一镜的平均引擎成本
REROLL_MIN_PER_TAKE = 6.0        # 一次重抽的人工等待 + 挑片分钟数


def cost_benefit(shot: Shot, *, hourly_usd: float = 25.0) -> str:
    """这一镜上 3D 底稿的成本与收益估算。给运营看的一段中文结论。

    收益侧的关键量是**期望重抽次数**：底稿把空间钉死之后，重抽主要只剩
    材质/光影问题，经验上能把重抽次数砍掉一半以上。所以算式是
    「省下的重抽成本 − 底稿成本」，而不是「底稿贵不贵」。
    """
    v = judge(shot)
    frames = max(2, int(round(shot.duration_s * shot.fps)))
    render_min = frames * PREVIZ_RENDER_S_PER_FRAME / 60
    previz_min = PREVIZ_SETUP_MIN + render_min
    previz_usd = previz_min / 60 * hourly_usd

    # 期望重抽次数随判定分升高：分越高说明模型越容易在这镜上翻车。
    takes_wo = 1.0 + max(0.0, v.score) * 0.55
    takes_w = 1.0 + max(0.0, v.score) * 0.22
    saved_takes = takes_wo - takes_w
    saved_usd = saved_takes * (SHOT_GEN_COST_USD + REROLL_MIN_PER_TAKE / 60 * hourly_usd)
    net = saved_usd - previz_usd

    verdict = "建议上底稿" if v.worth_it else "不建议上底稿"
    if v.worth_it and net < 0:
        verdict = "判定建议上底稿，但按当前单价不划算 —— 除非这镜是片子的门面"
    if not v.worth_it and net > 0:
        verdict = "判定不必上底稿，但成本上有余量，产能空闲时可以顺手做"

    return (
        f"{shot.id}（{shot.shot_size.name} / {shot.camera_move.name} / "
        f"{shot.duration_s:.1f}s / {frames} 帧 / {len(shot.subject_ids)} 主体）\n"
        f"  判定分 {v.score:.1f}（阈值 {PREVIZ_THRESHOLD}）：{'；'.join(v.reasons) or '无加分项'}\n"
        f"  成本：布景调机位 {PREVIZ_SETUP_MIN:.0f}min + 渲染 {render_min:.1f}min "
        f"= {previz_min:.1f}min ≈ ${previz_usd:.2f}\n"
        f"  收益：期望重抽 {takes_wo:.1f} 次 → {takes_w:.1f} 次，省 {saved_takes:.1f} 次 "
        f"≈ ${saved_usd:.2f}\n"
        f"  净收益 ${net:+.2f} —— {verdict}"
    )


# ---------------------------------------------------------------- Blender 脚本生成


#: 生成脚本的静态主体。所有随镜头变化的量都在 CONFIG 里，
#: 主体一个字不变 —— 这样脚本出 bug 时只需要看一份代码，
#: 而不是每镜生成出来的一份不同的代码。
_BLENDER_BODY = r'''
import json, math, os, sys

import bpy

ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT_DIR = os.path.abspath(ARGV[0] if ARGV else CONFIG["out_dir"])
os.makedirs(OUT_DIR, exist_ok=True)


def pick_engine():
    """EEVEE 在 4.2 起改名 BLENDER_EEVEE_NEXT。按运行时枚举挑，不按版本号猜 ——
    版本号判断会在下一个改名的版本里再坏一次。"""
    items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if name in items:
            return name
    return items[0]


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = pick_engine()
scene.render.resolution_x = CONFIG["resolution"][0]
scene.render.resolution_y = CONFIG["resolution"][1]
scene.render.resolution_percentage = 100
scene.render.fps = CONFIG["fps"]
scene.frame_start = CONFIG["frame_start"]
scene.frame_end = CONFIG["frame_end"]
scene.render.film_transparent = False

# ---- 灰模材质：统一无色散射，让渲染结果只剩几何信息 ----
mat = bpy.data.materials.new("previz_clay")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1.0)
if "Roughness" in bsdf.inputs:
    bsdf.inputs["Roughness"].default_value = 0.85
if "Metallic" in bsdf.inputs:
    bsdf.inputs["Metallic"].default_value = 0.0


def clay(obj):
    obj.data.materials.append(mat)
    return obj


# ---- 地面与主体代理 ----
bpy.ops.mesh.primitive_plane_add(size=CONFIG["ground_size"], location=(0, 0, 0))
clay(bpy.context.object).name = "previz_ground"

for proxy in CONFIG["subjects"]:
    x, y = proxy["xy"]
    h = proxy["height"]
    bpy.ops.mesh.primitive_cylinder_add(radius=0.19, depth=h * 0.82,
                                        location=(x, y, h * 0.41))
    body = clay(bpy.context.object)
    body.name = "proxy_%s_body" % proxy["id"]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=h * 0.072,
                                         location=(x, y, h * 0.93))
    head = clay(bpy.context.object)
    head.name = "proxy_%s_head" % proxy["id"]
    # 朝向标记：一个薄片贴在胸前，用来在灰模里看出人物面朝哪边。
    bpy.ops.mesh.primitive_cube_add(size=0.16, location=(x, y - 0.2, h * 0.72))
    clay(bpy.context.object).name = "proxy_%s_facing" % proxy["id"]

# ---- 光：三点布光的极简版。底稿不追求好看，只要法线方向能读出来 ----
for name, loc, energy in CONFIG["lights"]:
    light_data = bpy.data.lights.new(name, type="AREA")
    light_data.energy = energy
    light_data.size = 3.0
    obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    obj.rotation_euler = (math.radians(55), 0, math.atan2(-loc[0], loc[1]))

# ---- 相机 ----
cam_data = bpy.data.cameras.new("previz_cam")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = CONFIG["sensor_width_mm"]
cam_data.sensor_height = CONFIG["sensor_height_mm"]
cam_data.clip_start = 0.05
cam_data.clip_end = max(200.0, CONFIG["depth_range_m"][1] * 4)
cam = bpy.data.objects.new("previz_cam", cam_data)
bpy.context.collection.objects.link(cam)
scene.camera = cam

for k in CONFIG["keys"]:
    f = k["frame"]
    cam.location = k["location"]
    cam.rotation_euler = k["rotation_euler"]
    cam_data.lens = k["lens_mm"]
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
    cam_data.keyframe_insert("lens", frame=f)

# 缓入缓出：机械的匀速运镜一眼假。WHIP 例外，它要的就是突变。
interp = CONFIG["interpolation"]
for action in (cam.animation_data.action, cam_data.animation_data.action):
    for fcu in action.fcurves:
        for kp in fcu.keyframe_points:
            kp.interpolation = interp
            kp.easing = "EASE_IN_OUT"

# 手持抖动：给旋转曲线挂 NOISE modifier，而不是 K 死的关键帧。
if CONFIG["noise_scale"] > 0:
    for fcu in cam.animation_data.action.fcurves:
        if fcu.data_path != "rotation_euler":
            continue
        n = fcu.modifiers.new("NOISE")
        n.scale = 18.0
        n.strength = CONFIG["noise_scale"]
        n.phase = fcu.array_index * 7.3

# ---- 输出通道：gray / depth / normal ----
view_layer = scene.view_layers[0]
view_layer.use_pass_z = True
view_layer.use_pass_normal = True

scene.use_nodes = True
tree = scene.node_tree
for node in list(tree.nodes):
    tree.nodes.remove(node)
rl = tree.nodes.new("CompositorNodeRLayers")

def file_out(name, subdir, fmt="PNG", depth="8"):
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.name = name
    fo.base_path = os.path.join(OUT_DIR, subdir)
    fo.format.file_format = fmt
    fo.format.color_depth = depth
    fo.file_slots[0].path = subdir + "_"
    return fo

tree.links.new(rl.outputs["Image"], file_out("gray", "gray").inputs[0])

# depth 用固定区间做线性映射，不用 Normalize 节点：
# Normalize 是逐帧统计的，同一面墙在不同帧会被映射到不同灰度 → ControlNet 读到闪烁。
near, far = CONFIG["depth_range_m"]
mr = tree.nodes.new("CompositorNodeMapRange")
mr.inputs["From Min"].default_value = near
mr.inputs["From Max"].default_value = far
mr.inputs["To Min"].default_value = 1.0      # 近处白、远处黑，对齐主流 depth ControlNet 约定
mr.inputs["To Max"].default_value = 0.0
mr.use_clamp = True
tree.links.new(rl.outputs["Depth"], mr.inputs["Value"])
tree.links.new(mr.outputs["Value"], file_out("depth", "depth").inputs[0])

if "Normal" in rl.outputs:
    # 法线是 [-1,1]，直接存 PNG 会把负半轴全截成 0。先映射到 [0,1]。
    mix = tree.nodes.new("CompositorNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    mix.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    add = tree.nodes.new("CompositorNodeMixRGB")
    add.blend_type = "ADD"
    add.inputs[0].default_value = 1.0
    add.inputs[2].default_value = (0.5, 0.5, 0.5, 1.0)
    tree.links.new(rl.outputs["Normal"], mix.inputs[1])
    tree.links.new(mix.outputs[0], add.inputs[1])
    tree.links.new(add.outputs[0], file_out("normal", "normal").inputs[0])

bpy.ops.render.render(animation=True, write_still=False)

with open(os.path.join(OUT_DIR, "previz_manifest.json"), "w", encoding="utf-8") as fh:
    json.dump({"solve": CONFIG, "engine": scene.render.engine,
               "blender": list(bpy.app.version)}, fh, ensure_ascii=False, indent=2)
print("previz done:", OUT_DIR)
'''


def _subject_layout(shot: Shot) -> list[dict[str, Any]]:
    """主体代理的站位。单人站原点；多人沿 X 轴对称排开，间距按景别取景宽度算 ——
    双人镜把两个人摆在 3m 外是拍不进同一个画幅的。"""
    n = max(1, len(shot.subject_ids)) if shot.subject_ids else 1
    ids = shot.subject_ids or ["subject"]
    fr = FRAMING[shot.shot_size]
    spread = min(1.1, fr.frame_height_m * 0.55)
    out = []
    for i, sid in enumerate(ids[:n]):
        off = 0.0 if n == 1 else (i - (n - 1) / 2) * spread
        # 交错前后半步：一字排开在画面里会糊成一排，错开才有纵深。
        depth = 0.0 if n == 1 else (0.25 if i % 2 else -0.25)
        out.append({"id": sid, "xy": [round(off, 4), round(depth, 4)], "height": PROXY_HEIGHT_M})
    return out


def emit_blender_script(
    shot: Shot,
    storyboard: Storyboard | None = None,
    *,
    out_dir: str | Path = "previz_out",
    aspect: float | None = None,
    resolution: tuple[int, int] | None = None,
) -> str:
    """生成该镜的 Blender Python 脚本（无头渲染灰模 + depth + normal 序列）。

    脚本的可变部分全部收进头部的 ``CONFIG`` JSON，主体代码是固定文本。
    这样做有两个好处：生成逻辑只需断言 CONFIG 的数值对不对（本机可验证），
    以及脚本在 Blender 里报错时，栈上的行号永远指向同一份代码。
    """
    res = resolution or (storyboard.delivery.resolution if storyboard else (1280, 720))
    ar = aspect if aspect is not None else res[0] / max(res[1], 1)
    solve = solve_camera(shot, aspect=ar)

    ground = max(12.0, solve.distance_m * 4 + FRAMING[shot.shot_size].frame_height_m * 2)
    key_d = max(2.5, solve.distance_m * 1.4)
    cfg: dict[str, Any] = {
        **solve.to_dict(),
        "out_dir": str(out_dir),
        "resolution": [int(res[0]), int(res[1])],
        "sensor_width_mm": solve.sensor_width_mm,
        "sensor_height_mm": round(solve.sensor_height_mm, 5),
        "ground_size": round(ground, 3),
        "subjects": _subject_layout(shot),
        "lights": [
            ["key", [key_d * 0.7, -key_d, key_d * 0.9], 900.0],
            ["fill", [-key_d * 0.9, -key_d * 0.6, key_d * 0.5], 260.0],
            ["rim", [0.0, key_d, key_d * 1.1], 500.0],
        ],
        # 甩镜要的是突变，缓入缓出会把它抹成慢摇。
        "interpolation": "LINEAR" if shot.camera_move is CameraMove.WHIP else "BEZIER",
        "project": storyboard.project if storyboard else "",
        "episode": storyboard.episode if storyboard else "",
        "style_bible": storyboard.style_bible if storyboard else "",
    }
    header = (
        '# 由 longfilm.previz3d.emit_blender_script 生成 —— 不要手改，改生成器。\n'
        '# 运行：blender -b --factory-startup -noaudio --python <此脚本> -- <out_dir>\n'
        "import json\n"
        "CONFIG = json.loads(r'''" + json.dumps(cfg, ensure_ascii=False) + "''')\n"
    )
    return header + _BLENDER_BODY


# ---------------------------------------------------------------- 渲染调用


@dataclass
class PrevizRender:
    """渲染调用结果。**无 Blender 也是一个正常结果**，不是异常 ——
    编排机上本来就没有 Blender，产线要能在这里把任务转派给渲染节点。"""

    ok: bool
    available: bool
    out_dir: Path
    script_path: Path
    command: list[str]
    message: str = ""
    returncode: int | None = None
    log_tail: str = ""

    def command_line(self) -> str:
        return " ".join(self.command)


def render_previz(
    script: str,
    out_dir: str | Path,
    *,
    blender_bin: str | None = None,
    timeout_s: float = 1800.0,
    frames: tuple[int, int] | None = None,
) -> PrevizRender:
    """调用 Blender 无头渲染。找不到 Blender 就如实返回，不抛异常。

    参数顺序不能随意排：Blender 源码明说 "Arguments are executed in the order
    they are given"，``--python`` 会在被读到的那一刻执行，所以
    ``-b``、``--factory-startup`` 必须排在它前面，脚本自己的参数排在 ``--`` 之后。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    script_path = out / "previz_scene.py"
    script_path.write_text(script, encoding="utf-8")

    exe = blender_bin or shutil.which("blender")
    cmd = [
        exe or "blender",
        "-b",
        "--factory-startup",
        "-noaudio",
        "--python", str(script_path),
    ]
    if frames:
        cmd += ["-s", str(frames[0]), "-e", str(frames[1])]
    cmd += ["--", str(out)]

    resolved = exe if (exe and Path(exe).exists()) else shutil.which(cmd[0])
    if not resolved:
        msg = (
            "本机未找到 blender 可执行文件，已生成脚本但未渲染。\n"
            f"  脚本：{script_path}\n"
            f"  待执行：{' '.join(cmd)}\n"
            "  解决办法（任选其一）：\n"
            "    1) 装 Blender 4.x 后重试，或用 blender_bin= 显式指定路径；\n"
            "    2) 把 out_dir 整个目录同步到带 Blender 的渲染节点上执行同一条命令；\n"
            "    3) 该镜先不上底稿 —— 用 cost_benefit(shot) 复核一下是否真的必要。"
        )
        log.warning("render_previz: 未找到 blender，已降级为只生成脚本（%s）", script_path)
        return PrevizRender(False, False, out, script_path, cmd, msg)

    log.info("render_previz: %s", " ".join(cmd))
    try:
        proc = _sp_run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return PrevizRender(False, True, out, script_path, cmd,
                            f"Blender 渲染超过 {timeout_s}s 未结束，已终止", None)
    tail = "\n".join((proc.stdout or "").splitlines()[-20:])
    ok = proc.returncode == 0
    return PrevizRender(
        ok, True, out, script_path, cmd,
        "渲染完成" if ok else f"Blender 退出码 {proc.returncode}",
        proc.returncode,
        tail if ok else (proc.stderr or tail),
    )


# ---------------------------------------------------------------- 控制图打包


#: previz 输出目录下的通道子目录名 → ControlNet 侧的语义。
CONTROL_CHANNELS: dict[str, str] = {
    "depth": "depth",
    "normal": "normal",
    "gray": "reference",     # 灰模本身可当 tile/reference 或 img2img 底图
    "pose": "openpose",      # 由动捕/骨架烘焙出来的骨架图（若有）
}

_FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".exr", ".webp")


@dataclass
class ControlBundle:
    """一镜的 ControlNet 输入包。

    :meth:`to_video_refs` 出来的是 :class:`~longfilm.schema.VideoRef`（role=``previz``），
    可以直接塞进 Shot 的 RefPack —— 底稿从这里汇进主产线，不另起一套参考位机制。
    """

    shot_id: str
    root: Path
    channels: dict[str, Path] = field(default_factory=dict)
    frame_counts: dict[str, int] = field(default_factory=dict)
    fps: int = 24
    resolution: tuple[int, int] = (1280, 720)
    depth_range_m: tuple[float, float] = (0.0, 0.0)
    solve: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return min(self.frame_counts.values()) if self.frame_counts else 0

    @property
    def complete(self) -> bool:
        """至少有 depth，且各通道帧数一致。

        帧数不齐比缺通道更危险：ComfyUI 会按最短的那路静默截断，
        表现是「视频莫名其妙短了一截」，查起来要翻日志。
        """
        return bool(self.channels) and "depth" in self.channels and not self.problems

    def to_video_refs(self, *, weight: float = 1.0) -> list[VideoRef]:
        return [
            VideoRef(
                role="previz",
                uri=str(p),
                weight=weight,
                note=f"controlnet={CONTROL_CHANNELS.get(name, name)}; "
                     f"frames={self.frame_counts.get(name, 0)}",
            )
            for name, p in sorted(self.channels.items())
        ]

    def comfy_values(self) -> dict[str, Any]:
        """给 ComfyUI workflow 的语义名注入值，命名对齐 animate.py 的约定。"""
        v: dict[str, Any] = {
            "width": self.resolution[0],
            "height": self.resolution[1],
            "length": self.frame_count,
        }
        for name, p in self.channels.items():
            v[f"control_{CONTROL_CHANNELS.get(name, name)}"] = str(p)
        return v

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "root": str(self.root),
            "channels": {k: str(v) for k, v in self.channels.items()},
            "frame_counts": dict(self.frame_counts),
            "fps": self.fps,
            "resolution": list(self.resolution),
            "depth_range_m": list(self.depth_range_m),
            "complete": self.complete,
            "problems": list(self.problems),
        }


def previz_to_control(previz_dir: str | Path, *, shot_id: str = "") -> ControlBundle:
    """把渲染出的 depth/normal/gray/pose 序列打包成 ControlNet 输入。

    manifest 里有解算参数就读回来（fps、分辨率、depth 区间），没有就只报目录事实 ——
    不去猜：猜错的 fps 会让整镜时长对不上，而这种错到拼接时才暴露。
    """
    root = Path(previz_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"previz 目录不存在：{root}")

    bundle = ControlBundle(shot_id=shot_id or root.name, root=root)
    mf = root / "previz_manifest.json"
    if mf.exists():
        data = json.loads(mf.read_text(encoding="utf-8"))
        solve = data.get("solve", {})
        bundle.solve = solve
        bundle.shot_id = shot_id or solve.get("shot_id") or bundle.shot_id
        bundle.fps = int(solve.get("fps", bundle.fps))
        res = solve.get("resolution") or list(bundle.resolution)
        bundle.resolution = (int(res[0]), int(res[1]))
        dr = solve.get("depth_range_m") or [0.0, 0.0]
        bundle.depth_range_m = (float(dr[0]), float(dr[1]))
    else:
        bundle.problems.append("缺少 previz_manifest.json：fps 与 depth 归一化区间未知")

    for name in CONTROL_CHANNELS:
        d = root / name
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in _FRAME_SUFFIXES)
        if n == 0:
            bundle.problems.append(f"通道 {name} 目录存在但没有帧")
            continue
        bundle.channels[name] = d
        bundle.frame_counts[name] = n

    if not bundle.channels:
        bundle.problems.append(f"{root} 下没有任何可用通道目录（期望 {sorted(CONTROL_CHANNELS)}）")
    elif "depth" not in bundle.channels:
        bundle.problems.append("缺 depth 通道：depth 是空间约束的主力，没有它底稿价值有限")
    counts = set(bundle.frame_counts.values())
    if len(counts) > 1:
        bundle.problems.append(
            f"各通道帧数不一致 {bundle.frame_counts}；ComfyUI 会按最短的一路静默截断，"
            "请重渲或对齐后再送生成"
        )
    return bundle


# ---------------------------------------------------------------- 自测


def _demo_storyboard() -> Storyboard:
    from .schema import Appearance, CharacterBible, Continuity, DeliverySpec

    chars = [
        CharacterBible(id=cid, name=cid, age_statement="虚构角色，设定年龄 30 岁，成年",
                       appearance=Appearance(face="—"))
        for cid in ("lin", "qiu", "he")
    ]
    shots = [
        Shot(id="s1_01", scene_id="s1", index=0, duration_s=12.0, shot_size=ShotSize.FS,
             camera_move=CameraMove.ORBIT_L, lens_mm=35, subject_ids=["lin", "qiu"],
             action="两人对峙", continuity=Continuity(screen_direction="l2r")),
        Shot(id="s1_02", scene_id="s1", index=1, duration_s=3.0, shot_size=ShotSize.CU,
             camera_move=CameraMove.STATIC, lens_mm=85, subject_ids=["lin"]),
        Shot(id="s1_03", scene_id="s1", index=2, duration_s=2.0, shot_size=ShotSize.INSERT,
             camera_move=CameraMove.STATIC, lens_mm=50),
        Shot(id="s1_04", scene_id="s1", index=3, duration_s=9.0, shot_size=ShotSize.ELS,
             camera_move=CameraMove.CRANE, lens_mm=24, subject_ids=["lin", "qiu", "he"],
             engine_hint=EngineHint.HERO),
        Shot(id="s1_05", scene_id="s1", index=4, duration_s=6.0, shot_size=ShotSize.MS,
             camera_move=CameraMove.PUSH_PULL, lens_mm=35, subject_ids=["lin"]),
    ]
    return Storyboard(project="demo", episode="ep01", characters=chars,
                      scenes=[Scene(id="s1", title="对峙", shots=shots)],
                      delivery=DeliverySpec(resolution=(1280, 720)))


def _selftest() -> None:
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sb = _demo_storyboard()
    shots = {s.id: s for s in sb.all_shots()}

    # --- 1. 核心换算：35mm + MS 的机位距离必须落在实拍的合理区间 ---
    ms = solve_camera(shots["s1_05"], aspect=16 / 9)
    sensor_h = 36.0 / (16 / 9)
    assert abs(sensor_h - 20.25) < 1e-9, sensor_h
    expect = 35 * FRAMING[ShotSize.MS].frame_height_m / sensor_h
    assert abs(ms.distance_m - expect) < 1e-9, (ms.distance_m, expect)
    assert 1.5 <= ms.distance_m <= 2.2, f"35mm 中景机位距离 {ms.distance_m:.2f}m 不合理"
    assert abs(ms.hfov_deg - 54.43) < 0.1, ms.hfov_deg   # 35mm 全画幅水平视场角 ≈54.4°
    print(f"[1] MS@35mm: 距离 {ms.distance_m:.3f}m, 视场 {ms.hfov_deg:.2f}°x{ms.vfov_deg:.2f}°, "
          f"sensor_h {ms.sensor_height_mm:.2f}mm")

    # --- 2. 换算的单调性：焦段越长机位越远，景别越大机位越远 ---
    dists = [solve_camera(shots["s1_05"].model_copy(update={"lens_mm": f})).distance_m
             for f in (24, 35, 50, 85, 135)]
    assert dists == sorted(dists), dists
    sizes = [ShotSize.ECU, ShotSize.CU, ShotSize.MS, ShotSize.FS, ShotSize.LS, ShotSize.ELS]
    by_size = [solve_camera(shots["s1_05"].model_copy(update={"shot_size": z})).distance_m
               for z in sizes]
    assert by_size == sorted(by_size), by_size
    print(f"[2] 焦段单调 {[round(d,2) for d in dists]}; 景别单调 {[round(d,2) for d in by_size]}")

    # --- 3. look_at_euler 正确性：正前方机位应得到 rx=90°, rz=0 ---
    rx, ry, rz = look_at_euler((0.0, -2.0, 1.3), (0.0, 0.0, 1.3))
    assert abs(math.degrees(rx) - 90) < 1e-6 and abs(rz) < 1e-9 and ry == 0.0
    # 相机在 +X 侧看向原点（视线指向 -X）：绕 Z 转 +90°
    _, _, rz2 = look_at_euler((2.0, 0.0, 1.3), (0.0, 0.0, 1.3))
    assert abs(math.degrees(rz2) - 90) < 1e-6, math.degrees(rz2)
    # 反解回前向向量，闭环验证
    for loc in ((1.0, -2.0, 2.0), (-3.0, 1.5, 0.4)):
        a, _, c = look_at_euler(loc, (0, 0, 1.3))
        f = (-math.sin(c) * math.sin(a), math.cos(c) * math.sin(a), -math.cos(a))
        d = (0 - loc[0], 0 - loc[1], 1.3 - loc[2])
        n = math.sqrt(sum(v * v for v in d))
        assert all(abs(f[i] - d[i] / n) < 1e-9 for i in range(3)), (f, d)
    print(f"[3] look_at 闭环验证通过（正前方 rx={math.degrees(rx):.1f}°, rz={math.degrees(rz):.1f}°）")

    # --- 4. 运镜 → 关键帧：每种运镜都要能展开且帧号落在镜头区间内 ---
    covered = []
    for mv in CameraMove:
        sv = solve_camera(shots["s1_05"].model_copy(update={"camera_move": mv}))
        assert len(sv.keys) >= 2, mv
        assert sv.keys[0].frame == sv.frame_start and sv.keys[-1].frame == sv.frame_end, mv
        assert all(sv.frame_start <= k.frame <= sv.frame_end for k in sv.keys), mv
        covered.append((mv.name, len(sv.keys)))
    print(f"[4] 运镜覆盖 {len(covered)} 种，关键帧数 {dict(covered)}")

    # --- 5. 环绕要多关键帧；移轴变焦必须保持取景恒定 ---
    orb = solve_camera(shots["s1_01"])
    assert len(orb.keys) >= 5, "ORBIT 只给了首尾关键帧，中段会走弦而不是弧"
    radii = [math.hypot(k.location[0], k.location[1]) for k in orb.keys]
    assert max(radii) - min(radii) < 1e-6, f"环绕半径不恒定 {radii}"
    pp = solve_camera(shots["s1_05"])
    a, b = pp.keys[0], pp.keys[-1]
    ra = math.hypot(a.location[0], a.location[1]) / a.lens_mm
    rb = math.hypot(b.location[0], b.location[1]) / b.lens_mm
    assert abs(ra - rb) < 1e-9, f"移轴变焦未保持 distance/lens 恒定：{ra} vs {rb}"
    print(f"[5] ORBIT {len(orb.keys)} 关键帧半径恒定 {radii[0]:.3f}m；"
          f"PUSH_PULL 焦距 {a.lens_mm:.0f}→{b.lens_mm:.0f}mm，距离 "
          f"{math.hypot(*a.location[:2]):.2f}→{math.hypot(*b.location[:2]):.2f}m")

    # --- 6. 畸变告警：CU 配 35mm 必须报，配 85mm 不报 ---
    warn = solve_camera(shots["s1_02"].model_copy(update={"lens_mm": 35}))
    assert any("畸变" in w for w in warn.warnings), warn.warnings
    assert not solve_camera(shots["s1_02"]).warnings, solve_camera(shots["s1_02"]).warnings
    print(f"[6] 畸变告警: {warn.warnings[0]}")

    # --- 7. 底稿判定：长镜+环绕+双人入选，静态特写/插入不入选 ---
    plan = PrevizPlan.build(sb)
    sel = set(plan.shot_ids())
    assert "s1_01" in sel and "s1_04" in sel, sel
    assert "s1_02" not in sel and "s1_03" not in sel, sel
    print("[7] " + plan.summary())

    # --- 8. 生成的 Blender 脚本可被 Python 解析，且 CONFIG 可还原 ---
    script = emit_blender_script(shots["s1_01"], sb, out_dir="/tmp/previz_s1_01")
    compile(script, "previz_scene.py", "exec")
    cfg = json.loads(script.split("r'''", 1)[1].split("'''", 1)[0])
    assert cfg["shot_id"] == "s1_01" and cfg["resolution"] == [1280, 720]
    assert len(cfg["keys"]) == len(orb.keys) and cfg["fps"] == 24
    assert cfg["frame_end"] - cfg["frame_start"] + 1 == int(12.0 * 24)
    assert len(cfg["subjects"]) == 2 and cfg["subjects"][0]["xy"][0] < cfg["subjects"][1]["xy"][0]
    assert cfg["depth_range_m"][0] < cfg["depth_range_m"][1]
    assert "bpy" in script and "use_pass_z" in script and "CompositorNodeMapRange" in script
    assert "CompositorNodeNormalize" not in script, "不该用逐帧 Normalize 节点（depth 会闪烁）"
    print(f"[8] 脚本 {len(script)} 字节，可编译；CONFIG 键 {sorted(cfg)[:6]}... "
          f"帧 {cfg['frame_start']}-{cfg['frame_end']}")

    # --- 9. 无 Blender：返回结果而不是崩 ---
    with tempfile.TemporaryDirectory() as td:
        r = render_previz(script, Path(td) / "out", blender_bin="/nonexistent/blender")
        assert r.available is False and r.ok is False
        assert r.script_path.exists(), "脚本没写下来"
        assert "-b" in r.command and r.command.index("-b") < r.command.index("--python"), \
            "Blender 参数顺序错：-b 必须在 --python 之前"
        assert r.command[-1] == str(r.out_dir) and r.command[-2] == "--"
        print(f"[9] 无 Blender 降级: {r.message.splitlines()[0]}")
        print(f"    待执行: {r.command_line()}")

        # --- 10. previz_to_control：打包与帧数不齐的检出 ---
        out = r.out_dir
        (out / "previz_manifest.json").write_text(
            json.dumps({"solve": cfg}, ensure_ascii=False), encoding="utf-8")
        for ch, n in (("depth", 288), ("normal", 288), ("gray", 288)):
            (out / ch).mkdir()
            for i in range(n):
                (out / ch / f"{ch}_{i:04d}.png").write_bytes(b"")
        b = previz_to_control(out)
        assert b.complete and b.frame_count == 288 and b.fps == 24, b.to_dict()
        assert b.resolution == (1280, 720) and b.depth_range_m[1] > b.depth_range_m[0]
        refs = b.to_video_refs()
        assert len(refs) == 3 and all(r_.role == "previz" for r_ in refs)
        assert b.comfy_values()["control_depth"].endswith("depth")
        print(f"[10] ControlBundle: {b.frame_count} 帧 x {len(b.channels)} 通道 "
              f"{sorted(b.channels)}, depth 区间 {b.depth_range_m}")

        (out / "normal" / "normal_9999.png").write_bytes(b"")
        b2 = previz_to_control(out)
        assert not b2.complete and any("帧数不一致" in p for p in b2.problems), b2.problems
        print(f"[10] 帧数不齐检出: {b2.problems[0][:44]}...")

    # --- 11. 成本收益：长镜英雄镜为正，静态插入镜为负 ---
    hero = cost_benefit(shots["s1_04"])
    trivial = cost_benefit(shots["s1_03"])
    assert "建议上底稿" in hero and "不建议上底稿" in trivial
    print("[11] " + hero)
    print("     " + trivial.replace("\n", "\n     "))

    print("previz3d.py 自测通过")


if __name__ == "__main__":
    _selftest()
