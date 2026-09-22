"""超分与插帧后处理 —— 720p 出片，本地拉到 1080p/4K。

为什么把超分放在产线末端而不是让引擎直出高分辨率：
官方 API 按「秒 × 分辨率」计费，1080p 档普遍是 720p 档的 2~4 倍价；
开源侧 1080p 直出的显存占用同样是 720p 的 2 倍以上，还会把可生成时长压短。
先按 720p 把画面内容定下来（内容对了才值得花钱放大），再用后处理把像素补回去，
是当前性价比最高的路径 —— 代价是超分引擎必须**时域一致**，否则逐帧放大会闪烁。

本机（无 GPU、3GB 内存）只有 FFmpegUpscaler / FFmpegMinterpolate 这两条 CPU 基线能真跑，
demo 走它们；GPU 引擎全部延迟导入，缺硬件时抛 HardwareUnavailable 并把仓库地址和
显存门槛写进报错，而不是在 import 时炸掉整个产线。

外部事实查证日期 2026-09-19，来源见 EngineSpec.repo 与各 spec 的 note。
"""

from __future__ import annotations

import abc
import importlib
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal

from ._proc import popen as _sp_popen  # noqa: F401
from ._proc import run as _sp_run
from .schema import DeliverySpec, Transition
from .stitch import (
    ClipSpec,
    FFmpeg,
    concat_hard,
    concat_with_transitions,
    default_ffmpeg,
)

log = logging.getLogger(__name__)

__all__ = [
    "Hardware",
    "HardwareUnavailable",
    "EngineSpec",
    "VSR_ENGINES",
    "VFI_ENGINES",
    "Upscaler",
    "FFmpegUpscaler",
    "RealESRGANUpscaler",
    "SeedVRUpscaler",
    "ComfyUpscaler",
    "Interpolator",
    "FFmpegMinterpolate",
    "RifeInterpolator",
    "PostStep",
    "PostPlan",
    "plan_postprocess",
    "ChunkWindow",
    "plan_chunks",
    "chunked_process",
]


class HardwareUnavailable(RuntimeError):
    """所需的 GPU / 权重 / Python 依赖不在位。

    单独一个异常类型而不是复用 ProviderError：ProviderError 描述的是**远端 API**
    的失败语义（能不能重试、要不要换供应商），本地缺卡缺权重是部署问题，
    重试和换引擎都救不了，必须人工介入。
    """


# ---------------------------------------------------------------- 硬件


@dataclass(frozen=True)
class Hardware:
    """可用算力快照。plan_postprocess 的全部决策依据。"""

    has_cuda: bool = False
    vram_gb: float = 0.0          # 单卡显存，多卡取最大的那张
    gpu_count: int = 0
    gpu_name: str = ""
    cpu_cores: int = 1
    ram_gb: float = 0.0
    comfy_url: str = ""           # 远端 ComfyUI；有它就等于有一张别人的卡

    @classmethod
    def detect(cls) -> Hardware:
        """探测本机算力。

        刻意不用 torch.cuda 来探卡：这个函数会在没装 torch 的调度机上被调用，
        为了问一句「有没有卡」而拖进 2GB 的 torch 不可接受。nvidia-smi 是驱动自带的。
        """
        gpus = _nvidia_smi()
        ram_gb = 0.0
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            if m := re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo.read_text()):
                ram_gb = int(m.group(1)) / 1024 / 1024
        return cls(
            has_cuda=bool(gpus),
            vram_gb=max((g[1] for g in gpus), default=0.0),
            gpu_count=len(gpus),
            gpu_name=gpus[0][0] if gpus else "",
            cpu_cores=os.cpu_count() or 1,
            ram_gb=round(ram_gb, 1),
            comfy_url=os.environ.get("COMFY_URL", ""),
        )

    def describe(self) -> str:
        gpu = f"{self.gpu_count}×{self.gpu_name} {self.vram_gb:.0f}GB" if self.has_cuda else "无 GPU"
        remote = f"，远端 Comfy {self.comfy_url}" if self.comfy_url else ""
        return f"{gpu}，CPU {self.cpu_cores} 核，内存 {self.ram_gb:.1f}GB{remote}"


def _nvidia_smi() -> list[tuple[str, float]]:
    """返回 [(卡名, 显存GB)]。没有驱动/没有卡都返回空列表。"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = _sp_run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    gpus: list[tuple[str, float]] = []
    for line in out.stdout.strip().splitlines():
        name, _, mib = line.partition(",")
        try:
            gpus.append((name.strip(), float(mib.strip()) / 1024))
        except ValueError:
            continue
    return gpus


# ---------------------------------------------------------------- 引擎事实登记


@dataclass(frozen=True)
class EngineSpec:
    """一个后处理引擎的外部事实。

    显存门槛和吞吐必须来自官方 README 而不是感觉 —— plan_postprocess 拿这两个数
    做选型和排期，估错的直接后果是把一集片排进一张放不下的卡里跑三天。
    未查证到的字段写 0 并在 note 里说明，不要填一个像模像样的假数字。
    """

    key: str
    repo: str
    min_vram_gb: float                 # 官方给出的最低单卡显存；0 = 未查证到
    scales: tuple[int, ...] = (4,)     # 官方主打的放大倍数
    ref_out_px: int = 1920 * 1080      # 吞吐的参考输出分辨率
    ref_fps: float = 0.0               # 在 ref_out_px 上的输出帧率；0 = 未查证到
    temporal: bool = True              # 是否有时域一致性建模（无则逐帧放大会闪）
    verified_on: str = "2026-09-19"
    note: str = ""

    def estimate_s(self, frames: int, out_px: int) -> float:
        """按输出像素数线性外推耗时。0 吞吐时返回 0，调用方据此标「无法估算」。"""
        if self.ref_fps <= 0 or frames <= 0:
            return 0.0
        return frames / self.ref_fps * (out_px / self.ref_out_px)


# 超分引擎。数据来自各仓库 README，查证日期 2026-09-19。
VSR_ENGINES: dict[str, EngineSpec] = {
    "ffmpeg": EngineSpec(
        key="ffmpeg",
        repo="内置 bin/ffmpeg 7.0.2（scale=zimg + unsharp）",
        min_vram_gb=0.0,
        scales=(2, 3, 4),
        ref_fps=0.0,   # 纯 CPU，吞吐随核数变化，由 FFmpegUpscaler 按核数现算
        temporal=True,  # 逐帧确定性插值，不引入新细节，因此天然不闪
        note="唯一无 GPU 可跑的基线。只做重采样+锐化，不补细节，4x 会明显糊。",
    ),
    "realesrgan": EngineSpec(
        key="realesrgan",
        repo="https://github.com/xinntao/Real-ESRGAN",
        min_vram_gb=8.0,
        scales=(2, 4),
        ref_fps=2.0,
        temporal=False,
        note=(
            "README 的 inference_realesrgan_video.py + realesr-animevideov3；"
            "--tile 可切块降显存（'Tile size, 0 for no tile during testing'）。"
            "README 未给出具体显存数与 fps：min_vram_gb=8 与 ref_fps 是按 tile 模式的"
            "工程经验保守取值，上线前需按实卡实测校准。"
        ),
    ),
    "seedvr2": EngineSpec(
        key="seedvr2",
        repo="https://github.com/ByteDance-Seed/SeedVR",
        min_vram_gb=80.0,
        scales=(2, 4),
        ref_fps=0.0,
        temporal=True,
        note=(
            "README 原文：'1 H100-80G can handle videos with 100x720x1280. "
            "4 H100-80G further support 1080p and 2K videos'。"
            "即单卡 80G 只够把 100 帧做到 720p 级中间态，1080p 交付要 4 卡。"
            "one-step 扩散，README 未给 fps，故 ref_fps=0（无法估算耗时）。"
        ),
    ),
    "star": EngineSpec(
        key="star",
        repo="https://github.com/NJU-PCALab/STAR",
        min_vram_gb=24.0,
        scales=(4,),
        ref_fps=0.0,
        temporal=True,
        note=(
            "README 原文：72 帧 426×240 做 4x 默认设置需约 39GB 显存，"
            "并建议 '至少 24GB 显存'（靠缩短 frame_length 换显存）。"
            "对 720p→4K 这种量级显存需求远超单卡，本产线只登记不默认选用。"
        ),
    ),
    "flashvsr": EngineSpec(
        key="flashvsr",
        repo="https://github.com/OpenImagingLab/FlashVSR",
        min_vram_gb=0.0,
        scales=(4,),
        ref_out_px=768 * 1408,
        ref_fps=17.0,
        temporal=True,
        note=(
            "README 原文：单张 A100 上 768×1408 约 17 fps，'first diffusion-based "
            "one-step streaming framework towards real-time VSR'，仅针对 4x 优化。"
            "README 未公布显存数字，min_vram_gb=0 表示未查证 —— 排期前必须实测。"
        ),
    ),
    "upscale_a_video": EngineSpec(
        key="upscale_a_video",
        repo="https://github.com/sczhou/Upscale-A-Video",
        min_vram_gb=0.0,
        scales=(4,),
        ref_fps=0.0,
        temporal=True,
        note=(
            "CVPR2024，文本引导 + propagator 做时域传播。README 未给显存与速度，"
            "两项均未查证（2026-09-19）。权重需从 Google Drive 手动下载。"
        ),
    ),
    "comfy": EngineSpec(
        key="comfy",
        repo="ComfyUI 工作流（远端 GPU），见 providers/comfy_local.py",
        min_vram_gb=0.0,      # 显存在远端，不占本机
        scales=(2, 4),
        ref_fps=0.0,
        temporal=True,
        note="把上面任意一个 VSR 包成工作流；本地只发请求，显存门槛由远端节点决定。",
    ),
}

# 插帧引擎。
VFI_ENGINES: dict[str, EngineSpec] = {
    "minterpolate": EngineSpec(
        key="minterpolate",
        repo="内置 bin/ffmpeg 7.0.2（minterpolate 滤镜）",
        min_vram_gb=0.0,
        scales=(2,),
        ref_fps=0.0,
        temporal=True,
        note=(
            "块匹配光流，纯 CPU。快速运动和遮挡边缘会出果冻/撕裂，"
            "只适合 demo 和小幅补帧（24→30），24→60 建议上 RIFE。"
        ),
    ),
    "rife": EngineSpec(
        key="rife",
        repo="https://github.com/hzwer/Practical-RIFE",
        min_vram_gb=4.0,
        scales=(2, 4),
        ref_fps=30.0,
        temporal=True,
        note=(
            "README：最新模型 4.26（2024-09-21 发布），日常推荐 4.25；"
            "inference_video.py --multi=2/4 --video=x.mp4 --scale=0.5（4K 用）。"
            "README 未给显存与 fps，min_vram_gb/ref_fps 为保守估计，需实测校准。"
        ),
    ),
    "gimm_vfi": EngineSpec(
        key="gimm_vfi",
        repo="https://github.com/GSeanCDAT/GIMM-VFI",
        min_vram_gb=8.0,
        scales=(2, 4, 8),
        ref_fps=0.0,
        temporal=True,
        note=(
            "NeurIPS2024，连续时间戳插帧。README 实测（V100）："
            "2K 8x 在 DS_SCALE=0.5 下约 7932 MiB，4K 8x 在 DS_SCALE=0.25 下约 10922 MiB。"
            "故 min_vram_gb=8 是有出处的；README 未给 fps。"
        ),
    ),
}


# ---------------------------------------------------------------- 超分


class Upscaler(abc.ABC):
    """超分引擎统一接口。

    只有两件事是强制的：声明 spec（选型依据），以及把 src 变成 target_wh 写到 dst。
    引擎内部要不要切 tile、要不要分段，是引擎自己的事；
    跨**时间**的分段由 chunked_process 统一负责，避免每家各写一遍还各写错一遍。
    """

    spec: EngineSpec

    @property
    def name(self) -> str:
        return self.spec.key

    def available(self, hw: Hardware) -> tuple[bool, str]:
        """(能否在该硬件上跑, 不能跑的原因)。plan_postprocess 用它筛引擎。"""
        if self.spec.min_vram_gb > 0 and not hw.has_cuda:
            return False, f"{self.name} 需要 CUDA GPU（≥{self.spec.min_vram_gb:.0f}GB 显存），本机无 GPU"
        if self.spec.min_vram_gb > hw.vram_gb:
            return False, (
                f"{self.name} 需要 ≥{self.spec.min_vram_gb:.0f}GB 显存，"
                f"本机最大单卡 {hw.vram_gb:.0f}GB"
            )
        return True, ""

    def require(self, hw: Hardware) -> None:
        ok, why = self.available(hw)
        if not ok:
            raise HardwareUnavailable(f"{why}；引擎仓库：{self.spec.repo}")

    def estimate_s(self, frames: int, out_wh: tuple[int, int]) -> float:
        return self.spec.estimate_s(frames, out_wh[0] * out_wh[1])

    @abc.abstractmethod
    def upscale(
        self,
        src: str | Path,
        dst: str | Path,
        *,
        target_wh: tuple[int, int],
        ffmpeg: FFmpeg | None = None,
    ) -> Path:
        """把 src 放大/缩放到 target_wh，写入 dst，返回 dst。"""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} vram≥{self.spec.min_vram_gb:.0f}GB>"


def _even(n: int) -> int:
    """H.264/265 的 yuv420p 要求宽高为偶数，奇数会被编码器直接拒。"""
    return n if n % 2 == 0 else n + 1


class FFmpegUpscaler(Upscaler):
    """纯 CPU 基线：zimg 重采样 + unsharp 锐化。

    它不补细节（没有生成式先验），但有三个不可替代的优点：
    确定性、零显存、逐帧独立因而**绝不闪烁**。所以它既是无卡环境的唯一选择，
    也是所有 GPU 超分的兜底路径 —— 宁可出一版稍软的 1080p，也不要交付失败。

    默认 lanczos：对真人/实拍向素材锐利度最好；卡通线条用 spline 振铃更小。
    """

    def __init__(
        self,
        *,
        flags: Literal["lanczos", "spline", "bicubic"] = "lanczos",
        sharpen: float = 0.6,
        threads: int = 0,
        crf: int = 16,
        preset: str = "medium",
    ):
        # ref_fps 按核数现算：CPU 缩放几乎线性吃核，用固定值会把 16 核机器的排期估成 8 倍。
        cores = threads or (os.cpu_count() or 1)
        self.spec = replace(VSR_ENGINES["ffmpeg"], ref_fps=_CPU_FPS_PER_CORE_1080P * cores)
        self.flags = flags
        self.sharpen = sharpen
        self.threads = threads
        self.crf = crf
        self.preset = preset

    def filter_chain(self, target_wh: tuple[int, int]) -> str:
        w, h = _even(target_wh[0]), _even(target_wh[1])
        parts = [f"scale={w}:{h}:flags={self.flags}"]
        if self.sharpen > 0:
            # 5x5 luma 核只锐化亮度：色度锐化在 yuv420p 上会放大色带和块效应。
            parts.append(f"unsharp=5:5:{self.sharpen:.3f}:5:5:0.0")
        parts.append("format=yuv420p")
        return ",".join(parts)

    def upscale(
        self,
        src: str | Path,
        dst: str | Path,
        *,
        target_wh: tuple[int, int],
        ffmpeg: FFmpeg | None = None,
    ) -> Path:
        ff = ffmpeg or default_ffmpeg()
        out = Path(dst)
        out.parent.mkdir(parents=True, exist_ok=True)
        args = ["-y", "-i", str(src), "-vf", self.filter_chain(target_wh)]
        if self.threads:
            args += ["-threads", str(self.threads)]
        args += [
            "-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset,
            "-pix_fmt", "yuv420p", "-c:a", "copy", str(out),
        ]
        ff.run(args, timeout=7200.0)
        return out


# 本机 WSL2 实测（bin/ffmpeg 7.0.2，8 核，libx264 preset=medium，720p→1080p lanczos+unsharp）：
# 120 帧 19.0s，合 6.3 帧/秒 ≈ 0.79 帧/秒/核。瓶颈是 x264 编码而非缩放，
# 所以按核数线性外推偏乐观（x264 的多线程加速是次线性的），换机器/换 preset 后必须重测。
_CPU_FPS_PER_CORE_1080P = 0.79


class _GPUUpscaler(Upscaler):
    """GPU 超分的公共壳：延迟导入 + 统一的「缺硬件」报错。

    子类只实现 _run()，不用各写一遍 torch 探测 —— 三份探测代码必然有两份写歪。
    """

    def _require_torch_cuda(self) -> None:
        try:
            import torch  # noqa: PLC0415  故意延迟导入：无卡环境不装 torch
        except ImportError as e:
            raise HardwareUnavailable(
                f"{self.name} 需要 PyTorch，当前环境未安装。"
                f"请在有 CUDA 的机器上 `pip install torch`，引擎仓库：{self.spec.repo}"
            ) from e
        if not torch.cuda.is_available():
            raise HardwareUnavailable(
                f"{self.name} 需要 CUDA GPU，torch.cuda.is_available() 为 False。"
                f"引擎仓库：{self.spec.repo}"
            )

    def upscale(
        self,
        src: str | Path,
        dst: str | Path,
        *,
        target_wh: tuple[int, int],
        ffmpeg: FFmpeg | None = None,
    ) -> Path:
        self._require_torch_cuda()
        out = Path(dst)
        out.parent.mkdir(parents=True, exist_ok=True)
        return self._run(Path(src), out, target_wh, ffmpeg or default_ffmpeg())

    @abc.abstractmethod
    def _run(self, src: Path, dst: Path, target_wh: tuple[int, int], ff: FFmpeg) -> Path:
        ...


class RealESRGANUpscaler(_GPUUpscaler):
    """Real-ESRGAN 视频超分。

    逐帧模型（spec.temporal=False），跨帧没有约束，细节会在帧间跳 ——
    所以这里强制接一个 deflicker：它不能修复结构性抖动，但能压住整体亮度呼吸。
    真要时域稳定应该上 SeedVR2/FlashVSR，Real-ESRGAN 的定位是「便宜、随处可跑」。
    """

    def __init__(self, *, model: str = "realesr-animevideov3", tile: int = 512, denoise: float = 0.5):
        self.spec = VSR_ENGINES["realesrgan"]
        self.model = model
        self.tile = tile
        self.denoise = denoise

    def _run(self, src: Path, dst: Path, target_wh: tuple[int, int], ff: FFmpeg) -> Path:
        try:
            # import_module 而不是 `import realesrgan`：这里只是探测依赖在不在位，
            # 绑一个用不到的名字既多余又会被 linter 当成死代码。
            importlib.import_module("realesrgan")
        except ImportError as e:
            raise HardwareUnavailable(
                "未安装 realesrgan 包。请 `pip install realesrgan basicsr` 并下载权重"
                f"（-n {self.model}），仓库：{self.spec.repo}"
            ) from e
        # TODO(2026-09-19): 官方 inference_realesrgan_video.py 的完整 flag 表未在 README 中
        # 给全（只确认了 -n/-i/-o/--outscale/-t tile）。实机接入时按 `python
        # inference_realesrgan_video.py -h` 的真实输出补齐，不要照抄这里的猜测。
        raise HardwareUnavailable(
            f"{self.name} 的实机调用尚未接线（需要权重目录与 inference 脚本路径）。"
            f"本机无 GPU 无法验证，参见 {self.spec.repo}"
        )


class SeedVRUpscaler(_GPUUpscaler):
    """SeedVR2 —— 当前画质天花板，也是显存黑洞。

    README 明确：单张 H100-80G 只够 100 帧 × 720×1280，1080p/2K 要 4 张 H100。
    所以它在本产线里是「英雄镜专用」：整集全走 SeedVR2 的成本不合理，
    由 plan_postprocess 在显存够的时候才选它，并且**必须**配合 chunked_process 分段。
    """

    def __init__(self, *, variant: Literal["3b", "7b"] = "3b", frames_per_batch: int = 100):
        self.spec = VSR_ENGINES["seedvr2"]
        self.variant = variant
        # 100 帧来自 README 里 80G 单卡的实测上限，不是拍脑袋的默认值。
        self.frames_per_batch = frames_per_batch

    def _run(self, src: Path, dst: Path, target_wh: tuple[int, int], ff: FFmpeg) -> Path:
        raise HardwareUnavailable(
            f"{self.name}({self.variant}) 需要 ≥{self.spec.min_vram_gb:.0f}GB 显存的 H100 级设备，"
            f"且需先按仓库说明下载权重（--res_h/--res_w 指定输出尺寸）。仓库：{self.spec.repo}"
        )


class ComfyUpscaler(_GPUUpscaler):
    """把任意 VSR 包成 ComfyUI 工作流，跑在远端的卡上。

    本地不需要 GPU，所以它覆盖掉父类的 _require_torch_cuda ——
    对远端服务而言「本机有没有卡」是个无关问题，该校验的是 COMFY_URL 通不通。
    """

    def __init__(self, *, base_url: str | None = None, workflow: str = "vsr_4x.json"):
        self.spec = VSR_ENGINES["comfy"]
        self.base_url = (base_url or os.environ.get("COMFY_URL") or "").rstrip("/")
        self.workflow = workflow

    def available(self, hw: Hardware) -> tuple[bool, str]:
        url = self.base_url or hw.comfy_url
        if not url:
            return False, "未配置 COMFY_URL，无法访问远端 ComfyUI"
        return True, ""

    def _require_torch_cuda(self) -> None:
        if not (self.base_url or os.environ.get("COMFY_URL")):
            raise HardwareUnavailable("未配置 COMFY_URL，无法访问远端 ComfyUI 做超分")

    def _run(self, src: Path, dst: Path, target_wh: tuple[int, int], ff: FFmpeg) -> Path:
        # 复用已有的 ComfyProvider：它已经封好了 prompt 提交、ws 进度和产物下载，
        # 在这里重写一遍 HTTP 只会多出一套要同步维护的超时/重试逻辑。
        from .providers.comfy_local import ComfyProvider  # noqa: PLC0415  避免顶层循环导入

        raise HardwareUnavailable(
            f"{self.name} 需要远端 ComfyUI 上存在 {self.workflow} 工作流并暴露输入/输出节点。"
            f"当前 base_url={self.base_url or '(未配置)'}；"
            f"接线时用 {ComfyProvider.__name__} 提交工作流，不要另写 HTTP 客户端"
        )


# ---------------------------------------------------------------- 插帧


class Interpolator(abc.ABC):
    """插帧引擎统一接口。"""

    spec: EngineSpec

    @property
    def name(self) -> str:
        return self.spec.key

    def available(self, hw: Hardware) -> tuple[bool, str]:
        if self.spec.min_vram_gb > 0 and not hw.has_cuda:
            return False, f"{self.name} 需要 CUDA GPU（≥{self.spec.min_vram_gb:.0f}GB 显存），本机无 GPU"
        if self.spec.min_vram_gb > hw.vram_gb:
            return False, f"{self.name} 需要 ≥{self.spec.min_vram_gb:.0f}GB 显存，本机 {hw.vram_gb:.0f}GB"
        return True, ""

    def require(self, hw: Hardware) -> None:
        ok, why = self.available(hw)
        if not ok:
            raise HardwareUnavailable(f"{why}；引擎仓库：{self.spec.repo}")

    def estimate_s(self, out_frames: int, out_wh: tuple[int, int]) -> float:
        return self.spec.estimate_s(out_frames, out_wh[0] * out_wh[1])

    @abc.abstractmethod
    def interpolate(
        self,
        src: str | Path,
        dst: str | Path,
        *,
        target_fps: int,
        ffmpeg: FFmpeg | None = None,
    ) -> Path:
        ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class FFmpegMinterpolate(Interpolator):
    """CPU 基线插帧：minterpolate 的块匹配光流。

    mi_mode=mci + mc_mode=aobmc + me_mode=bidir 是画质最好的一组，也最慢。
    vsbmc=1 打开可变尺寸块匹配，对小物体运动明显更稳，代价再慢一截 ——
    但插帧本来就是离线步骤，宁可慢也不要在人脸边缘出撕裂。
    """

    def __init__(self, *, quality: Literal["fast", "good"] = "good", crf: int = 16, preset: str = "medium"):
        self.spec = replace(VFI_ENGINES["minterpolate"], ref_fps=_CPU_VFI_FPS_PER_CORE * (os.cpu_count() or 1))
        self.quality = quality
        self.crf = crf
        self.preset = preset

    def filter_chain(self, target_fps: int) -> str:
        if self.quality == "fast":
            return f"minterpolate=fps={target_fps}:mi_mode=blend"
        return (
            f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc"
            f":me_mode=bidir:me=epzs:vsbmc=1"
        )

    def interpolate(
        self,
        src: str | Path,
        dst: str | Path,
        *,
        target_fps: int,
        ffmpeg: FFmpeg | None = None,
    ) -> Path:
        ff = ffmpeg or default_ffmpeg()
        out = Path(dst)
        out.parent.mkdir(parents=True, exist_ok=True)
        ff.run(
            [
                "-y", "-i", str(src),
                "-vf", self.filter_chain(target_fps),
                "-r", str(target_fps),
                "-c:v", "libx264", "-crf", str(self.crf), "-preset", self.preset,
                "-pix_fmt", "yuv420p", "-c:a", "copy", str(out),
            ],
            timeout=7200.0,
        )
        return out


# 本机实测（8 核，mci+aobmc+vsbmc，720p 24→48）：96 帧 53.5s = 1.79 帧/秒，
# 折算到 1080p 输出约 0.80 帧/秒 ≈ 0.10 帧/秒/核 —— 比缩放慢一个量级，
# 光流估计本身就是瓶颈，所以整集插帧在无卡机上属于「排得进但等不起」。
_CPU_VFI_FPS_PER_CORE = 0.10


class RifeInterpolator(Interpolator):
    """Practical-RIFE。插帧里的默认选择：快、稳、显存小。"""

    def __init__(self, *, model: str = "4.25", scale: float = 1.0, repo_dir: str | None = None):
        self.spec = VFI_ENGINES["rife"]
        # README：'4.25 by default for most scenes'，最新为 4.26（2024-09-21）。
        self.model = model
        # README：4K 建议 --scale=0.5（等价 --UHD），否则光流估计的显存会翻倍。
        self.scale = scale
        self.repo_dir = repo_dir or os.environ.get("RIFE_DIR", "")

    def interpolate(
        self,
        src: str | Path,
        dst: str | Path,
        *,
        target_fps: int,
        ffmpeg: FFmpeg | None = None,
    ) -> Path:
        try:
            import torch  # noqa: PLC0415  故意延迟导入
        except ImportError as e:
            raise HardwareUnavailable(
                f"{self.name} 需要 PyTorch 与 CUDA GPU，当前环境未安装 torch。仓库：{self.spec.repo}"
            ) from e
        if not torch.cuda.is_available():
            raise HardwareUnavailable(
                f"{self.name} 需要 CUDA GPU，torch.cuda.is_available() 为 False。仓库：{self.spec.repo}"
            )
        if not self.repo_dir or not Path(self.repo_dir, "inference_video.py").exists():
            raise HardwareUnavailable(
                "未找到 Practical-RIFE 仓库：请 clone 后设 RIFE_DIR 指向含 inference_video.py 的目录，"
                f"并下载 {self.model} 权重到 train_log/。仓库：{self.spec.repo}"
            )
        # README 确认的 flag：--video/--output/--multi/--scale/--fps。
        # multi 只能是整数倍，非整数倍率（如 24→30）RIFE 官方脚本不支持，
        # 这种情况必须先 multi 到更高帧率再用 ffmpeg 抽帧，所以这里直接拒绝而不是偷偷取整。
        raise HardwareUnavailable(
            f"{self.name} 的实机调用需在 {self.repo_dir} 内执行 "
            f"`python inference_video.py --video=... --multi=N --scale={self.scale}`；"
            "本机无 GPU 无法验证，故未接线"
        )


# ---------------------------------------------------------------- 计划


@dataclass
class PostStep:
    """后处理链上的一步。"""

    kind: Literal["upscale", "interpolate"]
    engine: str
    detail: str
    est_s: float = 0.0
    est_known: bool = True     # False = 引擎吞吐未查证，est_s 不可信


@dataclass
class PostPlan:
    """一条后处理链路的完整决策记录。

    把「选了什么、为什么、要跑多久」写成数据而不是藏在 if 里，
    排期表和成本表才能直接读它，出片后也能复盘当时为什么降级。
    """

    source_wh: tuple[int, int]
    target_wh: tuple[int, int]
    source_fps: int
    target_fps: int
    duration_s: float
    hardware: Hardware = field(default_factory=Hardware)
    steps: list[PostStep] = field(default_factory=list)
    upscaler: Upscaler | None = None
    interpolator: Interpolator | None = None
    chunked: bool = False
    chunk_s: float = 0.0
    overlap_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def est_total_s(self) -> float:
        return round(sum(s.est_s for s in self.steps), 1)

    @property
    def est_reliable(self) -> bool:
        return all(s.est_known for s in self.steps)

    @property
    def noop(self) -> bool:
        return not self.steps

    def describe(self) -> str:
        head = (
            f"{self.source_wh[0]}x{self.source_wh[1]}@{self.source_fps} -> "
            f"{self.target_wh[0]}x{self.target_wh[1]}@{self.target_fps}"
        )
        if self.noop:
            return f"{head}：无需后处理"
        chain = " -> ".join(f"{s.kind}:{s.engine}" for s in self.steps)
        est = f"约 {self.est_total_s / 60:.1f} 分钟" if self.est_reliable else "耗时无法估算（引擎吞吐未查证）"
        seg = f"，分段 {self.chunk_s:.0f}s/重叠 {self.overlap_s:.1f}s" if self.chunked else ""
        return f"{head}：{chain}；{est}{seg}"

    def run(self, src: str | Path, out: str | Path, *, ffmpeg: FFmpeg | None = None) -> Path:
        """按计划执行。noop 时复制一份而不是原地返回 ——
        调用方拿到的永远是 out 这个路径，后续步骤不必再分支判断。"""
        ff = ffmpeg or default_ffmpeg()
        out_p = Path(out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if self.noop:
            shutil.copy2(src, out_p)
            return out_p

        with tempfile.TemporaryDirectory(prefix="longfilm_post_") as td:
            cur = Path(src)
            tmp = Path(td)
            stages: list[Path] = []
            for i, step in enumerate(self.steps):
                last = i == len(self.steps) - 1
                dst = out_p if last else tmp / f"stage{i}_{step.engine}.mp4"
                if step.kind == "upscale":
                    assert self.upscaler is not None
                    up = self.upscaler
                    if self.chunked:
                        chunked_process(
                            cur,
                            lambda a, b, _up=up: _up.upscale(a, b, target_wh=self.target_wh, ffmpeg=ff),
                            dst,
                            chunk_s=self.chunk_s,
                            overlap_s=self.overlap_s,
                            ffmpeg=ff,
                        )
                    else:
                        up.upscale(cur, dst, target_wh=self.target_wh, ffmpeg=ff)
                else:
                    assert self.interpolator is not None
                    self.interpolator.interpolate(cur, dst, target_fps=self.target_fps, ffmpeg=ff)
                stages.append(dst)
                cur = dst
        return out_p


# 分段的触发阈值：短于这个时长的片子整段跑，分段带来的接缝风险不值当。
CHUNK_THRESHOLD_S = 20.0


def plan_postprocess(
    delivery: DeliverySpec,
    available_hardware: Hardware | None = None,
    *,
    source_wh: tuple[int, int] | None = None,
    source_fps: int | None = None,
    duration_s: float = 0.0,
    prefer: Literal["auto", "cpu", "quality"] = "auto",
) -> PostPlan:
    """按交付规格和可用硬件选后处理链路，并估算耗时。

    选型顺序是**先看能不能跑，再看画质**：显存不够的引擎连候选都不进，
    因为「排进去再 OOM」在长片产线里的代价是整批重跑。
    所有落选原因都记进 warnings，排期表要能解释为什么没走最好的那条。
    """
    hw = available_hardware or Hardware.detect()
    src_wh = source_wh or tuple(delivery.resolution)
    tgt_wh = tuple(delivery.upscale_to or delivery.resolution)
    src_fps = source_fps or delivery.fps
    tgt_fps = delivery.interpolate_to_fps or src_fps

    plan = PostPlan(
        source_wh=(int(src_wh[0]), int(src_wh[1])),
        target_wh=(int(tgt_wh[0]), int(tgt_wh[1])),
        source_fps=src_fps,
        target_fps=tgt_fps,
        duration_s=duration_s,
        hardware=hw,
    )

    ratio = plan.target_wh[0] / max(plan.source_wh[0], 1)
    if ratio > 1.001:
        up, why = _pick_upscaler(hw, ratio, prefer)
        plan.warnings.extend(why)
        plan.upscaler = up
        frames = int(round(duration_s * src_fps))
        est = up.estimate_s(frames, plan.target_wh)
        plan.steps.append(
            PostStep(
                kind="upscale",
                engine=up.name,
                detail=f"{plan.source_wh[0]}x{plan.source_wh[1]} -> {plan.target_wh[0]}x{plan.target_wh[1]}（{ratio:.2f}x）",
                est_s=round(est, 1),
                est_known=est > 0,
            )
        )
        if not up.spec.temporal:
            plan.warnings.append(
                f"{up.name} 无时域一致性建模，逐帧放大可能闪烁；"
                "长镜头建议改用 seedvr2/flashvsr，或在超分后加 deflicker"
            )
        # 分段的真正理由是显存：扩散类 VSR 把整段序列一次性喂进去，
        # 长度直接决定峰值显存。CPU 基线逐帧流式处理，没有这个问题。
        if duration_s > CHUNK_THRESHOLD_S and up.spec.min_vram_gb > 0:
            plan.chunked = True
            plan.chunk_s = _chunk_seconds(up, src_fps)
            plan.overlap_s = round(max(2.0 / max(src_fps, 1), 0.25), 3)
    elif ratio < 0.999:
        plan.warnings.append(
            f"目标分辨率比源小（{ratio:.2f}x），这是降采样不是超分；"
            "若确实要缩，交给交付编码那步做，后处理链不动"
        )

    if tgt_fps > src_fps:
        interp, why = _pick_interpolator(hw, prefer)
        plan.warnings.extend(why)
        plan.interpolator = interp
        out_frames = int(round(duration_s * tgt_fps))
        est = interp.estimate_s(out_frames, plan.target_wh)
        plan.steps.append(
            PostStep(
                kind="interpolate",
                engine=interp.name,
                detail=f"{src_fps} -> {tgt_fps} fps（{tgt_fps / max(src_fps, 1):.2f}x）",
                est_s=round(est, 1),
                est_known=est > 0,
            )
        )
        if tgt_fps % src_fps != 0 and interp.name == "rife":
            plan.warnings.append(
                f"{src_fps}->{tgt_fps} 不是整数倍，Practical-RIFE 的 --multi 只接受整数；"
                "需先插到更高帧率再用 ffmpeg 抽帧对齐"
            )
    elif tgt_fps < src_fps:
        plan.warnings.append(f"目标帧率 {tgt_fps} 低于源 {src_fps}，插帧步骤跳过（抽帧交给交付编码）")

    log.info("后处理计划：%s（硬件：%s）", plan.describe(), hw.describe())
    return plan


def _chunk_seconds(up: Upscaler, fps: int) -> float:
    """按引擎公布的单次可处理帧数换算分段秒数。

    SeedVR2 的 100 帧来自 README 实测；其它引擎没有公开数字，
    统一退到 8s —— 与本产线的原子镜长度同量级，接缝正好落在镜头边界附近。
    """
    frames = getattr(up, "frames_per_batch", 0)
    if frames and fps > 0:
        return round(frames / fps, 2)
    return 8.0


def _pick_upscaler(hw: Hardware, ratio: float, prefer: str) -> tuple[Upscaler, list[str]]:
    """候选按画质降序排，返回第一个硬件跑得动的。"""
    notes: list[str] = []
    if prefer == "cpu":
        return FFmpegUpscaler(), ["按 prefer=cpu 强制走 CPU 基线"]
    candidates: list[Upscaler] = [SeedVRUpscaler(), ComfyUpscaler(), RealESRGANUpscaler()]
    if prefer == "quality":
        # quality 模式下远端 Comfy 优先级降到最后：远端画质取决于对方工作流，不可控。
        candidates = [SeedVRUpscaler(), RealESRGANUpscaler(), ComfyUpscaler()]
    for c in candidates:
        ok, why = c.available(hw)
        if ok:
            return c, notes
        notes.append(f"跳过 {c.name}：{why}")
    notes.append("所有 GPU 超分均不可用，降级到 ffmpeg CPU 基线（不补细节，4x 会偏软）")
    return FFmpegUpscaler(), notes


def _pick_interpolator(hw: Hardware, prefer: str) -> tuple[Interpolator, list[str]]:
    if prefer == "cpu":
        return FFmpegMinterpolate(), ["按 prefer=cpu 强制走 CPU 基线"]
    rife = RifeInterpolator()
    ok, why = rife.available(hw)
    if ok:
        return rife, []
    return FFmpegMinterpolate(), [f"跳过 rife：{why}；降级到 minterpolate（快速运动会有果冻）"]


# ---------------------------------------------------------------- 分块处理


@dataclass(frozen=True)
class ChunkWindow:
    """一个待处理分段在**原片时间轴**上的区间。

    相邻窗口刻意重叠 overlap_s：这段重叠同时干两件事 ——
    给时域模型提供前文上下文，以及在回接时作为 xfade 的混合区。
    用同一段做两件事是有意的：如果上下文区和混合区分开，
    总时长就不再等于原片时长，而时长漂移在整集拼接后表现为音画渐行渐远。
    """

    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def plan_chunks(
    duration_s: float,
    *,
    chunk_s: float = 8.0,
    overlap_s: float = 0.5,
    min_tail_s: float | None = None,
) -> list[ChunkWindow]:
    """切分窗口。不变量：Σ窗口时长 - (n-1)×overlap_s == duration_s。

    该不变量是整个分段处理正确性的根 —— 回接用 xfade，每个接缝吃掉 overlap_s，
    所以只要窗口按这个规则切，拼回来的总时长就自动等于原片。
    """
    if duration_s <= 0:
        raise ValueError(f"时长必须为正，收到 {duration_s}")
    if chunk_s <= 0:
        raise ValueError(f"chunk_s 必须为正，收到 {chunk_s}")
    if overlap_s < 0 or overlap_s >= chunk_s / 2:
        raise ValueError(f"overlap_s 需落在 [0, chunk_s/2)，收到 {overlap_s}（chunk_s={chunk_s}）")

    # 尾段太短时并进上一段：一个 0.2s 的尾巴既做不出上下文，xfade 也吃不动。
    min_tail = min_tail_s if min_tail_s is not None else max(2 * overlap_s, 1.0)
    if duration_s <= chunk_s + min_tail:
        return [ChunkWindow(0, 0.0, duration_s)]

    bounds = [i * chunk_s for i in range(int(math.ceil(duration_s / chunk_s)))]
    if duration_s - bounds[-1] < min_tail and len(bounds) > 1:
        bounds.pop()
    bounds.append(duration_s)

    return [
        ChunkWindow(i, bounds[i], min(duration_s, bounds[i + 1] + (overlap_s if i + 2 < len(bounds) else 0.0)))
        for i in range(len(bounds) - 1)
    ]


def chunked_process(
    video: str | Path,
    fn: Callable[[Path, Path], object],
    out: str | Path,
    *,
    chunk_s: float = 8.0,
    overlap_s: float = 0.5,
    ffmpeg: FFmpeg | None = None,
    workdir: str | Path | None = None,
    duration_tolerance_s: float = 0.25,
) -> Path:
    """把长视频切段、逐段调 fn、再无缝拼回。

    fn(输入分段, 输出分段) 必须**保持时长**（超分和插帧都满足，剪辑类操作不满足）；
    时长变了就直接报错，因为漂移会一路累积到整集音画不同步，越晚发现越贵。

    接缝处理：overlap_s > 0 时相邻分段按 overlap_s 做 xfade 而不是硬接。
    硬接在超分里一定看得见 —— 两段各自独立推理，边界两帧的锐度/亮度必然有落差，
    表现为规律性的「一跳一跳」。xfade 把落差摊到 overlap_s 上，肉眼就抓不住了。
    """
    ff = ffmpeg or default_ffmpeg()
    src = Path(video)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    info = ff.probe(src)
    if not info.has_video:
        raise ValueError(f"{src} 没有视频流，无法分段处理")
    windows = plan_chunks(info.duration_s, chunk_s=chunk_s, overlap_s=overlap_s)
    log.info("分段处理 %s：%.2fs 切成 %d 段（chunk=%.1fs overlap=%.2fs）",
             src.name, info.duration_s, len(windows), chunk_s, overlap_s)

    if len(windows) == 1:
        fn(src, out_p)
        return out_p

    tmp_ctx = None
    if workdir:
        work = Path(workdir)
        work.mkdir(parents=True, exist_ok=True)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="longfilm_chunk_")
        work = Path(tmp_ctx.name)

    try:
        done: list[Path] = []
        for w in windows:
            raw = work / f"in_{w.index:04d}.mp4"
            # -ss 放在 -i 前用关键帧快速定位，再配 -t 重编码保证帧级精度；
            # 分段一律去掉音轨：音频重采样在每个接缝都会引入一帧静音，
            # 最后从原片整轨 remux 回来反而更稳。
            ff.run(
                [
                    "-y", "-ss", f"{w.start_s:.4f}", "-t", f"{w.duration_s:.4f}", "-i", str(src),
                    "-an", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-fps_mode", "cfr", str(raw),
                ],
                timeout=1800.0,
            )
            cut = ff.probe(raw).duration_s
            processed = work / f"out_{w.index:04d}.mp4"
            fn(raw, processed)
            got = ff.probe(processed).duration_s
            if abs(got - cut) > duration_tolerance_s:
                raise ValueError(
                    f"第 {w.index} 段处理后时长从 {cut:.3f}s 变成 {got:.3f}s；"
                    "chunked_process 要求 fn 保持时长，否则整集音画会漂"
                )
            done.append(processed)

        clips = [
            ClipSpec(
                path=str(p),
                transition_in=Transition.CUT if i == 0 or overlap_s <= 0 else Transition.DISSOLVE,
                transition_s=overlap_s,
                shot_id=f"chunk{i:04d}",
            )
            for i, p in enumerate(done)
        ]
        joined = work / "joined.mp4"
        if overlap_s <= 0:
            concat_hard(clips, joined, ffmpeg=ff)
        else:
            concat_with_transitions(clips, out=joined, transition_s=overlap_s, ffmpeg=ff)

        if info.has_audio:
            # 画面时长已保证不变，所以原片音轨可以整条贴回来，不需要逐段对齐。
            ff.run(
                [
                    "-y", "-i", str(joined), "-i", str(src),
                    "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest", str(out_p),
                ],
                timeout=1800.0,
            )
        else:
            shutil.move(str(joined), str(out_p))
        return out_p
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


# ---------------------------------------------------------------- 自测


def _mk_clip(ff: FFmpeg, path: Path, *, dur: float, size: str, fps: int, audio: bool) -> Path:
    """造一段带运动和时间码的测试片。

    testsrc2 每帧都在变（有滚动条和计数器），超分/插帧的接缝和丢帧一眼可见，
    比纯色或静止图有用得多。
    """
    args = ["-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}:duration={dur}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}"]
    args += ["-c:v", "libx264", "-crf", "20", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(path)]
    ff.run(args, timeout=300.0)
    return path


def _selftest() -> None:
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ff = default_ffmpeg()

    # 1) 硬件探测不依赖任何重包，在无卡机器上必须给出 has_cuda=False 而不是崩。
    hw = Hardware.detect()
    print(f"[1] 硬件：{hw.describe()}")
    assert hw.cpu_cores >= 1
    no_gpu = Hardware(has_cuda=False, cpu_cores=4, ram_gb=3.0)

    # 2) 引擎事实登记表的自洽性
    for name, spec in {**VSR_ENGINES, **VFI_ENGINES}.items():
        assert spec.key == name, f"{name} 的 key 与字典键不一致"
        assert spec.repo, f"{name} 缺仓库出处"
        assert spec.verified_on == "2026-09-19"
        if spec.min_vram_gb == 0 and spec.key not in {"ffmpeg", "minterpolate", "comfy"}:
            assert "未查证" in spec.note, f"{name} 显存为 0 却没说明是未查证"
    print(f"[2] 引擎登记表 {len(VSR_ENGINES)} 个超分 + {len(VFI_ENGINES)} 个插帧，出处齐全")

    # 3) 无 GPU 时 GPU 引擎必须给清晰报错，而不是 import 崩或静默降级
    for eng in (SeedVRUpscaler(), RealESRGANUpscaler()):
        ok, why = eng.available(no_gpu)
        assert not ok and "显存" in why, why
        try:
            eng.require(no_gpu)
            raise AssertionError(f"{eng.name} 在无 GPU 时应抛 HardwareUnavailable")
        except HardwareUnavailable as e:
            assert eng.spec.repo in str(e)
    comfy = ComfyUpscaler(base_url="")
    assert comfy.available(no_gpu)[0] is False
    assert ComfyUpscaler(base_url="http://gpu-box:8188").available(no_gpu)[0] is True
    print("[3] GPU 引擎在无卡环境报错清晰，Comfy 按 COMFY_URL 判定可用性")

    # 4) 计划：无 GPU 必须降级到 CPU 基线，并把落选原因写进 warnings
    d = DeliverySpec(resolution=(1280, 720), upscale_to=(1920, 1080), fps=24, interpolate_to_fps=48)
    plan = plan_postprocess(d, no_gpu, duration_s=300.0)
    print(f"[4] {plan.describe()}")
    for w in plan.warnings:
        print(f"    warn: {w}")
    assert [s.engine for s in plan.steps] == ["ffmpeg", "minterpolate"], plan.steps
    assert plan.chunked is False, "CPU 基线是流式处理，不该被要求分段"
    assert plan.est_total_s > 0 and plan.est_reliable
    assert any("降级" in w for w in plan.warnings)

    big = Hardware(has_cuda=True, vram_gb=80.0, gpu_count=1, gpu_name="H100", cpu_cores=64, ram_gb=512.0)
    plan_gpu = plan_postprocess(d, big, duration_s=300.0)
    print(f"[4b] {plan_gpu.describe()}")
    assert plan_gpu.steps[0].engine == "seedvr2"
    assert plan_gpu.chunked and abs(plan_gpu.chunk_s - 100 / 24) < 0.05, plan_gpu.chunk_s
    assert plan_gpu.est_reliable is False, "SeedVR2 吞吐未查证，不该假装能估时间"
    assert plan_gpu.steps[1].engine == "rife"

    noop = plan_postprocess(DeliverySpec(resolution=(1920, 1080), upscale_to=None, fps=24), no_gpu, duration_s=10.0)
    assert noop.noop and noop.describe().endswith("无需后处理")
    print("[4c] 目标=源分辨率时计划为空")

    # 5) 分段窗口的时长不变量
    for dur, ck, ov in [(30.0, 8.0, 0.5), (100.0, 8.0, 0.5), (8.3, 8.0, 0.5), (3.0, 8.0, 0.5), (61.0, 20.0, 2.0)]:
        ws = plan_chunks(dur, chunk_s=ck, overlap_s=ov)
        total = sum(w.duration_s for w in ws) - (len(ws) - 1) * ov
        assert abs(total - dur) < 1e-6, f"dur={dur} ck={ck}: 不变量破了，{total} != {dur}"
        assert ws[0].start_s == 0.0 and abs(ws[-1].end_s - dur) < 1e-9
        for a, b in zip(ws, ws[1:]):
            assert b.start_s < a.end_s or ov == 0, "相邻窗口必须重叠"
    print("[5] plan_chunks 时长不变量在 5 组参数下成立")

    with tempfile.TemporaryDirectory(prefix="longfilm_upscale_test_") as td:
        tmp = Path(td)

        # 6) CPU 超分真跑：本机唯一能验证的路径，demo 就靠它
        src = _mk_clip(ff, tmp / "src.mp4", dur=6.0, size="640x360", fps=24, audio=True)
        up = FFmpegUpscaler(flags="lanczos", sharpen=0.6, preset="ultrafast")
        t0 = time.monotonic()
        dst = up.upscale(src, tmp / "up.mp4", target_wh=(1280, 720))
        took = time.monotonic() - t0
        i_src, i_dst = ff.probe(src), ff.probe(dst)
        print(f"[6] CPU 超分 {i_src.width}x{i_src.height} -> {i_dst.width}x{i_dst.height}，"
              f"{i_dst.duration_s:.2f}s，实耗 {took:.1f}s（{i_dst.n_frames / max(took, 1e-9):.1f} 帧/秒）")
        assert i_dst.size == (1280, 720)
        assert abs(i_dst.duration_s - i_src.duration_s) < 0.2
        assert i_dst.has_audio, "音轨必须原样带过去"

        # 7) CPU 插帧真跑
        vfi = FFmpegMinterpolate(quality="fast", preset="ultrafast")
        small = _mk_clip(ff, tmp / "small.mp4", dur=2.0, size="320x180", fps=24, audio=False)
        ip = vfi.interpolate(small, tmp / "ip.mp4", target_fps=48)
        i_ip = ff.probe(ip)
        print(f"[7] CPU 插帧 24 -> {i_ip.fps:.0f} fps，帧数 {ff.probe(small).n_frames} -> {i_ip.n_frames}")
        assert abs(i_ip.fps - 48) < 0.5
        assert abs(i_ip.duration_s - 2.0) < 0.2

        # 8) 分段处理真跑：切 3 段 + xfade 回接，总时长必须还原
        chunked_src = _mk_clip(ff, tmp / "long.mp4", dur=9.0, size="320x180", fps=24, audio=True)
        calls: list[tuple[str, str]] = []

        def _work(a: Path, b: Path) -> None:
            calls.append((a.name, b.name))
            FFmpegUpscaler(sharpen=0.4, preset="ultrafast").upscale(a, b, target_wh=(640, 360), ffmpeg=ff)

        joined = chunked_process(
            chunked_src, _work, tmp / "chunked.mp4", chunk_s=3.0, overlap_s=0.4, ffmpeg=ff
        )
        i_join = ff.probe(joined)
        print(f"[8] 分段处理：{len(calls)} 段，输出 {i_join.width}x{i_join.height} "
              f"{i_join.duration_s:.2f}s（原片 9.00s）")
        assert len(calls) == 3, calls
        assert i_join.size == (640, 360)
        assert abs(i_join.duration_s - 9.0) < 0.35, f"分段回接后时长漂了：{i_join.duration_s}"
        assert i_join.has_audio

        # 9) fn 改变时长时必须炸出来，不能让漂移流到下游
        def _bad(a: Path, b: Path) -> None:
            ff.run(["-y", "-i", str(a), "-t", "0.5", "-c", "copy", str(b)], timeout=120.0)

        try:
            chunked_process(chunked_src, _bad, tmp / "bad.mp4", chunk_s=3.0, overlap_s=0.4, ffmpeg=ff)
            raise AssertionError("fn 缩短了时长，chunked_process 应当报错")
        except ValueError as e:
            assert "保持时长" in str(e), e
        print("[9] fn 破坏时长时 chunked_process 正确拒绝")

        # 10) PostPlan.run 端到端
        p = plan_postprocess(
            DeliverySpec(resolution=(320, 180), upscale_to=(640, 360), fps=24),
            no_gpu, source_wh=(320, 180), duration_s=2.0, prefer="cpu",
        )
        p.upscaler = FFmpegUpscaler(sharpen=0.4, preset="ultrafast")
        final = p.run(small, tmp / "final.mp4", ffmpeg=ff)
        assert ff.probe(final).size == (640, 360)
        print(f"[10] PostPlan.run 端到端通过：{p.describe()}")

    print("upscale 自测全部通过")


if __name__ == "__main__":
    _selftest()
