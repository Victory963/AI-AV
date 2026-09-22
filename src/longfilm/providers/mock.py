"""离线 mock provider —— 整条产线的 CI 与 demo 底座。

本机没有 GPU、没有任何厂商密钥，端到端演示全靠它。所以 mock 不是「返回假 URL」，
而是真的用 ffmpeg lavfi 合成一段 mp4：时长/分辨率/fps 严格按请求走，
seed 决定配色、camera_move 决定画面怎么动、shot_size 决定主体多大、
首帧图存在时第 0 帧就是那张图、渲完自动导出尾帧 PNG。

这样做的理由：假 URL 能骗过 router，骗不过拼接器、QC 和首尾帧锁戏链。
只有产出真素材，下游模块才是被真正跑通的，而不是被 mock 掉的。

故障注入是**确定性**的（由 idempotency_key 哈希决定），
否则 router 的重试/降级测试无法复现。
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._fonts import cjk_font
from .._proc import popen as _sp_popen  # noqa: F401
from .._proc import run as _sp_run
from ..schema import CameraMove, ShotSize
from .base import (
    REGISTRY,
    Capabilities,
    FailureKind,
    GenRequest,
    GenResult,
    JobStatus,
    ProviderError,
    VideoProvider,
)

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def find_ffmpeg() -> str:
    """定位 ffmpeg。

    优先仓库自带的静态二进制：它的滤镜集是已知的（无 drawtext，有 libass/zoompan），
    系统 ffmpeg 版本不可控，渲染结果会随机器漂移，CI 就没法比对了。
    """
    env = os.environ.get("LONGFILM_FFMPEG")
    if env and Path(env).exists():
        return env
    local = _REPO_ROOT / "bin" / "ffmpeg"
    if local.exists():
        return str(local)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    found = shutil.which("ffmpeg")
    if not found:
        raise ProviderError(
            FailureKind.BAD_REQUEST,
            "找不到 ffmpeg：请把静态二进制放到 <仓库根>/bin/ffmpeg，"
            "或设置环境变量 LONGFILM_FFMPEG 指向它",
        )
    return found


# ---------------------------------------------------------------- 确定性随机


def _unit(key: str, salt: str) -> float:
    """把任意键映射到 [0,1)。

    用 sha256 而不是 random：故障注入必须可复现，
    同一个 idempotency_key 在任何机器、任何进程里都要得到同一个结果。
    """
    h = hashlib.sha256(f"{salt}|{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _palette(seed: int) -> tuple[str, str, str, str]:
    """seed → (底色, 中间色, 高光色, 主体色)。HSV 取色保证任意 seed 都不脏。

    返回 ffmpeg 认的 0xRRGGBB 字符串。
    """
    import colorsys

    base_h = _unit(str(seed), "hue")
    out: list[str] = []
    # 三段背景色沿色相轮小幅偏移 + 明度递增，得到有层次的渐变而不是糊成一团
    for dh, s, v in ((0.0, 0.55, 0.22), (0.08, 0.60, 0.48), (0.16, 0.45, 0.82)):
        r, g, b = colorsys.hsv_to_rgb((base_h + dh) % 1.0, s, v)
        out.append(f"0x{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    # 主体取补色，保证在任何背景上都跳得出来
    r, g, b = colorsys.hsv_to_rgb((base_h + 0.5) % 1.0, 0.75, 0.98)
    out.append(f"0x{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}")
    return out[0], out[1], out[2], out[3]


# ---------------------------------------------------------------- 镜头语法解析

# 景别 → 主体占画面短边的比例。ECU 几乎糊满画面，ELS 只剩一个点。
_SHOT_SCALE: dict[str, float] = {
    ShotSize.ECU.value: 0.92,
    ShotSize.CU.value: 0.70,
    ShotSize.MCU.value: 0.56,
    ShotSize.MS.value: 0.42,
    ShotSize.MLS.value: 0.32,
    ShotSize.FS.value: 0.25,
    ShotSize.LS.value: 0.16,
    ShotSize.ELS.value: 0.08,
    ShotSize.OTS.value: 0.50,
    ShotSize.POV.value: 0.62,
    ShotSize.INSERT.value: 0.66,
    ShotSize.TWO.value: 0.44,
}


def _detect_enum(text: str, values: list[str]) -> str | None:
    """从提示词里认出镜头语法术语。

    GenRequest 没有 shot_size/camera_move 的结构化字段（camera_move 是自由字符串），
    但提示词就是用 ShotSize/CameraMove 的枚举值原文拼出来的 ——
    所以从正文里反解是可靠的。优先匹配长串，免得 "close-up" 吃掉 "extreme close-up"。
    """
    low = text.lower()
    hits = [v for v in values if v.lower() in low]
    return max(hits, key=len) if hits else None


def resolve_shot_scale(req: GenRequest) -> float:
    raw = req.extra.get("shot_size") or _detect_enum(req.prompt, list(_SHOT_SCALE))
    return _SHOT_SCALE.get(str(raw or ""), _SHOT_SCALE[ShotSize.MS.value])


def resolve_camera_move(req: GenRequest) -> str:
    vals = [m.value for m in CameraMove]
    raw = req.camera_move or req.extra.get("camera_move") or _detect_enum(req.prompt, vals)
    return str(raw or CameraMove.STATIC.value)


# ---------------------------------------------------------------- 运镜 → 滤镜

# 每种运镜给一组 zoompan 表达式。zoompan 能同时做缩放和平移，
# 用一个滤镜覆盖推拉摇移，比 crop+scale 的动态尺寸链稳（后者会触发 scale 重初始化）。
# 表达式里 {n} 会被替换成总帧数，on 是输出帧号。
_CX = "iw/2-(iw/zoom/2)"          # 水平居中
_CY = "ih/2-(ih/zoom/2)"          # 垂直居中
_MAXX = "(iw-iw/zoom)"            # x 的最大可平移量
_MAXY = "(ih-ih/zoom)"

_CAMERA_RECIPES: dict[str, tuple[str, str, str]] = {
    CameraMove.STATIC.value: ("1", _CX, _CY),
    CameraMove.DOLLY_IN.value: ("1+0.38*on/{n}", _CX, _CY),
    CameraMove.ZOOM_IN.value: ("1+0.30*on/{n}", _CX, _CY),
    CameraMove.DOLLY_OUT.value: ("1.38-0.38*on/{n}", _CX, _CY),
    CameraMove.ZOOM_OUT.value: ("1.30-0.30*on/{n}", _CX, _CY),
    # 推拉变焦：一边推近一边把主体往边上甩，做出 vertigo 的滑动感
    CameraMove.PUSH_PULL.value: ("1+0.45*on/{n}", f"{_MAXX}*(0.5+0.35*on/{{n}})", _CY),
    CameraMove.PAN_L.value: ("1.28", f"{_MAXX}*(1-on/{{n}})", _CY),
    CameraMove.PAN_R.value: ("1.28", f"{_MAXX}*on/{{n}}", _CY),
    CameraMove.TRUCK_L.value: ("1.22", f"{_MAXX}*(1-on/{{n}})", _CY),
    CameraMove.TRUCK_R.value: ("1.22", f"{_MAXX}*on/{{n}}", _CY),
    CameraMove.TILT_U.value: ("1.28", _CX, f"{_MAXY}*(1-on/{{n}})"),
    CameraMove.TILT_D.value: ("1.28", _CX, f"{_MAXY}*on/{{n}}"),
    CameraMove.PEDESTAL_U.value: ("1.22", _CX, f"{_MAXY}*(1-on/{{n}})"),
    CameraMove.PEDESTAL_D.value: ("1.22", _CX, f"{_MAXY}*on/{{n}}"),
    # 升降：边升边拉开，模拟升起后视野变大
    CameraMove.CRANE.value: ("1.45-0.25*on/{n}", _CX, f"{_MAXY}*(1-on/{{n}})"),
    # 环绕：横向正弦走一圈，配一点点呼吸式缩放
    CameraMove.ORBIT_L.value: (
        "1.30+0.04*sin(2*PI*on/{n})",
        f"{_MAXX}*(0.5-0.5*sin(2*PI*on/{{n}}))",
        _CY,
    ),
    CameraMove.ORBIT_R.value: (
        "1.30+0.04*sin(2*PI*on/{n})",
        f"{_MAXX}*(0.5+0.5*sin(2*PI*on/{{n}}))",
        _CY,
    ),
    # 手持：两个不整除的频率叠加，避免抖动看起来是周期性的假抖
    CameraMove.HANDHELD.value: (
        "1.12",
        f"{_MAXX}*(0.5+0.5*sin(2*PI*on*7/{{n}}))",
        f"{_MAXY}*(0.5+0.5*sin(2*PI*on*11/{{n}}+1.1))",
    ),
    CameraMove.STEADICAM.value: (
        "1.16",
        f"{_MAXX}*(0.15+0.7*on/{{n}})",
        f"{_MAXY}*(0.5+0.12*sin(2*PI*on*3/{{n}}))",
    ),
    # 甩镜：三次方缓动，中段极快、两头几乎静止
    CameraMove.WHIP.value: ("1.25", f"{_MAXX}*pow(on/{{n}},3)", _CY),
}


def camera_filter(move: str, w: int, h: int, fps: int, frames: int) -> str:
    """运镜 → zoompan 滤镜串。未知运镜退回静止，并记一条日志。"""
    recipe = _CAMERA_RECIPES.get(move)
    if recipe is None:
        log.info("mock 不认识运镜 %r，按静止处理", move)
        recipe = _CAMERA_RECIPES[CameraMove.STATIC.value]
    n = max(frames - 1, 1)
    z, x, y = (part.format(n=n) for part in recipe)
    # d=1 + fps 让 zoompan 逐帧处理输入而不是在单帧上做 ken-burns
    return f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={w}x{h}:fps={fps}"


# ---------------------------------------------------------------- Pillow 贴图


def _load_font(px: int):
    from PIL import ImageFont

    p = cjk_font()
    if p is not None:
        return ImageFont.truetype(str(p), px)
    log.warning("未找到中文字体，水印将退回位图字体，中文会显示为方框（见 assets/fonts/README.md）")
    return ImageFont.load_default()


def render_overlay_png(path: Path, w: int, h: int, title: str, lines: list[str]) -> Path:
    """用 Pillow 画水印 PNG。

    本机 ffmpeg 没有 drawtext 滤镜，所以文字一律先出 RGBA 图再 overlay。
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = max(8, w // 60)

    title_font = _load_font(max(16, h // 12))
    info_font = _load_font(max(11, h // 34))

    # 顶部压一条半透明黑带，保证任何背景上文字都可读
    band = max(24, h // 9)
    d.rectangle([0, 0, w, band], fill=(0, 0, 0, 130))
    d.text((pad, pad // 2), title, font=title_font, fill=(255, 255, 255, 235))

    y = h - pad - len(lines) * (info_font.size + 4)
    d.rectangle([0, y - pad // 2, w, h], fill=(0, 0, 0, 120))
    for line in lines:
        d.text((pad, y), line, font=info_font, fill=(235, 235, 235, 220))
        y += info_font.size + 4

    # 取景框角标：让 QC 一眼看出这是 mock 素材，不会误当成真渲染
    c = max(10, w // 40)
    for (x0, y0, x1, y1) in (
        (pad, band + pad, pad + c, band + pad),
        (pad, band + pad, pad, band + pad + c),
        (w - pad - c, h - band - pad, w - pad, h - band - pad),
        (w - pad, h - band - pad - c, w - pad, h - band - pad),
    ):
        d.line([x0, y0, x1, y1], fill=(255, 255, 255, 180), width=max(1, w // 400))

    img.save(path)
    return path


def render_subject_png(path: Path, size: int, color: str) -> Path:
    """画「主体」精灵：一个带光晕的椭圆。

    用 Pillow 出图而不是 ffmpeg 的 geq —— geq 逐像素求值，大尺寸主体会把
    单镜渲染拖到十几秒，而 mock 要在 CI 里跑几十个镜头。
    """
    from PIL import Image, ImageDraw, ImageFilter

    rgb = tuple(int(color[2 + i * 2 : 4 + i * 2], 16) for i in range(3))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, size - 1, size - 1], fill=(*rgb, 210))
    img = img.filter(ImageFilter.GaussianBlur(max(1, size // 40)))
    d = ImageDraw.Draw(img)
    inset = size // 5
    d.ellipse([inset, inset, size - 1 - inset, size - 1 - inset], fill=(255, 255, 255, 120))
    img.save(path)
    return path


# ---------------------------------------------------------------- 作业状态


@dataclass
class _Job:
    job_id: str
    req: GenRequest
    status: JobStatus = JobStatus.PENDING
    result: GenResult | None = None
    started_at: float = field(default_factory=time.monotonic)
    submitted_at: float = field(default_factory=time.time)
    thread: threading.Thread | None = None


class MockProvider(VideoProvider):
    """离线假引擎，产出真 mp4。

    故障注入：
      fail_rate       —— 落 FAILED + SERVER（可重试，测退避）
      moderation_rate —— 落 REJECTED + MODERATION（不可重试，测换引擎）
    两者都由 idempotency_key 哈希决定，同一工单永远得到同一结果。
    另外 content_rating == "blocked" 一律拒单 —— 这是产线的内容门禁，不是随机故障。
    """

    def __init__(
        self,
        name: str = "mock",
        *,
        fail_rate: float = 0.0,
        moderation_rate: float = 0.0,
        latency_s: float = 0.0,
        out_dir: str | Path | None = None,
        native_audio: bool = True,
        quality_tier: int = 2,
        ffmpeg: str | None = None,
        **cfg: Any,
    ):
        if not 0.0 <= fail_rate <= 1.0 or not 0.0 <= moderation_rate <= 1.0:
            raise ValueError("fail_rate / moderation_rate 必须落在 [0,1]")
        self.fail_rate = fail_rate
        self.moderation_rate = moderation_rate
        self.latency_s = max(0.0, latency_s)
        self.native_audio = native_audio
        self._ffmpeg = ffmpeg
        self.out_dir = Path(
            out_dir
            or os.environ.get("LONGFILM_MOCK_DIR")
            or Path(tempfile.gettempdir()) / "longfilm_mock"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, _Job] = {}
        self._idem: dict[str, str] = {}
        self._lock = threading.Lock()

        caps = Capabilities(
            name=name,
            kind="open",                      # 无审核、可本地跑，路由上等同开源侧
            min_duration_s=1.0,
            max_duration_s=60.0,
            duration_steps=(),                # 连续时长：mock 不该成为时长约束的来源
            resolutions=((640, 360), (854, 480), (1280, 720), (1920, 1080)),
            fps_options=(12, 24, 25, 30),
            max_ref_images=9,
            max_ref_videos=3,
            max_ref_audios=3,
            supported_image_roles=frozenset(
                {
                    "identity", "wardrobe", "environment", "style", "prop",
                    "lighting", "composition", "first_frame", "last_frame",
                }
            ),
            supports_first_frame=True,
            supports_last_frame=True,
            supports_extend=True,
            supports_job_continue=True,
            supports_native_audio=native_audio,
            supports_lora=True,
            supports_seed=True,
            supports_negative_prompt=True,
            supports_camera_control=True,
            moderated=False,
            max_concurrency=int(cfg.get("max_concurrency", 8)),
            cost_per_second_usd=0.0,          # 本地渲染不花钱，但 router 要能排序
            typical_latency_s=max(latency_s, 2.0),
            quality_tier=quality_tier,
            notes="ffmpeg lavfi 合成，非真实生成；仅供 CI 与端到端 demo",
        )
        super().__init__(caps, **cfg)

    @property
    def ffmpeg(self) -> str:
        if self._ffmpeg is None:
            self._ffmpeg = find_ffmpeg()
        return self._ffmpeg

    def health(self) -> bool:
        try:
            return Path(self.ffmpeg).exists() or shutil.which(self.ffmpeg) is not None
        except ProviderError:
            return False

    # ------------------------------------------------------------ 提交

    def _inject(self, req: GenRequest) -> tuple[JobStatus, FailureKind, str] | None:
        """确定性故障注入。返回 None 表示放行。"""
        if req.content_rating == "blocked":
            return (
                JobStatus.REJECTED,
                FailureKind.MODERATION,
                "内容分级为 blocked，产线拒单（这是策略门禁，不是随机故障）",
            )
        key = req.idempotency_key or req.shot_id or req.prompt
        if self.moderation_rate and _unit(key, "moderation") < self.moderation_rate:
            return (JobStatus.REJECTED, FailureKind.MODERATION, "mock 注入的审核拒绝")
        if self.fail_rate and _unit(key, "fail") < self.fail_rate:
            return (JobStatus.FAILED, FailureKind.SERVER, "mock 注入的服务端错误")
        return None

    def submit(self, req: GenRequest) -> str:
        if req.duration_s <= 0:
            raise ProviderError(FailureKind.BAD_REQUEST, "duration_s 必须为正")

        with self._lock:
            if req.idempotency_key and req.idempotency_key in self._idem:
                job_id = self._idem[req.idempotency_key]
                log.info("mock 幂等命中 %s -> %s", req.idempotency_key, job_id)
                return job_id
            seq = len(self._jobs)
            digest = _unit(req.idempotency_key or req.shot_id or f"{req.prompt}{seq}", "job")
            job_id = f"mock-{int(digest * 2**40):010x}-{seq:03d}"
            job = _Job(job_id=job_id, req=req)
            self._jobs[job_id] = job
            if req.idempotency_key:
                self._idem[req.idempotency_key] = job_id

        t = threading.Thread(target=self._run, args=(job,), daemon=True, name=f"mock-{job_id}")
        job.thread = t
        t.start()
        return job_id

    def extend(self, req: GenRequest) -> str:
        """接上一个 job 的尾帧继续渲染。

        改写 first_frame_uri 而不是新增字段：下游拿到的仍是一条普通请求，
        首尾帧锁戏链在 mock 上跑通了，换成真 provider 也是同一条链路。
        """
        src = req.first_frame_uri
        if not src and req.extend_from_job:
            with self._lock:
                prev = self._jobs.get(req.extend_from_job)
            if prev is None:
                raise ProviderError(
                    FailureKind.BAD_REQUEST, f"extend 的源任务 {req.extend_from_job} 不在本进程"
                )
            if prev.result is None or not prev.result.last_frame_uri:
                raise ProviderError(
                    FailureKind.BAD_REQUEST,
                    f"源任务 {req.extend_from_job} 还没有尾帧（status={prev.status.value}）",
                )
            src = prev.result.last_frame_uri
        if not src and req.extend_from_video:
            src = str(self._grab_last_frame(Path(req.extend_from_video), self.out_dir))
        if not src:
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                "extend 需要 extend_from_job / extend_from_video / first_frame_uri 三者之一",
            )

        import dataclasses

        return self.submit(dataclasses.replace(req, first_frame_uri=src))

    # ------------------------------------------------------------ 轮询

    def poll(self, job_id: str) -> GenResult:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ProviderError(FailureKind.BAD_REQUEST, f"未知 job_id: {job_id}")
        if job.result is not None:
            return job.result
        # 模拟排队：latency 的前三成算排队，之后算渲染中。
        # router 的超时/进度逻辑需要看到 PENDING→RUNNING 的迁移，不能一上来就 RUNNING。
        elapsed = time.monotonic() - job.started_at
        status = JobStatus.PENDING if elapsed < self.latency_s * 0.3 else JobStatus.RUNNING
        return GenResult(
            job_id=job_id, status=status, provider=self.name, submitted_at=job.submitted_at
        )

    def cancel(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.result is not None:
            return
        job.result = GenResult(
            job_id=job_id,
            status=JobStatus.CANCELLED,
            provider=self.name,
            submitted_at=job.submitted_at,
            finished_at=time.time(),
            message="调用方取消",
        )

    # ------------------------------------------------------------ 渲染

    def _run(self, job: _Job) -> None:
        req = job.req
        if self.latency_s:
            time.sleep(self.latency_s)
        if job.result is not None:      # cancel() 抢先了
            return

        injected = self._inject(req)
        if injected is not None:
            status, kind, msg = injected
            job.result = GenResult(
                job_id=job.job_id,
                status=status,
                provider=self.name,
                failure=kind,
                message=msg,
                submitted_at=job.submitted_at,
                finished_at=time.time(),
            )
            log.info("mock 注入故障 job=%s %s/%s", job.job_id, status.value, kind.value)
            return

        try:
            video, last_frame, audio = self._render(job)
        except Exception as e:                      # noqa: BLE001 —— 渲染失败要变成结果而非炸线程
            log.exception("mock 渲染失败 job=%s", job.job_id)
            job.result = GenResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                provider=self.name,
                failure=FailureKind.SERVER,
                message=f"mock 渲染失败：{e}",
                submitted_at=job.submitted_at,
                finished_at=time.time(),
            )
            return

        job.result = GenResult(
            job_id=job.job_id,
            status=JobStatus.SUCCEEDED,
            provider=self.name,
            video_uri=str(video),
            last_frame_uri=str(last_frame) if last_frame else None,
            audio_uri=str(audio) if audio else None,
            duration_s=req.duration_s,
            cost_usd=0.0,
            submitted_at=job.submitted_at,
            finished_at=time.time(),
            raw={
                "camera_move": resolve_camera_move(req),
                "shot_scale": resolve_shot_scale(req),
                "seed": self._seed_of(req),
            },
        )

    @staticmethod
    def _seed_of(req: GenRequest) -> int:
        """没给 seed 就从工单派生一个，保证同工单配色稳定、跨工单配色不同。"""
        if req.seed is not None:
            return int(req.seed)
        return int(_unit(req.idempotency_key or req.shot_id or req.prompt, "seed") * 2**31)

    def _render(self, job: _Job) -> tuple[Path, Path | None, Path | None]:
        req = job.req
        w, h = self.caps.nearest_resolution(req.resolution)
        fps = min(self.caps.fps_options, key=lambda f: abs(f - req.fps))
        dur = round(max(self.caps.min_duration_s, min(self.caps.max_duration_s, req.duration_s)), 3)
        frames = max(1, int(round(dur * fps)))

        seed = self._seed_of(req)
        c0, c1, c2, accent = _palette(seed)
        move = resolve_camera_move(req)
        scale = resolve_shot_scale(req)

        work = self.out_dir / job.job_id
        work.mkdir(parents=True, exist_ok=True)

        subject_px = max(8, int(min(w, h) * scale))
        subject_png = render_subject_png(work / "subject.png", subject_px, accent)
        overlay_png = render_overlay_png(
            work / "overlay.png",
            w, h,
            req.shot_id or job.job_id,
            [
                f"MOCK RENDER · {self.name} · seed={seed}",
                f"{w}x{h} @ {fps}fps · {dur}s · {frames}f",
                f"move: {move}",
                f"scale: {scale:.2f} · rating: {req.content_rating}",
                (req.prompt or "")[:70],
            ],
        )

        cmd: list[str] = [self.ffmpeg, "-y", "-v", "error", "-nostdin"]
        # 0: 渐变背景。gradients 自带 seed，配色与运动都由它承担
        cmd += [
            "-f", "lavfi",
            "-i", (
                f"gradients=s={w}x{h}:c0={c0}:c1={c1}:c2={c2}:nb_colors=3"
                f":seed={seed % 65536}:d={dur}:r={fps}:speed=0.015"
            ),
        ]
        # 1: 主体精灵  2: 水印
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(dur), "-i", str(subject_png)]
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(dur), "-i", str(overlay_png)]

        idx = 3
        ff_idx: int | None = None
        first_frame = self._resolve_local(req.first_frame_uri)
        if first_frame is not None:
            cmd += ["-loop", "1", "-framerate", str(fps), "-t", str(dur), "-i", str(first_frame)]
            ff_idx, idx = idx, idx + 1

        aud_idx: int | None = None
        audio_src = self._resolve_local(req.audio_uri)
        if audio_src is not None:
            cmd += ["-i", str(audio_src)]
            aud_idx, idx = idx, idx + 1
        elif self.native_audio:
            # 音高跟着 seed 走：端到端 demo 里不同镜的音轨可听地不同
            freq = 180 + (seed % 12) * 40
            cmd += ["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={dur}:sample_rate=48000"]
            aud_idx, idx = idx, idx + 1

        # 主体飘移：即使机位静止，画面也得动，否则看不出 fps/时长是否生效。
        # 主体越大飘得越少，免得糊出画框。
        amp = (1.0 - scale) * 0.35
        vx = f"(W-w)/2+{amp:.3f}*W*sin(2*PI*t/{max(dur / 1.7, 0.7):.3f})"
        vy = f"(H-h)/2+{amp * 0.55:.3f}*H*sin(2*PI*t/{max(dur / 2.3, 0.9):.3f}+1.3)"

        chain = [
            "[0:v]format=rgba,setsar=1[bg]",
            f"[bg][1:v]overlay=x='{vx}':y='{vy}':shortest=1[mg]",
            f"[mg]{camera_filter(move, w, h, fps, frames)}[cam]",
        ]
        last = "cam"
        if ff_idx is not None:
            # 首帧图 alpha 从 1 线性降到 0：t=0 时完全覆盖，所以第 0 帧**就是**这张图，
            # 之后化入合成画面。首尾帧锁戏要的是「第一帧对得上」，不是「整段像它」。
            chain += [
                f"[{ff_idx}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},format=rgba,fade=t=out:st=0:d=0.5:alpha=1[ff]",
                f"[{last}][ff]overlay=0:0:shortest=1[ffo]",
            ]
            last = "ffo"
        chain += [f"[{last}][2:v]overlay=0:0:shortest=1,format=yuv420p[v]"]

        out = work / "render.mp4"
        cmd += ["-filter_complex", ";".join(chain), "-map", "[v]"]
        if aud_idx is not None:
            cmd += ["-map", f"{aud_idx}:a", "-c:a", "aac", "-b:a", "96k", "-ar", "48000"]
        cmd += [
            "-frames:v", str(frames),
            "-r", str(fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-shortest",
            str(out),
        ]

        proc = _sp_run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg 退出码 {proc.returncode}: {proc.stderr.strip()[-600:]}")

        last_frame = None
        # emit_last_frame：默认就导，尾帧是首尾帧锁戏链的输入，
        # 让调用方显式开启只会让链路在某些镜头上悄悄断掉。
        if req.extra.get("emit_last_frame", True):
            last_frame = self._grab_last_frame(out, work)
        return out, last_frame, (audio_src if aud_idx is not None and audio_src else None)

    def _grab_last_frame(self, video: Path, work: Path) -> Path:
        """导出尾帧 PNG。

        用 -sseof 从末尾倒着找而不是算帧号：mp4 的实际帧数可能因编码器补帧
        与请求值差 1，按帧号 select 会取空。
        """
        png = work / "last_frame.png"
        cmd = [
            self.ffmpeg, "-y", "-v", "error", "-nostdin",
            "-sseof", "-0.5", "-i", str(video),
            "-update", "1", "-frames:v", "1", str(png),
        ]
        proc = _sp_run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not png.exists():
            raise RuntimeError(f"尾帧导出失败：{proc.stderr.strip()[-400:]}")
        return png

    @staticmethod
    def _resolve_local(uri: str | None) -> Path | None:
        """mock 只吃本地文件。远程 URL 直接忽略而不是下载 ——
        mock 的存在前提就是无网络可用。"""
        if not uri:
            return None
        path = Path(uri[7:] if uri.startswith("file://") else uri)
        if path.exists():
            return path
        log.info("mock 忽略非本地素材 %s（离线环境不下载远程资源）", uri)
        return None


def register_defaults() -> list[MockProvider]:
    """注册三个档位：干净的、抖动的、纯拒单的，够 router 的测试矩阵用。"""
    out = [
        MockProvider("mock", quality_tier=2),
        MockProvider("mock-flaky", fail_rate=0.35, moderation_rate=0.15, quality_tier=2),
        MockProvider("mock-prude", moderation_rate=1.0, quality_tier=3),
    ]
    for p in out:
        REGISTRY.register(p)
    return out


register_defaults()


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    import re

    logging.basicConfig(level=logging.WARNING)
    root = Path(tempfile.mkdtemp(prefix="longfilm_mock_selftest_"))
    p = MockProvider("mock-selftest", out_dir=root, latency_s=0.0)
    assert p.health(), "找不到 ffmpeg，mock 无法工作"
    assert p.caps.moderated is False and p.caps.supports_extend

    def probe(path: str) -> dict[str, Any]:
        """用 ffmpeg 自己探测产物 —— 本机静态包不带 ffprobe，自测不能依赖它。"""
        meta = _sp_run(
            [p.ffmpeg, "-hide_banner", "-i", path], capture_output=True, text=True
        ).stderr
        count = _sp_run(
            [p.ffmpeg, "-v", "error", "-stats", "-i", path, "-map", "0:v:0", "-f", "null", "-"],
            capture_output=True, text=True,
        ).stderr
        wh = re.search(r"Video:.*?, (\d+)x(\d+)", meta)
        hms = re.search(r"Duration: (\d+):(\d+):([\d.]+)", meta)
        frames = re.findall(r"frame=\s*(\d+)", count)
        assert wh and hms and frames, f"探测失败\n{meta}\n{count}"
        return {
            "width": int(wh.group(1)),
            "height": int(wh.group(2)),
            "duration": int(hms.group(1)) * 3600 + int(hms.group(2)) * 60 + float(hms.group(3)),
            "frames": int(frames[-1]),
            "has_audio": "Audio:" in meta,
        }

    # ---- 1. 基本渲染：时长/分辨率/fps 严格按请求走
    req = GenRequest(
        prompt="medium close-up 雨夜天台，男人点燃最后一支烟, camera dollies in",
        duration_s=2.0,
        resolution=(640, 360),
        fps=24,
        seed=1234,
        shot_id="s01_sh001",
        idempotency_key="fp-a",
    )
    jid = p.submit(req)
    mid = p.poll(jid)
    assert mid.status in {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED}
    r = p.wait(jid, timeout_s=180, interval_s=0.2)
    assert r.ok, f"渲染失败：{r.status} {r.message}"
    assert Path(r.video_uri).exists() and Path(r.video_uri).stat().st_size > 2000

    v = probe(r.video_uri)
    assert (v["width"], v["height"]) == (640, 360), v
    assert v["frames"] == 48, f"2s@24fps 应为 48 帧，实得 {v['frames']}"
    assert abs(v["duration"] - 2.0) < 0.15, v
    assert v["has_audio"], "native_audio=True 应带音轨"

    # 运镜与景别确实从提示词里反解出来了
    assert r.raw["camera_move"] == CameraMove.DOLLY_IN.value, r.raw
    assert abs(r.raw["shot_scale"] - _SHOT_SCALE[ShotSize.MCU.value]) < 1e-9, r.raw

    # ---- 2. emit_last_frame：尾帧 PNG 必须落地且尺寸对得上
    assert r.last_frame_uri and Path(r.last_frame_uri).exists()
    from PIL import Image

    with Image.open(r.last_frame_uri) as im:
        assert im.size == (640, 360), im.size

    # ---- 3. 幂等：同 key 不产生第二个 job
    assert p.submit(req) == jid, "幂等键命中必须返回同一个 job_id"

    # ---- 4. seed 决定配色：换 seed 必须换出不同的画面
    def render_seed(seed: int, key: str) -> Path:
        q = GenRequest(
            prompt="full shot 走廊尽头, static locked-off camera",
            duration_s=1.0, resolution=(640, 360), fps=12, seed=seed,
            shot_id=f"seed{seed}", idempotency_key=key,
        )
        rr = p.wait(p.submit(q), timeout_s=180, interval_s=0.2)
        assert rr.ok, rr.message
        return Path(rr.last_frame_uri)

    a, b = render_seed(11, "fp-s11"), render_seed(99, "fp-s99")
    with Image.open(a) as ia, Image.open(b) as ib:
        pa, pb = ia.convert("RGB").getpixel((40, 300)), ib.convert("RGB").getpixel((40, 300))
    assert pa != pb, f"不同 seed 应产出不同配色，实得 {pa} vs {pb}"

    # ---- 5. camera_move 真的改变画面：静止 vs 推近，同 seed 下尾帧必须不同
    def render_move(move: str, key: str) -> Path:
        q = GenRequest(
            prompt="full shot 走廊尽头", camera_move=move,
            duration_s=1.0, resolution=(640, 360), fps=12, seed=7,
            shot_id=key, idempotency_key=key,
        )
        rr = p.wait(p.submit(q), timeout_s=180, interval_s=0.2)
        assert rr.ok, rr.message
        return Path(rr.last_frame_uri)

    still = render_move(CameraMove.STATIC.value, "fp-static")
    pushed = render_move(CameraMove.DOLLY_IN.value, "fp-dolly")
    panned = render_move(CameraMove.PAN_R.value, "fp-pan")
    with Image.open(still) as i1, Image.open(pushed) as i2, Image.open(panned) as i3:
        s1, s2, s3 = (i.convert("RGB").tobytes() for i in (i1, i2, i3))
    assert s1 != s2, "dolly in 的尾帧不该与静止机位相同"
    assert s1 != s3 and s2 != s3, "摇镜应产出与推镜不同的画面"

    # ---- 6. 首帧锁戏：第 0 帧必须就是给定的首帧图
    ffp = root / "given_first.png"
    Image.new("RGB", (640, 360), (0, 200, 60)).save(ffp)
    q = GenRequest(
        prompt="close-up 门把手", duration_s=1.5, resolution=(640, 360), fps=24, seed=5,
        first_frame_uri=str(ffp), shot_id="s01_sh009", idempotency_key="fp-ff",
    )
    rf = p.wait(p.submit(q), timeout_s=180, interval_s=0.2)
    assert rf.ok, rf.message
    frame0 = root / "frame0.png"
    _sp_run(
        [p.ffmpeg, "-y", "-v", "error", "-i", rf.video_uri, "-frames:v", "1", str(frame0)],
        check=True,
    )
    with Image.open(frame0) as im0:
        # 取水印带之外的中心点比色；h264 有量化误差，允许每通道差 12
        px = im0.convert("RGB").getpixel((320, 200))
    assert all(abs(c - t) <= 12 for c, t in zip(px, (0, 200, 60))), f"首帧未锁住，实得 {px}"

    # ---- 7. extend：接上一个 job 的尾帧继续渲
    ext = GenRequest(
        prompt="close-up 门把手转动", duration_s=1.5, resolution=(640, 360), fps=24, seed=5,
        extend_from_job=rf.job_id, shot_id="s01_sh010", idempotency_key="fp-ext",
    )
    re_ = p.wait(p.extend(ext), timeout_s=180, interval_s=0.2)
    assert re_.ok, re_.message
    assert re_.job_id != rf.job_id and Path(re_.video_uri).exists()
    # 续写镜的首帧应当贴近上一镜的尾帧
    f0 = root / "ext_frame0.png"
    _sp_run(
        [p.ffmpeg, "-y", "-v", "error", "-i", re_.video_uri, "-frames:v", "1", str(f0)], check=True
    )
    with Image.open(f0) as ia, Image.open(rf.last_frame_uri) as ib:
        ca = ia.convert("RGB").getpixel((320, 200))
        cb = ib.convert("RGB").getpixel((320, 200))
    assert all(abs(x - y) <= 24 for x, y in zip(ca, cb)), f"extend 未接上尾帧：{ca} vs {cb}"

    # extend 缺来源要明确报错
    try:
        p.extend(GenRequest(prompt="无来源", duration_s=1.0))
        raise AssertionError("extend 缺来源必须抛 BAD_REQUEST")
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST

    # ---- 8. 故障注入的确定性
    flaky = MockProvider("mock-flaky-test", fail_rate=0.5, moderation_rate=0.25, out_dir=root)
    kinds: dict[str, tuple[JobStatus, FailureKind]] = {}
    for i in range(24):
        k = f"fp-inject-{i}"
        rr = flaky.wait(
            flaky.submit(GenRequest(prompt="static locked-off camera 空镜", duration_s=1.0,
                                    resolution=(640, 360), fps=12, idempotency_key=k)),
            timeout_s=180, interval_s=0.2,
        )
        kinds[k] = (rr.status, rr.failure)
    rejected = [k for k, (s, _) in kinds.items() if s is JobStatus.REJECTED]
    failed = [k for k, (s, _) in kinds.items() if s is JobStatus.FAILED]
    okays = [k for k, (s, _) in kinds.items() if s is JobStatus.SUCCEEDED]
    assert rejected and failed and okays, (len(rejected), len(failed), len(okays))
    assert all(kinds[k][1] is FailureKind.MODERATION for k in rejected)
    assert all(kinds[k][1] is FailureKind.SERVER for k in failed)
    # 同一批键在一个全新实例上必须复现同一组结果
    flaky2 = MockProvider("mock-flaky-test2", fail_rate=0.5, moderation_rate=0.25, out_dir=root)
    for k in list(kinds)[:8]:
        inj = flaky2._inject(GenRequest(prompt="x", duration_s=1.0, idempotency_key=k))
        want = kinds[k]
        got = (inj[0], inj[1]) if inj else (JobStatus.SUCCEEDED, FailureKind.NONE)
        assert got == want, f"故障注入不可复现：{k} {got} != {want}"

    # ---- 9. 内容门禁：blocked 一律拒单，与随机率无关
    clean = MockProvider("mock-clean", out_dir=root)
    rb = clean.wait(
        clean.submit(GenRequest(prompt="x", duration_s=1.0, content_rating="blocked",
                                idempotency_key="fp-blocked")),
        timeout_s=30, interval_s=0.1,
    )
    assert rb.status is JobStatus.REJECTED and rb.failure is FailureKind.MODERATION
    assert rb.failure.should_failover and not rb.failure.retryable

    # ---- 10. 注册表
    assert "mock" in REGISTRY and "mock-flaky" in REGISTRY
    assert REGISTRY.get("mock").caps.supports_extend
    assert "mock" in [x.name for x in REGISTRY.capable(duration_s=8.0, unmoderated_only=True)]

    print("mock 自测通过：")
    print(f"  产物目录 {root}")
    print(f"  基准镜 {r.video_uri} ({Path(r.video_uri).stat().st_size} B, "
          f"{v['width']}x{v['height']} {v['frames']}f, 音轨={'有' if v['has_audio'] else '无'})")
    print(f"  尾帧 {r.last_frame_uri}")
    print(f"  seed 配色 seed11={pa} seed99={pb}（不同 → 生效）")
    print(f"  首帧锁戏中心像素 {px}（目标 (0, 200, 60) → 锁住）")
    print(f"  extend 接帧 {ca} vs 上一镜尾帧 {cb}")
    print(f"  故障注入 24 单：成功 {len(okays)} / 失败 {len(failed)} / 拒审 {len(rejected)}，"
          "跨实例复现一致")


if __name__ == "__main__":
    _selftest()
