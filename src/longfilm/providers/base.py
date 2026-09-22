"""引擎 Provider 抽象 —— 双引擎路由与多供应商容灾的地基。

设计要点：
1. Capabilities 是**声明式**的。router 只读 capabilities 做决策，
   绝不在 router 里写 `if provider == "seedance"` 这种硬编码分支 —— 
   换供应商时队列和工单不改。
2. 所有 provider 都是「提交 → 轮询」两段式。视频生成是分钟级任务，
   同步阻塞的接口在长片产线里必然拖垮并发。
3. 失败要分类。可重试的（限流/超时/5xx）和不可重试的（审核拒绝/参数错）
   走完全不同的降级路径：前者重试同一家，后者立刻换引擎。
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from ..schema import RefPack, Shot


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"      # 内容审核拒绝 —— 不可重试，必须换引擎
    CANCELLED = "cancelled"


class FailureKind(str, Enum):
    NONE = "none"
    RATE_LIMIT = "rate_limit"          # 可重试：退避
    TIMEOUT = "timeout"                # 可重试
    SERVER = "server"                  # 可重试
    QUOTA = "quota"                    # 换供应商
    MODERATION = "moderation"          # 换引擎（开源侧）
    BAD_REQUEST = "bad_request"        # 修工单，不重试
    AUTH = "auth"                      # 人工介入
    UNKNOWN = "unknown"

    @property
    def retryable(self) -> bool:
        return self in {FailureKind.RATE_LIMIT, FailureKind.TIMEOUT, FailureKind.SERVER}

    @property
    def should_failover(self) -> bool:
        return self in {
            FailureKind.QUOTA,
            FailureKind.MODERATION,
            FailureKind.AUTH,
            FailureKind.SERVER,
        }


@dataclass(frozen=True)
class Capabilities:
    """供应商能力声明。router 的全部决策依据。"""

    name: str
    kind: str                       # "official" | "open" | "aggregator"

    min_duration_s: float = 4.0
    max_duration_s: float = 15.0
    duration_steps: tuple[float, ...] = ()      # 空 = 连续；否则只能取这些档
    resolutions: tuple[tuple[int, int], ...] = ((1280, 720),)
    fps_options: tuple[int, ...] = (24,)

    max_ref_images: int = 0
    max_ref_videos: int = 0
    max_ref_audios: int = 0
    supported_image_roles: frozenset[str] = frozenset()

    supports_first_frame: bool = False
    supports_last_frame: bool = False
    supports_extend: bool = False          # 官方延长链
    supports_job_continue: bool = False    # 用上一个 job_id 续写
    supports_native_audio: bool = False    # 自带对白声
    supports_lora: bool = False            # 能挂自训 LoRA
    supports_seed: bool = True
    supports_negative_prompt: bool = True
    supports_camera_control: bool = False

    moderated: bool = True                 # 有内容审核
    max_concurrency: int = 4
    cost_per_second_usd: float = 0.0
    typical_latency_s: float = 120.0
    quality_tier: int = 3                  # 1..5，5 = 画质天花板

    notes: str = ""

    def fits_duration(self, d: float) -> bool:
        if not (self.min_duration_s <= d <= self.max_duration_s):
            return False
        if self.duration_steps:
            return any(abs(d - s) < 1e-6 for s in self.duration_steps)
        return True

    def clamp_duration(self, d: float) -> float:
        d = max(self.min_duration_s, min(self.max_duration_s, d))
        if self.duration_steps:
            return min(self.duration_steps, key=lambda s: abs(s - d))
        return d

    def nearest_resolution(self, want: tuple[int, int]) -> tuple[int, int]:
        wa = want[0] / max(want[1], 1)
        return min(
            self.resolutions,
            key=lambda r: (abs(r[0] * r[1] - want[0] * want[1]), abs(r[0] / max(r[1], 1) - wa)),
        )


@dataclass
class GenRequest:
    """一次生成请求。由 router 从 Shot 编译而来，provider 只负责翻译成自家 API。"""

    prompt: str
    duration_s: float
    resolution: tuple[int, int] = (1280, 720)
    fps: int = 24
    negative_prompt: str = ""
    seed: int | None = None
    refs: RefPack = field(default_factory=RefPack)

    first_frame_uri: str | None = None
    last_frame_uri: str | None = None
    extend_from_job: str | None = None
    extend_from_video: str | None = None

    audio_uri: str | None = None          # 音频先行：对白轨驱动
    camera_move: str | None = None
    lora: dict[str, Any] | None = None

    shot_id: str = ""
    content_rating: str = "g"
    idempotency_key: str = ""             # = Shot.fingerprint()，防重复扣费
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenResult:
    job_id: str
    status: JobStatus
    provider: str
    video_uri: str | None = None
    last_frame_uri: str | None = None     # provider 若能直出尾帧就填，省一次解码
    audio_uri: str | None = None
    duration_s: float = 0.0
    cost_usd: float = 0.0
    failure: FailureKind = FailureKind.NONE
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    submitted_at: float = 0.0
    finished_at: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is JobStatus.SUCCEEDED and bool(self.video_uri)

    @property
    def elapsed_s(self) -> float:
        if self.finished_at and self.submitted_at:
            return self.finished_at - self.submitted_at
        return 0.0


class ProviderError(RuntimeError):
    def __init__(self, kind: FailureKind, message: str, raw: dict[str, Any] | None = None):
        super().__init__(message)
        self.kind = kind
        self.raw = raw or {}


class VideoProvider(abc.ABC):
    """所有引擎的统一接口。

    实现者只需保证：submit 返回 job_id，poll 如实映射状态与 FailureKind。
    不要在 provider 内部做重试或降级 —— 那是 router 的职责，
    provider 内部偷偷重试会让成本核算和限流统计全部失真。
    """

    def __init__(self, caps: Capabilities, **cfg: Any):
        self.caps = caps
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.caps.name

    @abc.abstractmethod
    def submit(self, req: GenRequest) -> str:
        """提交任务，返回 job_id。失败抛 ProviderError。"""

    @abc.abstractmethod
    def poll(self, job_id: str) -> GenResult:
        """查询任务状态。不阻塞。"""

    def extend(self, req: GenRequest) -> str:
        """官方延长链。默认不支持。"""
        raise ProviderError(
            FailureKind.BAD_REQUEST, f"{self.name} 不支持 extend（caps.supports_extend=False）"
        )

    def cancel(self, job_id: str) -> None:
        return None

    def health(self) -> bool:
        """容灾探活。router 在降级前调用。"""
        return True

    def wait(
        self,
        job_id: str,
        timeout_s: float = 900.0,
        interval_s: float = 5.0,
        sleeper=time.sleep,
    ) -> GenResult:
        """阻塞等待。仅供 CLI/调试用，队列里走异步 poll。"""
        deadline = time.monotonic() + timeout_s
        while True:
            r = self.poll(job_id)
            if r.status in {
                JobStatus.SUCCEEDED,
                JobStatus.FAILED,
                JobStatus.REJECTED,
                JobStatus.CANCELLED,
            }:
                return r
            if time.monotonic() > deadline:
                r.status = JobStatus.FAILED
                r.failure = FailureKind.TIMEOUT
                r.message = f"等待 {timeout_s}s 超时"
                return r
            sleeper(interval_s)

    def estimate_cost(self, req: GenRequest) -> float:
        return round(self.caps.cost_per_second_usd * req.duration_s, 4)

    def __repr__(self) -> str:
        c = self.caps
        return (
            f"<{type(self).__name__} {c.name} kind={c.kind} "
            f"dur={c.min_duration_s}-{c.max_duration_s}s refs={c.max_ref_images}i/"
            f"{c.max_ref_videos}v/{c.max_ref_audios}a tier={c.quality_tier} "
            f"moderated={c.moderated}>"
        )


class ProviderRegistry:
    """供应商注册表。多供应商容灾的入口 —— 队列只认名字，不认实现。"""

    def __init__(self) -> None:
        self._providers: dict[str, VideoProvider] = {}

    def register(self, p: VideoProvider) -> VideoProvider:
        self._providers[p.name] = p
        return p

    def get(self, name: str) -> VideoProvider:
        if name not in self._providers:
            raise KeyError(f"未注册的 provider: {name}；已注册 {sorted(self._providers)}")
        return self._providers[name]

    def all(self) -> list[VideoProvider]:
        return list(self._providers.values())

    def of_kind(self, kind: str) -> list[VideoProvider]:
        return [p for p in self._providers.values() if p.caps.kind == kind]

    def capable(
        self,
        *,
        duration_s: float | None = None,
        needs_extend: bool = False,
        needs_last_frame: bool = False,
        needs_native_audio: bool = False,
        needs_lora: bool = False,
        unmoderated_only: bool = False,
        min_ref_images: int = 0,
    ) -> list[VideoProvider]:
        out: list[VideoProvider] = []
        for p in self._providers.values():
            c = p.caps
            if duration_s is not None and not c.fits_duration(duration_s):
                continue
            if needs_extend and not c.supports_extend:
                continue
            if needs_last_frame and not c.supports_last_frame:
                continue
            if needs_native_audio and not c.supports_native_audio:
                continue
            if needs_lora and not c.supports_lora:
                continue
            if unmoderated_only and c.moderated:
                continue
            if c.max_ref_images < min_ref_images:
                continue
            out.append(p)
        return out

    def __contains__(self, name: object) -> bool:
        return name in self._providers

    def __iter__(self) -> Iterable[VideoProvider]:
        return iter(self._providers.values())

    def __len__(self) -> int:
        return len(self._providers)


REGISTRY = ProviderRegistry()
