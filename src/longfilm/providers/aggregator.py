"""聚合平台 provider —— 主供应商挂掉时的第三备份。

产线的三层容灾：自建 ComfyUI（可控） → 官方 API（画质） → 聚合平台（兜底）。
聚合平台既不是最便宜也不是最好，它的价值只有一个：**当某一家官方通道限流、
欠费或区域性故障时，同一个模型还能从另一条链路发出去**。所以这里刻意不做
模型精选，只做「同一个 GenRequest 能不能原样换条路跑出来」。

聚合层有两个固有风险，本模块在类型上就把它们挡住：
1. 各家 input schema 不一样且会变。Replicate 上每个模型的入参字段都不同，
   写死字段名等于给自己埋雷 —— 所以 ReplicateModel 带一张可覆盖的字段映射表，
   并提供 fetch_schema() 在运行时核对。
2. 轮询地址不能靠拼。fal 提交时会返回 status_url/response_url/cancel_url，
   用返回的地址比自己拼 URL 稳得多（嵌套 model id 的路径规则是会变的实现细节）。

────────────────────────────────────────────────────────────────────────
查证日期 2026-09-19

【fal.ai queue API】来源 fal.ai/docs/model-apis/model-endpoints/queue
  提交   POST https://queue.fal.run/{model_id}[?fal_webhook=<url>]
         header: Authorization: Key $FAL_KEY
         返回   {request_id, response_url, status_url, cancel_url, queue_position}
  状态   GET  https://queue.fal.run/{model_id}/requests/{request_id}/status[?logs=1]
  结果   GET  https://queue.fal.run/{model_id}/requests/{request_id}
  取消   PUT  https://queue.fal.run/{model_id}/requests/{request_id}/cancel
  status ∈ IN_QUEUE / IN_PROGRESS / COMPLETED
  Wan I2V(+LoRA) 入参（fal.ai/models/fal-ai/wan-i2v-lora/api）：
    prompt, image_url, negative_prompt, num_frames[81..100], frames_per_second[5..24],
    seed, resolution{480p,720p}, aspect_ratio, loras[{path, weight_name, scale}],
    enable_safety_checker, enable_prompt_expansion
  产出：{"video": {"url": ...}, "seed": ...}

【Replicate HTTP API】来源 replicate.com/docs/reference/http
  提交   POST https://api.replicate.com/v1/predictions          body {version, input, webhook,
                                                                     webhook_events_filter}
         POST https://api.replicate.com/v1/models/{owner}/{name}/predictions   （官方模型）
         header: Authorization: Bearer $REPLICATE_API_TOKEN
         可选 header Prefer: wait=n（1..60 秒同步等待）—— 产线不用，队列要的是立即返回
  查询   GET  https://api.replicate.com/v1/predictions/{id}
  取消   POST https://api.replicate.com/v1/predictions/{id}/cancel
  status ∈ starting / processing / succeeded / failed / canceled
  限流   429 + "Request was throttled."
  视频模型 slug（replicate.com/collections/text-to-video 当日快照）：
    alibaba/wan-3, wan-video/wan-2.7-t2v, wan-video/wan-2.5-i2v-fast,
    kwaivgi/kling-v3-video, bytedance/seedance-2.0, minimax/hailuo-2.3,
    google/veo-3.1, openai/sora-2-pro
  TODO(2026-09-19)：**逐模型的 input schema 未查证**。默认字段映射按 Wan 系
    （prompt/image/duration/resolution/seed/negative_prompt）给，接新模型前
    先跑 ReplicateProvider.fetch_schema() 核对，不要凭默认值上线。
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import _http
from .base import (
    Capabilities,
    FailureKind,
    GenRequest,
    GenResult,
    JobStatus,
    ProviderError,
    VideoProvider,
)

log = logging.getLogger(__name__)

FAL_QUEUE_BASE = "https://queue.fal.run"
REPLICATE_BASE = "https://api.replicate.com/v1"

# 聚合平台背后仍然是各厂商的审核，分级路由规则与官方通道一致。
_BLOCKED_RATINGS = frozenset({"blocked", "r_violence", "r_suggestive"})

_FAL_STATUS: dict[str, JobStatus] = {
    "IN_QUEUE": JobStatus.PENDING,
    "IN_PROGRESS": JobStatus.RUNNING,
    "COMPLETED": JobStatus.SUCCEEDED,
}

_REPLICATE_STATUS: dict[str, JobStatus] = {
    "starting": JobStatus.PENDING,
    "processing": JobStatus.RUNNING,
    "succeeded": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
    "canceled": JobStatus.CANCELLED,
}

_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}


# ---------------------------------------------------------------- 公共工具


def _gate(req: GenRequest, who: str) -> None:
    """分级门禁：只做拒单与换引擎，不做任何审核规避。"""
    if req.content_rating in _BLOCKED_RATINGS:
        kind = FailureKind.BAD_REQUEST if req.content_rating == "blocked" else FailureKind.MODERATION
        raise ProviderError(
            kind, f"{req.shot_id or '该镜'} 分级 {req.content_rating} 不投 {who}（聚合平台同样有审核）"
        )


def _public_url(uri: str | None, what: str) -> str | None:
    if uri is None:
        return None
    if uri.startswith(("http://", "https://")):
        return uri
    raise ProviderError(
        FailureKind.BAD_REQUEST,
        f"{what} 必须是公网可达的 URL（收到 {uri!r}）；聚合平台读不到本机文件",
    )


def _first_frame(req: GenRequest) -> str | None:
    return req.first_frame_uri or next(
        (i.uri for i in req.refs.images if i.role == "first_frame"), None
    )


def _last_frame(req: GenRequest) -> str | None:
    return req.last_frame_uri or next(
        (i.uri for i in req.refs.images if i.role == "last_frame"), None
    )


def _dig_video_url(obj: Any, depth: int = 0) -> str | None:
    """从任意形状的输出里捞出视频 URL。

    各家输出结构完全不同（fal 是 {"video":{"url":...}}，Replicate 可能是裸字符串、
    字符串数组或 {"output": {...}}），与其为每个模型写一遍解析，不如做一次深度优先。
    """
    if depth > 6:
        return None
    if isinstance(obj, str):
        low = obj.split("?", 1)[0].lower()
        if obj.startswith(("http://", "https://")) and (
            Path(low).suffix in _VIDEO_SUFFIXES or "/video" in low
        ):
            return obj
        return None
    if isinstance(obj, Mapping):
        for key in ("video", "url", "output", "video_url", "videos"):
            if key in obj and (hit := _dig_video_url(obj[key], depth + 1)):
                return hit
        for v in obj.values():
            if hit := _dig_video_url(v, depth + 1):
                return hit
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            if hit := _dig_video_url(v, depth + 1):
                return hit
    return None


# ---------------------------------------------------------------- fal.ai


@dataclass(frozen=True)
class FalModel:
    model_id: str
    resolutions: tuple[tuple[int, int], ...] = ((832, 480), (1280, 720))
    fps_options: tuple[int, ...] = (16, 24)
    min_duration_s: float = 3.0
    max_duration_s: float = 6.0
    supports_lora: bool = False
    supports_last_frame: bool = False
    quality_tier: int = 3
    cost_per_second_usd: float = 0.0
    # num_frames 而不是 duration：Wan 在 fal 上按帧数下单（81..100 帧）
    frame_based: bool = True
    min_frames: int = 81
    max_frames: int = 100


FAL_MODELS: dict[str, FalModel] = {
    "fal-ai/wan-i2v": FalModel(model_id="fal-ai/wan-i2v", quality_tier=3),
    "fal-ai/wan-i2v-lora": FalModel(
        model_id="fal-ai/wan-i2v-lora", supports_lora=True, quality_tier=3
    ),
    "fal-ai/wan/v2.2-a14b/image-to-video": FalModel(
        model_id="fal-ai/wan/v2.2-a14b/image-to-video", quality_tier=4
    ),
    "fal-ai/wan/v2.2-5b/image-to-video": FalModel(
        model_id="fal-ai/wan/v2.2-5b/image-to-video", quality_tier=3
    ),
    # 2.6 支持 5/10/15s 与 720p/1080p，按帧数下单的老约定在这一代不适用
    "wan/v2.6/image-to-video": FalModel(
        model_id="wan/v2.6/image-to-video",
        resolutions=((1280, 720), (1920, 1080)),
        min_duration_s=5.0, max_duration_s=15.0,
        quality_tier=4, frame_based=False,
    ),
}


class FalProvider(VideoProvider):
    """fal.ai 队列 API。容灾链上的第三家，也是唯一一家能云端挂 LoRA 的聚合平台。"""

    def __init__(
        self,
        *,
        model: str = "fal-ai/wan-i2v-lora",
        api_key: str | None = None,
        base_url: str = FAL_QUEUE_BASE,
        webhook_url: str | None = None,
        out_dir: str | Path = "out/fal",
        name: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        if model not in FAL_MODELS:
            raise ValueError(f"未登记的 fal 模型 {model!r}；已登记 {sorted(FAL_MODELS)}")
        spec = FAL_MODELS[model]
        caps = Capabilities(
            # 用完整 model_id 派生名字：ProviderRegistry 按 name 去重，
            # 只取末段会让 fal-ai/wan/v2.2-5b/image-to-video 和 wan/v2.6/image-to-video 撞名
            name=name or "fal_" + model.replace("fal-ai/", "").replace("/", "_"),
            kind="aggregator",
            min_duration_s=spec.min_duration_s,
            max_duration_s=spec.max_duration_s,
            resolutions=spec.resolutions,
            fps_options=spec.fps_options,
            max_ref_images=2 if spec.supports_last_frame else 1,
            supported_image_roles=frozenset(
                {"first_frame", "last_frame"} if spec.supports_last_frame else {"first_frame"}
            ),
            supports_first_frame=True,
            supports_last_frame=spec.supports_last_frame,
            supports_lora=spec.supports_lora,
            supports_seed=True,
            supports_negative_prompt=True,
            moderated=True,
            max_concurrency=8,
            cost_per_second_usd=spec.cost_per_second_usd,
            typical_latency_s=120.0,
            quality_tier=spec.quality_tier,
            notes="fal.ai 队列；容灾第三顺位",
        )
        super().__init__(caps)
        self.spec = spec
        self.model = model
        self._api_key = api_key or os.environ.get("FAL_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.webhook_url = webhook_url
        self.out_dir = Path(out_dir)
        self.timeout_s = timeout_s
        self._jobs: dict[str, dict[str, Any]] = {}

    def _auth(self) -> str:
        if not self._api_key:
            raise ProviderError(FailureKind.AUTH, "缺少 FAL_KEY 环境变量，无法调用 fal.ai")
        return self._api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self._auth()}", "Content-Type": "application/json"}

    def health(self) -> bool:
        if not self._api_key:
            return False
        try:
            code, _ = _http.request_json(
                "GET",
                f"{self.base_url}/{self.model}/requests/longfilm-health-probe/status",
                headers=self._headers(), timeout=5.0,
            )
        except ProviderError:
            return False
        # 404 = 链路通、鉴权过、只是这个 request_id 不存在，正是想要的结果
        return code not in (401, 403) and code < 500

    # ---- job_id 编码：poll() 只拿得到一个字符串，但 fal 的地址依赖 model_id
    @staticmethod
    def _encode(model: str, request_id: str) -> str:
        return f"{model}#{request_id}"

    @staticmethod
    def _decode(job_id: str) -> tuple[str, str]:
        model, _, rid = job_id.rpartition("#")
        if not model or not rid:
            raise ProviderError(FailureKind.BAD_REQUEST, f"非法的 fal job_id：{job_id!r}")
        return model, rid

    def build_input(self, req: GenRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {"prompt": req.prompt}
        if req.negative_prompt:
            payload["negative_prompt"] = req.negative_prompt
        if img := _public_url(_first_frame(req), "首帧"):
            payload["image_url"] = img
        if req.seed is not None:
            payload["seed"] = int(req.seed) % 2147483648
        wh = self.caps.nearest_resolution(req.resolution)
        payload["resolution"] = f"{wh[1]}p"
        fps = req.fps if req.fps in self.caps.fps_options else self.caps.fps_options[-1]
        if self.spec.frame_based:
            frames = round(self.caps.clamp_duration(req.duration_s) * fps)
            payload["num_frames"] = max(self.spec.min_frames, min(self.spec.max_frames, frames))
            payload["frames_per_second"] = fps
        else:
            payload["duration"] = int(round(self.caps.clamp_duration(req.duration_s)))
        if req.lora:
            if not self.spec.supports_lora:
                raise ProviderError(
                    FailureKind.BAD_REQUEST, f"{self.model} 不支持 LoRA，请用 fal-ai/wan-i2v-lora"
                )
            payload["loras"] = _fal_loras(req.lora)
        # 刻意不设 enable_safety_checker：用平台默认值。
        # 产线的内容边界靠 router 分级门禁保证，不靠关掉别人的安全开关。
        return payload

    def submit(self, req: GenRequest) -> str:
        self._auth()
        _gate(req, "fal.ai")
        url = f"{self.base_url}/{self.model}"
        if self.webhook_url:
            from urllib.parse import quote

            url = f"{url}?fal_webhook={quote(self.webhook_url, safe='')}"
        code, body = _http.request_json(
            "POST", url, headers=self._headers(), json_body=self.build_input(req),
            timeout=self.timeout_s,
        )
        if code >= 400 or not isinstance(body, Mapping) or not body.get("request_id"):
            raise ProviderError(
                _fal_failure(code, body), f"fal 提交失败 HTTP {code}: {body}", raw=_raw(body)
            )
        rid = str(body["request_id"])
        job_id = self._encode(self.model, rid)
        self._jobs[job_id] = {
            "submitted_at": time.time(),
            "shot_id": req.shot_id,
            "duration_s": req.duration_s,
            # 用服务端返回的地址，别自己拼：嵌套 model id 的路径规则是实现细节
            "status_url": body.get("status_url"),
            "response_url": body.get("response_url"),
            "cancel_url": body.get("cancel_url"),
        }
        log.info("fal 已提交 %s model=%s request_id=%s", req.shot_id or "?", self.model, rid)
        return job_id

    def _urls(self, job_id: str) -> dict[str, str]:
        meta = self._jobs.get(job_id, {})
        model, rid = self._decode(job_id)
        base = f"{self.base_url}/{model}/requests/{rid}"
        return {
            "status": meta.get("status_url") or f"{base}/status",
            "response": meta.get("response_url") or base,
            "cancel": meta.get("cancel_url") or f"{base}/cancel",
        }

    def poll(self, job_id: str) -> GenResult:
        meta = self._jobs.get(job_id, {})
        urls = self._urls(job_id)
        res = GenResult(job_id=job_id, status=JobStatus.PENDING, provider=self.name,
                        submitted_at=meta.get("submitted_at", 0.0))
        code, body = _http.request_json(
            "GET", urls["status"], headers=self._headers(), timeout=self.timeout_s
        )
        if code >= 400 or not isinstance(body, Mapping):
            res.status, res.failure = JobStatus.FAILED, _fal_failure(code, body)
            res.message = f"fal 状态查询 HTTP {code}: {body}"
            return res
        res.raw = {"status": body.get("status"), "queue_position": body.get("queue_position")}
        status = _FAL_STATUS.get(str(body.get("status", "")).upper())
        if status is None:
            res.status, res.failure = JobStatus.FAILED, FailureKind.UNKNOWN
            res.message = f"未知 fal status: {body.get('status')}"
            return res
        if status is not JobStatus.SUCCEEDED:
            res.status = status
            return res

        code, out = _http.request_json(
            "GET", urls["response"], headers=self._headers(), timeout=self.timeout_s
        )
        res.finished_at = time.time()
        if code >= 400 or not isinstance(out, Mapping):
            res.status, res.failure = JobStatus.FAILED, _fal_failure(code, out)
            res.message = f"fal 取结果 HTTP {code}: {out}"
            return res
        if out.get("error") or out.get("error_type"):
            res.status = JobStatus.FAILED
            res.failure = _fal_failure(code, out)
            if res.failure is FailureKind.MODERATION:
                res.status = JobStatus.REJECTED
            res.message = f"{out.get('error_type')}: {out.get('error')}"
            return res
        url = _dig_video_url(out)
        if not url:
            res.status, res.failure = JobStatus.FAILED, FailureKind.UNKNOWN
            res.message = f"fal 返回里没找到视频 URL：{list(out)}"
            return res
        dest = self.out_dir / f"{meta.get('shot_id') or self._decode(job_id)[1]}.mp4"
        res.status = JobStatus.SUCCEEDED
        res.video_uri = str(_http.download(url, dest, timeout=max(self.timeout_s, 600.0)))
        res.duration_s = float(meta.get("duration_s", 0.0))
        res.cost_usd = round(self.caps.cost_per_second_usd * res.duration_s, 4)
        return res

    def cancel(self, job_id: str) -> None:
        _http.request_json(
            "PUT", self._urls(job_id)["cancel"], headers=self._headers(), timeout=self.timeout_s
        )


def _fal_loras(spec: Mapping[str, Any] | Sequence[Any]) -> list[dict[str, Any]]:
    """fal 的 LoRA 是 {path, scale}，path 必须是可下载的权重 URL，不是本地文件名。"""
    if isinstance(spec, Mapping):
        items: Sequence[Any] = spec.get("loras") or [spec]
    else:
        items = spec
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, Mapping):
            raise ProviderError(FailureKind.BAD_REQUEST, "LoRA 条目必须是 dict")
        path = it.get("url") or it.get("path")
        if not path or not str(path).startswith(("http://", "https://")):
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                f"fal 的 LoRA path 必须是可公开下载的权重 URL（收到 {path!r}）",
            )
        entry: dict[str, Any] = {"path": str(path), "scale": float(it.get("scale", it.get("strength", 0.85)))}
        if wn := it.get("weight_name"):
            entry["weight_name"] = str(wn)
        out.append(entry)
    return out


def _fal_failure(code: int, body: Any) -> FailureKind:
    txt = ""
    if isinstance(body, Mapping):
        txt = f"{body.get('error_type', '')} {body.get('error', '')} {body.get('detail', '')}".lower()
    if any(k in txt for k in ("nsfw", "safety", "content policy", "moderation")):
        return FailureKind.MODERATION
    return _http.status_to_failure(code)


# ---------------------------------------------------------------- Replicate


@dataclass(frozen=True)
class ReplicateModel:
    """Replicate 上一个视频模型的最小声明。

    input_map 是「产线语义名 → 该模型的入参字段名」。每个模型的 schema 都不一样，
    把它做成数据而不是代码，接新模型时就只加一行表项，不用改 provider。
    """

    owner: str
    name: str
    version: str = ""                       # 空 = 用 /models/{owner}/{name}/predictions
    resolutions: tuple[tuple[int, int], ...] = ((1280, 720),)
    min_duration_s: float = 5.0
    max_duration_s: float = 10.0
    duration_steps: tuple[float, ...] = (5.0, 10.0)
    supports_last_frame: bool = False
    quality_tier: int = 4
    cost_per_second_usd: float = 0.0
    input_map: Mapping[str, str] = field(default_factory=lambda: {
        "prompt": "prompt",
        "negative_prompt": "negative_prompt",
        "first_frame": "image",
        "last_frame": "last_frame_image",
        "duration": "duration",
        "resolution": "resolution",
        "seed": "seed",
    })

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.name}"


REPLICATE_MODELS: dict[str, ReplicateModel] = {
    "wan-video/wan-2.5-i2v-fast": ReplicateModel(
        owner="wan-video", name="wan-2.5-i2v-fast",
        resolutions=((832, 480), (1280, 720), (1920, 1080)), quality_tier=4,
    ),
    "alibaba/wan-3": ReplicateModel(
        owner="alibaba", name="wan-3",
        resolutions=((1280, 720), (1920, 1080)),
        min_duration_s=5.0, max_duration_s=10.0, quality_tier=5,
    ),
    "kwaivgi/kling-v3-video": ReplicateModel(
        owner="kwaivgi", name="kling-v3-video",
        resolutions=((1280, 720), (1920, 1080)),
        supports_last_frame=True, quality_tier=5,
    ),
    "bytedance/seedance-2.0": ReplicateModel(
        owner="bytedance", name="seedance-2.0",
        resolutions=((1280, 720), (1920, 1080)), quality_tier=5,
    ),
    "minimax/hailuo-2.3": ReplicateModel(
        owner="minimax", name="hailuo-2.3",
        resolutions=((1280, 720), (1920, 1080)),
        min_duration_s=6.0, max_duration_s=10.0, duration_steps=(6.0, 10.0), quality_tier=4,
    ),
    "google/veo-3.1": ReplicateModel(
        owner="google", name="veo-3.1",
        resolutions=((1280, 720), (1920, 1080)),
        min_duration_s=4.0, max_duration_s=8.0, duration_steps=(4.0, 6.0, 8.0), quality_tier=5,
    ),
}


class ReplicateProvider(VideoProvider):
    """Replicate predictions API。容灾链上覆盖模型最广的一家。"""

    def __init__(
        self,
        *,
        model: str = "wan-video/wan-2.5-i2v-fast",
        api_token: str | None = None,
        base_url: str = REPLICATE_BASE,
        webhook_url: str | None = None,
        webhook_events: Sequence[str] = ("completed",),
        out_dir: str | Path = "out/replicate",
        name: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        if model not in REPLICATE_MODELS:
            raise ValueError(f"未登记的 Replicate 模型 {model!r}；已登记 {sorted(REPLICATE_MODELS)}")
        spec = REPLICATE_MODELS[model]
        caps = Capabilities(
            name=name or f"replicate_{spec.name}",
            kind="aggregator",
            min_duration_s=spec.min_duration_s,
            max_duration_s=spec.max_duration_s,
            duration_steps=spec.duration_steps,
            resolutions=spec.resolutions,
            fps_options=(24,),
            max_ref_images=2 if spec.supports_last_frame else 1,
            supported_image_roles=frozenset(
                {"first_frame", "last_frame"} if spec.supports_last_frame else {"first_frame"}
            ),
            supports_first_frame=True,
            supports_last_frame=spec.supports_last_frame,
            supports_lora=False,        # 这几家都不开放挂权重
            supports_seed=True,
            supports_negative_prompt="negative_prompt" in spec.input_map,
            moderated=True,
            max_concurrency=8,
            cost_per_second_usd=spec.cost_per_second_usd,
            typical_latency_s=150.0,
            quality_tier=spec.quality_tier,
            notes="Replicate predictions；容灾第三顺位",
        )
        super().__init__(caps)
        self.spec = spec
        self.model = model
        self._api_token = api_token or os.environ.get("REPLICATE_API_TOKEN") or ""
        self.base_url = base_url.rstrip("/")
        self.webhook_url = webhook_url
        self.webhook_events = tuple(webhook_events)
        self.out_dir = Path(out_dir)
        self.timeout_s = timeout_s
        self._jobs: dict[str, dict[str, Any]] = {}

    def _auth(self) -> str:
        if not self._api_token:
            raise ProviderError(
                FailureKind.AUTH, "缺少 REPLICATE_API_TOKEN 环境变量，无法调用 Replicate"
            )
        return self._api_token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._auth()}", "Content-Type": "application/json"}

    def health(self) -> bool:
        if not self._api_token:
            return False
        try:
            code, _ = _http.request_json(
                "GET", f"{self.base_url}/models/{self.spec.slug}",
                headers=self._headers(), timeout=5.0,
            )
        except ProviderError:
            return False
        return code == 200

    def fetch_schema(self) -> dict[str, Any]:
        """拉取该模型的真实 input schema。

        接新模型前用它核对 input_map：Replicate 各模型字段名不统一且会随版本变，
        靠默认映射上线等于把「参数被静默忽略」这种 bug 留到成片以后才发现。
        """
        code, body = _http.request_json(
            "GET", f"{self.base_url}/models/{self.spec.slug}",
            headers=self._headers(), timeout=self.timeout_s,
        )
        if code >= 400 or not isinstance(body, Mapping):
            raise ProviderError(
                _http.status_to_failure(code), f"读取 {self.spec.slug} schema 失败 HTTP {code}"
            )
        latest = body.get("latest_version") if isinstance(body.get("latest_version"), Mapping) else {}
        schema = ((latest or {}).get("openapi_schema") or {})
        comps = schema.get("components", {}) if isinstance(schema, Mapping) else {}
        return dict(comps.get("schemas", {}).get("Input", {}))

    def build_input(self, req: GenRequest) -> dict[str, Any]:
        m = self.spec.input_map
        out: dict[str, Any] = {m["prompt"]: req.prompt}
        if req.negative_prompt and "negative_prompt" in m:
            out[m["negative_prompt"]] = req.negative_prompt
        if (ff := _public_url(_first_frame(req), "首帧")) and "first_frame" in m:
            out[m["first_frame"]] = ff
        if self.spec.supports_last_frame and (lf := _public_url(_last_frame(req), "尾帧")):
            out[m["last_frame"]] = lf
        if "duration" in m:
            out[m["duration"]] = int(round(self.caps.clamp_duration(req.duration_s)))
        if "resolution" in m:
            out[m["resolution"]] = f"{self.caps.nearest_resolution(req.resolution)[1]}p"
        if req.seed is not None and "seed" in m:
            out[m["seed"]] = int(req.seed) % 2147483648
        return out

    def submit(self, req: GenRequest) -> str:
        self._auth()
        _gate(req, "Replicate")
        payload: dict[str, Any] = {"input": self.build_input(req)}
        if self.webhook_url:
            payload["webhook"] = self.webhook_url
            payload["webhook_events_filter"] = list(self.webhook_events)
        if self.spec.version:
            url = f"{self.base_url}/predictions"
            payload["version"] = self.spec.version
        else:
            url = f"{self.base_url}/models/{self.spec.slug}/predictions"

        code, body = _http.request_json(
            "POST", url, headers=self._headers(), json_body=payload, timeout=self.timeout_s
        )
        if code >= 400 or not isinstance(body, Mapping) or not body.get("id"):
            raise ProviderError(
                _replicate_failure(code, body), f"Replicate 提交失败 HTTP {code}: {body}",
                raw=_raw(body),
            )
        pid = str(body["id"])
        self._jobs[pid] = {"submitted_at": time.time(), "shot_id": req.shot_id,
                           "duration_s": req.duration_s}
        log.info("Replicate 已提交 %s model=%s id=%s", req.shot_id or "?", self.spec.slug, pid)
        return pid

    def poll(self, job_id: str) -> GenResult:
        meta = self._jobs.get(job_id, {})
        res = GenResult(job_id=job_id, status=JobStatus.PENDING, provider=self.name,
                        submitted_at=meta.get("submitted_at", 0.0))
        code, body = _http.request_json(
            "GET", f"{self.base_url}/predictions/{job_id}",
            headers=self._headers(), timeout=self.timeout_s,
        )
        if code >= 400 or not isinstance(body, Mapping):
            res.status, res.failure = JobStatus.FAILED, _replicate_failure(code, body)
            res.message = f"Replicate 查询 HTTP {code}: {body}"
            return res
        res.raw = {"status": body.get("status"), "metrics": body.get("metrics")}
        status = _REPLICATE_STATUS.get(str(body.get("status", "")))
        if status is None:
            res.status, res.failure = JobStatus.FAILED, FailureKind.UNKNOWN
            res.message = f"未知 Replicate status: {body.get('status')}"
            return res
        if status in {JobStatus.PENDING, JobStatus.RUNNING}:
            res.status = status
            return res

        res.finished_at = time.time()
        if status is not JobStatus.SUCCEEDED:
            res.status = status
            res.failure = _replicate_failure(code, body)
            if res.failure is FailureKind.MODERATION:
                res.status = JobStatus.REJECTED
            res.message = str(body.get("error") or body.get("status"))
            return res

        url = _dig_video_url(body.get("output"))
        if not url:
            res.status, res.failure = JobStatus.FAILED, FailureKind.UNKNOWN
            res.message = f"Replicate 输出里没找到视频 URL：{body.get('output')!r}"
            return res
        dest = self.out_dir / f"{meta.get('shot_id') or job_id}.mp4"
        res.status = JobStatus.SUCCEEDED
        res.video_uri = str(_http.download(url, dest, timeout=max(self.timeout_s, 600.0)))
        res.duration_s = float(meta.get("duration_s", 0.0))
        res.cost_usd = round(self.caps.cost_per_second_usd * res.duration_s, 4)
        return res

    def cancel(self, job_id: str) -> None:
        _http.request_json(
            "POST", f"{self.base_url}/predictions/{job_id}/cancel",
            headers=self._headers(), json_body={}, timeout=self.timeout_s,
        )


def _replicate_failure(code: int, body: Any) -> FailureKind:
    err = str(body.get("error", "")).lower() if isinstance(body, Mapping) else ""
    if any(k in err for k in ("nsfw", "safety", "content policy", "flagged", "moderation")):
        return FailureKind.MODERATION
    if "throttled" in err:
        return FailureKind.RATE_LIMIT
    return _http.status_to_failure(code)


def _raw(body: Any) -> dict[str, Any]:
    return dict(body) if isinstance(body, Mapping) else {"body": body}


# ---------------------------------------------------------------- 容灾编队


def failover_chain(**kwargs: Any) -> list[VideoProvider]:
    """按质量档从高到低给出一条聚合层容灾链，供 ProviderRegistry 批量注册。"""
    provs: list[VideoProvider] = [
        ReplicateProvider(model=m, **kwargs) for m in ("alibaba/wan-3", "wan-video/wan-2.5-i2v-fast")
    ]
    provs += [FalProvider(model=m, **kwargs) for m in ("wan/v2.6/image-to-video", "fal-ai/wan-i2v-lora")]
    return sorted(provs, key=lambda p: -p.caps.quality_tier)


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    logging.basicConfig(level=logging.ERROR)

    req = GenRequest(
        prompt="a fictional adult archivist opens a brass door",
        negative_prompt="lowres, watermark",
        duration_s=6.0, resolution=(1280, 720), fps=24, seed=99,
        first_frame_uri="https://example.com/ff.png",
        shot_id="s03_sh02",
    )

    # ---- fal
    fal = FalProvider(model="fal-ai/wan-i2v-lora", api_key="")
    assert fal.caps.kind == "aggregator" and fal.caps.moderated is True
    assert fal.caps.supports_lora and not fal.caps.supports_last_frame
    assert fal.health() is False
    try:
        fal.submit(req)
    except ProviderError as e:
        assert e.kind is FailureKind.AUTH, e.kind
    else:
        raise AssertionError("fal 无 key 时应抛 AUTH")

    falk = FalProvider(model="fal-ai/wan-i2v-lora", api_key="fal-fake:selftest")
    body = falk.build_input(req)
    assert body["prompt"] == req.prompt and body["image_url"] == req.first_frame_uri
    assert body["resolution"] == "720p" and body["seed"] == 99
    assert body["frames_per_second"] == 24
    assert 81 <= body["num_frames"] <= 100, body["num_frames"]
    assert "enable_safety_checker" not in body, "不得代替平台关掉安全开关"

    lora_body = falk.build_input(
        GenRequest(prompt="x", duration_s=5.0, first_frame_uri="https://e.com/a.png",
                   lora={"loras": [{"url": "https://cdn.example.com/hero.safetensors", "scale": 0.7},
                                   {"path": "https://cdn.example.com/style.safetensors"}]})
    )
    assert lora_body["loras"] == [
        {"path": "https://cdn.example.com/hero.safetensors", "scale": 0.7},
        {"path": "https://cdn.example.com/style.safetensors", "scale": 0.85},
    ]
    try:
        falk.build_input(GenRequest(prompt="x", duration_s=5.0,
                                    first_frame_uri="https://e.com/a.png",
                                    lora={"path": "/local/hero.safetensors"}))
    except ProviderError as e:
        assert "可公开下载" in str(e)
    else:
        raise AssertionError("fal 的 LoRA 本地路径应当被拒")

    fal_nolora = FalProvider(model="fal-ai/wan-i2v", api_key="fal-fake:selftest")
    assert not fal_nolora.caps.supports_lora
    try:
        fal_nolora.build_input(GenRequest(prompt="x", duration_s=5.0,
                                          first_frame_uri="https://e.com/a.png",
                                          lora={"url": "https://e.com/x.safetensors"}))
    except ProviderError as e:
        assert "不支持 LoRA" in str(e)
    else:
        raise AssertionError("不支持 LoRA 的模型收到 lora 应当报错")

    # 2.6 按秒下单而不是按帧
    fal26 = FalProvider(model="wan/v2.6/image-to-video", api_key="fal-fake:selftest")
    b26 = fal26.build_input(GenRequest(prompt="x", duration_s=15.0, resolution=(1920, 1080),
                                       first_frame_uri="https://e.com/a.png"))
    assert b26["duration"] == 15 and "num_frames" not in b26 and b26["resolution"] == "1080p"

    # job_id 编解码要能还原出 model（poll 只拿得到一个字符串）
    jid = FalProvider._encode("fal-ai/wan/v2.2-5b/image-to-video", "req-123")
    assert FalProvider._decode(jid) == ("fal-ai/wan/v2.2-5b/image-to-video", "req-123")
    urls = falk._urls(jid)
    assert urls["status"].endswith("/requests/req-123/status")
    try:
        FalProvider._decode("no-hash")
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST
    else:
        raise AssertionError("非法 job_id 应当报错")

    # ---- replicate
    rep = ReplicateProvider(model="wan-video/wan-2.5-i2v-fast", api_token="")
    assert rep.caps.kind == "aggregator" and rep.caps.moderated and not rep.caps.supports_lora
    assert rep.health() is False
    try:
        rep.submit(req)
    except ProviderError as e:
        assert e.kind is FailureKind.AUTH, e.kind
    else:
        raise AssertionError("Replicate 无 token 时应抛 AUTH")

    repk = ReplicateProvider(model="kwaivgi/kling-v3-video", api_token="r8_fake_selftest")
    assert repk.caps.supports_last_frame
    inp = repk.build_input(
        GenRequest(prompt="x", negative_prompt="ng", duration_s=10.0, resolution=(1920, 1080),
                   seed=5, first_frame_uri="https://e.com/a.png",
                   last_frame_uri="https://e.com/b.png")
    )
    assert inp == {"prompt": "x", "negative_prompt": "ng", "image": "https://e.com/a.png",
                   "last_frame_image": "https://e.com/b.png", "duration": 10,
                   "resolution": "1080p", "seed": 5}
    # duration_steps 要把不合法时长吸附到档位上，而不是原样发出去
    assert repk.build_input(GenRequest(prompt="x", duration_s=7.0))["duration"] == 5
    assert ReplicateProvider(model="google/veo-3.1", api_token="r8_x").caps.clamp_duration(7.0) == 6.0

    try:
        repk.build_input(GenRequest(prompt="x", duration_s=5.0, first_frame_uri="/tmp/a.png"))
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST and "公网可达" in str(e)
    else:
        raise AssertionError("本地路径应当被拒")

    for bad, cls in (("fal-ai/nope", FalProvider), ("nobody/nope", ReplicateProvider)):
        try:
            cls(model=bad)
        except ValueError as e:
            assert "未登记" in str(e)
        else:
            raise AssertionError(f"{cls.__name__} 应当拒绝未登记的模型 {bad}")

    # ---- 分级门禁：聚合平台同样只做拒单，不做绕过
    for prov in (falk, repk):
        for rating, kind in (("blocked", FailureKind.BAD_REQUEST),
                             ("r_suggestive", FailureKind.MODERATION)):
            try:
                prov.submit(GenRequest(prompt="x", duration_s=5.0, content_rating=rating,
                                       first_frame_uri="https://e.com/a.png"))
            except ProviderError as e:
                assert e.kind is kind, (prov.name, rating, e.kind)
            else:
                raise AssertionError(f"{prov.name} 分级 {rating} 应当拒单")

    # ---- 输出解析：三家的结构都不一样，深度优先必须都能捞出来
    assert _dig_video_url({"video": {"url": "https://cdn/x.mp4"}}) == "https://cdn/x.mp4"
    assert _dig_video_url("https://cdn/y.mp4?sig=1") == "https://cdn/y.mp4?sig=1"
    assert _dig_video_url(["https://cdn/z.webm"]) == "https://cdn/z.webm"
    assert _dig_video_url({"output": {"videos": [{"url": "https://cdn/a.mov"}]}}) == "https://cdn/a.mov"
    assert _dig_video_url({"output": "https://cdn/thumb.png"}) is None
    assert _dig_video_url({"status": "processing"}) is None

    # ---- 失败分类：审核拒要能换引擎，限流只需退避
    assert _fal_failure(200, {"error_type": "NSFW_CONTENT"}) is FailureKind.MODERATION
    assert _fal_failure(429, {}) is FailureKind.RATE_LIMIT
    assert _replicate_failure(200, {"error": "flagged by safety checker"}) is FailureKind.MODERATION
    assert _replicate_failure(429, {"error": "Request was throttled."}) is FailureKind.RATE_LIMIT
    assert _replicate_failure(401, {}) is FailureKind.AUTH
    assert FailureKind.MODERATION.should_failover and not FailureKind.MODERATION.retryable

    # ---- 容灾链：按画质降序，且都是 aggregator
    chain = failover_chain()
    tiers = [p.caps.quality_tier for p in chain]
    assert tiers == sorted(tiers, reverse=True), tiers
    assert all(p.caps.kind == "aggregator" for p in chain)
    assert len({p.name for p in chain}) == len(chain), "注册表按 name 去重，名字不能撞"
    assert all(p.health() is False for p in chain), "无密钥时整条链都应探活失败"

    # ---- 密钥不得出现在任何异常文本里
    for prov, secret in ((falk, "fal-fake:selftest"), (repk, "r8_fake_selftest")):
        try:
            prov.submit(GenRequest(prompt="x", duration_s=5.0, first_frame_uri="/tmp/a.png"))
        except ProviderError as e:
            assert secret not in str(e)

    print("aggregator selftest OK：fal 5 模型 / Replicate 6 模型，"
          "AUTH 兜底、分级拒单、输出解析、失败分类、容灾链全部通过")


if __name__ == "__main__":
    _selftest()
