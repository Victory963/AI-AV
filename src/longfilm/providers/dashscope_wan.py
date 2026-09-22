"""阿里云百炼（DashScope）万相视频 provider + 官方 LoRA 定制训练入口。

这一家在本产线里的位置很特殊：它是**唯一同时提供「高画质闭源推理」和
「官方 LoRA 微调」的通道**。自建 ComfyUI 能微调但画质要自己调；其他官方 API
画质够但不让你训模型；只有万相两头都占。代价是它有内容审核，R 级镜头必须
由 router 路由到开源侧 —— 本模块只做**路由判断与拒单**，不做任何规避。

────────────────────────────────────────────────────────────────────────
查证日期 2026-09-19，来源：help.aliyun.com / alibabacloud.com 官方文档

【推理】异步两段式
  POST {base}/api/v1/services/aigc/video-generation/video-synthesis
       header: Authorization: Bearer $DASHSCOPE_API_KEY
               Content-Type: application/json
               X-DashScope-Async: enable      ← 缺这个直接报错
       body:   {"model":..., "input":{...}, "parameters":{...}}
  GET  {base}/api/v1/tasks/{task_id}
       output.task_status ∈ PENDING/RUNNING/SUCCEEDED/FAILED/CANCELED/UNKNOWN
       成功时 output.video_url（有效期 24h，必须当场转存）
  base: https://dashscope.aliyuncs.com（北京） / https://dashscope-intl.aliyuncs.com（新加坡）
        万相 2.7 文档另给了 workspace 维度的
        https://{WorkspaceId}.{region}.maas.aliyuncs.com 形式，用 DASHSCOPE_BASE_URL 覆盖即可。

  请求体有两代形态，本模块按 WanModel.api_style 分流：
    "media"       —— wan2.7：input.media = [{"type":"first_frame"|"last_frame"
                      |"driving_audio"|"first_clip","url":...}]，首尾帧/音驱/续写统一走这里
    "legacy_i2v"  —— wan2.1~2.6：input.img_url（+ 2.5/2.6 的 input.audio_url）
    "legacy_kf2v" —— wan2.2-kf2v-flash：input.first_frame_url / last_frame_url

【LoRA 定制训练】**证实存在**，不是云厂商宣传话术：
  可微调基座：图生视频-首帧 wan2.7-i2v / wan2.6-i2v / wan2.5-i2v-preview /
              wan2.2-i2v-flash；首尾帧 wan2.7-i2v（hyper_parameters.task_type="kf2v"）
              与 wan2.2-kf2v-flash
  数据集：zip（首帧图 + 训练视频 + data.jsonl），
          data.jsonl 每行 {"prompt","first_frame_path","video_path"[,"last_frame_path"]}
  上传：  POST {base}/compatible-mode/v1/files  multipart: file=@x.zip, purpose=fine-tune → id
  建任务：POST {base}/api/v1/fine-tunes
          {"model":"wan2.7-i2v","training_file_ids":[...],"training_type":"efficient_sft",
           "hyper_parameters":{"task_type":"i2v","n_epochs":50,"lora_rank":32,...}}
  查询：  GET  {base}/api/v1/fine-tunes/{job_id}        → job_id / status / finetuned_output
  列出：  GET  {base}/api/v1/fine-tunes
  检查点：GET  {base}/api/v1/fine-tunes/{job_id}/checkpoints → model_name
  取消：  POST {base}/api/v1/fine-tunes/{job_id}/cancel
  **产出的 LoRA 不可下载**：只能部署成在线服务后用 finetuned_output 当 model 名云端调用。
  计费 = 训练 Tokens 总量 × 单价；部署免费；调用按基座标准价。
  地域：微调功能**仅新加坡（dashscope-intl）可用**，必须用该地域的 API Key。

  → 产线含义：万相 LoRA 和自训 ComfyUI LoRA **不可互换**，角色一致性资产会分叉。
    CharacterBible.lora.path 在这里存的是 finetuned_output 模型名，不是文件路径。
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

BASE_CN = "https://dashscope.aliyuncs.com"
BASE_INTL = "https://dashscope-intl.aliyuncs.com"
SYNTH_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks/{task_id}"

# DashScope 的业务错误码 → FailureKind。
# 这张表是降级路径的分水岭：DataInspectionFailed 是审核拒，重试一万次也没用，
# 必须立刻换到开源侧；Throttling 则只需要退避。
_CODE_MAP: dict[str, FailureKind] = {
    "InvalidApiKey": FailureKind.AUTH,
    "Unauthorized": FailureKind.AUTH,
    "AccessDenied": FailureKind.AUTH,
    "Throttling": FailureKind.RATE_LIMIT,
    "Throttling.RateQuota": FailureKind.RATE_LIMIT,
    "Throttling.AllocationQuota": FailureKind.QUOTA,
    "Arrearage": FailureKind.QUOTA,
    "DataInspectionFailed": FailureKind.MODERATION,
    "InvalidParameter": FailureKind.BAD_REQUEST,
    "InvalidURL": FailureKind.BAD_REQUEST,
    "InternalError": FailureKind.SERVER,
}

_TASK_STATUS: dict[str, JobStatus] = {
    "PENDING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "SUCCEEDED": JobStatus.SUCCEEDED,
    "FAILED": JobStatus.FAILED,
    "CANCELED": JobStatus.CANCELLED,
}


@dataclass(frozen=True)
class WanModel:
    """一个万相模型的能力声明。router 不读这个，它只读编译出来的 Capabilities。"""

    model: str
    api_style: str                                   # media | legacy_i2v | legacy_kf2v
    resolutions: tuple[tuple[int, int], ...]
    min_duration_s: float
    max_duration_s: float
    duration_steps: tuple[float, ...] = ()
    supports_last_frame: bool = False
    supports_audio_drive: bool = False
    supports_extend: bool = False
    quality_tier: int = 4
    finetunable: bool = False


# 只登记查证到的模型名与参数范围。
# duration：wan2.7 文档明确写了 [2,15] 秒、默认 5；旧代模型文档只说
# "model-specific ranges" 没逐一列出（2026-09-19 查证），因此这里保守地
# 只声明官方默认的 5s 一档 —— 宁可少报能力让 router 换家，也不要报一个
# 会被服务端打回的时长。补齐前请勿放宽。
WAN_MODELS: dict[str, WanModel] = {
    "wan2.7-i2v": WanModel(
        model="wan2.7-i2v", api_style="media",
        resolutions=((1280, 720), (1920, 1080)),
        min_duration_s=2.0, max_duration_s=15.0,
        supports_last_frame=True, supports_audio_drive=True, supports_extend=True,
        quality_tier=5, finetunable=True,
    ),
    "wan2.6-i2v": WanModel(
        model="wan2.6-i2v", api_style="legacy_i2v",
        resolutions=((1280, 720), (1920, 1080)),
        min_duration_s=5.0, max_duration_s=5.0, duration_steps=(5.0,),
        supports_audio_drive=True, quality_tier=4, finetunable=True,
    ),
    "wan2.5-i2v-preview": WanModel(
        model="wan2.5-i2v-preview", api_style="legacy_i2v",
        resolutions=((854, 480), (1280, 720), (1920, 1080)),
        min_duration_s=5.0, max_duration_s=5.0, duration_steps=(5.0,),
        supports_audio_drive=True, quality_tier=4, finetunable=True,
    ),
    "wan2.2-i2v-flash": WanModel(
        model="wan2.2-i2v-flash", api_style="legacy_i2v",
        resolutions=((854, 480), (1280, 720)),
        min_duration_s=5.0, max_duration_s=5.0, duration_steps=(5.0,),
        quality_tier=3, finetunable=True,
    ),
    "wan2.2-i2v-plus": WanModel(
        model="wan2.2-i2v-plus", api_style="legacy_i2v",
        resolutions=((854, 480), (1280, 720)),
        min_duration_s=5.0, max_duration_s=5.0, duration_steps=(5.0,),
        quality_tier=4,
    ),
    "wan2.2-kf2v-flash": WanModel(
        model="wan2.2-kf2v-flash", api_style="legacy_kf2v",
        resolutions=((854, 480), (1280, 720)),
        min_duration_s=5.0, max_duration_s=5.0, duration_steps=(5.0,),
        supports_last_frame=True, quality_tier=3, finetunable=True,
    ),
    "wan2.1-i2v-turbo": WanModel(
        model="wan2.1-i2v-turbo", api_style="legacy_i2v",
        resolutions=((854, 480), (1280, 720)),
        min_duration_s=5.0, max_duration_s=5.0, duration_steps=(5.0,),
        quality_tier=3,
    ),
    "wan2.1-i2v-plus": WanModel(
        model="wan2.1-i2v-plus", api_style="legacy_i2v",
        resolutions=((1280, 720),),
        min_duration_s=5.0, max_duration_s=5.0, duration_steps=(5.0,),
        quality_tier=4,
    ),
}

# 官方通道接不了的分级。这里是**拒单**，不是审核绕过：
# 平台条款本来就不接这类单，投过去必被拒还连累账号信誉，不如在本地就换引擎。
_OFFICIAL_BLOCKED_RATINGS = frozenset({"blocked", "r_violence", "r_suggestive"})


def _resolution_label(wh: tuple[int, int]) -> str:
    """万相的 parameters.resolution 收的是 "480P"/"720P"/"1080P" 档位字符串。"""
    h = wh[1]
    if h >= 1000:
        return "1080P"
    if h >= 700:
        return "720P"
    return "480P"


def _require_public_url(uri: str, what: str) -> str:
    """云端 API 只能吃公网可达的 URL / OSS 地址，本地路径必须先上传。

    在这里拦掉，比让服务端返回一个语焉不详的 InvalidURL 再回来排查快得多。
    """
    if uri.startswith(("http://", "https://", "oss://")):
        return uri
    raise ProviderError(
        FailureKind.BAD_REQUEST,
        f"{what} 必须是公网可达的 URL（收到 {uri!r}）；本地文件请先经 assets 上传到 OSS",
    )


class DashScopeWanProvider(VideoProvider):
    """百炼万相。画质天花板之一，且是唯一支持官方 LoRA 的闭源通道。"""

    def __init__(
        self,
        *,
        model: str = "wan2.7-i2v",
        api_key: str | None = None,
        base_url: str | None = None,
        out_dir: str | Path = "out/dashscope",
        name: str | None = None,
        cost_per_second_usd: float = 0.0,
        watermark: bool = True,        # 合规：AI 生成标识默认开，交付端再决定是否去掉
        prompt_extend: bool = False,   # 产线自己有提示词 OS，让平台改写反而破坏一致性
        timeout_s: float = 60.0,
    ) -> None:
        if model not in WAN_MODELS:
            raise ValueError(f"未登记的万相模型 {model!r}；已登记 {sorted(WAN_MODELS)}")
        spec = WAN_MODELS[model]
        if cost_per_second_usd <= 0:
            # 成本为 0 会让 router 的成本打分把这家判成「免费」，进而永远优先选它。
            # 上线前必须从百炼计费页填真实单价（2026-09-19 未查证到逐模型价格）。
            log.warning("%s 未配置 cost_per_second_usd，router 的成本打分不可信", model)
        caps = Capabilities(
            name=name or f"dashscope_{model}",
            kind="official",
            min_duration_s=spec.min_duration_s,
            max_duration_s=spec.max_duration_s,
            duration_steps=spec.duration_steps,
            resolutions=spec.resolutions,
            fps_options=(24,),
            max_ref_images=2 if spec.supports_last_frame else 1,
            max_ref_audios=1 if spec.supports_audio_drive else 0,
            supported_image_roles=frozenset(
                {"first_frame", "last_frame"} if spec.supports_last_frame else {"first_frame"}
            ),
            supports_first_frame=True,
            supports_last_frame=spec.supports_last_frame,
            supports_extend=spec.supports_extend,
            supports_job_continue=False,       # 续写喂的是视频 URL（first_clip），不是 job_id
            supports_native_audio=spec.supports_audio_drive,
            supports_lora=spec.finetunable,    # 只能挂平台侧训出来的 finetuned_output
            supports_seed=True,
            supports_negative_prompt=True,
            supports_camera_control=False,
            moderated=True,
            max_concurrency=4,
            cost_per_second_usd=cost_per_second_usd,
            typical_latency_s=150.0,
            quality_tier=spec.quality_tier,
            notes="百炼万相；LoRA 需先在平台微调并部署，产出不可下载",
        )
        super().__init__(caps)
        self.spec = spec
        self.model = model
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("DASHSCOPE_BASE_URL") or BASE_CN).rstrip("/")
        self.out_dir = Path(out_dir)
        self.watermark = watermark
        self.prompt_extend = prompt_extend
        self.timeout_s = timeout_s
        self._jobs: dict[str, dict[str, Any]] = {}

    # ---- 鉴权与探活

    def _auth(self) -> str:
        if not self._api_key:
            raise ProviderError(
                FailureKind.AUTH,
                "缺少 DASHSCOPE_API_KEY 环境变量，无法调用百炼万相；"
                "微调相关接口另需新加坡地域的 Key",
            )
        return self._api_key

    def _headers(self, *, async_task: bool = False) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self._auth()}", "Content-Type": "application/json"}
        if async_task:
            h["X-DashScope-Async"] = "enable"
        return h

    def health(self) -> bool:
        if not self._api_key:
            return False
        # 查一个必然不存在的 task：只要服务端回了 4xx 业务错误就说明链路和鉴权是通的。
        try:
            code, _ = _http.request_json(
                "GET",
                self.base_url + TASK_PATH.format(task_id="longfilm-health-probe"),
                headers=self._headers(),
                timeout=5.0,
            )
        except ProviderError:
            return False
        return code not in (401, 403) and code < 500

    # ---- 请求编译

    def _gate(self, req: GenRequest) -> None:
        if req.content_rating in _OFFICIAL_BLOCKED_RATINGS:
            kind = (
                FailureKind.BAD_REQUEST if req.content_rating == "blocked" else FailureKind.MODERATION
            )
            raise ProviderError(
                kind,
                f"{req.shot_id or '该镜'} 分级 {req.content_rating} 不投官方通道"
                f"（{'产线拒单' if kind is FailureKind.BAD_REQUEST else '改由开源引擎承接'}）",
            )

    def build_payload(self, req: GenRequest) -> dict[str, Any]:
        """把 GenRequest 翻译成万相的请求体。离线可测，不触网。"""
        first = req.first_frame_uri or next(
            (i.uri for i in req.refs.images if i.role == "first_frame"), None
        )
        last = req.last_frame_uri or next(
            (i.uri for i in req.refs.images if i.role == "last_frame"), None
        )
        inp: dict[str, Any] = {"prompt": req.prompt}
        if req.negative_prompt:
            inp["negative_prompt"] = req.negative_prompt

        if self.spec.api_style == "media":
            media: list[dict[str, str]] = []
            if first:
                media.append({"type": "first_frame", "url": _require_public_url(first, "首帧")})
            if last:
                media.append({"type": "last_frame", "url": _require_public_url(last, "尾帧")})
            if req.audio_uri and self.spec.supports_audio_drive:
                media.append({"type": "driving_audio",
                              "url": _require_public_url(req.audio_uri, "驱动音频")})
            if req.extend_from_video and self.spec.supports_extend:
                media.append({"type": "first_clip",
                              "url": _require_public_url(req.extend_from_video, "续写源视频")})
            if media:
                inp["media"] = media
        elif self.spec.api_style == "legacy_kf2v":
            if not (first and last):
                raise ProviderError(
                    FailureKind.BAD_REQUEST, f"{self.model} 是首尾帧模型，首帧与尾帧都必须提供"
                )
            inp["first_frame_url"] = _require_public_url(first, "首帧")
            inp["last_frame_url"] = _require_public_url(last, "尾帧")
        else:
            if not first:
                raise ProviderError(FailureKind.BAD_REQUEST, f"{self.model} 是图生视频模型，缺少首帧")
            inp["img_url"] = _require_public_url(first, "首帧")
            if req.audio_uri and self.spec.supports_audio_drive:
                inp["audio_url"] = _require_public_url(req.audio_uri, "驱动音频")

        params: dict[str, Any] = {
            "resolution": _resolution_label(self.caps.nearest_resolution(req.resolution)),
            "duration": int(round(self.caps.clamp_duration(req.duration_s))),
            "prompt_extend": self.prompt_extend,
            "watermark": self.watermark,
        }
        if req.seed is not None:
            params["seed"] = int(req.seed) % 2147483648

        model = self.model
        if req.lora:
            # 平台侧 LoRA 不是一个可挂载的权重，而是一个**微调后部署出来的模型名**。
            # 所以它替换的是 model 字段本身，不是 parameters 里的某个开关。
            deployed = req.lora.get("deployed_model") or req.lora.get("finetuned_output")
            if not deployed:
                raise ProviderError(
                    FailureKind.BAD_REQUEST,
                    "百炼侧 LoRA 必须给出已部署的模型名（lora.deployed_model）；"
                    "平台不允许下载 LoRA 权重，本地文件路径在这里没有意义",
                )
            model = str(deployed)
        return {"model": model, "input": inp, "parameters": params}

    # ---- 提交 / 轮询

    def submit(self, req: GenRequest) -> str:
        self._auth()
        self._gate(req)
        payload = self.build_payload(req)
        code, body = _http.request_json(
            "POST", self.base_url + SYNTH_PATH,
            headers=self._headers(async_task=True), json_body=payload, timeout=self.timeout_s,
        )
        if code != 200 or not isinstance(body, Mapping):
            raise ProviderError(_failure_of(code, body), _describe(code, body), raw=_as_raw(body))
        task_id = ((body.get("output") or {}) if isinstance(body.get("output"), Mapping) else {}).get(
            "task_id"
        )
        if not task_id:
            raise ProviderError(FailureKind.UNKNOWN, f"万相未返回 task_id：{body}", raw=_as_raw(body))
        self._jobs[str(task_id)] = {
            "submitted_at": time.time(), "shot_id": req.shot_id,
            "duration_s": float(payload["parameters"]["duration"]),
        }
        log.info("万相已提交 %s model=%s task_id=%s", req.shot_id or "?", payload["model"], task_id)
        return str(task_id)

    def poll(self, job_id: str) -> GenResult:
        meta = self._jobs.get(job_id, {})
        res = GenResult(
            job_id=job_id, status=JobStatus.PENDING, provider=self.name,
            submitted_at=meta.get("submitted_at", 0.0),
        )
        code, body = _http.request_json(
            "GET", self.base_url + TASK_PATH.format(task_id=job_id),
            headers=self._headers(), timeout=self.timeout_s,
        )
        if code != 200 or not isinstance(body, Mapping):
            res.status, res.failure = JobStatus.FAILED, _failure_of(code, body)
            res.message = _describe(code, body)
            return res

        out = body.get("output") if isinstance(body.get("output"), Mapping) else {}
        out = out or {}
        res.raw = {"task_status": out.get("task_status"), "usage": body.get("usage")}
        status = _TASK_STATUS.get(str(out.get("task_status", "")).upper())
        if status is None:
            res.status, res.failure = JobStatus.FAILED, FailureKind.UNKNOWN
            res.message = f"未知 task_status: {out.get('task_status')}"
            return res
        res.status = status

        if status in {JobStatus.PENDING, JobStatus.RUNNING}:
            return res
        res.finished_at = time.time()
        if status is not JobStatus.SUCCEEDED:
            res.failure = _CODE_MAP.get(str(out.get("code", "")), FailureKind.UNKNOWN)
            if res.failure is FailureKind.MODERATION:
                res.status = JobStatus.REJECTED
            res.message = f"{out.get('code')}: {out.get('message')}"
            return res

        url = out.get("video_url")
        if not url:
            res.status, res.failure = JobStatus.FAILED, FailureKind.UNKNOWN
            res.message = "task_status=SUCCEEDED 但没有 video_url"
            return res
        # 视频 URL 只活 24h，成功的第一件事就是转存本地，不能留着 URL 当交付物
        dest = self.out_dir / f"{meta.get('shot_id') or job_id}.mp4"
        res.video_uri = str(_http.download(str(url), dest, timeout=max(self.timeout_s, 600.0)))
        res.duration_s = float(meta.get("duration_s", 0.0))
        res.cost_usd = round(self.caps.cost_per_second_usd * res.duration_s, 4)
        return res

    def extend(self, req: GenRequest) -> str:
        """视频续写：万相 2.7 用 media.first_clip 喂上一段视频，而不是 job_id。"""
        if not self.caps.supports_extend:
            return super().extend(req)
        if not req.extend_from_video:
            raise ProviderError(
                FailureKind.BAD_REQUEST, f"{self.model} 的续写需要 extend_from_video（上一段视频 URL）"
            )
        return self.submit(req)


# ---------------------------------------------------------------- 错误翻译


def _as_raw(body: Any) -> dict[str, Any]:
    return dict(body) if isinstance(body, Mapping) else {"body": body}


def _failure_of(code: int, body: Any) -> FailureKind:
    biz = str(body.get("code", "")) if isinstance(body, Mapping) else ""
    return _CODE_MAP.get(biz) or _http.status_to_failure(code)


def _describe(code: int, body: Any) -> str:
    if isinstance(body, Mapping) and body.get("code"):
        return f"万相 HTTP {code} {body.get('code')}: {body.get('message')}"
    return f"万相 HTTP {code}: {body}"


# ---------------------------------------------------------------- LoRA 训练


@dataclass
class TrainingJob:
    job_id: str
    status: str
    model: str = ""
    finetuned_output: str = ""      # 部署/调用时当 model 名用
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.status.upper() in {"SUCCEEDED", "FAILED", "CANCELED"}


class WanLoRATrainingClient:
    """百炼万相视频模型的官方 LoRA 微调客户端。

    **查证结论（2026-09-19）：接口真实存在，不是宣传话术。** 三点必须写进产线设计：

    1. **产出不可下载。** 微调只交付一个 ``finetuned_output`` 模型名，部署成在线
       服务后当 ``model`` 字段调用。所以百炼 LoRA 与自训 ComfyUI LoRA
       **不可互换**，角色一致性资产会按引擎分叉 —— CharacterBible 里同一个角色
       需要维护两份 LoRA 记录。
    2. **仅新加坡地域可用**，且必须用该地域的 API Key，与推理用的北京 Key 不通用。
       本类默认 base_url 因此是 dashscope-intl，不跟着 provider 走。
    3. 计费 = 训练 Tokens 总量 × 单价；模型部署免费，调用按基座标准价。
       具体单价 2026-09-19 未逐项查证，排期前请查百炼计费页。
    """

    FILES_PATH = "/compatible-mode/v1/files"
    JOBS_PATH = "/api/v1/fine-tunes"
    # TODO(2026-09-19)：部署接口的请求体字段 model_name 已查证，但 POST 路径
    # /api/v1/deployments 本次未在官方文档中逐字确认，上线前需核对。
    DEPLOY_PATH = "/api/v1/deployments"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = BASE_INTL,
        timeout_s: float = 120.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or ""
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _auth(self) -> str:
        if not self._api_key:
            raise ProviderError(
                FailureKind.AUTH,
                "缺少 DASHSCOPE_API_KEY，无法提交万相 LoRA 训练；"
                "注意微调仅新加坡地域可用，需该地域的 Key",
            )
        return self._api_key

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._auth()}", "Content-Type": "application/json"}

    # ---- 数据集

    def upload_dataset(self, zip_path: str | Path) -> str:
        """上传训练集 zip（首帧图 + 训练视频 + data.jsonl），返回 file_id。"""
        p = Path(zip_path)
        if not p.is_file():
            raise ProviderError(FailureKind.BAD_REQUEST, f"训练集不存在：{p}")
        code, body = _http.post_multipart(
            self.base_url + self.FILES_PATH,
            fields={"purpose": "fine-tune"},
            files={"file": (p.name, p.read_bytes())},
            headers={"Authorization": f"Bearer {self._auth()}"},
            timeout=max(self.timeout_s, 900.0),
        )
        if code >= 400 or not isinstance(body, Mapping) or not body.get("id"):
            raise ProviderError(_failure_of(code, body), f"上传训练集失败 HTTP {code}: {body}")
        return str(body["id"])

    @staticmethod
    def build_manifest(
        rows: Sequence[Mapping[str, str]], *, task_type: str = "i2v"
    ) -> str:
        """生成 data.jsonl 内容。

        每行：``{"prompt","first_frame_path","video_path"}``；
        首尾帧任务（task_type="kf2v"）额外要求 ``last_frame_path``。
        prompt 里必须带触发词，否则训出来的 LoRA 无法被定向唤起。
        """
        import json  # noqa: PLC0415  只在这条路径上用，不污染模块级导入

        need = {"prompt", "first_frame_path", "video_path"}
        if task_type == "kf2v":
            need = need | {"last_frame_path"}
        lines: list[str] = []
        for i, row in enumerate(rows):
            missing = need - set(row)
            if missing:
                raise ProviderError(
                    FailureKind.BAD_REQUEST, f"data.jsonl 第 {i + 1} 行缺字段 {sorted(missing)}"
                )
            lines.append(json.dumps({k: row[k] for k in need}, ensure_ascii=False))
        return "\n".join(lines) + "\n"

    # ---- 任务

    def create(
        self,
        *,
        model: str = "wan2.7-i2v",
        training_file_ids: Sequence[str],
        task_type: str = "i2v",
        validation_file_ids: Sequence[str] = (),
        hyper_parameters: Mapping[str, Any] | None = None,
        job_name: str = "",
    ) -> TrainingJob:
        """创建 LoRA 微调任务。hyper_parameters 的默认值取自官方示例。"""
        spec = WAN_MODELS.get(model)
        if spec is None or not spec.finetunable:
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                f"{model} 不在可微调基座列表；可微调："
                f"{sorted(m for m, s in WAN_MODELS.items() if s.finetunable)}",
            )
        if task_type not in {"i2v", "kf2v"}:
            raise ProviderError(FailureKind.BAD_REQUEST, f"task_type 只能是 i2v/kf2v，收到 {task_type!r}")
        if not training_file_ids:
            raise ProviderError(FailureKind.BAD_REQUEST, "training_file_ids 不能为空")

        hp: dict[str, Any] = {
            "task_type": task_type,
            "n_epochs": 50,
            "batch_size": 1,
            "learning_rate": 2e-5,
            "split": 0.9,
            "lora_rank": 32,
            "lora_alpha": 32,
        }
        hp.update(hyper_parameters or {})
        payload: dict[str, Any] = {
            "model": model,
            "training_file_ids": list(training_file_ids),
            "training_type": "efficient_sft",     # 万相视频侧的 LoRA 训练类型
            "hyper_parameters": hp,
        }
        if validation_file_ids:
            payload["validation_file_ids"] = list(validation_file_ids)
        if job_name:
            payload["job_name"] = job_name

        code, body = _http.request_json(
            "POST", self.base_url + self.JOBS_PATH,
            headers=self._headers(), json_body=payload, timeout=self.timeout_s,
        )
        return self._job_from(code, body)

    def get(self, job_id: str) -> TrainingJob:
        code, body = _http.request_json(
            "GET", f"{self.base_url}{self.JOBS_PATH}/{job_id}",
            headers=self._headers(), timeout=self.timeout_s,
        )
        return self._job_from(code, body)

    def list(self, *, page_no: int = 1, page_size: int = 20) -> list[TrainingJob]:
        code, body = _http.request_json(
            "GET", f"{self.base_url}{self.JOBS_PATH}?page_no={page_no}&page_size={page_size}",
            headers=self._headers(), timeout=self.timeout_s,
        )
        if code >= 400 or not isinstance(body, Mapping):
            raise ProviderError(_failure_of(code, body), f"列出微调任务失败 HTTP {code}: {body}")
        out = body.get("output") if isinstance(body.get("output"), Mapping) else {}
        jobs = (out or {}).get("jobs") or []
        return [self._job_of(j) for j in jobs if isinstance(j, Mapping)]

    def checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        """列出检查点。每项的 model_name 才是部署/调用时用的名字。"""
        code, body = _http.request_json(
            "GET", f"{self.base_url}{self.JOBS_PATH}/{job_id}/checkpoints",
            headers=self._headers(), timeout=self.timeout_s,
        )
        if code >= 400 or not isinstance(body, Mapping):
            raise ProviderError(_failure_of(code, body), f"查询检查点失败 HTTP {code}: {body}")
        out = body.get("output") if isinstance(body.get("output"), Mapping) else {}
        return [dict(c) for c in ((out or {}).get("checkpoints") or []) if isinstance(c, Mapping)]

    def cancel(self, job_id: str) -> TrainingJob:
        code, body = _http.request_json(
            "POST", f"{self.base_url}{self.JOBS_PATH}/{job_id}/cancel",
            headers=self._headers(), json_body={}, timeout=self.timeout_s,
        )
        return self._job_from(code, body)

    def deploy(self, model_name: str) -> dict[str, Any]:
        """把训好的检查点部署成在线服务。部署免费，调用按基座价计费。"""
        code, body = _http.request_json(
            "POST", self.base_url + self.DEPLOY_PATH,
            headers=self._headers(), json_body={"model_name": model_name},
            timeout=self.timeout_s,
        )
        if code >= 400 or not isinstance(body, Mapping):
            raise ProviderError(_failure_of(code, body), f"部署失败 HTTP {code}: {body}")
        return dict(body)

    def download_weights(self, job_id: str) -> Path:
        """百炼**不提供** LoRA 权重下载 —— 这不是没实现，是平台侧不存在这条路径。"""
        raise NotImplementedError(
            "百炼万相微调产出的 LoRA 不可下载（2026-09-19 查证官方文档确认）："
            "只能部署成在线服务后用 finetuned_output 模型名云端调用。"
            f"需要可迁移的权重请走自建 ComfyUI 侧训练（job_id={job_id}）"
        )

    def _job_from(self, code: int, body: Any) -> TrainingJob:
        if code >= 400 or not isinstance(body, Mapping):
            raise ProviderError(_failure_of(code, body), f"微调接口 HTTP {code}: {body}")
        out = body.get("output") if isinstance(body.get("output"), Mapping) else body
        return self._job_of(out if isinstance(out, Mapping) else {})

    @staticmethod
    def _job_of(d: Mapping[str, Any]) -> TrainingJob:
        return TrainingJob(
            job_id=str(d.get("job_id", "")),
            status=str(d.get("status", "UNKNOWN")),
            model=str(d.get("model", "")),
            finetuned_output=str(d.get("finetuned_output", "")),
            message=str(d.get("message", "")),
            raw=dict(d),
        )


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    logging.basicConfig(level=logging.ERROR)   # 静掉成本告警，自测本来就没配价格

    # 1) caps 声明按模型分化，且 router 关心的位都对
    p27 = DashScopeWanProvider(model="wan2.7-i2v", api_key="")
    assert p27.caps.kind == "official" and p27.caps.moderated is True
    assert p27.caps.supports_first_frame and p27.caps.supports_last_frame
    assert p27.caps.supports_extend and p27.caps.supports_native_audio
    assert p27.caps.supports_lora, "wan2.7-i2v 可微调，应声明 supports_lora"
    assert p27.caps.fits_duration(8.0) and not p27.caps.fits_duration(20.0)

    p21 = DashScopeWanProvider(model="wan2.1-i2v-turbo", api_key="")
    assert not p21.caps.supports_last_frame and not p21.caps.supports_lora
    assert p21.caps.duration_steps == (5.0,) and p21.caps.clamp_duration(8.0) == 5.0

    pkf = DashScopeWanProvider(model="wan2.2-kf2v-flash", api_key="")
    assert pkf.caps.supports_last_frame and pkf.spec.api_style == "legacy_kf2v"

    try:
        DashScopeWanProvider(model="wanx-nope")
    except ValueError as e:
        assert "未登记" in str(e)
    else:
        raise AssertionError("未登记的模型名应当拒绝构造")

    # 2) 无 key：health() 为 False，submit 抛 AUTH（不是崩）
    req = GenRequest(
        prompt="a fictional adult courier crossing a rain-lit bridge",
        negative_prompt="lowres", duration_s=8.0, resolution=(1920, 1080), seed=7,
        first_frame_uri="https://example.com/ff.png",
        last_frame_uri="https://example.com/lf.png",
        shot_id="s02_sh01",
    )
    assert p27.health() is False
    for prov in (p27, p21, pkf):
        try:
            prov.submit(req)
        except ProviderError as e:
            assert e.kind is FailureKind.AUTH, (prov.name, e.kind)
        else:
            raise AssertionError(f"{prov.name} 无 key 时应抛 AUTH")

    # 3) 请求体编译：2.7 走 media 数组，旧代走 img_url
    p27k = DashScopeWanProvider(model="wan2.7-i2v", api_key="sk-fake-for-selftest")
    body = p27k.build_payload(req)
    assert body["model"] == "wan2.7-i2v"
    assert [m["type"] for m in body["input"]["media"]] == ["first_frame", "last_frame"]
    assert body["parameters"] == {
        "resolution": "1080P", "duration": 8, "prompt_extend": False,
        "watermark": True, "seed": 7,
    }
    p22k = DashScopeWanProvider(model="wan2.2-i2v-flash", api_key="sk-fake-for-selftest")
    body22 = p22k.build_payload(req)
    assert body22["input"]["img_url"] == "https://example.com/ff.png"
    assert "media" not in body22["input"]
    assert body22["parameters"]["duration"] == 5 and body22["parameters"]["resolution"] == "720P"

    # 4) 本地路径必须被拦下（云端 API 吃不到本机文件）
    try:
        p27k.build_payload(GenRequest(prompt="x", duration_s=5.0, first_frame_uri="/tmp/a.png"))
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST and "公网可达" in str(e)
    else:
        raise AssertionError("本地路径应当被拒")

    # 5) 首尾帧模型缺尾帧时报清楚的错
    try:
        pkf_k = DashScopeWanProvider(model="wan2.2-kf2v-flash", api_key="sk-fake-for-selftest")
        pkf_k.build_payload(GenRequest(prompt="x", duration_s=5.0,
                                       first_frame_uri="https://e.com/a.png"))
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST and "尾帧" in str(e)
    else:
        raise AssertionError("kf2v 缺尾帧应当报错")

    # 6) LoRA：必须给已部署模型名，本地路径在这条通道上没有意义
    body_lora = p27k.build_payload(
        GenRequest(prompt="x", duration_s=5.0, first_frame_uri="https://e.com/a.png",
                   lora={"deployed_model": "ft-wan27-hero-v3"})
    )
    assert body_lora["model"] == "ft-wan27-hero-v3"
    try:
        p27k.build_payload(GenRequest(prompt="x", duration_s=5.0,
                                      first_frame_uri="https://e.com/a.png",
                                      lora={"name": "hero.safetensors"}))
    except ProviderError as e:
        assert "已部署的模型名" in str(e)
    else:
        raise AssertionError("本地 LoRA 文件名在百炼侧应当被拒")

    # 7) 内容门禁：官方通道只做路由与拒单
    for rating, kind in (("blocked", FailureKind.BAD_REQUEST),
                         ("r_violence", FailureKind.MODERATION),
                         ("r_suggestive", FailureKind.MODERATION)):
        try:
            p27k.submit(GenRequest(prompt="x", duration_s=5.0, content_rating=rating,
                                   first_frame_uri="https://e.com/a.png", shot_id="s9"))
        except ProviderError as e:
            assert e.kind is kind, (rating, e.kind)
            assert e.kind.should_failover or rating == "blocked"
        else:
            raise AssertionError(f"分级 {rating} 应当被官方通道拒单")

    # 8) 错误码映射：审核拒不可重试、限流可重试
    assert _CODE_MAP["DataInspectionFailed"] is FailureKind.MODERATION
    assert not FailureKind.MODERATION.retryable and FailureKind.MODERATION.should_failover
    assert _CODE_MAP["Throttling"].retryable
    assert _failure_of(200, {"code": "DataInspectionFailed"}) is FailureKind.MODERATION
    assert _failure_of(401, {}) is FailureKind.AUTH
    assert _failure_of(500, {}) is FailureKind.SERVER

    # 9) 分辨率档位
    assert _resolution_label((1920, 1080)) == "1080P"
    assert _resolution_label((1280, 720)) == "720P"
    assert _resolution_label((854, 480)) == "480P"

    # 10) LoRA 训练客户端：无 key 全线抛 AUTH，且下载权重永远 NotImplementedError
    tc = WanLoRATrainingClient(api_key="")
    assert tc.base_url == BASE_INTL, "微调仅新加坡地域可用，默认必须指向 intl"
    for call in (
        lambda: tc.create(model="wan2.7-i2v", training_file_ids=["f-1"]),
        lambda: tc.get("job-1"),
        lambda: tc.list(),
        lambda: tc.checkpoints("job-1"),
        lambda: tc.cancel("job-1"),
        lambda: tc.deploy("ft-x"),
    ):
        try:
            call()
        except ProviderError as e:
            assert e.kind is FailureKind.AUTH, e.kind
        else:
            raise AssertionError("无 key 时微调接口应抛 AUTH")

    try:
        tc.download_weights("job-1")
    except NotImplementedError as e:
        assert "不可下载" in str(e) and "2026-09-19" in str(e)
    else:
        raise AssertionError("百炼不提供权重下载，必须显式 NotImplementedError")

    # 11) 训练参数与 data.jsonl 校验（纯本地，不触网）
    tck = WanLoRATrainingClient(api_key="sk-fake-for-selftest")
    try:
        tck.create(model="wan2.2-i2v-plus", training_file_ids=["f-1"])
    except ProviderError as e:
        assert "不在可微调基座列表" in str(e)
    else:
        raise AssertionError("不可微调的基座应当被拒")
    try:
        tck.create(model="wan2.7-i2v", training_file_ids=[])
    except ProviderError as e:
        assert "不能为空" in str(e)
    else:
        raise AssertionError("空训练集应当被拒")

    jsonl = WanLoRATrainingClient.build_manifest(
        [{"prompt": "hero_trigger 一个虚构成年角色在雨夜行走",
          "first_frame_path": "image/1.jpg", "video_path": "video/1.mp4"}]
    )
    assert jsonl.endswith("\n") and "hero_trigger" in jsonl
    try:
        WanLoRATrainingClient.build_manifest(
            [{"prompt": "x", "first_frame_path": "a.jpg", "video_path": "v.mp4"}],
            task_type="kf2v",
        )
    except ProviderError as e:
        assert "last_frame_path" in str(e)
    else:
        raise AssertionError("kf2v 缺尾帧路径应当被拒")

    # 12) 密钥绝不出现在任何异常文本里
    secret = "sk-fake-for-selftest"
    try:
        p27k.build_payload(GenRequest(prompt="x", duration_s=5.0, first_frame_uri="/tmp/a.png"))
    except ProviderError as e:
        assert secret not in str(e)

    print("dashscope_wan selftest OK：8 个模型 caps / 两代请求体 / 错误码映射 / "
          "LoRA 训练链路（证实存在，产出不可下载）全部通过")


if __name__ == "__main__":
    _selftest()
