"""火山方舟（Volcengine Ark）Seedance 视频生成 provider。

规格查证日期：2026-09-19。来源见 _RESEARCH_NOTES。
方舟是本产线的「画质天花板 + 强审核」那一侧：G/PG13 的常规镜走它，
审核拒绝时 router 读 FailureKind.MODERATION 把镜头甩给开源侧，
所以本文件里**状态映射的准确性比功能多寡更重要**。

未查证到官方文档明确说明的字段一律标 TODO，不臆造。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..schema import ImageRef
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

_RESEARCH_NOTES = """查证于 2026-09-19：
- 创建任务  POST {base}/contents/generations/tasks
  头 Authorization: Bearer $ARK_API_KEY, Content-Type: application/json
  体 {model, content[], resolution?, ratio?, duration?, seed?, watermark?,
      camera_fixed?, generate_audio?, return_last_frame?, callback_url?,
      execution_expires_after?, priority?, safety_identifier?}
  content 元素：{type:"text",text} / {type:"image_url",image_url:{url},role}
                {type:"video_url",video_url:{url},role} / {type:"audio_url",...}
  image role ∈ first_frame | last_frame | reference_image
  video role = reference_video；audio role = reference_audio
  来源 doubao.apifox.cn/265914813e0、docs.aitoearn.ai（方舟原生格式透传文档）、
       docs.console.zenlayer.com Seedance create-video
- 查询任务  GET {base}/contents/generations/tasks/{task_id}
  status ∈ queued | running | succeeded | failed（+ cancelled，见 TODO）
  来源 docs.omove.ai/9049227m0（方舟原生格式透传）
- 公共错误码（doubao.apifox.cn/6107465m0）：
  AuthenticationError 401 / AccountOverdueError 403 / AccessDenied 403
  RateLimitExceeded.EndpointRPMExceeded|EndpointTPMExceeded 429
  ModelAccountRpmRateLimitExceeded / ModelAccountTpmRateLimitExceeded 429
  QuotaExceeded 429（免费额度耗尽）/ ModelLoadingError 429 / ServerOverloaded 429
  MissingParameter 400 / InvalidParameter 400 / InvalidEndpoint.NotFound 404
- 审核码：InputImageSensitiveContentDetected 已被实际使用者报告（2026-02 起
  人脸图会命中）。同族的 InputTextSensitiveContentDetected /
  OutputVideoSensitiveContentDetected 未在公共错误码页出现 —— 见 TODO，
  因此审核判定**不只靠码名枚举**，还兜底匹配 SensitiveContent/Sensitive 子串。
- 计费：token 数 = 宽 × 高 × 帧率 × 时长 / 1024，仅成功任务计费，
  审核失败不计费（volcengine.com/docs/82379/1099320 模型价格页口径）。
- 时长：1.0 系列 2-12s，预设 5/10s；lite-i2v 分辨率 480p/720p/1080p，
  pro 480p/1080p（302.ai 产品详情页）。
"""

# ---------------------------------------------------------------- HTTP 小封装

_TIMEOUT_S = 60.0


def _http_json(
    method: str,
    url: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> tuple[int, dict[str, Any]]:
    """发一次 JSON 请求，返回 (http_status, 解析后的 body)。

    优先 requests，没装则退到标准库 urllib —— 本产线不为一个 POST 引入依赖。
    非 2xx **不抛异常**：方舟把审核拒绝、限流、欠费都编码在响应体的 error.code 里，
    调用方需要拿到码名才能分类，异常会把这个信息吞掉。
    """
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        import requests  # 延迟导入：缺它也能 import 本模块并跑翻译层自测
    except ImportError:
        requests = None  # type: ignore[assignment]

    if requests is not None:
        resp = requests.request(
            method, url, headers=headers, data=payload, timeout=timeout_s
        )
        return resp.status_code, _loads(resp.text)

    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return r.status, _loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return e.code, _loads(e.read().decode("utf-8", "replace"))
    except urllib.error.URLError as e:
        raise ProviderError(FailureKind.TIMEOUT, f"方舟网络不可达：{e.reason}") from e


def _loads(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"_raw_text": text[:2000]}
    return obj if isinstance(obj, dict) else {"_raw": obj}


# ---------------------------------------------------------------- 模型目录


@dataclass(frozen=True)
class _ModelSpec:
    """一个方舟模型接入点的能力档位。

    模型 ID 带日期后缀（doubao-<系列>-<版本>-<变体>-<YYMMDD>），
    方舟会新增后缀而不改旧的，所以这里写死具体后缀比写别名更可复现。
    """

    model_id: str
    resolutions: tuple[tuple[int, int], ...]
    duration_steps: tuple[float, ...]
    min_duration_s: float
    max_duration_s: float
    max_ref_images: int
    supports_first_frame: bool
    supports_last_frame: bool
    supports_ref_video: bool
    supports_native_audio: bool
    quality_tier: int
    # 元/百万 token。token = 宽×高×帧率×时长/1024（查证 2026-09-19）
    price_cny_per_mtoken: float
    note: str = ""


# 分辨率档位按方舟的 480p/720p/1080p 语义给出 16:9 像素，实际下发的是档位字符串。
_R480 = (854, 480)
_R720 = (1280, 720)
_R1080 = (1920, 1080)

MODEL_CATALOG: dict[str, _ModelSpec] = {
    "doubao-seedance-1-0-pro-250528": _ModelSpec(
        model_id="doubao-seedance-1-0-pro-250528",
        resolutions=(_R480, _R1080),
        duration_steps=(5.0, 10.0),
        min_duration_s=2.0,
        max_duration_s=12.0,
        max_ref_images=4,
        supports_first_frame=True,
        supports_last_frame=True,
        supports_ref_video=False,
        supports_native_audio=False,
        quality_tier=5,
        price_cny_per_mtoken=15.0,  # 0.015 元/千 token
        note="1.0 主力档。预设 5/10s；2-12s 连续档位未在官方页确认 → 见 TODO",
    ),
    "doubao-seedance-1-0-lite-i2v-250428": _ModelSpec(
        model_id="doubao-seedance-1-0-lite-i2v-250428",
        resolutions=(_R480, _R720, _R1080),
        duration_steps=(5.0, 10.0),
        min_duration_s=2.0,
        max_duration_s=12.0,
        max_ref_images=4,
        supports_first_frame=True,
        supports_last_frame=True,
        supports_ref_video=False,
        supports_native_audio=False,
        quality_tier=4,
        price_cny_per_mtoken=14.0,
        note="图生视频 lite，预演/草稿档",
    ),
    "doubao-seedance-1-0-lite-t2v-250428": _ModelSpec(
        model_id="doubao-seedance-1-0-lite-t2v-250428",
        resolutions=(_R480, _R720, _R1080),
        duration_steps=(5.0, 10.0),
        min_duration_s=2.0,
        max_duration_s=12.0,
        max_ref_images=0,
        supports_first_frame=False,
        supports_last_frame=False,
        supports_ref_video=False,
        supports_native_audio=False,
        quality_tier=4,
        price_cny_per_mtoken=14.0,
        note="纯文生视频 lite，不吃参考图",
    ),
    "doubao-seedance-2-0-260128": _ModelSpec(
        model_id="doubao-seedance-2-0-260128",
        resolutions=(_R480, _R720, _R1080),
        duration_steps=(),
        min_duration_s=4.0,
        max_duration_s=15.0,
        max_ref_images=9,
        supports_first_frame=True,
        supports_last_frame=True,
        supports_ref_video=True,       # 2.0 的「视频延长/接着拍」走 reference_video
        supports_native_audio=True,    # generate_audio
        quality_tier=5,
        price_cny_per_mtoken=46.0,     # 无视频输入的纯生成档；有视频输入 28 元/百万
        note="2.0：原生音频 + 参考视频续拍，extend 链靠它",
    ),
}

DEFAULT_MODEL = "doubao-seedance-1-0-pro-250528"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# TODO(2026-09-19)：以下几项官方文档页是 JS 渲染、抓不到正文，用二手镜像交叉验证后写入，
#   接到真 key 后需各跑一次确认：
#   1. pro-250528 是否只接受 5/10s 两档（镜像说"预设 5/10、范围 2-12"，语义有歧义）；
#   2. text 内嵌参数的正式拼写：镜像同时出现 --dur/--duration、--cf/--camerafixed、
#      --wm/--watermark 两套。本实现默认走**顶层 JSON 字段**（各镜像一致），
#      text 内嵌仅作为 inline_params=True 的备选；
#   3. 查询接口是否返回 cancelled 状态、以及取消任务的 DELETE 路径；
#   4. 审核拒绝落在 HTTP 400 的 error.code 还是任务 status=failed 的 error.code
#      （本实现两条路都映射到 MODERATION）；
#   5. 真实单价（元/百万 token）以控制台账单为准，此处取公开报价。

# ---------------------------------------------------------------- 错误码映射

# 精确码名 → FailureKind。前缀匹配在 _classify 里另做，这里只放全等命中的。
_ERROR_CODE_MAP: dict[str, FailureKind] = {
    "AuthenticationError": FailureKind.AUTH,
    "AccessDenied": FailureKind.AUTH,
    "AccountOverdueError": FailureKind.QUOTA,
    "QuotaExceeded": FailureKind.QUOTA,
    "ModelAccountRpmRateLimitExceeded": FailureKind.RATE_LIMIT,
    "ModelAccountTpmRateLimitExceeded": FailureKind.RATE_LIMIT,
    "ModelLoadingError": FailureKind.RATE_LIMIT,
    "ServerOverloaded": FailureKind.RATE_LIMIT,
    "MissingParameter": FailureKind.BAD_REQUEST,
    "InvalidParameter": FailureKind.BAD_REQUEST,
    "InvalidEndpoint.NotFound": FailureKind.BAD_REQUEST,
    "InvalidEndpointOrModel.NotFound": FailureKind.BAD_REQUEST,
    "InternalServiceError": FailureKind.SERVER,
    "InputImageSensitiveContentDetected": FailureKind.MODERATION,
    "InputTextSensitiveContentDetected": FailureKind.MODERATION,
    "OutputVideoSensitiveContentDetected": FailureKind.MODERATION,
    "SensitiveContentDetected": FailureKind.MODERATION,
}


def classify_error(code: str, http_status: int = 0) -> FailureKind:
    """方舟错误码 → FailureKind。

    码名枚举不完整（审核族的码方舟没有集中列出），所以先全等、再前缀、
    最后按子串兜底。审核这条必须宁可多判也不能漏判：漏判会让 router
    把一个永远过不了审的镜头在方舟上反复重试烧钱。
    """
    if code in _ERROR_CODE_MAP:
        return _ERROR_CODE_MAP[code]
    if code.startswith("RateLimitExceeded"):
        return FailureKind.RATE_LIMIT
    low = code.lower()
    if "sensitive" in low or "moderation" in low or "contentpolicy" in low:
        return FailureKind.MODERATION
    if "ratelimit" in low or "toomanyrequests" in low:
        return FailureKind.RATE_LIMIT
    if "overdue" in low or "quota" in low or "insufficientbalance" in low:
        return FailureKind.QUOTA
    if "auth" in low or "apikey" in low:
        return FailureKind.AUTH
    if "timeout" in low:
        return FailureKind.TIMEOUT
    return _classify_by_http(http_status)


def _classify_by_http(http_status: int) -> FailureKind:
    if http_status == 401:
        return FailureKind.AUTH
    if http_status == 403:
        return FailureKind.QUOTA
    if http_status == 429:
        return FailureKind.RATE_LIMIT
    if http_status == 408:
        return FailureKind.TIMEOUT
    if 500 <= http_status < 600:
        return FailureKind.SERVER
    if 400 <= http_status < 500:
        return FailureKind.BAD_REQUEST
    return FailureKind.UNKNOWN


# 方舟 status → 本产线 JobStatus。cancelled 未在文档确认，先按字面兜住。
_STATUS_MAP: dict[str, JobStatus] = {
    "queued": JobStatus.PENDING,
    "pending": JobStatus.PENDING,
    "running": JobStatus.RUNNING,
    "processing": JobStatus.RUNNING,
    "succeeded": JobStatus.SUCCEEDED,
    "success": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
    "cancelled": JobStatus.CANCELLED,
    "canceled": JobStatus.CANCELLED,
}

# ---------------------------------------------------------------- 参数翻译


def _resolution_tag(res: tuple[int, int]) -> str:
    """像素尺寸 → 方舟的分辨率档位字符串。按短边归档，竖屏也能落对档。"""
    short = min(res)
    for limit, tag in ((512, "480p"), (768, "720p"), (1152, "1080p")):
        if short <= limit:
            return tag
    return "4k"


_RATIO_TABLE: tuple[tuple[float, str], ...] = (
    (21 / 9, "21:9"),
    (16 / 9, "16:9"),
    (4 / 3, "4:3"),
    (1.0, "1:1"),
    (3 / 4, "3:4"),
    (9 / 16, "9:16"),
)


def _ratio_tag(res: tuple[int, int]) -> str:
    """像素尺寸 → 最接近的官方 ratio 档。方舟只认枚举值，不认任意比例。"""
    want = res[0] / max(res[1], 1)
    return min(_RATIO_TABLE, key=lambda kv: abs(kv[0] - want))[1]


# ImageRef.role → 方舟 image_url 的 role 字段。
# 产线有 9 种 role，方舟只有 3 种 —— 身份/服装/场景/风格/道具这些统统压成
# reference_image，压缩关系写在这里而不是散落在 submit 里，方便换供应商时替换。
_IMAGE_ROLE_MAP: dict[str, str] = {
    "first_frame": "first_frame",
    "last_frame": "last_frame",
    "identity": "reference_image",
    "wardrobe": "reference_image",
    "environment": "reference_image",
    "style": "reference_image",
    "prop": "reference_image",
    "lighting": "reference_image",
    "composition": "reference_image",
}

# 本产线的 CameraMove 值 → 方舟的运镜描述。方舟没有结构化运镜参数（只有
# camera_fixed 开关），运镜只能写进提示词，所以这里做的是**文案映射**。
_CAMERA_FIXED_MOVES = frozenset({"static locked-off camera"})


class ArkSeedanceProvider(VideoProvider):
    """火山方舟 Seedance。

    幂等：req.idempotency_key（= Shot.fingerprint()）命中则直接返回旧 job_id，
    不再发第二次 POST —— 视频任务是按成功产出计费的，重复提交等于重复扣费。
    幂等表默认只在进程内；长期产线应把 _idem_store 换成 Redis/DB，
    构造时传 idem_store=<dict-like> 即可。
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        inline_params: bool = False,
        generate_audio: bool = False,
        watermark: bool = False,
        idem_store: dict[str, str] | None = None,
        **cfg: Any,
    ):
        if model not in MODEL_CATALOG:
            raise ValueError(
                f"未知的方舟模型 {model!r}；已登记 {sorted(MODEL_CATALOG)}。"
                "新模型请先查证官方文档后补进 MODEL_CATALOG，不要直接透传。"
            )
        spec = MODEL_CATALOG[model]
        self.spec = spec
        self.inline_params = inline_params
        self.generate_audio = generate_audio and spec.supports_native_audio
        self.watermark = watermark
        self._idem: dict[str, str] = idem_store if idem_store is not None else {}
        # 提交时记下预估时长，poll 拿不到时长时用它算成本
        self._submitted: dict[str, tuple[float, GenRequest]] = {}

        self._api_key_override = api_key
        self._base_url = (base_url or os.environ.get("ARK_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")

        roles = {"first_frame", "last_frame", "reference_image"} if spec.max_ref_images else set()
        caps = Capabilities(
            name=f"ark:{model}",
            kind="official",
            min_duration_s=spec.min_duration_s,
            max_duration_s=spec.max_duration_s,
            duration_steps=spec.duration_steps,
            resolutions=spec.resolutions,
            fps_options=(24,),                       # 方舟输出固定 24fps
            max_ref_images=spec.max_ref_images,
            max_ref_videos=1 if spec.supports_ref_video else 0,
            max_ref_audios=1 if spec.supports_native_audio else 0,
            supported_image_roles=frozenset(roles),
            supports_first_frame=spec.supports_first_frame,
            supports_last_frame=spec.supports_last_frame,
            supports_extend=spec.supports_ref_video,
            supports_job_continue=False,             # 方舟按素材续，不按 job_id 续
            supports_native_audio=spec.supports_native_audio,
            supports_lora=False,                     # 闭源，挂不了自训 LoRA
            supports_seed=True,
            supports_negative_prompt=False,          # 无独立 negative 字段，见 _compose_prompt
            supports_camera_control=True,            # 仅 camera_fixed 开关 + 提示词
            moderated=True,
            max_concurrency=int(cfg.get("max_concurrency", 4)),
            cost_per_second_usd=0.0,                 # 按 token 计费，见 estimate_cost
            typical_latency_s=float(cfg.get("typical_latency_s", 150.0)),
            quality_tier=spec.quality_tier,
            notes=spec.note,
        )
        super().__init__(caps, **cfg)

    # ------------------------------------------------------------ 鉴权

    @property
    def api_key(self) -> str | None:
        return self._api_key_override or os.environ.get("ARK_API_KEY")

    def _require_key(self) -> str:
        key = self.api_key
        if not key:
            raise ProviderError(
                FailureKind.AUTH,
                "缺少 ARK_API_KEY：请设置环境变量或构造时传 api_key。"
                "本产线不把密钥写进代码或分镜文件。",
            )
        return key

    def health(self) -> bool:
        """探活只看密钥在不在。

        不发真请求：方舟没有免费的 ping 接口，任何真请求都可能计费或占 RPM 配额，
        而 router 在每次降级前都会调 health()。
        """
        return bool(self.api_key)

    # ------------------------------------------------------------ 请求体翻译

    def _compose_prompt(self, req: GenRequest) -> str:
        """拼提示词。

        方舟没有独立的 negative_prompt 字段（caps 已声明 False），
        router 若仍塞了 negative，这里以「避免出现 …」并进正文，
        总比静默丢弃强 —— 静默丢弃会让 QC 查不出为什么负面词没生效。
        """
        parts = [req.prompt.strip()]
        move = (req.camera_move or "").strip()
        if move and move not in _CAMERA_FIXED_MOVES:
            parts.append(move)
        neg = req.negative_prompt.strip()
        if neg:
            parts.append(f"避免出现：{neg}")
        text = ", ".join(p for p in parts if p)

        if self.inline_params:
            # 备选路径：把标量参数写进 text 尾部。拼写以二手镜像为准，见文件头 TODO。
            text += (
                f" --resolution {_resolution_tag(req.resolution)}"
                f" --ratio {_ratio_tag(req.resolution)}"
                f" --dur {int(round(self.caps.clamp_duration(req.duration_s)))}"
                f" --fps {req.fps}"
                f" --wm {'true' if self.watermark else 'false'}"
                f" --cf {'true' if move in _CAMERA_FIXED_MOVES else 'false'}"
            )
            if req.seed is not None:
                text += f" --seed {req.seed}"
        return text

    def _image_content(self, req: GenRequest) -> list[dict[str, Any]]:
        """把 RefPack + first/last_frame_uri 摊成方舟的 content 图片元素。

        顺序有意义：方舟的提示词可以用「图 1 / 图 2」指代，顺序变了效果就变，
        所以首帧固定排第一、尾帧第二，其余参考图按 RefPack 原序跟在后面。
        """
        spec = self.spec
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def push(uri: str, role: str) -> None:
            if uri in seen:
                return
            seen.add(uri)
            out.append({"type": "image_url", "image_url": {"url": uri}, "role": role})

        if req.first_frame_uri and spec.supports_first_frame:
            push(req.first_frame_uri, "first_frame")
        if req.last_frame_uri and spec.supports_last_frame:
            push(req.last_frame_uri, "last_frame")

        for ref in req.refs.images:
            if not isinstance(ref, ImageRef):
                continue
            role = _IMAGE_ROLE_MAP.get(ref.role)
            if role is None:
                continue
            if role == "first_frame" and not spec.supports_first_frame:
                continue
            if role == "last_frame" and not spec.supports_last_frame:
                continue
            push(ref.uri, role)

        if len(out) > spec.max_ref_images and spec.max_ref_images >= 0:
            # 超限直接报错而不是截断：截断会悄悄丢掉身份锚，
            # 裁剪是 refpack.fit_to_budget() 的职责，不在 provider 里猜。
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                f"{self.name} 最多接受 {spec.max_ref_images} 张图，本次 {len(out)} 张；"
                "请先用 refpack.fit_to_budget() 裁剪",
            )
        return out

    def build_body(self, req: GenRequest) -> dict[str, Any]:
        """GenRequest → 方舟请求体。纯函数，便于离线自测与 diff 复现。"""
        spec = self.spec
        duration = self.caps.clamp_duration(req.duration_s)
        res = self.caps.nearest_resolution(req.resolution)

        content: list[dict[str, Any]] = [{"type": "text", "text": self._compose_prompt(req)}]
        content.extend(self._image_content(req))

        if spec.supports_ref_video:
            src = req.extend_from_video or next(
                (v.uri for v in req.refs.videos if v.role in ("motion", "previz")), None
            )
            if src:
                content.append(
                    {"type": "video_url", "video_url": {"url": src}, "role": "reference_video"}
                )
        if spec.supports_native_audio and req.audio_uri:
            content.append(
                {"type": "audio_url", "audio_url": {"url": req.audio_uri}, "role": "reference_audio"}
            )

        body: dict[str, Any] = {"model": spec.model_id, "content": content}
        if not self.inline_params:
            body.update(
                {
                    "resolution": _resolution_tag(res),
                    "ratio": _ratio_tag(res),
                    "duration": int(round(duration)),
                    "watermark": self.watermark,
                    "camera_fixed": (req.camera_move or "") in _CAMERA_FIXED_MOVES,
                    "return_last_frame": True,   # 首尾帧锁戏链要拿尾帧，能直出就别再解码
                }
            )
            if req.seed is not None:
                body["seed"] = int(req.seed)
            if spec.supports_native_audio:
                body["generate_audio"] = self.generate_audio
        if req.shot_id:
            # safety_identifier 是方舟的终端用户标识位，用镜头 ID 便于在审核工单里回溯
            body["safety_identifier"] = req.shot_id
        return body

    # ------------------------------------------------------------ 提交 / 轮询

    def submit(self, req: GenRequest) -> str:
        key = self._require_key()
        if req.idempotency_key and req.idempotency_key in self._idem:
            job_id = self._idem[req.idempotency_key]
            log.info("方舟幂等命中 %s -> %s，跳过重复提交", req.idempotency_key, job_id)
            return job_id

        body = self.build_body(req)
        status, data = _http_json(
            "POST", f"{self._base_url}/contents/generations/tasks", key, body
        )
        job_id = data.get("id")
        if status >= 400 or not job_id:
            err = data.get("error") or {}
            code = str(err.get("code") or data.get("code") or "")
            raise ProviderError(
                classify_error(code, status),
                f"方舟创建任务失败 http={status} code={code or '<无>'}: "
                f"{err.get('message') or data.get('message') or data}",
                raw=data,
            )

        self._submitted[job_id] = (self.caps.clamp_duration(req.duration_s), req)
        if req.idempotency_key:
            self._idem[req.idempotency_key] = job_id
        log.info("方舟任务已提交 shot=%s job=%s model=%s", req.shot_id, job_id, self.spec.model_id)
        return job_id

    def poll(self, job_id: str) -> GenResult:
        key = self._require_key()
        status, data = _http_json(
            "GET", f"{self._base_url}/contents/generations/tasks/{job_id}", key
        )
        return self._to_result(job_id, status, data)

    def _to_result(self, job_id: str, http_status: int, data: dict[str, Any]) -> GenResult:
        """查询响应 → GenResult。审核拒绝的映射是这里唯一不能错的一条。"""
        res = GenResult(job_id=job_id, status=JobStatus.PENDING, provider=self.name, raw=data)
        err = data.get("error") or {}
        code = str(err.get("code") or "")

        if http_status >= 400:
            res.failure = classify_error(code, http_status)
            res.message = str(err.get("message") or data)
            # 审核拒绝要落 REJECTED 而不是 FAILED：router 靠这个状态直接换引擎，
            # 落成 FAILED 会被当成可重试，在方舟上白烧配额。
            res.status = (
                JobStatus.REJECTED if res.failure is FailureKind.MODERATION else JobStatus.FAILED
            )
            return res

        raw_status = str(data.get("status") or "").lower()
        res.status = _STATUS_MAP.get(raw_status, JobStatus.PENDING)
        if raw_status and raw_status not in _STATUS_MAP:
            log.warning("方舟返回未知状态 %r（job=%s），暂按 PENDING 处理", raw_status, job_id)

        if res.status is JobStatus.FAILED:
            res.failure = classify_error(code, 0) if code else FailureKind.UNKNOWN
            res.message = str(err.get("message") or "方舟任务失败但未给出 error.message")
            if res.failure is FailureKind.MODERATION:
                res.status = JobStatus.REJECTED

        content = data.get("content") or {}
        if isinstance(content, dict):
            res.video_uri = content.get("video_url")
            res.last_frame_uri = content.get("last_frame_url")
            res.audio_uri = content.get("audio_url")

        dur, _req = self._submitted.get(job_id, (0.0, None))
        res.duration_s = float(data.get("duration") or dur or 0.0)
        res.submitted_at = _epoch(data.get("created_at"))
        res.finished_at = _epoch(data.get("updated_at"))

        if res.status is JobStatus.SUCCEEDED:
            # 仅成功任务计费 —— 失败/审核拒绝不扣费，成本必须挂在这一支
            usage = data.get("usage") or {}
            tokens = usage.get("completion_tokens") or usage.get("total_tokens")
            res.cost_usd = (
                self._tokens_to_usd(float(tokens))
                if tokens
                else self._tokens_to_usd(self._estimate_tokens(res.duration_s, self.caps.resolutions[-1], 24))
            )
            if not res.video_uri:
                res.status = JobStatus.FAILED
                res.failure = FailureKind.SERVER
                res.message = "方舟报 succeeded 但 content.video_url 为空"
        return res

    def cancel(self, job_id: str) -> None:
        # TODO(2026-09-19)：取消任务的 DELETE 路径未在可抓取的文档里确认。
        # 贸然 DELETE 到猜的路径可能命中别的资源，所以这里只记日志不发请求。
        log.warning("方舟 cancel 未实现（接口路径待查证），job=%s 将自行跑完", job_id)

    # ------------------------------------------------------------ extend

    def extend(self, req: GenRequest) -> str:
        """官方延长链。

        方舟没有独立的 extend 接口：2.0 系列把「视频延长/接着拍」做成了
        reference_video 素材 + 提示词，走的还是创建任务那个 endpoint。
        1.x 系列不支持参考视频，caps.supports_extend 已为 False。
        """
        if not self.spec.supports_ref_video:
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                f"{self.name} 不支持 extend：{self.spec.model_id} 无 reference_video 位，"
                "请用 seedance 2.0 系列，或改走首尾帧续接",
            )
        if not (req.extend_from_video or req.refs.videos):
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                "extend 需要 extend_from_video（上一段成片 URL）或 refs.videos 里的 motion 参考",
            )
        return self.submit(req)

    # ------------------------------------------------------------ 成本

    @staticmethod
    def _estimate_tokens(duration_s: float, res: tuple[int, int], fps: int) -> float:
        """方舟视频 token 公式：宽 × 高 × 帧率 × 时长 / 1024（查证 2026-09-19）。"""
        return res[0] * res[1] * fps * duration_s / 1024.0

    # 成本按 USD 记（GenResult.cost_usd），方舟报价是人民币，需折算。
    # TODO(2026-09-19)：汇率写死不合适，接生产时应从配置或汇率服务读。
    CNY_PER_USD = 7.1

    def _tokens_to_usd(self, tokens: float) -> float:
        cny = tokens / 1_000_000.0 * self.spec.price_cny_per_mtoken
        return round(cny / self.CNY_PER_USD, 6)

    def estimate_cost(self, req: GenRequest) -> float:
        """覆写基类的「按秒计价」：方舟按 token 计，分辨率对价格的影响是平方级的。"""
        res = self.caps.nearest_resolution(req.resolution)
        dur = self.caps.clamp_duration(req.duration_s)
        return self._tokens_to_usd(self._estimate_tokens(dur, res, 24))


def _epoch(v: Any) -> float:
    """方舟的 created_at/updated_at 是秒级 Unix 时间戳；非数字一律当作未知。"""
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def register_defaults() -> list[ArkSeedanceProvider]:
    """把目录里的模型全部注册进 REGISTRY。

    无 key 也注册：router 的候选集是静态的，能力筛选靠 caps，
    密钥缺失由 health() 在降级时暴露，而不是让 provider 在启动期消失 ——
    后者会让「为什么没走方舟」变成一个查不出来的问题。
    """
    out = []
    for model in MODEL_CATALOG:
        p = ArkSeedanceProvider(model)
        REGISTRY.register(p)
        out.append(p)
    return out


register_defaults()


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    from ..schema import ImageRef, RefPack

    logging.basicConfig(level=logging.WARNING)
    key_backup = os.environ.pop("ARK_API_KEY", None)

    pro = ArkSeedanceProvider(DEFAULT_MODEL)
    assert pro.name == "ark:doubao-seedance-1-0-pro-250528"
    assert pro.caps.kind == "official" and pro.caps.moderated and pro.caps.quality_tier == 5
    assert pro.caps.supports_first_frame and pro.caps.supports_last_frame
    assert not pro.caps.supports_extend, "1.0 pro 没有参考视频位，不应声明 extend"
    assert not pro.caps.supports_lora and not pro.caps.supports_native_audio

    # --- 无 key：health 为假，submit 抛 AUTH
    assert pro.health() is False
    req = GenRequest(
        prompt="雨夜天台，男人点燃最后一支烟",
        duration_s=7.3,
        resolution=(1920, 1080),
        fps=24,
        seed=42,
        negative_prompt="模糊, 多余手指",
        camera_move="camera dollies in",
        shot_id="s01_sh003",
        idempotency_key="fp-deadbeef",
        first_frame_uri="https://cdn.example.com/ff.png",
        refs=RefPack(
            images=[
                ImageRef(role="identity", uri="https://cdn.example.com/id.png", subject_id="c1"),
                ImageRef(role="environment", uri="https://cdn.example.com/env.png"),
            ]
        ),
    )
    try:
        pro.submit(req)
        raise AssertionError("无 ARK_API_KEY 时 submit 必须抛 AUTH")
    except ProviderError as e:
        assert e.kind is FailureKind.AUTH, e.kind

    # --- 请求体翻译（纯函数，不联网）
    body = pro.build_body(req)
    assert body["model"] == "doubao-seedance-1-0-pro-250528"
    assert body["duration"] == 5, f"7.3s 应吸附到 5/10 档中的 5，实得 {body['duration']}"
    assert body["resolution"] == "1080p" and body["ratio"] == "16:9"
    assert body["seed"] == 42 and body["return_last_frame"] is True
    assert body["camera_fixed"] is False, "dolly in 不是固定机位"
    assert body["safety_identifier"] == "s01_sh003"
    assert "generate_audio" not in body, "1.0 pro 无原生音频，不应下发该字段"

    texts = [c for c in body["content"] if c["type"] == "text"]
    assert len(texts) == 1 and texts[0]["text"].startswith("雨夜天台")
    assert "camera dollies in" in texts[0]["text"]
    assert "避免出现：模糊, 多余手指" in texts[0]["text"], "negative 必须并入正文而非静默丢弃"

    imgs = [c for c in body["content"] if c["type"] == "image_url"]
    assert [c["role"] for c in imgs] == ["first_frame", "reference_image", "reference_image"], imgs
    assert imgs[0]["image_url"]["url"] == "https://cdn.example.com/ff.png"

    # --- 超参考位上限要报错而不是静默截断
    fat = GenRequest(
        prompt="x",
        duration_s=5,
        refs=RefPack(images=[ImageRef(role="prop", uri=f"u{i}") for i in range(6)]),
    )
    try:
        pro.build_body(fat)
        raise AssertionError("超过 max_ref_images 必须抛 BAD_REQUEST")
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST

    # --- inline_params 备选路径
    inline = ArkSeedanceProvider(DEFAULT_MODEL, inline_params=True)
    ib = inline.build_body(req)
    itext = ib["content"][0]["text"]
    assert "--resolution 1080p" in itext and "--ratio 16:9" in itext
    assert "--dur 5" in itext and "--seed 42" in itext and "--cf false" in itext
    assert "duration" not in ib, "inline 模式下标量不应再重复走顶层字段"

    # --- 2.0：原生音频 + 参考视频 + extend
    v2 = ArkSeedanceProvider("doubao-seedance-2-0-260128", generate_audio=True)
    assert v2.caps.supports_extend and v2.caps.supports_native_audio
    assert v2.caps.max_ref_videos == 1 and v2.caps.max_ref_images == 9
    ereq = GenRequest(
        prompt="接上一镜继续往前走",
        duration_s=8,
        extend_from_video="https://cdn.example.com/prev.mp4",
        audio_uri="https://cdn.example.com/line.wav",
        idempotency_key="fp-ext",
    )
    eb = v2.build_body(ereq)
    vids = [c for c in eb["content"] if c["type"] == "video_url"]
    auds = [c for c in eb["content"] if c["type"] == "audio_url"]
    assert vids and vids[0]["role"] == "reference_video"
    assert auds and auds[0]["role"] == "reference_audio"
    assert eb["generate_audio"] is True and eb["duration"] == 8
    # 1.0 pro 上 extend 必须明确拒绝
    try:
        pro.extend(ereq)
        raise AssertionError("1.0 pro 不支持 extend，必须抛错")
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST
    # 2.0 上缺素材也要拒绝
    try:
        v2.extend(GenRequest(prompt="无素材", duration_s=8))
        raise AssertionError("extend 缺 extend_from_video 必须抛错")
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST

    # --- 错误码分类：审核这条是路由降级的命门
    assert classify_error("InputImageSensitiveContentDetected") is FailureKind.MODERATION
    assert classify_error("OutputVideoSensitiveContentDetected") is FailureKind.MODERATION
    assert classify_error("SomeNewSensitiveThing") is FailureKind.MODERATION, "未知审核码要兜底"
    assert classify_error("RateLimitExceeded.EndpointRPMExceeded") is FailureKind.RATE_LIMIT
    assert classify_error("ModelAccountTpmRateLimitExceeded") is FailureKind.RATE_LIMIT
    assert classify_error("AccountOverdueError") is FailureKind.QUOTA
    assert classify_error("QuotaExceeded") is FailureKind.QUOTA
    assert classify_error("AuthenticationError") is FailureKind.AUTH
    assert classify_error("InvalidParameter") is FailureKind.BAD_REQUEST
    assert classify_error("", 503) is FailureKind.SERVER
    assert FailureKind.MODERATION.should_failover and not FailureKind.MODERATION.retryable

    # --- 状态映射（喂假响应，不联网）
    pro._submitted["cgt-x"] = (10.0, req)
    running = pro._to_result("cgt-x", 200, {"status": "running"})
    assert running.status is JobStatus.RUNNING and running.failure is FailureKind.NONE

    ok = pro._to_result(
        "cgt-x",
        200,
        {
            "status": "succeeded",
            "content": {
                "video_url": "https://o.example.com/v.mp4",
                "last_frame_url": "https://o.example.com/last.png",
            },
            "usage": {"completion_tokens": 486000},
            "created_at": 1_790_000_000,
            "updated_at": 1_790_000_160,
        },
    )
    assert ok.ok and ok.last_frame_uri.endswith("last.png")
    assert ok.elapsed_s == 160.0 and ok.cost_usd > 0
    assert ok.duration_s == 10.0

    rej = pro._to_result(
        "cgt-x",
        200,
        {"status": "failed", "error": {"code": "OutputVideoSensitiveContentDetected", "message": "审核未通过"}},
    )
    assert rej.status is JobStatus.REJECTED, "审核拒绝必须是 REJECTED，否则 router 会当可重试"
    assert rej.failure is FailureKind.MODERATION and rej.cost_usd == 0.0

    limited = pro._to_result("cgt-x", 429, {"error": {"code": "RateLimitExceeded.EndpointRPMExceeded"}})
    assert limited.status is JobStatus.FAILED and limited.failure.retryable

    empty_ok = pro._to_result("cgt-x", 200, {"status": "succeeded", "content": {}})
    assert empty_ok.status is JobStatus.FAILED and empty_ok.failure is FailureKind.SERVER

    # --- 计费：token 公式 + 分辨率的平方级影响
    c1080 = pro.estimate_cost(GenRequest(prompt="p", duration_s=10, resolution=(1920, 1080)))
    c480 = pro.estimate_cost(GenRequest(prompt="p", duration_s=10, resolution=(854, 480)))
    assert c1080 > c480 > 0, (c1080, c480)
    tok = ArkSeedanceProvider._estimate_tokens(10.0, (1920, 1080), 24)
    assert abs(tok - 1920 * 1080 * 24 * 10 / 1024) < 1e-6

    # --- 幂等：命中后不再发第二次 POST（有 key 才走到这一步）
    os.environ["ARK_API_KEY"] = "sk-selftest-not-a-real-key"
    idem_pro = ArkSeedanceProvider(DEFAULT_MODEL, idem_store={"fp-deadbeef": "cgt-cached"})
    assert idem_pro.health() is True
    assert idem_pro.submit(req) == "cgt-cached", "幂等键命中必须直接返回旧 job_id"
    os.environ.pop("ARK_API_KEY")
    if key_backup is not None:
        os.environ["ARK_API_KEY"] = key_backup

    # --- 注册表
    assert "ark:doubao-seedance-1-0-pro-250528" in REGISTRY
    assert len(REGISTRY.of_kind("official")) >= len(MODEL_CATALOG)
    ext_capable = [p.name for p in REGISTRY.capable(duration_s=8.0, needs_extend=True)]
    assert "ark:doubao-seedance-2-0-260128" in ext_capable
    assert "ark:doubao-seedance-1-0-pro-250528" not in ext_capable

    print("ark_seedance 自测通过：")
    print(f"  已注册模型 {len(MODEL_CATALOG)} 个，默认 {DEFAULT_MODEL}")
    print(f"  请求体样例 {json.dumps(body, ensure_ascii=False)[:180]}...")
    print(f"  10s/1080p 预估成本 ${c1080}  10s/480p ${c480}")
    print(f"  审核拒绝映射 {rej.status.value}/{rej.failure.value}（should_failover={rej.failure.should_failover}）")


if __name__ == "__main__":
    _selftest()
