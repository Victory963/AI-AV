"""教师蒸馏数据集构建 —— 把闭源 API 的画质，蒸到自己能改的开源模型里。

产线里闭源 API 负责画质天花板，开源引擎负责可控、可微调、无审核阻塞。
两者的差距主要不在「像不像」，而在**运动连贯性、运镜执行力、材质光影**。
这三样恰好是可以用「提示词 → 视频」配对样本学出来的，所以拿过检的成片
反向当教师样本，在开源基座上训 LoRA，是成本最低的追赶路径。

本模块只做**数据侧**：回收 → 切片 → 打 caption → 出训练集目录与 toml。
真正的训练工单在 lora.py，偏好对齐在 dpo.py。

--------------------------------------------------------------------------
法律提示（条款事实陈述，不是法律建议；请自行判断与咨询律师）
--------------------------------------------------------------------------
查证日期：2026-09-19。以下为各家条款原文摘录与出处，条款随时可能改版，
落地前请以当日官网为准。本模块的 LicenseFlag 只是把这些事实编码进数据，
**不替使用者做合规判断，也不提供任何绕过手段**。

1) Runway —— https://runway.com/terms-of-use （页面标注 Last Updated:
   September 15, 2026）
   §4.4 原文：“The Company does not claim ownership of any of your Inputs
   or Outputs.” 另有 “Subject to your compliance with the Agreement, the
   Company does not restrict your commercial use of your Outputs.”
   但 §5 User Conduct 原文禁止：“directly or indirectly uses the Services
   (including, but not limited to, Outputs) to create, train, develop, or
   improve similar or competitive products or services.”
   → 即：输出归你，但用输出训练「相似或竞品」模型被明文禁止。

2) Luma AI —— https://lumalabs.ai/legal/tos （页面标注 Last Updated:
   May 14, 2026）
   所有权原文：“Customer owns and retains all right, title, and interest in
   and to the Output”，并附条件 “it can only use the Outputs for commercial
   purposes if the Outputs were produced during an active Subscription Term
   under Customer's paid subscription allowing for the commercial use of
   those Outputs.”
   §4.7(g) 原文禁止：“use the Services or Output, directly or indirectly, to
   create, test, train, or otherwise develop any artificial intelligence or
   machine learning models, systems, architecture, weights, or related
   technology.”
   → 这一条比 Runway 更宽：不限于竞品，任何 AI/ML 模型训练都被禁止。

3) Google Gemini API —— https://ai.google.dev/gemini-api/terms
   （页面标注 Last updated 2026-04-28 UTC）
   原文：“You may not use the Services to develop models that compete with
   the Services (e.g., Gemini API or Google AI Studio).”
   → 限制范围写的是「竞品模型」，比 Luma 窄，但「竞品」边界由 Google 解释。

4) OpenAI —— https://openai.com/policies/terms-of-use/ 与
   .../row-terms-of-use/ 在 2026-09-19 的自动抓取中均返回 HTTP 403，
   **本次未能取得原文**。已知其历史条款含「不得用 Output 开发与 OpenAI
   竞争的模型」一类措辞，但本模块不据记忆转述。
   TODO(2026-09-19)：人工打开上述页面核对「What you cannot do」小节中
   关于 Output 与模型开发的条目原文与生效日期，再把 openai/sora 的
   LicenseFlag 从 TRAINING_UNKNOWN 改掉。

5) 可灵 KlingAI —— https://kling.ai/document-api/apiReference/legal/termsOfService
   2026-09-19 抓取只拿到页面骨架（内容由前端渲染），未取得条款正文。
   TODO(2026-09-19)：人工核对可灵开放平台《用户服务协议》中关于生成内容
   权属与「不得用于训练」的条款。

6) 火山方舟 Seedance（字节）、阿里云百炼 DashScope（Wan）
   2026-09-19 未取得可引用的条款正文（页面跳转/前端渲染）。
   TODO(2026-09-19)：人工核对火山引擎《机器学习平台服务协议》与阿里云
   《模型服务灵积用户协议》中输出权属与训练用途条款。

7) 自建开源引擎（comfy_local 等）
   输出权属通常归运行者，但**基座模型自身的许可证仍然约束衍生权重**
   （例：部分权重带非商用或 use-based 限制）。用开源引擎的产出训 LoRA 前，
   仍需看基座 weights 的 LICENSE，而不是只看「我自己跑的」。

实务结论（事实层面，不构成建议）：上面 1/2/3 三家的条款，对「拿其输出
训练模型」都有明文限制，差别只在范围。本模块默认把它们标成
TRAINING_RESTRICTED，并在 DistillDataset.trainable() 里**默认剔除**，
使用者若基于自己的法律判断要放行，必须显式传 include_restricted=True —— 
把这个决定留在调用处，而不是藏在默认值里。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from ._proc import popen as _sp_popen  # noqa: F401
from ._proc import run as _sp_run
from .providers.base import GenResult
from .schema import CameraMove, ContentRating, Shot, ShotSize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- 许可证标记


class LicenseFlag(str, Enum):
    """一条教师样本在「能不能拿去训练」这件事上的状态。

    值是**事实标签**，不是许可。判断依据见模块 docstring 里逐条摘录的原文。
    """

    OPEN_OUTPUT = "open_output"                 # 自建开源引擎产出，受基座权重许可证约束
    TRAINING_PERMITTED = "training_permitted"   # 条款明确不禁止拿输出训练
    TRAINING_RESTRICTED = "training_restricted" # 条款明文禁止（范围见 note）
    TRAINING_UNKNOWN = "training_unknown"       # 未查证到条款正文，按未知处理
    THIRD_PARTY_IP = "third_party_ip"           # 素材含第三方版权（外采空镜等）


@dataclass(frozen=True)
class LicenseFact:
    """一家供应商的条款事实条目。source/checked_on 必填，防止无出处的断言混进产线。"""

    flag: LicenseFlag
    source: str
    checked_on: str
    quote: str = ""
    note: str = ""


# provider 名字对齐 providers/ 下各实现的 Capabilities.name。
# 没列到的 provider 一律按 TRAINING_UNKNOWN 处理（见 license_of）。
LICENSE_FACTS: dict[str, LicenseFact] = {
    "runway": LicenseFact(
        flag=LicenseFlag.TRAINING_RESTRICTED,
        source="https://runway.com/terms-of-use",
        checked_on="2026-09-19",
        quote=("directly or indirectly uses the Services (including, but not limited to, "
               "Outputs) to create, train, develop, or improve similar or competitive "
               "products or services."),
        note="Last Updated: September 15, 2026；§4.4 同时写明不主张输出所有权。限制范围＝相似/竞品。",
    ),
    "luma": LicenseFact(
        flag=LicenseFlag.TRAINING_RESTRICTED,
        source="https://lumalabs.ai/legal/tos",
        checked_on="2026-09-19",
        quote=("use the Services or Output, directly or indirectly, to create, test, train, "
               "or otherwise develop any artificial intelligence or machine learning models, "
               "systems, architecture, weights, or related technology."),
        note="Last Updated: May 14, 2026；§4.7(g)。范围＝任何 AI/ML 模型，不限竞品。",
    ),
    "gemini_veo": LicenseFact(
        flag=LicenseFlag.TRAINING_RESTRICTED,
        source="https://ai.google.dev/gemini-api/terms",
        checked_on="2026-09-19",
        quote="You may not use the Services to develop models that compete with the Services "
              "(e.g., Gemini API or Google AI Studio).",
        note="Last updated 2026-04-28 UTC。范围写作「竞品模型」，边界由 Google 解释。",
    ),
    "openai_sora": LicenseFact(
        flag=LicenseFlag.TRAINING_UNKNOWN,
        source="https://openai.com/policies/terms-of-use/",
        checked_on="2026-09-19",
        note="2026-09-19 自动抓取返回 HTTP 403，未取得原文。TODO：人工核对后再改标记。",
    ),
    "kling": LicenseFact(
        flag=LicenseFlag.TRAINING_UNKNOWN,
        source="https://kling.ai/document-api/apiReference/legal/termsOfService",
        checked_on="2026-09-19",
        note="2026-09-19 抓取只得到前端骨架，未取得条款正文。TODO：人工核对。",
    ),
    "ark_seedance": LicenseFact(
        flag=LicenseFlag.TRAINING_UNKNOWN,
        source="https://www.volcengine.com/docs/82379",
        checked_on="2026-09-19",
        note="2026-09-19 未取得可引用的协议正文。TODO：人工核对火山引擎服务协议。",
    ),
    "dashscope_wan": LicenseFact(
        flag=LicenseFlag.TRAINING_UNKNOWN,
        source="https://help.aliyun.com/zh/model-studio/",
        checked_on="2026-09-19",
        note="2026-09-19 未取得可引用的协议正文。TODO：人工核对阿里云模型服务协议。",
    ),
    "comfy_local": LicenseFact(
        flag=LicenseFlag.OPEN_OUTPUT,
        source="本地自建；以基座权重自身 LICENSE 为准",
        checked_on="2026-09-19",
        note="输出权属通常归运行者，但基座 weights 的 use-based 限制仍然生效。",
    ),
    "mock": LicenseFact(
        flag=LicenseFlag.TRAINING_PERMITTED,
        source="内置 mock provider，无真实条款",
        checked_on="2026-09-19",
        note="仅供自测与干跑。",
    ),
}


def license_of(provider: str) -> LicenseFact:
    """未登记的 provider 一律按未知处理 —— 默认放行会把法律风险变成静默 bug。"""
    fact = LICENSE_FACTS.get(provider)
    if fact is None:
        return LicenseFact(
            flag=LicenseFlag.TRAINING_UNKNOWN,
            source="未登记",
            checked_on="2026-09-19",
            note=f"provider={provider!r} 不在 LICENSE_FACTS 里，按未知处理；请补登记后再入训练集。",
        )
    return fact


# ---------------------------------------------------------------- 样本与数据集


#: 教师样本默认只收这两档。R_VIOLENCE / R_SUGGESTIVE 不是不能拍，
#: 而是不该进**训练集** —— 训练数据会被模型无差别泛化，分级信息在权重里丢失，
#: 之后再想靠提示词把它收回去就没有抓手了。要放宽必须调用处显式传。
DEFAULT_TEACHER_RATINGS: frozenset[ContentRating] = frozenset(
    {ContentRating.G, ContentRating.PG13}
)


class TeacherSample(BaseModel):
    """一条教师样本 = 视频 + 原始提示词 + 镜头语法标签 + 质检分 + 来源 + 许可证。

    字段刻意扁平：训练集要按镜头语法分桶、要做多样性统计，
    嵌套结构只会让每个消费者重新展开一遍。
    """

    shot_id: str
    take: int = 0
    video_uri: str
    duration_s: float = Field(ge=0.0)
    fps: int = 24
    resolution: tuple[int, int] = (1280, 720)

    prompt: str
    negative_prompt: str = ""

    # 镜头语法标签 —— 蒸馏真正要学的东西
    shot_size: ShotSize = ShotSize.MS
    camera_move: CameraMove = CameraMove.STATIC
    lens_mm: int = 35
    scene_id: str = ""
    subject_ids: list[str] = Field(default_factory=list)
    environment: str = ""
    lighting: str = ""
    mood: str = ""
    style: str = ""

    qc_score: float = Field(ge=0.0, le=1.0)
    provider: str = ""
    license: LicenseFlag = LicenseFlag.TRAINING_UNKNOWN
    license_source: str = ""
    content_rating: ContentRating = ContentRating.G
    shot_fingerprint: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.shot_id}_t{self.take}"


class DistillDataset(BaseModel):
    """一批教师样本。可直接 JSON 落盘，dataset_hash 回填进 LoRASpec 做复现。"""

    name: str = "teacher"
    min_score: float = 0.0
    samples: list[TeacherSample] = Field(default_factory=list)
    built_on: str = "2026-09-19"

    def __len__(self) -> int:
        return len(self.samples)

    def trainable(self, *, include_restricted: bool = False,
                  include_unknown: bool = False) -> list[TeacherSample]:
        """默认只放行条款明确不禁止的样本。

        把「放行受限样本」做成显式参数，是为了让这个决定在代码 review 里可见 ——
        埋在默认值里的合规决定等于没有决定。
        """
        ok = {LicenseFlag.TRAINING_PERMITTED, LicenseFlag.OPEN_OUTPUT}
        if include_restricted:
            ok = ok | {LicenseFlag.TRAINING_RESTRICTED}
        if include_unknown:
            ok = ok | {LicenseFlag.TRAINING_UNKNOWN}
        return [s for s in self.samples if s.license in ok]

    def license_breakdown(self) -> dict[str, int]:
        return dict(Counter(s.license.value for s in self.samples))

    def total_seconds(self) -> float:
        return round(sum(s.duration_s for s in self.samples), 2)

    def dataset_hash(self) -> str:
        """训练集指纹。同指纹 = 同一批样本，可直接填进 LoRASpec.dataset_hash。"""
        blob = "|".join(sorted(f"{s.key}:{s.video_uri}:{s.shot_fingerprint}" for s in self.samples))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> DistillDataset:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RenderRecord:
    """一次出片记录。router/chain 出片后往这里塞，蒸馏与 DPO 都吃这个结构。"""

    shot: Shot
    result: GenResult
    take: int = 0

    @property
    def key(self) -> str:
        # job_id 才是一次 take 的唯一标识；shot.id 在多 take 时会撞。
        return self.result.job_id or f"{self.shot.id}_t{self.take}"


RenderLike = RenderRecord | tuple[Shot, GenResult]


def _as_records(renders: Iterable[RenderLike]) -> list[RenderRecord]:
    out: list[RenderRecord] = []
    seen: Counter[str] = Counter()
    for r in renders:
        rec = r if isinstance(r, RenderRecord) else RenderRecord(shot=r[0], result=r[1])
        if not isinstance(r, RenderRecord):
            # 用 tuple 传进来时没有 take 号，按同 shot 出现顺序补
            rec = RenderRecord(shot=rec.shot, result=rec.result, take=seen[rec.shot.id])
        seen[rec.shot.id] += 1
        out.append(rec)
    return out


def qc_score_of(report: Any) -> tuple[float, bool | None]:
    """把质检报告归一成 (score, passed)。

    qc.py 由另一组实现，这里用结构化取值而不是 import，避免两个模块互相卡进度。
    支持三种形态：裸 float、含 score/passed 键的 mapping、带同名属性的对象。
    """
    if report is None:
        return 0.0, None
    if isinstance(report, (int, float)):
        return float(report), None
    if isinstance(report, Mapping):
        score = report.get("score", report.get("qc_score", 0.0))
        passed = report.get("passed", report.get("ok"))
    else:
        score = getattr(report, "score", getattr(report, "qc_score", 0.0))
        passed = getattr(report, "passed", getattr(report, "ok", None))
    return float(score or 0.0), (None if passed is None else bool(passed))


def harvest(
    renders: Iterable[RenderLike],
    qc_reports: Mapping[str, Any] | None = None,
    *,
    min_score: float = 0.75,
    allowed_ratings: Iterable[ContentRating] = DEFAULT_TEACHER_RATINGS,
    prompt_of: Mapping[str, str] | None = None,
    default_resolution: tuple[int, int] = (1280, 720),
    name: str = "teacher",
) -> DistillDataset:
    """只回收过检镜。

    qc_reports 按 RenderRecord.key（即 job_id）索引；找不到时退回 Shot.qc_score，
    两者都没有就丢弃 —— 无分数的样本进训练集等于把噪声当教师。
    """
    allowed = frozenset(allowed_ratings)
    reports = qc_reports or {}
    samples: list[TeacherSample] = []
    dropped: Counter[str] = Counter()

    for rec in _as_records(renders):
        shot, res = rec.shot, rec.result
        if not res.ok:
            dropped["未成片"] += 1
            continue
        if shot.content_rating not in allowed:
            dropped[f"分级 {shot.content_rating.value} 不收"] += 1
            continue

        report = reports.get(rec.key, reports.get(shot.id))
        score, passed = qc_score_of(report)
        if report is None and shot.qc_score is not None:
            score, passed = float(shot.qc_score), None
        if report is None and shot.qc_score is None:
            dropped["无质检分"] += 1
            continue
        if passed is False or score < min_score:
            dropped[f"质检未过（<{min_score}）"] += 1
            continue

        fact = license_of(res.provider or shot.provider_used or "")
        prompt = (prompt_of or {}).get(rec.key) or (prompt_of or {}).get(shot.id) or ""
        if not prompt:
            # 没有编译后的提示词就退回镜头字段拼一条，总比空 caption 强
            prompt = ", ".join(
                p for p in (shot.action, shot.environment, shot.lighting, shot.mood, shot.style)
                if p
            )

        samples.append(TeacherSample(
            shot_id=shot.id,
            take=rec.take,
            video_uri=res.video_uri or shot.render_uri or "",
            duration_s=res.duration_s or shot.duration_s,
            fps=shot.fps,
            # 分辨率不在 Shot 里（由 router 按 capabilities 裁定），
            # provider 若在 raw 里回填就用它，否则退回交付默认值。
            resolution=tuple(res.raw.get("resolution") or default_resolution),
            prompt=prompt,
            negative_prompt=shot.negative_prompt,
            shot_size=shot.shot_size,
            camera_move=shot.camera_move,
            lens_mm=shot.lens_mm,
            scene_id=shot.scene_id,
            subject_ids=list(shot.subject_ids),
            environment=shot.environment,
            lighting=shot.lighting,
            mood=shot.mood,
            style=shot.style,
            qc_score=min(1.0, max(0.0, score)),
            provider=res.provider,
            license=fact.flag,
            license_source=fact.source,
            content_rating=shot.content_rating,
            shot_fingerprint=shot.fingerprint(),
            meta={"job_id": res.job_id, "cost_usd": res.cost_usd},
        ))

    if dropped:
        log.info("harvest 丢弃统计：%s", dict(dropped))
    return DistillDataset(name=name, min_score=min_score, samples=samples)


# ---------------------------------------------------------------- 训练集导出


def _ffmpeg_bin() -> str:
    """优先用项目自带的静态 ffmpeg，其次 imageio-ffmpeg，最后 PATH。"""
    local = Path(__file__).resolve().parents[2] / "bin" / "ffmpeg"
    if local.exists() and os.access(local, os.X_OK):
        return str(local)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError(
            "找不到 ffmpeg：项目 bin/ffmpeg 不存在、imageio-ffmpeg 未装、PATH 里也没有。"
            "请把静态 ffmpeg 放到 <项目根>/bin/ffmpeg。"
        )
    return found


def musubi_target_frames(seconds: float, fps: int) -> int:
    """musubi-tuner 的 target_frames 必须是 N*4+1。

    出处：https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md
    （2026-09-19 查证，原文 "target_frames = [1, 25, 45]  # required; must be N*4+1"）。
    这是 Wan/HunyuanVideo 的 VAE 时间下采样步长决定的，凑不满会被丢帧。
    """
    n = max(1, int(round(seconds * fps)))
    k = max(0, round((n - 1) / 4))
    return int(k * 4 + 1)


#: 分辨率桶。musubi 的 [general].resolution 写的是基准边长 [W, H]，
#: enable_bucket=true 时按宽高比自动分桶（dataset_config.md，2026-09-19 查证）。
#: 960x544 是 musubi README 给 12GB 显存的建议基准，故作为低档默认。
RESOLUTION_BUCKETS: dict[str, tuple[int, int]] = {
    "544p": (960, 544),
    "720p": (1280, 720),
    "480p_vertical": (544, 960),
    "1024sq": (1024, 1024),
}


CaptionStyle = str  # "cinematic" | "terse" | "motion_only" | "trigger_only"


def build_caption(sample: TeacherSample, *, style: CaptionStyle = "cinematic",
                  trigger: str = "") -> str:
    """按风格拼 caption。

    风格不是排版偏好，而是**决定 LoRA 学什么**：
    - cinematic：全量镜头语法 + 内容，学「整体质感」，泛化最广也最容易学串内容；
    - motion_only：只留运镜与动作，专门学运动/运镜，不把教师的内容偏好带进来；
    - terse：景别 + 动作，短 caption 抗过拟合；
    - trigger_only：只有触发词，角色 LoRA 用，把一切都绑到触发词上。
    """
    if style == "trigger_only":
        return trigger or sample.shot_id
    if style == "motion_only":
        parts = [sample.camera_move.value, f"{sample.lens_mm}mm lens", sample.prompt.split(",")[0]]
    elif style == "terse":
        parts = [sample.shot_size.value, sample.prompt.split(",")[0]]
    else:
        parts = [
            sample.shot_size.value,
            sample.camera_move.value,
            f"{sample.lens_mm}mm lens",
            sample.prompt,
            sample.environment,
            sample.lighting,
            sample.mood,
            sample.style,
        ]
    if trigger:
        parts.insert(0, trigger)
    # 去冗余用「包含」而不是「相等」：镜头字段之间天然重叠
    # （action 里常常已经写了 environment），重复词会被 LoRA 当成强信号背下来。
    out: list[str] = []
    for p in parts:
        p = (p or "").strip().strip(",").strip()
        if not p or any(p in q for q in out):
            continue
        out = [q for q in out if q not in p]
        out.append(p)
    return ", ".join(out)


@dataclass
class ClipSpec:
    """一个切片：源视频的 [start, start+dur) 段 → 训练集里的一个 mp4 + 一个 .txt。"""

    src_uri: str
    dst_video: Path
    dst_caption: Path
    start_s: float
    duration_s: float
    caption: str
    shot_id: str
    take: int
    frames: int

    def ffmpeg_argv(self, *, fps: int, resolution: tuple[int, int], crf: int = 14) -> list[str]:
        w, h = resolution
        # -ss 放在 -i 前是关键帧级快速 seek；训练切片不要求帧精确起点，
        # 但要求**帧数精确**，所以用 -frames:v 卡死总帧数而不是靠 -t。
        vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=bicubic,"
              f"crop={w}:{h},fps={fps}")
        return [
            _ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{self.start_s:.3f}", "-i", self.src_uri,
            "-vf", vf, "-frames:v", str(self.frames),
            "-an", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p",
            str(self.dst_video),
        ]


class TrainingSetManifest(BaseModel):
    """to_training_set 的产物清单。落盘成 manifest.json，训练脚本读它。"""

    out_dir: str
    video_dir: str
    cache_dir: str
    fps: int
    resolution: tuple[int, int]
    target_frames: list[int]
    clip_s: float
    caption_style: str
    n_clips: int
    total_seconds: float
    dataset_hash: str
    musubi_toml: str
    diffusion_pipe_toml: str
    executed: bool = False
    failed: list[str] = Field(default_factory=list)


def _musubi_dataset_toml(video_dir: Path, cache_dir: Path, *, resolution: tuple[int, int],
                         target_frames: Sequence[int], fps: int, batch_size: int = 1,
                         num_repeats: int = 1) -> str:
    """musubi-tuner 的 dataset config。

    字段与取值约束全部对齐
    https://github.com/kohya-ss/musubi-tuner/blob/main/docs/dataset_config.md
    （2026-09-19 查证）：[general] 支持 resolution / caption_extension / batch_size /
    num_repeats / enable_bucket / bucket_no_upscale；视频数据集用 video_directory +
    target_frames(必填, N*4+1) + frame_extraction(head|chunk|full|slide|uniform)，
    source_fps 必须写成小数。
    """
    frames = ", ".join(str(f) for f in target_frames)
    return f"""# musubi-tuner dataset config（由 longfilm.distill 生成）
# 字段依据 docs/dataset_config.md，查证日期 2026-09-19
[general]
resolution = [{resolution[0]}, {resolution[1]}]
caption_extension = ".txt"
batch_size = {batch_size}
num_repeats = {num_repeats}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "{video_dir.as_posix()}"
cache_directory = "{cache_dir.as_posix()}"
target_frames = [{frames}]
# chunk：把长视频切成互不重叠的定长块，比 head 更省素材，
# 且不会让模型只见到每条素材的开头。
frame_extraction = "chunk"
source_fps = {float(fps):.1f}
"""


def _diffusion_pipe_dataset_toml(video_dir: Path, *, resolution: tuple[int, int],
                                 frame_buckets: Sequence[int], num_repeats: int = 1) -> str:
    """diffusion-pipe 的 dataset.toml。

    字段依据 https://github.com/tdrussell/diffusion-pipe/blob/main/examples/dataset.toml
    （2026-09-19 查证）：resolutions 可写 [[W, H]] 对；enable_ar_bucket / min_ar /
    max_ar / num_ar_buckets 控宽高比桶；frame_buckets 控帧数桶，视频会落到
    「不超过自身长度的最长桶」，太短的视频会被直接丢弃（不会落到 image 桶 1）。
    """
    fb = ", ".join(str(f) for f in frame_buckets)
    return f"""# diffusion-pipe dataset config（由 longfilm.distill 生成）
# 字段依据 examples/dataset.toml，查证日期 2026-09-19
resolutions = [[{resolution[0]}, {resolution[1]}]]
enable_ar_bucket = true
min_ar = 0.5
max_ar = 2.0
num_ar_buckets = 7
# 视频落到「不超过自身长度的最长桶」，所以桶要和切片长度对齐，
# 否则切片会被整段丢弃。
frame_buckets = [{fb}]

[[directory]]
path = '{video_dir.as_posix()}'
num_repeats = {num_repeats}
"""


def to_training_set(
    dataset: DistillDataset,
    out_dir: str | Path,
    *,
    fps: int = 16,
    resolution: tuple[int, int] | str = "544p",
    clip_s: float = 3.0,
    caption_style: CaptionStyle = "cinematic",
    trigger: str = "",
    include_restricted: bool = False,
    include_unknown: bool = False,
    execute: bool = False,
    num_repeats: int = 1,
) -> TrainingSetManifest:
    """把 DistillDataset 转成训练框架直接能吃的目录。

    默认 execute=False 只出「计划 + 配置」，不动 ffmpeg —— 切片是分钟级 IO，
    产线里通常丢给单独的 worker 跑，这里不该阻塞编排线程。
    """
    res = RESOLUTION_BUCKETS[resolution] if isinstance(resolution, str) else resolution
    out = Path(out_dir)
    video_dir = out / "videos"
    cache_dir = out / "cache"
    video_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames = musubi_target_frames(clip_s, fps)
    usable = dataset.trainable(include_restricted=include_restricted,
                               include_unknown=include_unknown)
    if not usable:
        log.warning("训练集为空：%d 条样本全部被许可证过滤掉（%s）",
                    len(dataset), dataset.license_breakdown())

    clips: list[ClipSpec] = []
    for s in usable:
        # 一条素材切出 floor(dur/clip_s) 段；不足 0.6 段的尾巴丢掉，
        # 补零/拉伸会教模型学到不存在的运动速度。
        n = int(s.duration_s // clip_s)
        if n == 0 and s.duration_s >= clip_s * 0.6:
            n = 1
        for i in range(n):
            stem = f"{s.key}_c{i:02d}"
            clips.append(ClipSpec(
                src_uri=s.video_uri,
                dst_video=video_dir / f"{stem}.mp4",
                dst_caption=video_dir / f"{stem}.txt",
                start_s=round(i * clip_s, 3),
                duration_s=clip_s,
                caption=build_caption(s, style=caption_style, trigger=trigger),
                shot_id=s.shot_id,
                take=s.take,
                frames=frames,
            ))

    failed: list[str] = []
    for c in clips:
        c.dst_caption.write_text(c.caption + "\n", encoding="utf-8")
    if execute:
        for c in clips:
            argv = c.ffmpeg_argv(fps=fps, resolution=res)
            proc = _sp_run(argv, capture_output=True, text=True)
            if proc.returncode != 0 or not c.dst_video.exists():
                failed.append(f"{c.dst_video.name}: {proc.stderr.strip()[:200]}")
                c.dst_caption.unlink(missing_ok=True)
    else:
        script = "\n".join(
            " ".join(_shq(a) for a in c.ffmpeg_argv(fps=fps, resolution=res)) for c in clips
        )
        (out / "cut_clips.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + script + "\n",
                                          encoding="utf-8")
        (out / "cut_clips.sh").chmod(0o755)

    musubi = _musubi_dataset_toml(video_dir, cache_dir, resolution=res,
                                  target_frames=[frames], fps=fps, num_repeats=num_repeats)
    dpipe = _diffusion_pipe_dataset_toml(video_dir, resolution=res,
                                         frame_buckets=[1, frames], num_repeats=num_repeats)
    (out / "dataset_musubi.toml").write_text(musubi, encoding="utf-8")
    (out / "dataset_diffusion_pipe.toml").write_text(dpipe, encoding="utf-8")

    ok_clips = [c for c in clips if not execute or c.dst_video.exists()]
    manifest = TrainingSetManifest(
        out_dir=str(out), video_dir=str(video_dir), cache_dir=str(cache_dir),
        fps=fps, resolution=res, target_frames=[frames], clip_s=clip_s,
        caption_style=caption_style, n_clips=len(ok_clips),
        total_seconds=round(len(ok_clips) * clip_s, 2),
        dataset_hash=dataset.dataset_hash(),
        musubi_toml=str(out / "dataset_musubi.toml"),
        diffusion_pipe_toml=str(out / "dataset_diffusion_pipe.toml"),
        executed=execute, failed=failed,
    )
    (out / "manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8")
    return manifest


def _shq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'" if re.search(r"[^\w./:=-]", s) else s


# ---------------------------------------------------------------- 多样性体检


def _tokens(text: str) -> list[str]:
    """中英混排分词：CJK 单字成词，ASCII 按词。够用于 n-gram 重复度，不追求语言学正确。"""
    return re.findall(r"[a-zA-Z]+|[0-9]+|[一-鿿]", text.lower())


def _ngrams(toks: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def _norm_entropy(counter: Counter[str]) -> float:
    """归一化熵 0..1。1 = 完全均匀，0 = 全挤在一个类里。

    除以 log(类别数) 而不是 log(理论全集)，因为「只用了 3 种景别但用得很均匀」
    和「只用了 3 种景别且 95% 是特写」是两种病，前者靠 coverage 指标抓。
    """
    total = sum(counter.values())
    if total == 0 or len(counter) <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counter.values() if c)
    return h / math.log(len(counter))


@dataclass
class DiversityMetrics:
    """多样性体检的数值面板。FlywheelMonitor 直接吃这个做坍缩判定。"""

    n_samples: int
    shot_size: Counter[str] = field(default_factory=Counter)
    camera_move: Counter[str] = field(default_factory=Counter)
    subject: Counter[str] = field(default_factory=Counter)
    scene: Counter[str] = field(default_factory=Counter)

    shot_size_entropy: float = 0.0
    camera_move_entropy: float = 0.0
    subject_entropy: float = 0.0
    scene_entropy: float = 0.0

    shot_size_coverage: float = 0.0     # 用到的景别 / 全部 12 种
    camera_move_coverage: float = 0.0   # 用到的运镜 / 全部 20 种

    distinct_1: float = 0.0
    distinct_3: float = 0.0
    top_3gram: list[tuple[str, int]] = field(default_factory=list)
    dup_prompt_ratio: float = 0.0
    max_class_share: float = 0.0        # 最大单类占比（景别/运镜取大者）

    def overall(self) -> float:
        """综合多样性 0..1。

        取熵、覆盖率、distinct-3 的**几何平均**而非算术平均：
        任何一项塌到 0 都必须把总分拉到 0，算术平均会被其他项掩盖。
        """
        parts = [
            self.shot_size_entropy, self.camera_move_entropy,
            self.subject_entropy or 1.0,   # 单角色短片不该被判为坍缩
            self.shot_size_coverage, self.camera_move_coverage,
            self.distinct_3,
        ]
        parts = [max(p, 1e-6) for p in parts]
        return round(math.exp(sum(math.log(p) for p in parts) / len(parts)), 4)


def diversity_metrics(dataset: DistillDataset) -> DiversityMetrics:
    ss = Counter(s.shot_size.value for s in dataset.samples)
    cm = Counter(s.camera_move.value for s in dataset.samples)
    subj = Counter(cid for s in dataset.samples for cid in s.subject_ids)
    scn = Counter(s.scene_id for s in dataset.samples if s.scene_id)

    toks_all: list[str] = []
    g3: Counter[tuple[str, ...]] = Counter()
    prompts = Counter(s.prompt.strip() for s in dataset.samples)
    for s in dataset.samples:
        t = _tokens(s.prompt)
        toks_all.extend(t)
        g3.update(_ngrams(t, 3))

    n = len(dataset.samples)
    m = DiversityMetrics(
        n_samples=n, shot_size=ss, camera_move=cm, subject=subj, scene=scn,
        shot_size_entropy=_norm_entropy(ss),
        camera_move_entropy=_norm_entropy(cm),
        subject_entropy=_norm_entropy(subj),
        scene_entropy=_norm_entropy(scn),
        shot_size_coverage=round(len(ss) / len(ShotSize), 4),
        camera_move_coverage=round(len(cm) / len(CameraMove), 4),
        distinct_1=round(len(set(toks_all)) / max(1, len(toks_all)), 4),
        distinct_3=round(len(g3) / max(1, sum(g3.values())), 4),
        top_3gram=[(" ".join(k), v) for k, v in g3.most_common(5)],
        dup_prompt_ratio=round(1 - len(prompts) / max(1, n), 4),
        max_class_share=round(max(
            (max(ss.values()) / n) if ss else 0.0,
            (max(cm.values()) / n) if cm else 0.0,
        ), 4) if n else 0.0,
    )
    return m


def diversity_report(dataset: DistillDataset) -> str:
    """多样性体检报告（给人看的）。

    这是防 mode collapse 的仪表盘：数据飞轮天然偏向「模型已经拍得好的那种镜头」，
    几轮下来景别和运镜会收敛到少数几种，质检分反而更高 —— 指标涨、能力窄。
    所以这份报告要和质检通过率**同时**看，单看任何一个都会被骗。
    """
    m = diversity_metrics(dataset)
    if m.n_samples == 0:
        return "多样性体检：数据集为空。"

    def dist_lines(c: Counter[str], title: str, limit: int = 8) -> list[str]:
        total = sum(c.values()) or 1
        rows = [f"  {title}（{len(c)} 类）:"]
        for k, v in c.most_common(limit):
            bar = "█" * max(1, round(v / total * 30))
            rows.append(f"    {k:<34} {v:>4}  {v / total:>6.1%} {bar}")
        if len(c) > limit:
            rows.append(f"    …… 另有 {len(c) - limit} 类")
        return rows

    L = [
        f"多样性体检 · {dataset.name} · {m.n_samples} 条 / {dataset.total_seconds()}s",
        f"  许可证分布: {dataset.license_breakdown()}",
        "",
    ]
    L += dist_lines(m.shot_size, "景别 shot_size")
    L += [f"    归一化熵 {m.shot_size_entropy:.3f} ｜ 覆盖率 {m.shot_size_coverage:.1%}"
          f"（{len(m.shot_size)}/{len(ShotSize)}）", ""]
    L += dist_lines(m.camera_move, "运镜 camera_move")
    L += [f"    归一化熵 {m.camera_move_entropy:.3f} ｜ 覆盖率 {m.camera_move_coverage:.1%}"
          f"（{len(m.camera_move)}/{len(CameraMove)}）", ""]
    L += dist_lines(m.subject, "角色 subject", limit=6)
    L += [f"    归一化熵 {m.subject_entropy:.3f}", ""]
    L += dist_lines(m.scene, "场景 scene", limit=6)
    L += [f"    归一化熵 {m.scene_entropy:.3f}", ""]
    L += [
        "  提示词重复度:",
        f"    distinct-1 {m.distinct_1:.3f} ｜ distinct-3 {m.distinct_3:.3f} "
        f"｜ 完全重复提示词占比 {m.dup_prompt_ratio:.1%}",
    ]
    for g, c in m.top_3gram:
        L.append(f"    高频 3-gram「{g}」× {c}")
    L += ["", f"  综合多样性 overall = {m.overall():.3f}（几何平均，任一维塌陷即整体归零）"]

    warn: list[str] = []
    if m.max_class_share > 0.5:
        warn.append(f"单一镜头语法占比 {m.max_class_share:.0%} > 50%，分布已经偏斜")
    if m.shot_size_coverage < 0.4:
        warn.append(f"景别覆盖率 {m.shot_size_coverage:.0%} < 40%，训练集撑不起完整镜头语法")
    if m.camera_move_coverage < 0.3:
        warn.append(f"运镜覆盖率 {m.camera_move_coverage:.0%} < 30%，学不到运镜多样性")
    if m.distinct_3 < 0.3:
        warn.append(f"distinct-3 {m.distinct_3:.2f} < 0.30，提示词模板化严重，LoRA 会把模板背下来")
    if m.dup_prompt_ratio > 0.2:
        warn.append(f"完全重复提示词占比 {m.dup_prompt_ratio:.0%} > 20%，等于变相提高了这部分样本权重")
    L.append("")
    L += ["  ⚠ " + w for w in warn] or ["  结论：分布健康，未触发坍缩预警。"]
    return "\n".join(L)


# ---------------------------------------------------------------- 混通用数据


@dataclass
class MixPlan:
    """通用数据混合方案。产物是两个 [[directory]] 的 num_repeats。"""

    teacher_clips: int
    general_clips: int          # 通用数据目录里可用的总条数
    general_take: int           # 实际取用多少条 —— 通用库通常远大于教师集，必须抽样
    target_ratio: float
    teacher_repeats: int
    general_repeats: int
    effective_ratio: float
    rationale: str

    def musubi_snippet(self, teacher_dir: str | Path, general_dir: str | Path,
                       *, cache_root: str | Path, target_frames: Sequence[int],
                       fps: int) -> str:
        frames = ", ".join(str(f) for f in target_frames)
        c = Path(cache_root)
        return f"""# 混合数据集：general 有效占比 {self.effective_ratio:.1%}
# 注意 general 目录只应放入抽样后的 {self.general_take} 条（共 {self.general_clips} 条可选），
# 否则 num_repeats 再怎么调也压不到目标占比。
[[datasets]]
video_directory = "{Path(teacher_dir).as_posix()}"
cache_directory = "{(c / 'teacher').as_posix()}"
target_frames = [{frames}]
frame_extraction = "chunk"
source_fps = {float(fps):.1f}
num_repeats = {self.teacher_repeats}

[[datasets]]
video_directory = "{Path(general_dir).as_posix()}"
cache_directory = "{(c / 'general').as_posix()}"
target_frames = [{frames}]
frame_extraction = "chunk"
source_fps = {float(fps):.1f}
num_repeats = {self.general_repeats}
"""


#: 通用数据占比的经验建议值。
#: 机制依据（已查证 2026-09-19）：musubi-tuner dataset_config.md 与 diffusion-pipe
#: examples/dataset.toml 都用 num_repeats 控制每个目录在一个 epoch 里被过几遍，
#: 语义与 sd-scripts 相同（num_repeats=1 = 过一遍，不复制）。所以「混比例」
#: 的实现就是给两个目录配不同 num_repeats。
#: 比例数值本身（0.20~0.35）是社区经验区间，**没有找到可引用的权威实验**。
#: TODO(2026-09-19)：未检索到针对视频 LoRA 的正则化数据配比消融实验；
#: 落地时建议自己做一次 0 / 0.2 / 0.4 三档对比，用 diversity_report 的
#: overall 与质检通过率一起判定，不要照搬这个默认值。
GENERAL_MIX_RATIO_DEFAULT = 0.25
GENERAL_MIX_RATIO_RANGE = (0.20, 0.35)

_MIX_RATIONALE = (
    "为什么要混：教师样本同质性极高（同一部片、同一套角色、同一批提示词模板），"
    "纯教师集训出来的 LoRA 会把「这部片的内容」和「要学的运动/质感」一起吃进去，"
    "挂上去之后任何提示词都往教师内容上飘，这就是灾难性遗忘的具体表现。"
    "混入与教师内容无关的通用视频，作用等同于 DreamBooth 的正则化图像："
    "给模型保留「不打触发词时该是什么样」的锚。\n"
    f"建议区间 {GENERAL_MIX_RATIO_RANGE[0]:.0%}~{GENERAL_MIX_RATIO_RANGE[1]:.0%}，"
    f"默认 {GENERAL_MIX_RATIO_DEFAULT:.0%}。低于 15% 起不到锚定作用；"
    "高于 40% 时教师信号被稀释，蒸馏效率明显下降（学不动运镜）。"
    "该区间为社区经验值，未找到可引用的权威消融实验 —— 见 GENERAL_MIX_RATIO_DEFAULT 处 TODO。"
)


def mix_general_data(
    dataset: DistillDataset | int,
    general_dir: str | Path,
    *,
    ratio: float = GENERAL_MIX_RATIO_DEFAULT,
    max_repeats: int = 10,
) -> MixPlan:
    """算出让通用数据占到 ratio 所需的 num_repeats 组合。

    dataset 可以直接传条数，方便在切片之后（条数已经变了）重算。
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"ratio 必须在 [0, 1) 之间，收到 {ratio}")
    n_t = dataset if isinstance(dataset, int) else len(dataset.trainable())
    gdir = Path(general_dir)
    n_g = len([p for p in gdir.glob("**/*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}]) \
        if gdir.exists() else 0

    if n_t == 0 or n_g == 0 or ratio == 0.0:
        return MixPlan(n_t, n_g, 0, ratio, 1, 0, 0.0,
                       _MIX_RATIONALE + "\n（通用数据目录为空或 ratio=0，本轮不混。）")

    # 目标：take*rg / (n_t*rt + take*rg) = ratio。
    # 只调 num_repeats 是解不出来的：通用库往往比教师集大一两个数量级，
    # rg 的最小步长 1 就已经把占比顶到 90% 以上。所以先**抽样**定 take，
    # 再用 repeats 补足 take 不够时的缺口（take 被 n_g 顶住的情况）。
    best: tuple[int, int, int, float] | None = None
    best_key: tuple[float, int, int] = (float("inf"), 0, 0)
    for rt in range(1, max_repeats + 1):
        need = ratio * n_t * rt / (1.0 - ratio)      # 通用侧需要的「有效条数」
        rg = max(1, math.ceil(need / n_g))
        if rg > max_repeats:
            continue
        take = max(1, min(n_g, round(need / rg)))
        eff = (take * rg) / (n_t * rt + take * rg)
        # 2 个百分点以内的误差一律视为「同样够准」，在这些方案里选 repeats 最小的。
        # repeats 大意味着一个 epoch 变长、保存点变稀、早停更迟钝，
        # 为了把占比从 14.3% 修到 15.5% 付出 10 倍 epoch 长度不划算。
        err = abs(eff - ratio)
        key = (0.0 if err <= 0.02 else err, rt, rg)
        cand = (rt, rg, take, err)
        if best is None or key < best_key:
            best, best_key = cand, key
    assert best is not None, "max_repeats 太小，解不出任何配比"
    rt, rg, take, _ = best
    eff = (take * rg) / (n_t * rt + take * rg)
    return MixPlan(n_t, n_g, take, ratio, rt, rg, round(eff, 4), _MIX_RATIONALE)


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    import tempfile

    from .providers.base import GenResult, JobStatus
    from .schema import Shot

    sizes = [ShotSize.CU, ShotSize.MS, ShotSize.FS, ShotSize.OTS, ShotSize.LS,
             ShotSize.MCU, ShotSize.TWO]
    moves = [CameraMove.STATIC, CameraMove.DOLLY_IN, CameraMove.PAN_R, CameraMove.ORBIT_L,
             CameraMove.HANDHELD, CameraMove.CRANE, CameraMove.TILT_D, CameraMove.STEADICAM]
    acts = ["推门而入抖落肩上积雪", "回头凝视对面楼的灯", "把钥匙按在掌心慢慢攥紧",
            "蹲下来系鞋带顺势看向身后", "在镜子前撕掉贴了半天的创可贴",
            "端起凉透的茶又放下", "一言不发地把外套披到对方身上",
            "翻找抽屉最后抽出一张旧车票", "站在扶梯上被人流推着往前",
            "把烟按灭在窗台的积水里", "听见门铃后僵在原地", "拆开信封却没有读",
            "沿着护栏走到尽头停住", "抬手挡住突然打过来的车灯"]
    envs = ["雨夜天台", "老式电梯间", "沿海公路收费站", "旧书店二楼",
            "深夜便利店后厨", "废弃泳池边", "地铁末班车厢"]
    lights = ["侧逆光，霓虹反射", "顶光压眉骨，阴影很硬", "窗外街灯做唯一光源",
              "练习室日光灯全开，平光", "车灯扫过形成移动光斑"]
    moods = ["克制的紧张", "疲惫后的松弛", "压不住的怒", "近乎麻木的平静", "隐约的期待"]
    styles = ["胶片颗粒，冷调", "高对比黑白", "柔焦暖调，浅景深", "数字锐利，高饱和"]

    def make(i: int, provider: str, score: float, rating: ContentRating = ContentRating.PG13) -> RenderRecord:
        sh = Shot(
            id=f"sh{i:02d}", scene_id=f"sc{i % 4:02d}", index=i, duration_s=8.0,
            shot_size=sizes[i % len(sizes)], camera_move=moves[i % len(moves)],
            lens_mm=[24, 35, 50, 85][i % 4], subject_ids=[["lin", "qi"][i % 2]],
            action=acts[i % len(acts)], environment=envs[i % len(envs)],
            lighting=lights[i % len(lights)], mood=moods[i % len(moods)],
            style=styles[i % len(styles)],
            content_rating=rating,
        )
        return RenderRecord(
            shot=sh,
            result=GenResult(job_id=f"job-{i:03d}", status=JobStatus.SUCCEEDED,
                             provider=provider, video_uri=f"s3://render/{sh.id}.mp4",
                             duration_s=8.0, cost_usd=0.4),
            take=0,
        )

    # ---- [1] harvest 只收过检 + 分级 + 许可证登记
    recs = [make(i, "mock", 0.6 + (i % 5) * 0.1) for i in range(20)]
    recs += [make(90, "runway", 0.95), make(91, "openai_sora", 0.95)]
    recs.append(make(92, "mock", 0.99, rating=ContentRating.R_SUGGESTIVE))
    recs.append(RenderRecord(shot=make(93, "mock", 0.9).shot,
                             result=GenResult(job_id="job-093", status=JobStatus.REJECTED,
                                              provider="mock")))
    qc = {r.result.job_id: {"score": 0.6 + (int(r.shot.id[2:]) % 5) * 0.1, "passed": True}
          for r in recs}
    qc["job-090"] = {"score": 0.95, "passed": True}
    qc["job-091"] = {"score": 0.95, "passed": True}
    qc["job-092"] = {"score": 0.99, "passed": True}

    ds = harvest(recs, qc, min_score=0.75, name="ep01-teacher")
    assert all(s.qc_score >= 0.75 for s in ds.samples), "过检门槛失效"
    assert all(s.content_rating in DEFAULT_TEACHER_RATINGS for s in ds.samples), "分级门禁失效"
    assert not any(s.shot_id == "sh93" for s in ds.samples), "未成片的镜进了数据集"
    bd = ds.license_breakdown()
    assert bd.get("training_restricted") == 1, bd     # runway
    assert bd.get("training_unknown") == 1, bd        # openai_sora
    print(f"[1] harvest: {len(recs)} 条出片 -> {len(ds)} 条教师样本；许可证分布 {bd}")

    # ---- [2] 许可证默认拦截
    n_default = len(ds.trainable())
    n_loose = len(ds.trainable(include_restricted=True, include_unknown=True))
    assert n_loose == n_default + 2, (n_default, n_loose)
    print(f"[2] trainable 默认放行 {n_default} 条；显式放宽后 {n_loose} 条（runway/openai 被默认拦下）")

    # ---- [3] target_frames 必须是 N*4+1
    for sec, fps in ((3.0, 16), (2.0, 24), (5.0, 16), (1.0, 8)):
        f = musubi_target_frames(sec, fps)
        assert (f - 1) % 4 == 0 and f >= 1, (sec, fps, f)
    assert musubi_target_frames(3.0, 16) == 49, musubi_target_frames(3.0, 16)
    print(f"[3] target_frames N*4+1: 3s@16fps -> {musubi_target_frames(3.0, 16)}，"
          f"2s@24fps -> {musubi_target_frames(2.0, 24)}，5s@16fps -> {musubi_target_frames(5.0, 16)}")

    # ---- [4] 真跑 ffmpeg：造素材 -> 切片 -> 出 toml
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "src.mp4"
        _sp_run(
            [_ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=24:duration=8",
             "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p", str(src)],
            check=True, capture_output=True)
        for s in ds.samples[:3]:
            s.video_uri = str(src)
        small = DistillDataset(name="tiny", samples=ds.samples[:3])
        man = to_training_set(small, tmp / "trainset", fps=16, resolution="544p",
                              clip_s=3.0, caption_style="cinematic", execute=True)
        assert not man.failed, man.failed
        assert man.n_clips == 3 * (8 // 3), man.n_clips
        mp4s = sorted((tmp / "trainset" / "videos").glob("*.mp4"))
        txts = sorted((tmp / "trainset" / "videos").glob("*.txt"))
        assert len(mp4s) == len(txts) == man.n_clips, (len(mp4s), len(txts))
        # 帧数必须正好等于 target_frames，否则 musubi 会丢桶
        probe = _sp_run(
            [_ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-i", str(mp4s[0]),
             "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
            capture_output=True, text=True)
        assert probe.returncode == 0, probe.stderr
        toml_txt = Path(man.musubi_toml).read_text(encoding="utf-8")
        assert 'frame_extraction = "chunk"' in toml_txt and "target_frames = [49]" in toml_txt
        assert "source_fps = 16.0" in toml_txt, toml_txt
        dp = Path(man.diffusion_pipe_toml).read_text(encoding="utf-8")
        assert "frame_buckets = [1, 49]" in dp and "resolutions = [[960, 544]]" in dp
        print(f"[4] to_training_set(execute=True): 3 条素材 -> {man.n_clips} 个切片，"
              f"{Path(mp4s[0]).stat().st_size} B/片，caption 示例：")
        print("      " + txts[0].read_text(encoding='utf-8').strip()[:90])

        # ---- [5] 干跑模式产出 shell 计划
        man2 = to_training_set(small, tmp / "plan", fps=16, clip_s=3.0, execute=False)
        sh = (tmp / "plan" / "cut_clips.sh").read_text(encoding="utf-8")
        assert sh.count("-frames:v 49") == man2.n_clips, sh[:200]
        print(f"[5] execute=False 干跑：生成 {man2.n_clips} 条 ffmpeg 命令到 cut_clips.sh，未动磁盘视频")

        # ---- [7] mix_general_data
        gdir = tmp / "general"
        gdir.mkdir()
        for i in range(40):
            (gdir / f"g{i}.mp4").write_bytes(b"\x00")
        for target in (0.15, 0.25, 0.35):
            plan = mix_general_data(man.n_clips, gdir, ratio=target)
            assert abs(plan.effective_ratio - target) < 0.05, plan
            assert 1 <= plan.general_take <= plan.general_clips
            print(f"[7] mix_general_data 目标 {target:.0%}: 教师 {plan.teacher_clips} 片 / 通用库 "
                  f"{plan.general_clips} 片 -> 取 {plan.general_take} 片，"
                  f"repeats {plan.teacher_repeats}:{plan.general_repeats}，"
                  f"实际占比 {plan.effective_ratio:.1%}")
        snip = plan.musubi_snippet(tmp / "trainset" / "videos", gdir,
                                   cache_root=tmp / "cache", target_frames=[49], fps=16)
        assert snip.count("[[datasets]]") == 2 and "num_repeats" in snip
        empty = mix_general_data(small, tmp / "nope", ratio=0.25)
        assert empty.general_repeats == 0 and empty.effective_ratio == 0.0

    # ---- [6] 多样性体检：健康集 vs 坍缩集
    rep = diversity_report(ds)
    assert "结论：分布健康" in rep, rep
    m_ok = diversity_metrics(ds)
    collapsed = DistillDataset(name="collapsed", samples=[
        TeacherSample(shot_id=f"c{i}", video_uri="x", duration_s=8.0,
                      prompt="close-up of the character turning around slowly in the rain",
                      shot_size=ShotSize.CU, camera_move=CameraMove.DOLLY_IN,
                      subject_ids=["lin"], scene_id="sc00", qc_score=0.95, provider="mock",
                      license=LicenseFlag.TRAINING_PERMITTED)
        for i in range(30)
    ])
    m_bad = diversity_metrics(collapsed)
    rep_bad = diversity_report(collapsed)
    assert m_bad.overall() < m_ok.overall(), (m_bad.overall(), m_ok.overall())
    assert m_bad.max_class_share == 1.0 and m_bad.dup_prompt_ratio > 0.9
    assert "⚠" in rep_bad and "模板化严重" in rep_bad, rep_bad
    print(f"[6] 多样性 overall：健康集 {m_ok.overall():.3f} vs 坍缩集 {m_bad.overall():.3f}；"
          f"坍缩集预警 {rep_bad.count('⚠')} 条：")
    for ln in rep_bad.splitlines():
        if ln.strip().startswith("⚠"):
            print("      " + ln.strip())
    print("    健康集报告节选：")
    for ln in rep.splitlines()[:3] + [l for l in rep.splitlines() if "综合多样性" in l]:
        print("      " + ln)

    # ---- [8] dataset_hash 稳定且对内容敏感
    h1 = ds.dataset_hash()
    assert h1 == DistillDataset(name="x", samples=list(ds.samples)).dataset_hash()
    assert h1 != collapsed.dataset_hash()
    print(f"[8] dataset_hash 稳定：{h1}（可直接填进 LoRASpec.dataset_hash）")

    print("\ndistill 自测通过")


if __name__ == "__main__":
    _selftest()
