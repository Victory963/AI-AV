"""合规与产物溯源 —— 显式/隐式标识、C2PA、发片前门禁。

三条必须同时满足的法规线（查证日期 2026-09-19）：

1. 中国《人工智能生成合成内容标识办法》（国信办等四部门，2025-09-01 施行）
   第四条：视频类「在视频起始画面和视频播放周边的适当位置添加显著的提示标识，
   可以在视频末尾和中间适当位置添加显著的提示标识」——> add_visible_disclosure。
   第五条：「在生成合成内容的文件元数据中添加隐式标识，隐式标识包含生成合成内容
   属性信息、服务提供者名称或者编码、内容编号等制作要素信息」——> write_metadata。
   来源：https://www.cac.gov.cn/2025-03/14/c_1743654684782215.htm
2. EU AI Act 第 50 条（2026-08-02 起适用）：
   第 2 款要求生成式系统的输出以**机器可读**形式标注为人工生成/篡改；
   第 4 款要求深度伪造的部署者披露内容系人工生成 ——> write_metadata + c2pa_stub。
   来源：https://artificialintelligenceact.eu/article/50/
3. C2PA 2.1：manifest = claim + assertions，AI 生成用 c2pa.actions 里的
   digitalSourceType=trainedAlgorithmicMedia 表达 ——> c2pa_stub。
   来源：https://spec.c2pa.org/specifications/specifications/2.1/specs/C2PA_Specification.html

设计立场：本模块只做**标注与拒单**。ProvenanceManifest 既是给监管看的合规材料，
也是给自己看的复现依据 —— 两者要的字段其实是同一批（引擎、模型版本、LoRA、
参考图指纹、seed、提示词哈希），所以只维护一份，不做两套。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ._fonts import ASSET_FONTS, cjk_font, family_name
from .schema import AudioRef, ContentRating, ImageRef, Shot, Storyboard, VideoRef
from .stitch import FFmpeg, default_ffmpeg
# 滤镜参数里的 \ : ' 转义规则只有一套，拼接器已经写对了；再抄一份迟早会漂。
from .stitch import _escape_filter_value as _esc

log = logging.getLogger(__name__)

__all__ = [
    "DISCLOSURE_ZH",
    "DISCLOSURE_EN",
    "IPTC_TRAINED_ALGORITHMIC_MEDIA",
    "SourceType",
    "RefDigest",
    "ShotProvenance",
    "ProvenanceManifest",
    "add_visible_disclosure",
    "write_metadata",
    "read_metadata",
    "c2pa_stub",
    "compliance_check",
    "sha256_file",
    "sha256_text",
    "parse_source_type",
]


# 办法第四条只要求「显著的提示标识」，没有规定文案。用中英双语：
# 中文满足境内监管，英文满足 AI Act 第 50 条第 4 款对披露的可理解性要求。
DISCLOSURE_ZH = "本视频由人工智能生成"
DISCLOSURE_EN = "AI-generated"

# IPTC DigitalSourceType 受控词表，C2PA 的 digitalSourceType 取值就来自这里。
IPTC_TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)
IPTC_COMPOSITE_WITH_TRAINED = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
)

# 字体不随仓库分发，按 _fonts 的查找顺序定位；一个都没有时保留原路径，调用时报错给出取得途径。
DEFAULT_FONT = cjk_font() or (ASSET_FONTS / "msyh.ttc")

DisclosurePosition = Literal[
    "top_left", "top_center", "top_right",
    "bottom_left", "bottom_center", "bottom_right",
]
# ASS 的 \an 对齐编号：7/8/9 顶部左中右，1/2/3 底部左中右。
_ALIGN: dict[str, int] = {
    "top_left": 7, "top_center": 8, "top_right": 9,
    "bottom_left": 1, "bottom_center": 2, "bottom_right": 3,
}

# 标识文字高度占画面高度的比例。
# TODO(2026-09-19): 办法本身未规定字号；配套强制性国标 GB 45438-2025
# 《网络安全技术 人工智能生成合成内容标识方法》规定了具体标注方法，但截至查证日
# 未能通过公开渠道取得全文，因此「5% 画面高度」是工程上取的可辩护下限而非引用条款。
# 取得国标全文后必须按其数值校准本常量与 _IMPLICIT_LABEL_KEYS 的字段名。
DISCLOSURE_HEIGHT_RATIO = 0.05


# ---------------------------------------------------------------- 指纹


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """流式哈希。视频文件动辄几百 MB，本机只有 3GB 内存，不能整读。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 溯源清单


SourceType = Literal[
    "ai_generated",      # 本产线或其它模型生成
    "own_creation",      # 自有拍摄/绘制
    "licensed",          # 已获授权的素材
    "public_domain",     # 公有领域
    "real_person_photo", # 真人肖像照片 —— 本产线禁止用作参考
    "unknown",           # 未标注来源，视同不合规
]

# 参考素材的来源类型写在 note 里（schema 不为此单开字段：
# note 是自由文本，加一个受控标签比改契约便宜，且不影响已有分镜 JSON）。
# 大小写不敏感：标注人手写 "Source=Licensed" 是常态，把它判成 unknown 只会
# 在发片前刷出一堆假告警，而真正该拦的 real_person_photo 反而被淹没。
_SOURCE_RE = re.compile(r"source\s*=\s*([A-Za-z_]+)", re.IGNORECASE)
_VALID_SOURCES = {
    "ai_generated", "own_creation", "licensed", "public_domain", "real_person_photo",
}

# ImageRef / VideoRef / AudioRef 三者的结构在 schema 里是平行的（role/uri/weight/note），
# 来源标注规则对它们完全一样，所以这里按「有 note 的参考位」统一处理，
# 不给每种类型抄一份判定逻辑。
AnyRef = ImageRef | VideoRef | AudioRef


def parse_source_type(ref: AnyRef) -> SourceType:
    """从参考位的 note 里解析 `source=xxx` 标签。缺失或非法一律算 unknown。

    刻意不做「看起来像 AI 生成就当 ai_generated」的猜测：
    来源声明是法律责任，猜错的成本由用户承担，必须由标注人显式写下。
    """
    if m := _SOURCE_RE.search(ref.note or ""):
        v = m.group(1).lower()
        if v in _VALID_SOURCES:
            return v  # type: ignore[return-value]
    return "unknown"


class RefDigest(BaseModel):
    """一张参考图在产物里的留痕。"""

    role: str
    uri: str
    sha256: str = ""            # 本地取不到文件时留空，并在 note 里说明
    source_type: SourceType = "unknown"
    weight: float = 1.0
    subject_id: str | None = None
    note: str = ""


class ShotProvenance(BaseModel):
    """单镜溯源。字段选取标准：能不能靠它把这一镜重跑出来。"""

    shot_id: str
    scene_id: str = ""
    provider: str = ""              # Shot.provider_used
    model_version: str = ""         # 引擎侧的模型版本号，由调用方按 job 回填
    job_id: str | None = None
    seed: int | None = None

    prompt_sha256: str = ""
    negative_prompt_sha256: str = ""
    shot_fingerprint: str = ""      # Shot.fingerprint()，缓存键与复现锚

    lora_path: str = ""
    lora_trigger: str = ""
    lora_strength: float = 0.0
    lora_dataset_hash: str = ""

    refs: list[RefDigest] = Field(default_factory=list)
    subject_ids: list[str] = Field(default_factory=list)
    age_statements: dict[str, str] = Field(default_factory=dict)

    content_rating: str = ContentRating.G.value
    duration_s: float = 0.0
    fps: int = 24
    resolution: tuple[int, int] = (1280, 720)

    render_uri: str | None = None
    render_sha256: str = ""


class ProvenanceManifest(BaseModel):
    """一集片的完整溯源清单。

    这份 JSON 有三个去处：落盘存档、压进视频元数据（隐式标识）、
    喂给 c2pa_stub 生成 C2PA manifest。三处同源，避免「归档说 A、元数据说 B」。
    """

    spec_version: Literal["longfilm-provenance/1"] = "longfilm-provenance/1"
    project: str
    episode: str = "ep01"
    created_at: str = ""

    # 办法第五条点名要的三样：属性信息、服务提供者名称或编码、内容编号。
    content_producer: str = ""      # 服务提供者名称
    producer_code: str = ""         # 服务提供者编码（统一社会信用代码等）
    produce_id: str = ""            # 内容编号；留空时用清单摘要兜底

    tool_name: str = "longfilm"
    tool_version: str = "0.1.0"

    ai_disclosure: bool = True
    disclosure_text: str = DISCLOSURE_ZH
    shots: list[ShotProvenance] = Field(default_factory=list)

    @classmethod
    def from_storyboard(
        cls,
        sb: Storyboard,
        *,
        content_producer: str,
        producer_code: str = "",
        produce_id: str = "",
        model_versions: dict[str, str] | None = None,
        prompts: dict[str, str] | None = None,
        asset_root: str | Path | None = None,
    ) -> ProvenanceManifest:
        """从分镜构建清单。

        prompts 传的是**真正送进引擎的那条提示词**（由 prompt_os 拼装）；
        不传时退回镜头语义字段的哈希 —— 后者能证明分镜没被改过，
        但证明不了提示词没被改过，所以生产链路必须把 prompts 传进来。
        """
        root = Path(asset_root) if asset_root else None
        mv = model_versions or {}
        shots: list[ShotProvenance] = []
        for sc in sb.scenes:
            for s in sc.shots:
                prompt = prompts.get(s.id) if prompts else None
                shots.append(
                    ShotProvenance(
                        shot_id=s.id,
                        scene_id=s.scene_id,
                        provider=s.provider_used or "",
                        model_version=mv.get(s.provider_used or "", ""),
                        job_id=s.job_id,
                        seed=s.seed,
                        prompt_sha256=sha256_text(prompt if prompt is not None else _semantic_prompt(s)),
                        negative_prompt_sha256=sha256_text(s.negative_prompt or sb.global_negative),
                        shot_fingerprint=s.fingerprint(),
                        **_lora_fields(sb, s),
                        # 参考视频/音频同样是生成输入，溯源不能只记图：
                        # 音色参考正是「这段声音是谁的」唯一可核验线索。
                        refs=[
                            _ref_digest(r, root)
                            for r in (*s.refs.images, *s.refs.videos, *s.refs.audios)
                        ],
                        subject_ids=list(s.subject_ids),
                        age_statements={
                            cid: (c.age_statement if (c := sb.character(cid)) else "")
                            for cid in s.subject_ids
                        },
                        content_rating=s.content_rating.value,
                        duration_s=s.duration_s,
                        fps=s.fps,
                        resolution=tuple(sb.delivery.upscale_to or sb.delivery.resolution),
                        render_uri=s.render_uri,
                        render_sha256=(
                            sha256_file(p) if s.render_uri and (p := _resolve(s.render_uri, root)) else ""
                        ),
                    )
                )
        m = cls(
            project=sb.project,
            episode=sb.episode,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            content_producer=content_producer,
            producer_code=producer_code,
            tool_name="longfilm",
            ai_disclosure=sb.delivery.ai_disclosure,
        )
        m.shots = shots
        # 内容编号缺省用清单摘要：它对同一份产物稳定、对任何字段改动敏感，
        # 天然满足「编号唯一且可核验」，比随机 UUID 更有用。
        m.produce_id = produce_id or f"{sb.project}-{sb.episode}-{m.digest()[:16]}"
        return m

    def digest(self) -> str:
        """清单指纹。刻意排除 produce_id 自身，否则自引用算不出稳定值。"""
        payload = self.model_dump(mode="json", exclude={"produce_id", "created_at"})
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def implicit_label(self) -> dict[str, object]:
        """办法第五条要求写进文件元数据的隐式标识块。

        字段名沿用业界对 GB 45438-2025 的通行实现（Label / ContentProducer /
        ProduceID / ...）。
        TODO(2026-09-19): 国标全文未能取得，字段名与取值编码方式属**待核对项**；
        拿到标准后按其附录校正，不要因为「现在能跑」就认为已经合规。
        """
        return {
            "AIGC": {
                "Label": "1",                       # 1 = 人工智能生成合成
                "ContentProducer": self.producer_code or self.content_producer,
                "ProduceID": self.produce_id,
                "ReservedCode1": "",
                "ContentPropagator": "",
                "PropagateID": "",
            },
            "longfilm": {
                "spec": self.spec_version,
                "project": self.project,
                "episode": self.episode,
                "created_at": self.created_at,
                "tool": f"{self.tool_name}/{self.tool_version}",
                "shots": len(self.shots),
                "manifest_sha256": self.digest(),
            },
        }

    def engines_used(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.shots:
            key = f"{s.provider}@{s.model_version}" if s.model_version else (s.provider or "unknown")
            out[key] = out.get(key, 0) + 1
        return out

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=indent)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> ProvenanceManifest:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _semantic_prompt(s: Shot) -> str:
    """镜头语义字段的规范化串，作为提示词哈希的兜底输入。"""
    return " | ".join(
        [s.shot_size.value, s.camera_move.value, s.action, s.environment, s.lighting, s.mood, s.style]
    )


def _lora_fields(sb: Storyboard, s: Shot) -> dict[str, object]:
    """取该镜主角的 LoRA。多主角时取第一个有 LoRA 的 ——
    一镜挂多 LoRA 在当前引擎里并不常见，真出现时清单会漏记，所以这里留日志。"""
    for cid in s.subject_ids:
        c = sb.character(cid)
        if c and c.lora:
            return {
                "lora_path": c.lora.path,
                "lora_trigger": c.lora.trigger_word,
                "lora_strength": c.lora.strength,
                "lora_dataset_hash": c.lora.dataset_hash or "",
            }
    return {}


def _resolve(uri: str, root: Path | None) -> Path | None:
    """只处理本地可读文件；远端 URI 返回 None（哈希留空并如实标注）。"""
    if "://" in uri and not uri.startswith("file://"):
        return None
    p = Path(uri[7:] if uri.startswith("file://") else uri)
    if not p.is_absolute() and root:
        p = root / p
    return p if p.is_file() else None


def _ref_digest(ref: AnyRef, root: Path | None) -> RefDigest:
    p = _resolve(ref.uri, root)
    return RefDigest(
        role=ref.role,
        uri=ref.uri,
        sha256=sha256_file(p) if p else "",
        source_type=parse_source_type(ref),
        weight=ref.weight,
        # VideoRef/AudioRef 没有 subject_id，缺了就留空而不是崩掉。
        subject_id=getattr(ref, "subject_id", None),
        note="" if p else "本地不可读，未计算哈希",
    )


def iter_refs(sb: Storyboard) -> list[tuple[str, AnyRef]]:
    """列出全片所有需要核验来源的参考位，附一句「它挂在哪」的定位串。

    角色圣经里的定妆/转身/动态参考同样要过门禁：refpack 正是从 portraits 里
    挑 identity 锚的，一张真人照放在那里，比放在某一镜的参考位上影响更大
    —— 它会进**每一个**用到该角色的镜头。
    """
    out: list[tuple[str, AnyRef]] = []
    for c in sb.characters:
        for label, refs in (
            ("portraits", c.portraits), ("turnaround", c.turnaround),
            ("motion_refs", c.motion_refs),
        ):
            out += [(f"角色 {c.id}.{label}", r) for r in refs]
    for s in sb.all_shots():
        for r in s.refs.images:
            out.append((f"{s.id} 参考图", r))
        for r in s.refs.videos:
            out.append((f"{s.id} 参考视频", r))
        for r in s.refs.audios:
            out.append((f"{s.id} 参考音频", r))
    return out


# ---------------------------------------------------------------- 显式标识


def _font_family(font: Path) -> str:
    """读出字体的 family name。libass 按 family 匹配，不认文件名。"""
    return family_name(font)


def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def build_disclosure_ass(
    text: str,
    *,
    width: int,
    height: int,
    position: DisclosurePosition,
    font_family: str,
    font_px: int,
    start_s: float,
    end_s: float,
    box: bool,
) -> str:
    """生成显式标识用的 ASS 字幕。

    独立成函数是为了能在不跑 ffmpeg 的情况下断言字号/对齐 ——
    标识不显著在法规上是实质缺陷，而渲染完再用肉眼看是查不出 1px 差别的。
    """
    align = _ALIGN[position]
    margin = max(8, round(height * 0.02))
    # BorderStyle=3 是不透明底框：标识要在任何画面上都可读，
    # 只靠描边在高频纹理（树叶、人群）上会糊掉，达不到「显著」。
    border_style, outline = (3, round(font_px * 0.25)) if box else (1, max(2, round(font_px * 0.08)))
    # ASS 的 { } 是特效标签起止符，正文里出现会被吞掉甚至吃掉后面整行。
    safe = text.replace("{", "（").replace("}", "）").replace("\n", r"\N")
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Disclosure,{font_family},{font_px},&H00FFFFFF,&H000000FF,&H00000000,"
            f"&HA0000000,1,0,0,0,100,100,0,0,{border_style},{outline},0,{align},"
            f"{margin},{margin},{margin},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            f"Dialogue: 0,{_ass_time(start_s)},{_ass_time(end_s)},Disclosure,,0,0,0,,{safe}",
            "",
        ]
    )


def add_visible_disclosure(
    video: str | Path,
    out: str | Path,
    *,
    text: str = DISCLOSURE_ZH,
    position: DisclosurePosition = "top_left",
    font: str | Path = DEFAULT_FONT,
    height_ratio: float = DISCLOSURE_HEIGHT_RATIO,
    start_s: float = 0.0,
    end_s: float | None = None,
    box: bool = True,
    crf: int = 18,
    preset: str = "medium",
    ffmpeg: FFmpeg | None = None,
) -> Path:
    """烧录显式 AI 生成标识。

    走 libass（ass 滤镜）而不是 drawtext：本项目的静态 ffmpeg 没有 drawtext，
    而且 libass 的中文换行、描边和底框质量本来就比 drawtext 好。

    默认覆盖全片（end_s=None）。办法第四条对视频只强制要求「起始画面」有标识，
    中间和末尾是「可以」；但产线产物会被二次剪辑和搬运，只标开头等于标了个寂寞，
    所以默认常驻，需要只标开头的场景显式传 end_s。
    """
    ff = ffmpeg or default_ffmpeg()
    src, out_p, font_p = Path(video), Path(out), Path(font)
    if not font_p.is_file():
        raise FileNotFoundError(f"字体不存在：{font_p}；显式标识必须能渲染中文，不接受回退到方框。"
                                "取得途径见 assets/fonts/README.md")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    info = ff.probe(src)
    if not info.has_video:
        raise ValueError(f"{src} 没有视频流，无法烧录显式标识")
    font_px = max(12, round(info.height * height_ratio))
    # 时间窗兜底：容器头里没有 Duration（某些引擎直出的 fragmented mp4 就是这样）
    # 时 probe 会给 0，照它写 Dialogue 的结束时间就是「0 秒后消失」——
    # 渲染不会报错，产物却一个标识都没有，等于漏标。宁可标到天荒地老。
    stop = end_s if end_s is not None else info.duration_s
    if stop <= start_s:
        log.warning("%s 时长探测为 %.3fs，显式标识改为常驻到片尾（起 %.3fs）",
                    src, info.duration_s, start_s)
        stop = start_s + 359_999.0   # ASS 时间码上限 9:59:59.99，够任何长片
    ass = build_disclosure_ass(
        text,
        width=info.width,
        height=info.height,
        position=position,
        font_family=_font_family(font_p),
        font_px=font_px,
        start_s=start_s,
        end_s=stop,
        box=box,
    )
    ass_path = out_p.with_suffix(".disclosure.ass")
    ass_path.write_text(ass, encoding="utf-8")

    vf = f"ass=f='{_esc(str(ass_path))}':fontsdir='{_esc(str(font_p.parent))}'"
    ff.run(
        [
            "-y", "-i", str(src), "-vf", vf,
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
            "-c:a", "copy", str(out_p),
        ],
        timeout=7200.0,
    )
    log.info("显式标识已烧录：%s（%s，字号 %dpx = 画面高度 %.1f%%）",
             out_p, position, font_px, height_ratio * 100)
    return out_p


# ---------------------------------------------------------------- 隐式标识


def write_metadata(
    video: str | Path,
    out: str | Path,
    manifest: ProvenanceManifest,
    *,
    embed_full_manifest: bool = False,
    ffmpeg: FFmpeg | None = None,
) -> Path:
    """写入隐式标识（办法第五条）。

    `-c copy` 不重编码：隐式标识是容器层的事，为了写几行文字重压一遍画质说不过去。

    comment 字段放 AIGC JSON 块 —— 办法要求的落点就是「文件元数据」，
    而 comment/description 是 mp4/mkv/mov 都支持、绝大多数工具都能读出来的公共字段。
    完整清单默认不内嵌：一集片的 manifest 有几十 KB，塞进 udta 会让部分播放器
    解析变慢甚至截断；归档走 manifest.save()，元数据里只留 manifest_sha256 做锚。
    """
    ff = ffmpeg or default_ffmpeg()
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    label = manifest.implicit_label()
    if embed_full_manifest:
        label["longfilm_manifest"] = json.loads(manifest.to_json(indent=0))
    blob = json.dumps(label, ensure_ascii=False, separators=(",", ":"))

    engines = "、".join(manifest.engines_used()) or "unknown"
    description = (
        f"{manifest.disclosure_text}。制作者：{manifest.content_producer or '未声明'}；"
        f"内容编号：{manifest.produce_id}；生成引擎：{engines}；"
        f"清单摘要 sha256:{manifest.digest()[:16]}"
    )
    ff.run(
        [
            "-y", "-i", str(video), "-map", "0", "-c", "copy",
            "-movflags", "use_metadata_tags",
            "-metadata", f"title={manifest.project} {manifest.episode}",
            "-metadata", f"comment={blob}",
            "-metadata", f"description={description}",
            "-metadata", f"artist={manifest.content_producer}",
            str(out_p),
        ],
        timeout=1800.0,
    )
    log.info("隐式标识已写入：%s（%d 字节元数据）", out_p, len(blob))
    return out_p


def read_metadata(video: str | Path, *, ffmpeg: FFmpeg | None = None) -> dict[str, str]:
    """回读容器元数据。写完必须能读回来才算写成功 ——
    ffmpeg 对不支持的 tag 是静默丢弃的，不回读就永远不知道标识没落地。

    走 `-f ffmetadata` 而不是解析 `ffmpeg -i` 的 stderr：后者**会把长标签截断显示**
    （本机实测 comment 被切在 255 字符左右），照它解析出来的 JSON 是断的，
    会让人误判成隐式标识没写进去。ffmetadata 是无损的机器格式。
    """
    ff = ffmpeg or default_ffmpeg()
    raw = ff.run_capture(["-i", str(video), "-f", "ffmetadata", "-"], timeout=120.0)
    return _parse_ffmetadata(raw.decode("utf-8", "replace"))


def _unescape_ffmeta(v: str) -> str:
    """ffmetadata 用反斜杠转义 = ; # \\ 和换行。"""
    return re.sub(r"\\(.)", r"\1", v)


def _split_ffmeta(line: str) -> tuple[str, str] | None:
    """按第一个**未转义**的 = 分割；转义过的 = 属于键名本身。"""
    i = 0
    while i < len(line):
        if line[i] == "\\":
            i += 2
            continue
        if line[i] == "=":
            return line[:i], line[i + 1:]
        i += 1
    return None


def _parse_ffmetadata(text: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    key: str | None = None
    val = ""
    for line in text.splitlines():
        if key is None:
            if not line or line[0] in ";#":
                continue
            if line[0] == "[":
                break   # 进入 [STREAM]/[CHAPTER] 段，全局标签已经读完
            split = _split_ffmeta(line)
            if split is None:
                continue
            key, val = split
        else:
            val += "\n" + line
        if (len(val) - len(val.rstrip("\\"))) % 2 == 1:
            val = val[:-1]
            continue    # 行尾单个反斜杠 = 值续到下一行
        tags[_unescape_ffmeta(key).lower()] = _unescape_ffmeta(val)
        key, val = None, ""
    return tags


# ---------------------------------------------------------------- C2PA


def c2pa_stub(manifest: ProvenanceManifest) -> dict[str, object]:
    """生成 C2PA manifest definition（未签名）。

    返回的结构对齐 c2patool / c2pa-python 吃的 manifest definition：
    claim_generator_info + title + format + instance_id + assertions + ingredients。
    签名配置（alg / private_key / sign_cert）刻意不在这里 ——
    私钥不该经过一个纯数据函数。

    TODO(2026-09-19): 真正签名需要 `pip install c2pa-python` 与一张符合 C2PA
    信任列表的证书（测试可用 c2patool 自带的 es256 测试证书，但这种签名
    在验证端会标记为不受信任）。本函数只保证结构正确，不产生任何法律效力的声明。
    """
    when = manifest.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 有参考图参与的产物在 IPTC 词表里属于「合成」而非纯生成：
    # 用错 digitalSourceType 等于做了一份不准确的声明，比不声明更糟。
    has_refs = any(s.refs for s in manifest.shots)
    source_type = IPTC_COMPOSITE_WITH_TRAINED if has_refs else IPTC_TRAINED_ALGORITHMIC_MEDIA

    actions = [
        {
            "action": "c2pa.created",
            "when": when,
            "digitalSourceType": source_type,
            "softwareAgent": {
                "name": manifest.tool_name,
                "version": manifest.tool_version,
            },
        }
    ]
    # 每个引擎记一条 placed 动作：溯源的核心问题是「哪一段是谁生成的」，
    # 把引擎聚合成一条会把这个信息抹掉。
    for engine, count in manifest.engines_used().items():
        name, _, version = engine.partition("@")
        actions.append(
            {
                "action": "c2pa.placed",
                "when": when,
                "digitalSourceType": IPTC_TRAINED_ALGORITHMIC_MEDIA,
                "softwareAgent": {"name": name, "version": version or "unknown"},
                "parameters": {"org.longfilm.shot_count": count},
            }
        )

    # Ingredient 的字段集由 c2pa-rs 的 ManifestDefinition 定死（title/format/
    # instance_id/relationship/hash/description/...），塞自定义顶层键会被解析器拒收。
    # role 与 source_type 写进 description，机器可读的那份在
    # org.longfilm.provenance assertion 里已经有了，不必在这里再挂一份。
    # relationship 用 inputTo：C2PA 规范正是把「喂给生成模型的提示词/种子图」
    # 定义为 inputTo（componentOf 是素材合成，parentOf 是编辑前的原图）。
    # TODO(2026-09-20): C2PA 的 ingredient.hash 是二进制摘要的 base64，
    # 这里放的是十六进制串；真正签名前必须转码，否则验证端会判 hash 不匹配。
    ingredients = [
        {
            "title": Path(r.uri).name or r.uri,
            "relationship": "inputTo",
            "format": _guess_mime(r.uri),
            "instance_id": f"xmp:iid:{r.sha256[:32]}" if r.sha256 else f"uri:{r.uri}",
            "hash": r.sha256,
            "description": f"role={r.role}; source_type={r.source_type}",
        }
        for r in _unique_refs(manifest)
    ]

    return {
        "claim_generator_info": [
            {"name": manifest.tool_name, "version": manifest.tool_version}
        ],
        "title": f"{manifest.project} {manifest.episode}",
        "format": "video/mp4",
        "instance_id": f"xmp:iid:{manifest.digest()[:32]}",
        "assertions": [
            {"label": "c2pa.actions.v2", "data": {"actions": actions}},
            {
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "CreativeWork",
                    "producer": {"@type": "Organization", "name": manifest.content_producer},
                    "identifier": manifest.produce_id,
                    "dateCreated": when,
                },
            },
            {
                "label": "cawg.training-mining",
                "data": {
                    # 产物本身不希望被再拿去训练，这是 CAWG 定义的标准表达方式。
                    "entries": {
                        "cawg.ai_generative_training": {"use": "notAllowed"},
                        "cawg.ai_training": {"use": "notAllowed"},
                    }
                },
            },
            {
                "label": "org.longfilm.provenance",
                "data": json.loads(manifest.to_json(indent=0)),
            },
        ],
        "ingredients": ingredients,
    }


def _guess_mime(uri: str) -> str:
    """按扩展名猜 MIME。C2PA 的 format 要求是**具体**的媒体类型，
    早前写死的 "image/*" 既不是合法 MIME，也把参考视频/音频全标成了图片。"""
    import mimetypes  # noqa: PLC0415  只在生成 C2PA 结构时用得上

    mime, _ = mimetypes.guess_type(uri.split("?", 1)[0])
    return mime or "application/octet-stream"


def _unique_refs(manifest: ProvenanceManifest) -> list[RefDigest]:
    seen: dict[str, RefDigest] = {}
    for s in manifest.shots:
        for r in s.refs:
            seen.setdefault(r.sha256 or r.uri, r)
    return list(seen.values())


# ---------------------------------------------------------------- 发片门禁


# "未成年" 里含 "成年"，先判否定词再判肯定词，顺序反了会把未成年声明判成合规。
# 英文一律加词边界：不加的话 "eighteen"/"nineteen" 里的 "teen"、
# "childhood friend" 里的 "child"、"minority" 里的 "minor" 都会被判成未成年表述，
# 把**合法的成年声明**拦下来 —— 误拦和漏放一样是事故。
_MINOR_RE = re.compile(
    r"未成年|未满|\bminors?\b|\bunderage\b|\bchild(?:ren)?\b|\bteen(?:s|ager|agers|aged)?\b",
    re.IGNORECASE,
)
_ADULT_RE = re.compile(r"成年|成人|\badults?\b|18\+", re.IGNORECASE)
_AGE_RE = re.compile(r"(\d{1,3})\s*(?:岁|years?\s*old|yo)")


def _age_problem(c_id: str, statement: str) -> str | None:
    """校验 age_statement 是否构成有效的成年声明。"""
    if _MINOR_RE.search(statement):
        return f"角色 {c_id}: age_statement 含未成年表述「{statement}」，产线拒单"
    ages = [int(a) for a in _AGE_RE.findall(statement)]
    if any(a < 18 for a in ages):
        return f"角色 {c_id}: age_statement 中的年龄 {min(ages)} 小于 18，产线拒单"
    if not ages and not _ADULT_RE.search(statement):
        return (
            f"角色 {c_id}: age_statement「{statement}」既没有明确年龄也没有成年表述，"
            "无法作为合规材料"
        )
    if "虚构" not in statement and "fiction" not in statement.lower():
        return f"角色 {c_id}: age_statement 未声明角色为虚构，无法排除真人指涉"
    return None


def compliance_check(storyboard: Storyboard) -> list[str]:
    """发片前门禁。返回问题清单，空列表 = 可以发。

    与 Storyboard.validate_continuity 一样不抛异常：合规问题要**一次列全**，
    抛第一个异常会让人改一条跑一次，改到第五条才发现第六条。
    """
    problems: list[str] = []

    if not storyboard.characters:
        problems.append("角色圣经为空：无法核验任何年龄声明")
    for c in storyboard.characters:
        if p := _age_problem(c.id, c.age_statement):
            problems.append(p)

    known = {c.id for c in storyboard.characters}
    shots = storyboard.all_shots()
    for s in shots:
        for cid in s.subject_ids:
            if cid not in known:
                problems.append(f"{s.id}: 角色 {cid} 不在角色圣经里，无法核验年龄声明")
        if s.content_rating is ContentRating.BLOCKED:
            problems.append(f"{s.id}: 内容分级 blocked，产线拒单")

    # 真人素材参考：既是肖像权/声音权问题，也直接落进 AI Act 第 50 条第 4 款的
    # deep fake 定义（对真实人物的可辨识再现），本产线一律不做。
    # 覆盖面必须是**全部**参考位 —— 只查镜头参考图的话，一张真人定妆照放进
    # CharacterBible.portraits 就能绕过门禁，而它会进每一个用到该角色的镜头；
    # 参考视频（动作）和参考音频（音色克隆）同理，后者正是声音深度伪造的入口。
    sources = "|".join(sorted(_VALID_SOURCES))
    for where, ref in iter_refs(storyboard):
        st = parse_source_type(ref)
        if st == "real_person_photo":
            problems.append(
                f"{where} {ref.uri} 标注为真人素材（source=real_person_photo），"
                "本产线只做虚构角色，拒单"
            )
        elif st == "unknown":
            problems.append(
                f"{where} {ref.uri} 未标注来源类型，请在 note 里写 source=<{sources}>"
            )

    d = storyboard.delivery
    if not d.ai_disclosure:
        problems.append(
            "delivery.ai_disclosure=False：《人工智能生成合成内容标识办法》第四条要求"
            "在视频起始画面添加显著提示标识，不得关闭"
        )
    if not d.c2pa:
        problems.append(
            "delivery.c2pa=False：EU AI Act 第 50 条第 2 款要求输出带机器可读标记，"
            "面向欧盟发行时必须开启"
        )
    if not shots:
        problems.append("分镜里没有任何镜头，无可交付内容")

    return problems


# ---------------------------------------------------------------- 自测


def _demo_storyboard() -> Storyboard:
    from .schema import (
        Appearance, CharacterBible, DeliverySpec, LoRASpec, RefPack, Scene, ShotSize,
    )

    lin = CharacterBible(
        id="lin",
        name="林决",
        age_statement="虚构角色，设定年龄 29 岁",
        appearance=Appearance(face="棱角分明", hair="短寸", distinguishing="左眉断疤"),
        lora=LoRASpec(
            base_model="wan2.2-i2v-a14b", path="/lora/lin.safetensors",
            trigger_word="linjue_ref", strength=0.8, dataset_hash="abc123",
        ),
    )
    shot = Shot(
        id="s1_01", scene_id="s1", index=0, duration_s=8.0, shot_size=ShotSize.MCU,
        subject_ids=["lin"], action="推开铁门", environment="废弃厂房", seed=42,
        provider_used="ark_seedance", job_id="job-001",
        refs=RefPack(images=[
            ImageRef(role="identity", uri="assets/lin_id.png", note="定妆 source=ai_generated", subject_id="lin"),
            ImageRef(role="environment", uri="assets/factory.png", note="source=licensed"),
        ]),
    )
    return Storyboard(
        project="dark_alley", episode="ep01", characters=[lin],
        scenes=[Scene(id="s1", title="开场", shots=[shot])],
        delivery=DeliverySpec(resolution=(640, 360), upscale_to=None, ai_disclosure=True, c2pa=True),
    )


def _selftest() -> None:
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ff = default_ffmpeg()
    sb = _demo_storyboard()

    # 1) 门禁：干净的分镜必须零问题
    assert compliance_check(sb) == [], compliance_check(sb)
    print("[1] 合规门禁：干净分镜放行")

    # 2) 门禁：每类违规都要被单独抓到，而不是抓到一条就停
    bad = sb.model_copy(deep=True)
    bad.characters[0].age_statement = "17 岁的高中生"
    bad.scenes[0].shots[0].refs.images[1].note = "source=real_person_photo"
    bad.scenes[0].shots[0].refs.images.append(ImageRef(role="style", uri="x.png", note="随手截的"))
    bad.delivery.ai_disclosure = False
    bad.delivery.c2pa = False
    issues = compliance_check(bad)
    for i in issues:
        print(f"    - {i}")
    assert len(issues) == 5, issues
    assert any("小于 18" in i for i in issues)
    assert any("真人素材" in i for i in issues)
    assert any("未标注来源类型" in i for i in issues)
    assert any("第四条" in i for i in issues) and any("第 50 条" in i for i in issues)
    # "未成年" 不能被 "成年" 的肯定规则蒙混过关
    assert _age_problem("x", "虚构角色，未成年") is not None
    assert _age_problem("x", "虚构角色，设定年龄 25 岁") is None
    assert _age_problem("x", "设定年龄 25 岁") is not None, "缺少虚构声明应当拦截"
    assert _age_problem("x", "虚构角色，未满 18 岁") is not None
    # 英文成年声明里的 eighteen/nineteen 含 "teen"、childhood 含 "child"、
    # minority 含 "minor"：不加词边界就会把合法的成年声明当未成年表述拦下来。
    for word in ("eighteen", "nineteen", "canteen", "childhood", "minority"):
        assert not _MINOR_RE.search(word), word
    for ok_text in (
        "fictional adult character, eighteen years old",
        "虚构角色，19 岁，childhood friend of the lead",
        "fictional adult, nineteen",
    ):
        assert _age_problem("x", ok_text) is None, ok_text
    for bad_text in ("fictional minor", "fictional teenager, 16 yo", "虚构角色，未成年"):
        assert _age_problem("x", bad_text) is not None, bad_text
    print(f"[2] 合规门禁：抓到 {len(issues)} 类违规；eighteen/nineteen/childhood 不误拦")

    # 2b) 真人素材可以藏在三个地方，一个都不能漏：
    #     角色定妆照（会进每一个用到该角色的镜头）、参考视频（动作）、参考音频（音色克隆）
    from .schema import AudioRef, RefPack as RP, VideoRef
    sneak = sb.model_copy(deep=True)
    sneak.characters[0].portraits.append(
        ImageRef(role="identity", uri="real_star.jpg", note="source=real_person_photo",
                 subject_id="lin")
    )
    sneak.scenes[0].shots[0].refs.videos.append(
        VideoRef(role="motion", uri="real_dance.mp4", note="source=real_person_photo")
    )
    sneak.scenes[0].shots[0].refs.audios.append(
        AudioRef(role="voice_timbre", uri="real_voice.wav", note="source=real_person_photo")
    )
    sneaked = compliance_check(sneak)
    for i in sneaked:
        print(f"    - {i}")
    assert len(sneaked) == 3, sneaked
    assert any("portraits" in i for i in sneaked), "定妆照里的真人照必须被抓到"
    assert any("参考视频" in i for i in sneaked) and any("参考音频" in i for i in sneaked)
    # 来源标签大小写不敏感，否则手写 "Source=Licensed" 会刷出假告警淹没真问题
    assert parse_source_type(ImageRef(role="style", uri="a.png", note="Source=Licensed")) \
        == "licensed"
    assert parse_source_type(VideoRef(role="motion", uri="a.mp4", note="source=ai_generated")) \
        == "ai_generated"
    assert parse_source_type(ImageRef(role="style", uri="a.png", note="随手截的")) == "unknown"
    assert len(iter_refs(sb)) == 2 and len(iter_refs(sneak)) == 5
    print(f"[2b] 定妆照/参考视频/参考音频里的真人素材各抓到一条，共 {len(sneaked)} 条")

    # 2c) 空分镜：不崩，且两类缺失都要报
    empty = compliance_check(Storyboard(project="empty"))
    assert any("角色圣经为空" in i for i in empty) and any("没有任何镜头" in i for i in empty)
    empty_m = ProvenanceManifest.from_storyboard(Storyboard(project="empty"),
                                                 content_producer="x")
    assert empty_m.shots == [] and empty_m.engines_used() == {}
    assert json.dumps(c2pa_stub(empty_m), ensure_ascii=False)
    print("[2c] 空分镜：门禁报缺角色/缺镜头，空清单仍能出 C2PA 结构")

    # 3) 溯源清单
    m = ProvenanceManifest.from_storyboard(
        sb, content_producer="示例影视工作室", producer_code="91310000MA1K00000X",
        model_versions={"ark_seedance": "seedance-1.0-pro"},
        prompts={"s1_01": "linjue_ref, medium close-up, 推开铁门"},
    )
    print(f"[3] 清单：produce_id={m.produce_id} digest={m.digest()[:16]} 引擎={m.engines_used()}")
    sp = m.shots[0]
    assert sp.seed == 42 and sp.job_id == "job-001"
    assert sp.lora_trigger == "linjue_ref" and sp.lora_dataset_hash == "abc123"
    assert sp.age_statements == {"lin": "虚构角色，设定年龄 29 岁"}
    assert [r.source_type for r in sp.refs] == ["ai_generated", "licensed"]
    assert sp.prompt_sha256 == sha256_text("linjue_ref, medium close-up, 推开铁门")
    assert sp.shot_fingerprint == sb.all_shots()[0].fingerprint()
    # digest 必须对内容敏感、对 produce_id 自身不敏感（否则自引用算不出稳定值）
    d0 = m.digest()
    probe = m.model_copy(deep=True)
    probe.produce_id = "换个编号"
    assert probe.digest() == d0
    probe.shots[0].seed = 43
    assert probe.digest() != d0
    print("[3b] 清单摘要对内容敏感、对编号不敏感")

    # 4) C2PA 结构
    c = c2pa_stub(m)
    labels = [a["label"] for a in c["assertions"]]
    print(f"[4] C2PA assertions: {labels}")
    assert labels[0] == "c2pa.actions.v2"
    acts = c["assertions"][0]["data"]["actions"]
    assert acts[0]["action"] == "c2pa.created"
    # 有参考图 -> compositeWithTrainedAlgorithmicMedia，不能一律写成纯生成
    assert acts[0]["digitalSourceType"] == IPTC_COMPOSITE_WITH_TRAINED, acts[0]
    assert any(a["action"] == "c2pa.placed" and "seedance" in a["softwareAgent"]["name"] for a in acts)
    assert c["claim_generator_info"][0]["name"] == "longfilm"
    assert len(c["ingredients"]) == 2 and all(i["relationship"] == "inputTo" for i in c["ingredients"])
    # ingredient 的 format 必须是具体 MIME（"image/*" 不是合法媒体类型），
    # 且不能出现 c2pa-rs 不认识的自定义顶层键
    allowed_ing_keys = {"title", "relationship", "format", "instance_id", "hash", "description"}
    for ing in c["ingredients"]:
        assert set(ing) <= allowed_ing_keys, ing
        assert ing["format"] == "image/png", ing
        assert "role=" in ing["description"] and "source_type=" in ing["description"]
    # 参考视频/音频也要进清单与 ingredients，并拿到各自的 MIME
    mm = ProvenanceManifest.from_storyboard(
        sneak, content_producer="示例影视工作室",
        model_versions={"ark_seedance": "seedance-1.0-pro"},
    )
    mimes = {i["format"] for i in c2pa_stub(mm)["ingredients"]}
    assert {"image/png", "video/mp4", "audio/x-wav"} <= mimes, mimes
    # 清单只记**镜头**参考位：角色圣经里的定妆照是候选池，真正被选中的那几张
    # 由 refpack 编进 shot.refs.images，没被选中的没有参与生成，记进溯源反而是噪音。
    # （门禁不同：候选池里有真人照就必须拦，见 iter_refs。）
    assert not any(i["title"] == "real_star.jpg" for i in c2pa_stub(mm)["ingredients"])
    print(f"[4c] ingredient 字段合法，MIME 按扩展名推断：{sorted(mimes)}")
    assert json.dumps(c, ensure_ascii=False), "C2PA stub 必须可序列化"
    no_ref = m.model_copy(deep=True)
    for s in no_ref.shots:
        s.refs = []
    assert c2pa_stub(no_ref)["assertions"][0]["data"]["actions"][0]["digitalSourceType"] == (
        IPTC_TRAINED_ALGORITHMIC_MEDIA
    )
    print("[4b] 无参考图时降为 trainedAlgorithmicMedia")

    with tempfile.TemporaryDirectory(prefix="longfilm_compliance_") as td:
        tmp = Path(td)

        # 5) 清单落盘与回读
        mp = m.save(tmp / "manifest.json")
        assert ProvenanceManifest.load(mp).digest() == m.digest()
        print(f"[5] 清单落盘 {mp.stat().st_size} 字节，回读摘要一致")

        # 6) ASS 生成：字号必须按画面高度算出来，对齐必须对
        ass = build_disclosure_ass(
            "本视频由人工智能生成", width=640, height=360, position="top_left",
            font_family="Microsoft YaHei", font_px=18, start_s=0, end_s=3, box=True,
        )
        assert "PlayResY: 360" in ass and "Microsoft YaHei,18," in ass
        assert ass.rstrip().endswith("本视频由人工智能生成")
        assert ",3,4,0,7," in ass, "BorderStyle=3 + 顶部左对齐(7)"
        brace = build_disclosure_ass("a{b}c", width=8, height=8, position="bottom_right",
                                     font_family="F", font_px=4, start_s=0, end_s=1, box=False)
        assert "a（b）c" in brace, "花括号必须被中和，否则 libass 会当成特效标签"
        print("[6] ASS 生成：字号/对齐/花括号转义均正确")

        # 7) 真渲染：烧录显式标识，并用平均亮度证明确实画上去了
        src = tmp / "src.mp4"
        ff.run(["-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=24:d=3",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-c:v", "libx264", "-crf", "20", "-preset", "ultrafast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(src)], timeout=300)
        marked = add_visible_disclosure(src, tmp / "marked.mp4", preset="ultrafast")
        i0, i1 = ff.probe(src), ff.probe(marked)
        assert i1.size == i0.size and abs(i1.duration_s - i0.duration_s) < 0.2
        assert i1.has_audio, "标识烧录不应丢音轨"
        # 底片全黑，所以标识区域的平均亮度就是「字到底画上没有」的直接证据。
        # 只量左上角那块而不是整帧：整帧会被大片黑底稀释到 16.6（limited range 的
        # 黑是 16），和「完全没画」区分不开；分区量还能顺带验证位置对不对。
        def _yavg(crop: str) -> float:
            out = ff.run(["-i", str(marked), "-vf",
                          f"crop={crop},signalstats,metadata=print:key=lavfi.signalstats.YAVG",
                          "-f", "null", "-"], timeout=300)
            return max(float(v) for v in re.findall(r"YAVG=([\d.]+)", out))

        w, h = i1.width, i1.height
        top_left = _yavg(f"{w // 2}:{h // 4}:0:0")
        bottom_right = _yavg(f"{w // 2}:{h // 4}:{w // 2}:{h - h // 4}")
        print(f"[7] 显式标识烧录完成：{w}x{h}，左上角 YAVG={top_left:.1f} "
              f"右下角 YAVG={bottom_right:.1f}（黑底基准 16.0）")
        # 纯黑底实测 21.2（白色字形占该区域约 2.4% 面积）；阈值取 18 留足余量，
        # 但仍远高于「一个字都没画」的 16.0，能区分渲染失败和字体回退成空白。
        assert top_left > 18, f"左上角仍是黑的（YAVG={top_left}），libass 没把字渲染出来"
        assert bottom_right < 17, f"右下角不该有内容（YAVG={bottom_right}），position 没生效"
        assert (tmp / "marked.disclosure.ass").exists(), "ASS 应留档，便于复核标识是否达标"

        # 8) 隐式标识：写进去必须能读回来
        final = write_metadata(marked, tmp / "final.mp4", m)
        tags = read_metadata(final)
        blob = json.loads(tags["comment"])
        print(f"[8] 元数据回读：title={tags.get('title')!r} AIGC={blob['AIGC']}")
        assert blob["AIGC"]["Label"] == "1"
        assert blob["AIGC"]["ProduceID"] == m.produce_id
        assert blob["AIGC"]["ContentProducer"] == "91310000MA1K00000X"
        assert blob["longfilm"]["manifest_sha256"] == m.digest()
        assert "人工智能生成" in tags["description"]
        assert abs(ff.probe(final).duration_s - i1.duration_s) < 0.05, "-c copy 不应改变时长"

        # 8b) 路径含空格/中文/单引号：滤镜参数转义没写对的话 ffmpeg 会直接报
        #     "Unable to parse option value"，这是拼滤镜链最常踩的坑
        odd = tmp / "有 空格 的 目录"
        odd.mkdir()
        odd_src = odd / "源 片's 素材.mp4"
        ff.run(["-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:r=24:d=2",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                str(odd_src)], timeout=300)
        odd_out = add_visible_disclosure(odd_src, odd / "输 出's.mp4", preset="ultrafast")
        assert ff.probe(odd_out).size == (320, 180)
        print(f"[8b] 含空格/中文/单引号的路径烧录成功：{odd_out.name}")

        # 8c) 探不到时长时标识必须常驻，而不是 0 秒后消失（渲染不报错但等于漏标）
        span = build_disclosure_ass("x", width=320, height=180, position="top_left",
                                    font_family="F", font_px=9, start_s=0.0,
                                    end_s=359_999.0, box=True)
        assert "9:59:59.00" in span, span.splitlines()[-2]

        # 9) 缺字体时必须明确报错而不是渲染成方框
        try:
            add_visible_disclosure(src, tmp / "x.mp4", font=tmp / "nope.ttc")
            raise AssertionError("字体不存在时应抛 FileNotFoundError")
        except FileNotFoundError as e:
            assert "字体不存在" in str(e)
        print("[9] 缺字体时拒绝渲染，不静默回退")

    print("compliance 自测全部通过")


if __name__ == "__main__":
    _selftest()
