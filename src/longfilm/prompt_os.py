"""提示词操作系统 —— 分镜 JSON → 提示词的**确定性**编译器。

核心要求只有一条：同一个 Shot 编译两次必须逐字相同。
理由是整条产线的复现能力都挂在这上面 ——
`Shot.fingerprint()` 用来命中缓存、跳过已生成的镜；A/B 实验靠「只改一个变量」
才能归因；线上出了坏镜要能用同一份 JSON 复现出同一条提示词才谈得上排查。
只要编译过程里混进一个集合迭代序、一次 `random`、一个时间戳，这三件事同时失效。
所以本模块：不读时钟、不用 set 参与输出排序、不做任何「随机同义词替换」。

分层拼装顺序固定为
    [镜头语法] → [参考位绑定] → [主体身份] → [动作] → [环境] → [光线] → [风格圣经] → [画质词]

为什么身份要靠前（这是全模块最关键的一个决定）：
1. 文本编码器有硬性长度上限（T5/CLIP 一类都是定长），**超出部分从尾部直接截断**。
   放在前面的层永远不会被截掉，放在最后的画质词丢了只是糙一点，身份丢了整镜作废。
2. 扩散采样的早期步决定主体结构与布局，晚期步才补细节；靠前的条件词在早期步
   影响更大，正好对应「先把人定住」。
3. 身份是唯一后期修不回来的属性 —— 光可以调色，抖动可以插帧，脸换了只能重抽。
镜头语法排在身份之前，是因为景别决定主体在画幅里占多大比例：它是身份描述的
「容器」。容器没定，模型会先自己猜一个构图，再把身份词往里塞，主体比例就飘。

外部查证（2026-09-19）：
- `Emily2040/seedance-2.0` **确实存在**（https://github.com/Emily2040/seedance-2.0，
  7.3k stars，默认分支 main，最后推送 2026-09-08，描述 "Comprehensive production
  pipeline for quad-modal AI filmmaking with Seedance 2.0"）。其
  references/prompt-compiler.md 明确要求：最终提示词保持可读散文，**不要**把内部
  JSON 标签（power shift / hidden want / subtext 之类）泄漏进提示词；
  references/reference-workflow.md 给出 `@Image1..9 / @Video1..3 / @Audio1..3` 的
  标签绑定语法与「transfer / ignore」写法。本模块的 seedance 方言据此设计：
  发短句散文 + 逐字标签，不发结构化标签。
- Wan 系提示词风格：官方与社区一致的说法是「用自然语言、长而详细的描述效果最好」，
  I2V 时重点写运动/相机/场景演化而不是复述图里已有的静态信息
  （Wan-AI/Wan2.2-I2V-A14B 讨论区、huggingface.co/blog/MonsterMMORPG/
  how-to-prompt-wan-models-full-tutorial-and-guide，查证日期 2026-09-19）。
  故 wan 方言拼成一条连贯长句。
- TODO(2026-09-19)：各闭源 API 是否提供独立 negative_prompt 字段、上限多少 token，
  未逐家查证。本模块始终单独产出 `negative`，由 router 按
  `Capabilities.supports_negative_prompt` 决定是下发还是丢弃，不在这里猜。

内容策略：本模块只做虚构成年角色。`ContentRating.BLOCKED` 的镜直接拒编，
不提供任何改写以绕过平台审核的路径。
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from . import refpack as _refpack
from .schema import (
    CameraMove,
    ContentRating,
    Scene,
    Shot,
    ShotSize,
    Storyboard,
)

logger = logging.getLogger(__name__)

Dialect = str   # "wan" | "seedance" | "generic"


# ---------------------------------------------------------------- 负面词库

# 按**失败模式**分组，而不是按词性分组。分组是为了能按镜关闭某一组：
# 比如刻意要做旧胶片颗粒的镜，就得关掉 compression 组，否则 "grain" 被压掉。
NEGATIVE_BANK: dict[str, str] = {
    # 身份漂移：同一个人在片内变脸/变年龄/变人数。长片最致命的一类。
    "identity_drift": (
        "face morphing between frames, inconsistent facial features, changing hairstyle, "
        "changing eye color, different person, face swap artifacts, age shift, "
        "duplicate character, twin of the same person, inconsistent wardrobe"
    ),
    # 解剖崩坏：手指/肢体/关节。生成模型的结构性顽疾，单帧就能看出来。
    "anatomy": (
        "extra fingers, missing fingers, fused fingers, malformed hands, extra limbs, "
        "extra arms, broken joints, twisted torso, disconnected limbs, distorted anatomy, "
        "asymmetric eyes, deformed teeth"
    ),
    # 时序闪烁：单帧都对，连起来在抖。只有视频模型才有，图像负面词库里通常没有。
    "temporal": (
        "flickering, temporal instability, strobing, jittery motion, frame popping, "
        "texture crawling, morphing background, sliding feet, foot sliding, "
        "inconsistent lighting between frames, warping geometry"
    ),
    # 水印文字：模型从训练数据里学来的 logo/字幕/署名，以及乱码文字。
    # 本产线的字幕一律后期用 libass 烧，画面里出现任何文字都是缺陷。
    "watermark_text": (
        "watermark, signature, logo, text overlay, subtitles, caption, timestamp, "
        "gibberish text, garbled letters, chinese characters on screen, ui elements, "
        "stock photo watermark"
    ),
    # 压缩/退化：低码率块效应、过锐、糊。影响交付画质，不影响一致性，最可选。
    "compression_artifact": (
        "blocky compression artifacts, banding, macroblocking, oversharpened halos, "
        "heavy denoise smearing, plastic skin, blurry, low resolution, jpeg artifacts"
    ),
}

# 每组针对什么失败模式、为什么值得单开一组。与 NEGATIVE_BANK 并列而不是塞进
# 同一个 dict，是因为 NEGATIVE_BANK 要能直接 join 进提示词。
NEGATIVE_RATIONALE: dict[str, str] = {
    "identity_drift": "镜内/跨镜变脸、变发型、人数增殖。长片一致性的头号杀手，后期无法修复。",
    "anatomy": "手指与肢体结构崩坏。单帧可见，观众容错率最低，QC 里属于直接打回。",
    "temporal": "逐帧都合理但连起来闪烁/抖动/脚底打滑。视频专有，图像负面词库覆盖不到。",
    "watermark_text": "模型幻觉出的水印、logo、乱码字幕。本产线字幕走后期 libass，画面内文字一律视为缺陷。",
    "compression_artifact": "块效应、过锐、塑料皮肤等画质退化。可选组：刻意做旧胶片颗粒时要关掉。",
}

DEFAULT_NEGATIVE_GROUPS: tuple[str, ...] = (
    "identity_drift",
    "anatomy",
    "temporal",
    "watermark_text",
    "compression_artifact",
)


# ---------------------------------------------------------------- 方言适配层


@dataclass(frozen=True)
class PromptDialect:
    """一种目标模型家族的提示词口味。

    把差异集中成数据，而不是散在 compile_prompt 里的 if：新增一种引擎口味时
    只加一条 DIALECTS 记录，编译主流程一行不用改。这也是提示词能做 A/B 的前提
    —— 「换方言」必须是一个可枚举的变量，而不是一堆分支的组合。
    """

    name: str
    layer_order: tuple[str, ...]
    joiner: str                       # 层与层之间的连接符
    quality_terms: tuple[str, ...]
    camera_style: str                 # "prose" = 自然语言描述；"terms" = 术语短语
    ref_tags: bool                    # 是否在提示词里写 @ImageN 绑定指令
    ref_lang: str = "en"
    max_chars: int = 1800
    tail: str = "."


DIALECTS: dict[str, PromptDialect] = {
    # 开源 Wan 系：吃长句自然语言，官方样例普遍 30-150 词的连贯描述。
    # 参考图通过 pipeline 参数传入而不是提示词内的 @ 标签，故 ref_tags=False。
    "wan": PromptDialect(
        name="wan",
        layer_order=("camera", "identity", "action", "environment", "light", "style", "quality"),
        joiner=", ",
        quality_terms=(
            "cinematic film still",
            "natural film grain",
            "physically plausible motion",
            "consistent identity across all frames",
            "high detail",
        ),
        camera_style="prose",
        ref_tags=False,
        max_chars=2200,
    ),
    # 闭源 Seedance 类：吃结构化的镜头术语 + 逐字参考位标签。
    # 短句分号/句号断开，每句只讲一件事，便于平台侧解析与人工 diff。
    "seedance": PromptDialect(
        name="seedance",
        layer_order=(
            "camera", "refs", "identity", "action", "environment", "light", "style", "quality",
        ),
        joiner=". ",
        quality_terms=("cinematic grade", "high detail", "natural motion blur"),
        camera_style="terms",
        ref_tags=True,
        ref_lang="en",
        max_chars=1500,
    ),
    # 兜底：新供应商还没摸清口味时用，取两者中间值。
    "generic": PromptDialect(
        name="generic",
        layer_order=("camera", "identity", "action", "environment", "light", "style", "quality"),
        joiner=", ",
        quality_terms=("cinematic", "high detail", "consistent identity"),
        camera_style="terms",
        ref_tags=False,
        max_chars=1200,
    ),
}


# ---------------------------------------------------------------- 输出类型


@dataclass(frozen=True)
class CompiledPrompt:
    """一次编译的全部产物。冻结是为了防止调用方就地改字符串后再去算指纹。"""

    positive: str
    negative: str
    camera_directive: str
    audio_directive: str
    token_estimate: int
    dialect: str
    layers: tuple[tuple[str, str], ...] = field(default=())
    truncated: bool = False

    def fingerprint(self) -> str:
        """提示词指纹。A/B 实验记录里存它，比存全文便宜且能直接比对。"""
        blob = "\n".join(
            (self.dialect, self.positive, self.negative,
             self.camera_directive, self.audio_directive)
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def layer(self, name: str) -> str:
        return next((v for k, v in self.layers if k == name), "")


# ---------------------------------------------------------------- 镜头语法

# 焦段 → 透视语言。写成区间表而不是 if 链，是为了让「换一套镜头语言」变成改数据。
_LENS_BANDS: tuple[tuple[int, str], ...] = (
    (20, "ultra wide-angle, exaggerated perspective"),
    (28, "wide-angle, expansive spatial depth"),
    (50, "normal lens, natural perspective"),
    (85, "short telephoto, mild compression"),
    (10_000, "telephoto, strongly compressed perspective, isolated subject"),
)

# 光圈 → 景深语言。阈值取常规电影镜头习惯：T2 以下算浅景深，f/8 以上算全景深。
_APERTURE_BANDS: tuple[tuple[float, str], ...] = (
    (2.0, "very shallow depth of field, creamy bokeh"),
    (4.0, "shallow depth of field"),
    (8.0, "moderate depth of field"),
    (1e9, "deep focus, everything sharp"),
)

_APERTURE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _aperture_value(aperture: str) -> float:
    """从 'f/2.8' / 'T1.5' 里取数值。取不到按 2.8 处理 —— 这是最常见的默认光圈，
    猜错的代价只是景深描述略偏，不值得为此抛异常打断整条产线。"""
    m = _APERTURE_RE.search(aperture)
    return float(m.group(1)) if m else 2.8


def camera_syntax(shot: Shot, *, style: str = "terms") -> str:
    """把 ShotSize / CameraMove / lens_mm / aperture 编译成镜头语法短语。

    顺序是「景别 → 焦段透视 → 景深 → 运镜」：从最外层的画幅约束往里收。
    这个顺序与摄影师报机位的口头顺序一致，模型的训练语料里也是这个序。
    """
    lens_word = next(w for mm, w in _LENS_BANDS if shot.lens_mm <= mm)
    dof_word = next(w for a, w in _APERTURE_BANDS if _aperture_value(shot.aperture) <= a)
    bits = [
        shot.shot_size.value,
        f"{shot.lens_mm}mm {lens_word}",
        f"{shot.aperture} {dof_word}",
        shot.camera_move.value,
    ]
    if style == "prose":
        return f"A {bits[0]} captured on {bits[1]}, {bits[2]}, {bits[3]}"
    return ", ".join(bits)


# ---------------------------------------------------------------- 分层构造


def _scene_of(shot: Shot, sb: Storyboard) -> Scene | None:
    return next((sc for sc in sb.scenes if sc.id == shot.scene_id), None)


def _identity_layer(shot: Shot, sb: Storyboard) -> str:
    """主体身份层。多角色时用 ' and ' 连接并保留 subject_ids 的书写顺序 ——
    顺序即主次，模型对先出现的主体分配的注意力更多。"""
    bits: list[str] = []
    for sid in shot.subject_ids:
        char = sb.character(sid)
        if char is None:
            logger.warning("%s: 角色 %s 不在角色圣经里，身份层缺锚", shot.id, sid)
            bits.append(sid)
            continue
        desc = char.identity_prompt()
        bits.append(f"{char.name}（{desc}）" if desc else char.name)
    return " and ".join(bits)


def _environment_layer(shot: Shot, sb: Storyboard) -> str:
    """环境层。镜头自述优先，缺了才回落到场景的 location/time_of_day ——
    分镜作者在镜上写的环境是针对这一镜的，比场景级描述精确。"""
    if shot.environment:
        return shot.environment
    sc = _scene_of(shot, sb)
    if sc is None:
        return ""
    return ", ".join(p for p in (sc.location, sc.time_of_day) if p)


def _light_layer(shot: Shot) -> str:
    return ", ".join(p for p in (shot.lighting, shot.mood) if p)


def _style_layer(shot: Shot, sb: Storyboard) -> str:
    """风格层。镜级 style 与全片 style_bible 叠加而不是二选一：
    style_bible 保证全片统一，镜级 style 做局部偏移（比如回忆段落）。"""
    return ", ".join(p for p in (sb.style_bible, shot.style) if p)


def _refs_layer(shot: Shot, dialect: PromptDialect) -> str:
    """参考位绑定层。直接复用 refpack.binding_directives —— 标签编号必须与
    实际提交给 API 的参考位顺序严格一致，两处各算一次迟早错位。"""
    if not dialect.ref_tags:
        return ""
    return " ".join(_refpack.binding_directives(shot.refs, lang=dialect.ref_lang))


def _dialogue_text(shot: Shot, sb: Storyboard) -> str:
    bits: list[str] = []
    for line in shot.dialogue:
        char = sb.character(line.speaker_id)
        who = char.name if char else line.speaker_id
        bits.append(f"{who}（{line.emotion}）：“{line.text}”")
    return " ".join(bits)


def audio_directive(shot: Shot, sb: Storyboard) -> str:
    """音频指令。始终产出完整音频计划（对白 + 音色 + 音效 + 响度），
    因为它同时喂给 TTS/Foley 侧，而不只是喂给自带声音的视频模型。"""
    bits: list[str] = []
    if shot.dialogue:
        bits.append("对白：" + _dialogue_text(shot, sb))
        voices = []
        for line in shot.dialogue:
            char = sb.character(line.speaker_id)
            if char is not None:
                voices.append(
                    f"{char.name}={char.voice.tts_engine}"
                    f"/{char.voice.tts_voice_id or 'default'}"
                    f"@{char.voice.language}"
                )
        # dict 去重保序：同一角色在一镜里说两句不该出现两次音色声明
        uniq = list(dict.fromkeys(voices))
        if uniq:
            bits.append("音色：" + "，".join(uniq))
    if shot.sfx:
        bits.append("音效：" + "，".join(shot.sfx))
    bits.append(f"目标响度：{sb.audio.loudness_lufs} LUFS")
    if sb.audio.audio_first:
        bits.append("音频先行：对白轨时长为画面时长基准")
    return "；".join(bits)


def compile_negative(
    shot: Shot,
    sb: Storyboard,
    *,
    groups: Sequence[str] = DEFAULT_NEGATIVE_GROUPS,
) -> str:
    """负面提示词 = 词库分组 + 全片负面 + 角色负面 + 镜级负面。

    叠加顺序从通用到具体，去重保序。去重是因为重复词在部分实现里会被当成
    加权（重复 = 提权），无意中重复会让某一类负面压过头，画面发僵。
    """
    chunks = [NEGATIVE_BANK[g] for g in groups if g in NEGATIVE_BANK]
    unknown = [g for g in groups if g not in NEGATIVE_BANK]
    if unknown:
        logger.warning("未知负面词组 %s，已忽略；可用分组 %s", unknown, sorted(NEGATIVE_BANK))
    if sb.global_negative:
        chunks.append(sb.global_negative)
    for sid in shot.subject_ids:
        char = sb.character(sid)
        if char is not None and char.negative_prompt:
            chunks.append(char.negative_prompt)
    if shot.negative_prompt:
        chunks.append(shot.negative_prompt)

    terms: list[str] = []
    for chunk in chunks:
        for t in chunk.split(","):
            t = t.strip()
            if t:
                terms.append(t)
    return ", ".join(dict.fromkeys(terms))


# ---------------------------------------------------------------- token 估算


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 每字约 1 token，英文每词约 1.3 token，标点各算 1。

    这里要的不是准确值而是**确定性上界感知** —— 用途是在超出方言长度预算时
    提前告警，避免文本编码器从尾部截断。真正的 tokenizer 各家不同且要装依赖，
    为一个告警阈值引入一整个 tokenizer 包不划算。
    """
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    words = len(_WORD_RE.findall(text))
    punct = sum(1 for ch in text if ch in ",.;:()/-—、，。；：“”！？")
    return cjk + int(words * 1.3) + punct


# ---------------------------------------------------------------- 主编译


def compile_prompt(
    shot: Shot,
    storyboard: Storyboard,
    *,
    dialect: Dialect = "wan",
    include_dialogue: bool = False,
    negative_groups: Sequence[str] = DEFAULT_NEGATIVE_GROUPS,
) -> CompiledPrompt:
    """把一个 Shot 编译成提示词。同输入必然同输出（逐字节）。

    `include_dialogue` 只控制**台词是否进画面提示词**：自带原生声音的模型需要
    看到台词才能对口型，不带声音的模型看到台词只会在画面里幻觉出字幕。
    无论取值如何，`audio_directive` 里始终有完整对白计划，因为 TTS 侧要用。
    """
    if shot.content_rating is ContentRating.BLOCKED:
        raise ValueError(
            f"{shot.id}: content_rating=blocked，产线拒单，不编译提示词"
        )
    if dialect not in DIALECTS:
        raise ValueError(f"未知方言 {dialect!r}；可用 {sorted(DIALECTS)}")
    d = DIALECTS[dialect]

    built: dict[str, str] = {
        "camera": camera_syntax(shot, style=d.camera_style),
        "refs": _refs_layer(shot, d),
        "identity": _identity_layer(shot, storyboard),
        "action": shot.action,
        "environment": _environment_layer(shot, storyboard),
        "light": _light_layer(shot),
        "style": _style_layer(shot, storyboard),
        "quality": ", ".join(d.quality_terms),
    }
    if include_dialogue and shot.dialogue:
        # 台词紧跟动作层：说话本身就是这一镜的动作，分开会让口型与动作脱节。
        built["dialogue"] = "角色说话并对口型，台词：" + _dialogue_text(shot, storyboard)

    order = list(d.layer_order)
    if "dialogue" in built:
        order.insert(order.index("action") + 1, "dialogue")

    # 逐层剥掉行尾标点再拼：方言的 joiner 自己负责断句，层内带的句号会和 joiner
    # 叠成 "。. " 这种脏尾巴，而脏尾巴在 A/B 的字节比对里是纯噪声。
    layers = tuple((k, built[k].rstrip(" ,;.，；。")) for k in order if built.get(k).strip())
    positive = d.joiner.join(v for _, v in layers)
    if d.tail and not positive.endswith(d.tail):
        positive += d.tail

    truncated = len(positive) > d.max_chars
    if truncated:
        # 不在这里截断：截断等于悄悄丢层，正是本模块要避免的事。
        # 只告警，让调用方决定是精简分镜还是换一个上下文更长的引擎。
        logger.warning(
            "%s: %s 方言提示词 %d 字符，超过建议上限 %d，尾部（画质词/风格层）有被编码器截断的风险",
            shot.id, d.name, len(positive), d.max_chars,
        )

    return CompiledPrompt(
        positive=positive,
        negative=compile_negative(shot, storyboard, groups=negative_groups),
        camera_directive=camera_syntax(shot, style="terms"),
        audio_directive=audio_directive(shot, storyboard),
        token_estimate=estimate_tokens(positive),
        dialect=d.name,
        layers=layers,
        truncated=truncated,
    )


# ---------------------------------------------------------------- A/B 差异


def _ab_lines(p: CompiledPrompt, tag: str) -> list[str]:
    lines = [f"dialect: {p.dialect}", f"tokens~: {p.token_estimate}",
             f"fingerprint: {p.fingerprint()}"]
    lines += [f"[{k}] {v}" for k, v in p.layers]
    lines.append(f"[negative] {p.negative}")
    lines.append(f"[camera] {p.camera_directive}")
    lines.append(f"[audio] {p.audio_directive}")
    return [f"{ln}\n" for ln in lines]


def diff_prompts(a: CompiledPrompt, b: CompiledPrompt, *,
                 label_a: str = "A", label_b: str = "B") -> str:
    """两条提示词的差异报告，A/B 实验归因用。

    按**层**做 diff 而不是按字符：A/B 的唯一有用结论是「改了哪一层导致的差异」，
    字符级 diff 在长句方言下会把整段标成一行改动，等于没说。
    """
    diff = difflib.unified_diff(
        _ab_lines(a, label_a), _ab_lines(b, label_b),
        fromfile=label_a, tofile=label_b, n=0,
    )
    body = "".join(diff).rstrip()
    return body or f"{label_a} 与 {label_b} 完全相同（fingerprint {a.fingerprint()}）"


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    from .schema import (
        Appearance,
        CharacterBible,
        Continuity,
        DialogueLine,
        ImageRef,
        LoRASpec,
        RefPack,
        Storyboard as SB,
        VoiceProfile,
    )

    lin = CharacterBible(
        id="lin", name="LIN YUE", age_statement="虚构角色，设定年龄 27 岁",
        appearance=Appearance(face="oval face, sharp jawline", hair="short black hair",
                              distinguishing="scar over left eyebrow"),
        voice=VoiceProfile(tts_voice_id="v_lin", timbre_ref_uri="s3://voice/lin.wav"),
        portraits=[ImageRef(role="identity", uri="s3://bible/lin/front.png", note="正脸 face")],
        lora=LoRASpec(base_model="wan2.2-i2v-a14b", path="/m/lin.safetensors",
                      trigger_word="linyue_v3"),
        negative_prompt="beard",
    )
    qi = CharacterBible(
        id="qi", name="QI CHUAN", age_statement="虚构角色，设定年龄 34 岁",
        appearance=Appearance(face="square face", hair="grey temples"),
        portraits=[ImageRef(role="identity", uri="s3://bible/qi/front.png", note="正脸 face")],
    )

    shot = Shot(
        id="sh02", scene_id="sc01", index=1, duration_s=8.0,
        shot_size=ShotSize.MCU, camera_move=CameraMove.DOLLY_IN, lens_mm=85, aperture="f/1.8",
        subject_ids=["lin", "qi"],
        action="lin turns from the rain-streaked window and speaks without looking at qi",
        environment="a cramped noodle shop, steam on the glass",
        lighting="single practical overhead, hard falloff", mood="tense, held breath",
        style="muted teal shadows",
        dialogue=[DialogueLine(speaker_id="lin", text="你早该走了。", emotion="cold")],
        sfx=["rain on tin roof", "boiling water"],
        continuity=Continuity(prev_shot_id="sh01", inherit_last_frame=True),
    )
    sb = SB(project="t", characters=[lin, qi],
            scenes=[Scene(id="sc01", location="noodle shop", time_of_day="night",
                          shots=[shot])],
            style_bible="neo-noir, anamorphic flare, 1990s Hong Kong palette",
            global_negative="modern smartphones")

    # 1) 幂等性：同一 Shot 编译两次必须字节相同
    a1 = compile_prompt(shot, sb, dialect="wan")
    a2 = compile_prompt(shot, sb, dialect="wan")
    assert a1 == a2, "wan 方言不幂等"
    assert a1.positive.encode() == a2.positive.encode()
    assert a1.fingerprint() == a2.fingerprint()

    # 2) 分层顺序：镜头语法在最前，身份在动作之前
    names = [k for k, _ in a1.layers]
    assert names[0] == "camera", names
    assert names.index("identity") < names.index("action") < names.index("quality"), names
    assert "linyue_v3" in a1.positive, "LoRA 触发词必须进身份层"
    assert "LIN YUE" in a1.positive and "QI CHUAN" in a1.positive

    # 3) 方言差异：seedance 带 @ 标签，wan 不带
    shot.refs = _refpack.build_refpack(
        shot, sb, prev_last_frame_uri="s3://out/sh01_last.png"
    )
    s1 = compile_prompt(shot, sb, dialect="seedance")
    assert "@Image1" in s1.positive and "只控制" in s1.positive
    assert "@Image" not in compile_prompt(shot, sb, dialect="wan").positive
    assert "refs" in [k for k, _ in s1.layers]
    assert compile_prompt(shot, sb, dialect="seedance") == s1   # 方言也要幂等

    # 4) 镜头语法
    cs = camera_syntax(shot)
    assert "medium close-up" in cs and "85mm" in cs and "f/1.8" in cs
    assert "short telephoto" in cs and "very shallow depth of field" in cs
    assert "camera dollies in" in cs
    wide = camera_syntax(shot.model_copy(update={"lens_mm": 18, "aperture": "f/11"}))
    assert "ultra wide-angle" in wide and "deep focus" in wide

    # 5) 台词开关
    no_d = compile_prompt(shot, sb, dialect="wan", include_dialogue=False)
    with_d = compile_prompt(shot, sb, dialect="wan", include_dialogue=True)
    assert "你早该走了。" not in no_d.positive
    assert "你早该走了。" in with_d.positive
    # audio_directive 与开关无关，TTS 侧永远拿得到完整计划
    assert "你早该走了。" in no_d.audio_directive == with_d.audio_directive
    assert "indextts2/v_lin@zh" in no_d.audio_directive
    assert "rain on tin roof" in no_d.audio_directive and "-16.0 LUFS" in no_d.audio_directive

    # 6) 负面词库
    assert "flickering" in a1.negative and "extra fingers" in a1.negative
    assert "watermark" in a1.negative and "face morphing between frames" in a1.negative
    assert "modern smartphones" in a1.negative and "beard" in a1.negative
    neg_terms = [t.strip() for t in a1.negative.split(",")]
    assert len(neg_terms) == len(set(neg_terms)), "负面词去重失效（重复=隐式提权）"
    grainy = compile_prompt(shot, sb, negative_groups=("identity_drift", "temporal"))
    assert "jpeg artifacts" not in grainy.negative
    assert set(NEGATIVE_RATIONALE) == set(NEGATIVE_BANK)

    # 7) A/B 差异报告
    rep = diff_prompts(a1, s1, label_a="wan", label_b="seedance")
    assert "dialect" in rep and "+" in rep
    assert "完全相同" in diff_prompts(a1, a2)

    # 8) 内容分级：blocked 直接拒编
    blocked = shot.model_copy(update={"content_rating": ContentRating.BLOCKED})
    try:
        compile_prompt(blocked, sb)
        raise AssertionError("blocked 镜头必须拒编")
    except ValueError as e:
        assert "拒单" in str(e)

    assert a1.token_estimate > 0 and not a1.truncated

    print("=== wan positive ===")
    print(a1.positive)
    print("\n=== seedance positive ===")
    print(s1.positive)
    print("\n=== negative ===")
    print(a1.negative)
    print("\n=== camera / audio ===")
    print(a1.camera_directive)
    print(a1.audio_directive)
    print(f"\ntokens~{a1.token_estimate}  fp={a1.fingerprint()}")
    print("\n=== A/B diff (wan vs seedance) ===")
    print(rep[:900])
    print("\nprompt_os 自测通过")


if __name__ == "__main__":
    _selftest()
