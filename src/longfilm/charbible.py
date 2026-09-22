"""角色圣经构建 —— 把「一句话角色描述」变成可投产的身份锚资产。

跨镜身份漂移是长片 AI 产线最贵的缺陷：调色能救光，插帧能救抖动，
**没有任何后期能把 A 的脸换回 B**，只能整镜重抽。所以防漂移必须发生在
拍片之前，也就是这一层：

1. **定妆图是矩阵，不是几张好看的头像。** 视角 × 光位 × 表情三个维度，
   每一维都对应一类模型必须外推的场景。缺哪一维，模型就在那一维上自己编，
   编出来的就是另一张脸。`CharacterBibleBuilder.shot_list()` 给出最小覆盖集
   和每个组合的存在理由，`matrix_rationale()` 解释为什么是这些组合、最少几张。

2. **特征锚点要有可判定的「够不够独特」标准。** 「清秀的年轻女性」在模型的
   隐空间里指向一大片人，等于没有约束；「左眉尾一道 2cm 弧形旧疤」只指向很小
   一片。`validate_bible()` 把这件事做成可执行判据（位置限定词 + 不可移除体征
   的数量下限），而不是留给人「感觉一下够不够」。

3. **锚点有优先级。** 提示词长度有限、参考位只有 9 个，什么先进什么后进要有序。
   `consistency_anchors()` 按「不可移除 > 结构 > 颜色 > 服装」排出锚点词，
   供 prompt_os 优先拼入。

本模块只做**虚构成年角色**。`validate_bible()` 复用
`compliance._age_problem()` 的成年声明判据，不另造一套平行实现 ——
两处判据不一致时，发片门禁和角色构建会互相打架。
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from .schema import (
    Appearance,
    CharacterBible,
    ImageRef,
    LoRASpec,
    VideoRef,
    VoiceProfile,
)

logger = logging.getLogger(__name__)


# ================================================================= 矩阵维度

class View(str, Enum):
    """拍摄视角。值是直接拼进文生图提示词的英文短语。"""

    FRONT = "frontal view, facing camera directly"
    TQ_RIGHT = "three-quarter view turned to their right, 45 degrees"
    TQ_LEFT = "three-quarter view turned to their left, 45 degrees"
    PROFILE = "full profile view, 90 degrees to camera"
    BACK_TQ = "rear three-quarter view, 135 degrees, face partially visible"
    BACK = "back view, 180 degrees, head and shoulders from behind"
    LOW_ANGLE = "low angle looking up at the subject"
    HIGH_ANGLE = "high angle looking down at the subject"


class Light(str, Enum):
    """光位。决定模型学到「这张脸的骨骼在光下怎么投影」。"""

    FLAT = "flat even frontal lighting, soft box, minimal shadow"
    KEY_LEFT = "key light from camera left at 45 degrees, soft fill on the right"
    KEY_RIGHT = "key light from camera right at 45 degrees, soft fill on the left"
    RIM = "rim light from behind, subject edge-lit against a dark background"
    TOP = "top light from directly above, shadows under the brow and nose"
    LOW_KEY = "low key lighting, single hard source, deep shadows, high contrast"
    PRACTICAL = "lit by a single warm practical lamp in frame"
    BACKLIT = "strong backlight, subject partly in silhouette"


class Expression(str, Enum):
    NEUTRAL = "neutral relaxed expression, eyes open, mouth closed"
    SLIGHT_SMILE = "slight closed-mouth smile, eyes softly creased"
    CONCERN = "concerned expression, brows drawn together, lips pressed"
    ANGER = "controlled anger, jaw set, eyes hard"
    SURPRISE = "surprise, eyes widened, brows raised, mouth slightly open"
    EYES_CLOSED = "eyes closed, calm face"
    SPEAKING = "mid-speech, mouth open on a vowel, natural asymmetry"


class Framing(str, Enum):
    HEADSHOT = "tight head and shoulders, head fills the frame"
    BUST = "bust framing, head to mid-chest"
    FULL_BODY = "full body, head to feet, standing"
    EYES = "extreme close-up of the eyes and brow only"


# 每张定妆图的用途分级。裁预算时从 OPTIONAL 开始砍。
class Tier(str, Enum):
    CORE = "core"                 # 最小覆盖集，少一张就有一类镜头会漂
    RECOMMENDED = "recommended"   # 覆盖常见但非必然出现的机位/光位
    OPTIONAL = "optional"         # 只在训 LoRA 或特殊镜型时需要


@dataclass(frozen=True)
class PortraitSpec:
    """定妆清单里的一格：一个 视角×光位×表情×景别 组合 + 它为什么必须存在。"""

    view: View
    light: Light
    expression: Expression
    framing: Framing
    tier: Tier
    role: str                     # ImageRef 的 role：identity / wardrobe
    purpose: str                  # 为什么需要这一格 —— 缺了会在哪类镜头上出问题
    turnaround: bool = False      # 归入 CharacterBible.turnaround 而非 portraits

    @property
    def key(self) -> str:
        """矩阵坐标。同时是 ImageRef.note 里的机器可读标签和 seed 的输入。"""
        return (f"view={self.view.name};light={self.light.name};"
                f"expr={self.expression.name};frame={self.framing.name}")

    def label(self) -> str:
        return f"{_CN_VIEW[self.view]} · {_CN_LIGHT[self.light]} · " \
               f"{_CN_EXPR[self.expression]} · {_CN_FRAME[self.framing]}"


_CN_VIEW = {
    View.FRONT: "正面", View.TQ_RIGHT: "右 3/4 侧", View.TQ_LEFT: "左 3/4 侧",
    View.PROFILE: "正侧", View.BACK_TQ: "后 3/4 侧", View.BACK: "背面",
    View.LOW_ANGLE: "仰角", View.HIGH_ANGLE: "俯角",
}
_CN_LIGHT = {
    Light.FLAT: "平光", Light.KEY_LEFT: "主光左", Light.KEY_RIGHT: "主光右",
    Light.RIM: "轮廓光", Light.TOP: "顶光", Light.LOW_KEY: "低调硬光",
    Light.PRACTICAL: "实用光源", Light.BACKLIT: "逆光",
}
_CN_EXPR = {
    Expression.NEUTRAL: "中性", Expression.SLIGHT_SMILE: "微笑",
    Expression.CONCERN: "忧虑", Expression.ANGER: "隐忍怒",
    Expression.SURPRISE: "惊讶", Expression.EYES_CLOSED: "闭眼",
    Expression.SPEAKING: "说话中",
}
_CN_FRAME = {
    Framing.HEADSHOT: "头肩", Framing.BUST: "半身",
    Framing.FULL_BODY: "全身", Framing.EYES: "眼部特写",
}


# ---------------------------------------------------------------- 最小覆盖集

# 为什么是这 16 格：每一格对应一类「模型不得不外推」的情况，
# 外推的地方就是漂移发生的地方。三个维度各自解决一类外推：
#
#   视角 —— i2v 的起幅机位由分镜决定，不由定妆决定。只给正面锚，
#           模型在 3/4 侧和正侧就得自己脑补颧骨与鼻梁的投影关系。
#   光位 —— 平光是「底图」，它定义脸的**材质**；方向光定义脸的**骨骼**。
#           只有平光的角色一进暗场（本产线大量夜景）就会变形。
#   表情 —— CU 情绪镜要求模型做表情形变。没有表情样本，它从训练集里
#           别的脸上借形变，借来的就是别人的脸。
#
# 景别单列一维是因为 ECU 眼部镜和 FS 全身镜吃的是完全不同的细节层级：
# 半身锚在 ECU 上会被放大到糊，全身锚在 ECU 上等于没有信息。
_CORE_MATRIX: tuple[PortraitSpec, ...] = (
    # —— 视角轴（平光中性作底图）
    PortraitSpec(View.FRONT, Light.FLAT, Expression.NEUTRAL, Framing.BUST,
                 Tier.CORE, "identity",
                 "身份底图。所有光效与表情都是在它之上的形变；底图脏了后面全脏，"
                 "refpack 的 identity 锚默认优先取它"),
    PortraitSpec(View.TQ_RIGHT, Light.FLAT, Expression.NEUTRAL, Framing.BUST,
                 Tier.CORE, "identity",
                 "3/4 侧是对话戏最常见的机位（过肩镜的被摄方几乎总是 3/4），"
                 "缺了它所有 OTS 镜都在外推"),
    PortraitSpec(View.PROFILE, Light.FLAT, Expression.NEUTRAL, Framing.BUST,
                 Tier.CORE, "identity",
                 "正侧锁定鼻梁-下颌-后脑的轮廓线，这是正面图完全没有的信息；"
                 "缺了它侧身镜的侧脸会变成另一个人"),
    # —— 光位轴
    PortraitSpec(View.TQ_RIGHT, Light.KEY_LEFT, Expression.NEUTRAL, Framing.BUST,
                 Tier.CORE, "identity",
                 "逆侧主光：光从脸的远侧来，暴露颧骨与鼻影的骨骼关系"),
    PortraitSpec(View.TQ_LEFT, Light.KEY_RIGHT, Expression.NEUTRAL, Framing.BUST,
                 Tier.CORE, "identity",
                 "镜像光位。两张合起来告诉模型这张脸左右并不对称——"
                 "真人脸都不对称，强行对称会立刻显得像 AI"),
    PortraitSpec(View.FRONT, Light.RIM, Expression.NEUTRAL, Framing.BUST,
                 Tier.CORE, "identity",
                 "轮廓光/暗场锚。本产线夜戏比重高，只有平光锚的角色一进暗场就变形"),
    # —— 表情轴
    PortraitSpec(View.FRONT, Light.FLAT, Expression.SLIGHT_SMILE, Framing.HEADSHOT,
                 Tier.CORE, "identity", "正向情绪的形变样本"),
    PortraitSpec(View.FRONT, Light.FLAT, Expression.CONCERN, Framing.HEADSHOT,
                 Tier.CORE, "identity", "负向情绪的形变样本（眉心与唇线）"),
    PortraitSpec(View.FRONT, Light.FLAT, Expression.SPEAKING, Framing.HEADSHOT,
                 Tier.CORE, "identity",
                 "张口样本。音频先行产线里口型由 TTS 驱动，没有张口锚，"
                 "模型会在每次开口时重画牙齿和下颌"),
    # —— 景别轴
    PortraitSpec(View.FRONT, Light.FLAT, Expression.NEUTRAL, Framing.EYES,
                 Tier.CORE, "identity",
                 "ECU/CU 专用。眼型、睫毛、虹膜纹理在半身图上只有几十个像素，"
                 "放到特写镜里就是糊的"),
    # —— turnaround（i2v 起幅机位不定，转身序列是最便宜的全角度覆盖）
    # 0° 这一张同时充当 wardrobe 锚：它和「正面平光中性全身」是同一张照片，
    # 拆成两格只会让摄影拍两次一模一样的图，也会让两个 ImageRef 指向同一个文件。
    PortraitSpec(View.FRONT, Light.FLAT, Expression.NEUTRAL, Framing.FULL_BODY,
                 Tier.CORE, "wardrobe",
                 "转身 0° 兼服装全身锚：身材比例与服装细节的唯一来源，"
                 "wardrobe 跨镜跳变观众一眼看出", turnaround=True),
    PortraitSpec(View.TQ_RIGHT, Light.FLAT, Expression.NEUTRAL, Framing.FULL_BODY,
                 Tier.CORE, "identity", "转身 45°", turnaround=True),
    PortraitSpec(View.PROFILE, Light.FLAT, Expression.NEUTRAL, Framing.FULL_BODY,
                 Tier.CORE, "identity", "转身 90°", turnaround=True),
    PortraitSpec(View.BACK_TQ, Light.FLAT, Expression.NEUTRAL, Framing.FULL_BODY,
                 Tier.CORE, "identity",
                 "转身 135°：背身镜的后脑与发型轮廓", turnaround=True),
    PortraitSpec(View.BACK, Light.FLAT, Expression.NEUTRAL, Framing.FULL_BODY,
                 Tier.CORE, "identity",
                 "转身 180°：走远镜/背影镜，缺了它背影会换个人", turnaround=True),
)

_RECOMMENDED_MATRIX: tuple[PortraitSpec, ...] = (
    PortraitSpec(View.TQ_LEFT, Light.FLAT, Expression.NEUTRAL, Framing.BUST,
                 Tier.RECOMMENDED, "identity", "左 3/4 平光，补齐视角轴的另一半"),
    PortraitSpec(View.FRONT, Light.LOW_KEY, Expression.NEUTRAL, Framing.HEADSHOT,
                 Tier.RECOMMENDED, "identity", "低调硬光：悬疑/夜戏的高对比场景"),
    PortraitSpec(View.FRONT, Light.TOP, Expression.NEUTRAL, Framing.HEADSHOT,
                 Tier.RECOMMENDED, "identity", "顶光：眉弓与鼻下阴影，室内实景常见"),
    PortraitSpec(View.TQ_RIGHT, Light.PRACTICAL, Expression.NEUTRAL, Framing.BUST,
                 Tier.RECOMMENDED, "identity", "实用光源：画内灯具照明的场景"),
    PortraitSpec(View.FRONT, Light.FLAT, Expression.ANGER, Framing.HEADSHOT,
                 Tier.RECOMMENDED, "identity", "强情绪形变样本"),
    PortraitSpec(View.FRONT, Light.FLAT, Expression.SURPRISE, Framing.HEADSHOT,
                 Tier.RECOMMENDED, "identity", "瞬时反应镜的形变样本"),
    PortraitSpec(View.LOW_ANGLE, Light.FLAT, Expression.NEUTRAL, Framing.BUST,
                 Tier.RECOMMENDED, "identity", "仰角：下颌与颈部结构，权力关系镜常用"),
    PortraitSpec(View.HIGH_ANGLE, Light.FLAT, Expression.NEUTRAL, Framing.BUST,
                 Tier.RECOMMENDED, "identity", "俯角：头顶发旋与发际线"),
)

_OPTIONAL_MATRIX: tuple[PortraitSpec, ...] = (
    PortraitSpec(View.PROFILE, Light.BACKLIT, Expression.NEUTRAL, Framing.BUST,
                 Tier.OPTIONAL, "identity", "逆光剪影：只在有剪影镜时需要"),
    PortraitSpec(View.FRONT, Light.FLAT, Expression.EYES_CLOSED, Framing.HEADSHOT,
                 Tier.OPTIONAL, "identity", "闭眼：睡眠/冥想/死亡镜"),
    PortraitSpec(View.TQ_LEFT, Light.KEY_LEFT, Expression.SPEAKING, Framing.BUST,
                 Tier.OPTIONAL, "identity", "侧面张口：侧拍对话镜"),
    PortraitSpec(View.TQ_RIGHT, Light.RIM, Expression.CONCERN, Framing.HEADSHOT,
                 Tier.OPTIONAL, "identity", "轮廓光 + 情绪：夜戏情绪镜"),
    PortraitSpec(View.FRONT, Light.KEY_LEFT, Expression.NEUTRAL, Framing.FULL_BODY,
                 Tier.OPTIONAL, "wardrobe", "带光位的全身：服装材质在方向光下的表现"),
    PortraitSpec(View.TQ_RIGHT, Light.FLAT, Expression.NEUTRAL, Framing.EYES,
                 Tier.OPTIONAL, "identity", "侧向眼部特写"),
    PortraitSpec(View.BACK_TQ, Light.RIM, Expression.NEUTRAL, Framing.BUST,
                 Tier.OPTIONAL, "identity", "背身轮廓光：转身回头镜的前半段"),
    PortraitSpec(View.FRONT, Light.LOW_KEY, Expression.ANGER, Framing.HEADSHOT,
                 Tier.OPTIONAL, "identity", "硬光 + 强情绪：冲突高潮镜"),
)

MATRIX_LEVELS: dict[str, tuple[PortraitSpec, ...]] = {
    "core": _CORE_MATRIX,
    "recommended": _CORE_MATRIX + _RECOMMENDED_MATRIX,
    "lora": _CORE_MATRIX + _RECOMMENDED_MATRIX + _OPTIONAL_MATRIX,
}

MIN_CORE_SHOTS = len(_CORE_MATRIX)


# ================================================================= 构建器

# 文生图的技术后缀。所有定妆图共用，保证「只有矩阵维度在变」。
#
# f/4 而不是 f/1.4：定妆图要的是**可读的特征**，不是好看的虚化。
# 浅景深会把耳廓、发际线、颈侧这些锚点糊掉，而它们恰恰是跨镜最稳的锚。
# 中灰无缝背景：背景一旦有内容，模型会把它当成角色身份的一部分带进每个镜。
PORTRAIT_TECH_SUFFIX = (
    "character reference sheet, neutral mid-grey seamless backdrop, "
    "85mm portrait lens, f/4, even exposure, sharp focus on the eyes, "
    "no motion blur, photographic, full color, high detail skin texture"
)

# 定妆图的通用负面词。每一条都在挡一种「把锚点挡住」的失败：
# 帽子挡发际线、墨镜挡眼型、手挡下半脸、重阴影吞掉骨骼、多人混淆身份。
PORTRAIT_NEGATIVE = (
    "hat, cap, hood, sunglasses, face mask, hand covering face, hair over eyes, "
    "heavy shadow hiding facial features, multiple people, crowd, "
    "extreme wide angle distortion, fisheye, text, watermark, logo, signature, "
    "cropped head, out of frame, blurry, jpeg artifacts, extra fingers"
)

_SLOT_TAG_RE = re.compile(
    r"view=(?P<view>\w+);light=(?P<light>\w+);expr=(?P<expr>\w+);frame=(?P<frame>\w+)"
)


def _slot_seed(character_id: str, key: str) -> int:
    """定妆图的种子必须可复现：补拍一张时要能拿到和原来一致的脸。"""
    h = hashlib.sha256(f"{character_id}|{key}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


class CharacterBibleBuilder:
    """从角色描述构建完整 CharacterBible，并给出定妆图拍摄清单。

    用法：

        b = (CharacterBibleBuilder("shen_mu", "沈牧", age=34)
             .describe(face="…", hair="…", distinguishing="…")
             .with_voice(tts_voice_id="v_shen")
             .with_lora(base_model="wan2.2-i2v-a14b", trigger_word="shenmu_char"))
        print(b.matrix_rationale())
        char = b.build(level="core")

    `build()` 生成的 `portraits` / `turnaround` 里的 `uri` 是**待拍**的目标路径，
    不是已存在的文件。`emit_portrait_prompts()` 按这些路径出图，
    `validate_bible()` 检查覆盖度。这三者共用 `PortraitSpec.key` 做对齐，
    所以拍摄、出图、校验永远不会对不上号。
    """

    def __init__(
        self,
        character_id: str,
        name: str,
        *,
        age: int,
        occupation: str = "",
        persona: str = "",
        asset_root: str = "assets/cast",
        ext: str = "png",
    ) -> None:
        if age < 18:
            # 不是「警告」而是直接拒绝：本产线只做虚构成年角色，
            # 让一个未成年设定先构建出来、再指望下游门禁拦住，是在赌门禁不漏。
            raise ValueError(f"角色 {character_id}: 设定年龄 {age} < 18，本产线只做虚构成年角色")
        self.character_id = character_id
        self.name = name
        self.age = age
        self.occupation = occupation
        self.persona = persona
        self.asset_root = asset_root.rstrip("/")
        self.ext = ext.lstrip(".")
        self._appearance = Appearance()
        self._voice = VoiceProfile()
        self._lora: LoRASpec | None = None
        self._negative = ""
        self._motion_refs: list[VideoRef] = []

    # ---- 描述

    def describe(
        self,
        *,
        face: str = "",
        hair: str = "",
        body: str = "",
        skin: str = "",
        wardrobe: str = "",
        distinguishing: str = "",
    ) -> CharacterBibleBuilder:
        self._appearance = Appearance(
            face=face or self._appearance.face,
            hair=hair or self._appearance.hair,
            body=body or self._appearance.body,
            skin=skin or self._appearance.skin,
            wardrobe=wardrobe or self._appearance.wardrobe,
            distinguishing=distinguishing or self._appearance.distinguishing,
        )
        return self

    def with_voice(self, **kw: object) -> CharacterBibleBuilder:
        self._voice = VoiceProfile.model_validate(
            {**self._voice.model_dump(), **{k: v for k, v in kw.items() if v is not None}}
        )
        return self

    def with_lora(self, *, base_model: str, trigger_word: str,
                  path: str | None = None, strength: float = 0.85) -> CharacterBibleBuilder:
        self._lora = LoRASpec(
            base_model=base_model,
            path=path or f"loras/{self.character_id}_v1.safetensors",
            trigger_word=trigger_word,
            strength=strength,
        )
        return self

    def with_negative(self, negative: str) -> CharacterBibleBuilder:
        self._negative = negative
        return self

    def with_motion_ref(self, uri: str, note: str = "") -> CharacterBibleBuilder:
        self._motion_refs.append(VideoRef(role="motion", uri=uri, weight=0.8, note=note))
        return self

    # ---- 清单

    def shot_list(self, level: str = "core") -> list[PortraitSpec]:
        """定妆图拍摄清单。level ∈ core / recommended / lora。

        `core` 是最小覆盖集；`lora` 用于要训角色 LoRA 的情况
        （张数对齐 `lora.recommend_character_lora()` 的素材要求，两边不要各算各的）。
        """
        if level not in MATRIX_LEVELS:
            raise ValueError(f"未知清单等级 {level!r}，可选 {sorted(MATRIX_LEVELS)}")
        return list(MATRIX_LEVELS[level])

    def age_statement(self) -> str:
        occ = f"，{self.occupation}" if self.occupation else ""
        return f"虚构角色，设定年龄 {self.age} 岁，成年{occ}"

    def uri_for(self, spec: PortraitSpec) -> str:
        """目标文件路径。用矩阵坐标而不是序号命名：

        序号命名（p01.png）在补拍或插入一格之后会整体错位，
        而错位的 identity 锚不会报错，只会让脸悄悄漂 —— 最难查的一类 bug。
        """
        stem = (f"{spec.view.name}_{spec.light.name}_{spec.expression.name}_"
                f"{spec.framing.name}").lower()
        sub = "turn" if spec.turnaround else "portrait"
        return f"{self.asset_root}/{self.character_id}/{sub}/{stem}.{self.ext}"

    def _ref_for(self, spec: PortraitSpec, weight: float) -> ImageRef:
        return ImageRef(
            role=spec.role,
            uri=self.uri_for(spec),
            weight=weight,
            subject_id=self.character_id,
            # note 同时承载人读标签和机器可读的矩阵坐标。不给 schema 新增字段，
            # 是因为 schema 是全产线契约，不能为单个模块改；note 是它留的自由位。
            note=f"{spec.label()} [{spec.key}]",
        )

    @staticmethod
    def _weight_for(spec: PortraitSpec) -> float:
        """参考位权重：底图最重，越外推的组合越轻。

        权重不是「重要性」而是「可信度」：底图是直接观测，
        侧面/暗光/强表情本身就带更多不确定性，给高权重会把不确定性也锁进去。
        """
        if spec.role == "wardrobe":
            return 0.7
        if spec.turnaround:
            return 0.6
        base = 1.0
        if spec.view is not View.FRONT:
            base -= 0.08
        if spec.light is not Light.FLAT:
            base -= 0.10
        if spec.expression is not Expression.NEUTRAL:
            base -= 0.08
        if spec.framing is Framing.EYES:
            base += 0.05          # 特写镜的身份判定几乎全靠眼部
        return round(max(0.5, min(1.0, base)), 2)

    def matrix_rationale(self, level: str = "core") -> str:
        """解释这份清单为什么是这些组合、最少几张。给美术/摄影看的说明书。"""
        specs = self.shot_list(level)
        views = {s.view for s in specs}
        lights = {s.light for s in specs}
        exprs = {s.expression for s in specs}
        frames = {s.framing for s in specs}
        turn = [s for s in specs if s.turnaround]

        L = [
            f"定妆拍摄清单 · {self.name}（{self.character_id}）· 等级 {level}",
            f"  共 {len(specs)} 张 = 定妆 {len(specs)-len(turn)} + 转身序列 {len(turn)}",
            f"  维度覆盖：视角 {len(views)} / 光位 {len(lights)} / "
            f"表情 {len(exprs)} / 景别 {len(frames)}",
            "",
            "为什么是三个维度（每一维都对应一类『模型不得不外推』的情况，",
            "而外推发生的地方就是漂移发生的地方）：",
            "  · 视角 —— i2v 的起幅机位由分镜决定，不由定妆决定。只给正面锚，",
            "            模型在 3/4 侧与正侧要自己脑补颧骨和鼻梁的投影关系。",
            "  · 光位 —— 平光定义脸的**材质**，方向光定义脸的**骨骼**。",
            "            只有平光锚的角色一进夜戏暗场就会变形。",
            "  · 表情 —— 特写情绪镜要求模型做表情形变；没有表情样本，",
            "            它会从训练集里别的脸上借形变，借来的就是别人的脸。",
            "  景别单列是因为 ECU 眼部镜和全身镜吃的细节层级完全不同：",
            "  半身锚放大到特写会糊，全身锚在特写里等于没有信息。",
            "",
            f"最少几张：核心集 {MIN_CORE_SHOTS} 张，构成如下 ——",
            "  3 张 视角底图（正面 / 右 3/4 / 正侧，全部平光中性）",
            "  3 张 光位变体（逆侧主光 ×2 + 轮廓光，用来交代骨骼与暗场表现）",
            "  3 张 表情变体（微笑 / 忧虑 / 张口说话，张口那张给音频先行的口型用）",
            "  1 张 眼部特写（ECU/CU 专用）",
            "  5 张 转身序列 0/45/90/135/180（覆盖任意起幅机位与背影镜；",
            "        其中 0° 那张同时就是服装全身锚，不另拍一张一模一样的图）",
            "  再少就会出现『某类镜头没有锚』，而那类镜头的返工成本远高于补拍一张图。",
            "",
            "逐格清单：",
        ]
        for i, s in enumerate(specs, 1):
            L.append(f"  {i:>2}. [{s.tier.value:<11}] {s.label():<26} "
                     f"w={self._weight_for(s):.2f}  {s.purpose}")
            L.append(f"      → {self.uri_for(s)}")
        return "\n".join(L)

    # ---- 产出

    def build(self, level: str = "core") -> CharacterBible:
        """产出完整 CharacterBible。"""
        specs = self.shot_list(level)
        portraits = [self._ref_for(s, self._weight_for(s)) for s in specs if not s.turnaround]
        turnaround = [self._ref_for(s, self._weight_for(s)) for s in specs if s.turnaround]

        persona = self.persona
        if self.occupation and self.occupation not in persona:
            persona = f"{self.occupation}。{persona}" if persona else self.occupation

        char = CharacterBible(
            id=self.character_id,
            name=self.name,
            age_statement=self.age_statement(),
            persona=persona,
            appearance=self._appearance,
            voice=self._voice,
            portraits=portraits,
            turnaround=turnaround,
            motion_refs=list(self._motion_refs),
            lora=self._lora,
            negative_prompt=self._negative,
        )
        problems = validate_bible(char)
        if problems:
            # 只记日志不抛：构建过程中途检查是正常的（还没填完），
            # 真正的门禁在 validate_bible 的调用方和 compliance_check。
            logger.info("角色 %s 构建完成，但有 %d 条待办：%s",
                        char.id, len(problems), problems[0])
        return char


# ================================================================= 出图工单

@dataclass
class PortraitJob:
    """一条文生图工单。可直接喂给 FLUX / SDXL 的批量脚本。"""

    id: str
    character_id: str
    key: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int
    target_uri: str
    role: str
    tier: Tier
    purpose: str
    label: str
    covered_by: str | None = None     # 已有成品时填其 uri；批量脚本据此跳过

    def render(self) -> str:
        mark = "已有" if self.covered_by else "待拍"
        return (f"[{mark}] {self.id}  {self.label}  {self.width}x{self.height} "
                f"seed={self.seed}\n  → {self.target_uri}\n  {self.prompt}")


# 各景别的出图分辨率。眼部特写用方图：眼睛是正方形构图，
# 用 3:4 会在上下留大量无用像素，而 SDXL/FLUX 的像素预算是固定的。
_FRAMING_SIZE: dict[Framing, tuple[int, int]] = {
    Framing.HEADSHOT: (1024, 1024),
    Framing.BUST: (896, 1152),
    Framing.FULL_BODY: (832, 1216),
    Framing.EYES: (1024, 1024),
}


def _parse_slot_key(note: str) -> str | None:
    m = _SLOT_TAG_RE.search(note or "")
    return m.group(0) if m else None


def emit_portrait_prompts(
    character: CharacterBible,
    *,
    level: str = "core",
    extra_style: str = "",
) -> list[PortraitJob]:
    """为每个 视角/光位/表情 组合出一条文生图提示词。

    提示词分层拼装，层序是固定的（身份 → 视角 → 光位 → 表情 → 景别 → 技术后缀）：
    文生图模型对前置 token 更敏感，身份锚必须在最前面，技术后缀最后 ——
    把 "85mm f/4" 放前面会让模型先满足镜头参数再考虑长得像谁。

    已经有成品的组合（`character.portraits` 里带同一矩阵坐标的 ImageRef）
    会标 `covered_by` 而不是从清单里删掉：清单要保持完整才能一眼看出覆盖率，
    删掉已拍的格子会让「还缺什么」和「一共要什么」混成一个数。
    """
    have: dict[str, str] = {}
    for ref in (*character.portraits, *character.turnaround):
        k = _parse_slot_key(ref.note)
        if k:
            have[k] = ref.uri
    if not have and (character.portraits or character.turnaround):
        # 手写的角色圣经（如 configs/demo_episode.json 里的）note 是自由文本，
        # 无法对齐矩阵坐标。此时不猜，全部按待拍处理并留痕。
        logger.info("角色 %s 的定妆图 note 里没有矩阵坐标，清单按全部待拍输出",
                    character.id)

    identity = character.identity_prompt() or character.name
    negative = ", ".join(x for x in (PORTRAIT_NEGATIVE, character.negative_prompt) if x.strip())

    jobs: list[PortraitJob] = []
    for spec in MATRIX_LEVELS[level]:
        w, h = _FRAMING_SIZE[spec.framing]
        layers = [
            identity,
            spec.view.value,
            spec.light.value,
            spec.expression.value,
            spec.framing.value,
        ]
        if extra_style:
            layers.append(extra_style)
        layers.append(PORTRAIT_TECH_SUFFIX)
        stem = (f"{spec.view.name}_{spec.light.name}_{spec.expression.name}_"
                f"{spec.framing.name}").lower()
        jobs.append(PortraitJob(
            id=f"{character.id}__{stem}",
            character_id=character.id,
            key=spec.key,
            prompt=", ".join(p.strip() for p in layers if p and p.strip()),
            negative_prompt=negative,
            width=w, height=h,
            seed=_slot_seed(character.id, spec.key),
            target_uri=(f"assets/cast/{character.id}/"
                        f"{'turn' if spec.turnaround else 'portrait'}/{stem}.png"),
            role=spec.role,
            tier=spec.tier,
            purpose=spec.purpose,
            label=spec.label(),
            covered_by=have.get(spec.key),
        ))
    return jobs


# ================================================================= 校验

# 「太泛」的描述词。它们在模型隐空间里指向一大片人，等于没有约束。
# 判据不是「这些词不好」，而是「这些词不可判定」：
# 两个人都符合「清秀的年轻女性」，但没有两个人有同一道疤。
_GENERIC_WORDS = frozenset({
    "漂亮", "好看", "帅气", "英俊", "美丽", "清秀", "秀气", "普通", "一般", "标准",
    "年轻", "干净", "精致", "气质好", "有魅力", "迷人", "阳光", "温柔",
    "beautiful", "handsome", "pretty", "attractive", "good-looking", "gorgeous",
    "average", "normal", "typical", "ordinary", "young", "nice", "cute", "clean",
})

# 不可移除的体征。它们是最强的锚，因为换装、换发型、换光都带不走。
_IMMUTABLE_RE = re.compile(
    r"疤|痣|纹身|胎记|雀斑|酒窝|虎牙|断眉|缺口|异色瞳|义眼|耳洞|唇钉|发际线|"
    r"接缝|格栅|义体|假肢|白化|胎痕|牙|骨钉|"
    r"scar|mole|tattoo|birthmark|freckle|dimple|heterochromia|prosthe|seam|vent",
    re.IGNORECASE,
)

# 位置限定词。有位置的描述才可判定：「一道疤」没法验证，「左眉尾一道疤」可以。
_LOCATOR_RE = re.compile(
    r"左|右|上|下|中|前|后|眉|眼|鼻|唇|嘴|颊|颧|颌|下巴|额|耳|颈|喉|肩|锁骨|"
    r"手|腕|指|虎口|太阳穴|眼角|嘴角|发|鬓|头顶|胸|背|腰|腿|"
    r"left|right|upper|lower|brow|eye|nose|lip|cheek|jaw|chin|forehead|ear|"
    r"neck|throat|shoulder|collar|hand|wrist|finger|temple|hair",
    re.IGNORECASE,
)

# 可判定的限定，按**类别**分开。拆开的理由见 AnchorScore.strong：
# 一个部位配一个形容词（「长发」）指向的人太多，算不上身份锚；
# 要判定为强锚点，至少要命中两类限定（颜色 + 形状、形状 + 尺寸……）。
_QUAL_COLOR_RE = re.compile(
    r"黑|白|灰|银|金|棕|褐|红|橙|黄|绿|青|蓝|紫|粉|钴|琥珀|"
    r"black|white|grey|gray|silver|gold|brown|red|orange|amber|green|blue|"
    r"cobalt|purple|pink",
    re.IGNORECASE,
)
_QUAL_SHAPE_RE = re.compile(
    r"弧形|直|卷|波浪|方|圆|尖|斜|锯齿|新月|环形|"
    r"curl|wavy|straight|square|round|sharp|crescent|zigzag",
    re.IGNORECASE,
)
_QUAL_SIZE_RE = re.compile(
    r"[0-9一二两三四五六七八九十]+|齐肩|及腰|过肩|齐耳|及膝|齐眉|"
    r"长|短|粗|细|深|浅|厚|薄|大|小|"
    r"thin|thick|long|short|deep|shallow",
    re.IGNORECASE,
)

# 兼容旧调用：任意一类命中即算「有限定」
_QUALIFIER_RE = re.compile(
    "|".join(r.pattern for r in (_QUAL_COLOR_RE, _QUAL_SHAPE_RE, _QUAL_SIZE_RE)),
    re.IGNORECASE,
)

_ANCHOR_SPLIT_RE = re.compile(r"[、，,;；/＋+|]| and | with ")

# 判据阈值。选这几个数的理由：
#   3 个强锚点 —— 少于 3 个时，任意一个被服装/发型/光遮住就没有锚了；
#   1 个不可移除体征 —— 全靠可变特征（发色、服装）的角色一换装就是新人；
#   2 个部位不同的锚 —— 全集中在脸上的锚在全身镜/背影镜里全部失效。
MIN_STRONG_ANCHORS = 3
MIN_IMMUTABLE_ANCHORS = 1
MIN_ANCHOR_REGIONS = 2


@dataclass
class AnchorScore:
    """单个锚点短语的判定结果。"""

    text: str
    immutable: bool
    located: bool
    qualified: bool
    generic: bool
    qualifier_kinds: int = 0     # 命中了几类限定（颜色 / 形状 / 尺寸数量）

    @property
    def strong(self) -> bool:
        """强锚点判据：不是泛词，且（不可移除 或 有位置 + 至少两类限定）。

        为什么要两类而不是一类：「长发」满足「有位置（发）+ 有限定（长）」，
        但它在模型隐空间里指向一大片人，起不到身份约束的作用。
        「黑色齐肩直发」命中颜色 + 尺寸 + 形状三类，才真的把人群收窄了。
        不可移除体征（疤/痣/异色瞳）单独成立，因为它本身就足够罕见。
        """
        if self.generic or len(self.text.strip()) < 2:
            return False
        return self.immutable or (self.located and self.qualifier_kinds >= 2)

    @property
    def rank(self) -> int:
        """排序用的强度。不可移除 > 有位置有限定 > 只有限定。"""
        return (40 if self.immutable else 0) + (20 if self.located else 0) \
            + (10 if self.qualified else 0) - (50 if self.generic else 0)


def _split_anchor_phrases(text: str) -> list[str]:
    return [p.strip() for p in _ANCHOR_SPLIT_RE.split(text or "") if p.strip()]


def score_anchor(text: str) -> AnchorScore:
    """判定一个锚点短语够不够独特。判据见 AnchorScore.strong。"""
    low = text.lower()
    kinds = sum(bool(r.search(text))
                for r in (_QUAL_COLOR_RE, _QUAL_SHAPE_RE, _QUAL_SIZE_RE))
    return AnchorScore(
        text=text,
        immutable=bool(_IMMUTABLE_RE.search(text)),
        located=bool(_LOCATOR_RE.search(text)),
        qualified=kinds >= 1,
        generic=any(g in low for g in _GENERIC_WORDS),
        qualifier_kinds=kinds,
    )


def _region_of(text: str) -> str:
    """锚点落在哪个身体区域。用于「别把锚全堆在脸上」的检查。"""
    for region, pat in (
        ("head", r"眉|眼|鼻|唇|嘴|颊|颧|颌|下巴|额|耳|太阳穴|虹膜|瞳|牙|brow|eye|nose|lip|"
                 r"cheek|jaw|chin|forehead|ear|iris|tooth"),
        ("hair", r"发|鬓|头顶|辫|刘海|hair|bang|braid"),
        ("neck", r"颈|喉|锁骨|neck|throat|collar"),
        ("torso", r"肩|胸|背|腰|shoulder|chest|back|waist"),
        ("limb", r"手|腕|指|虎口|臂|腿|hand|wrist|finger|arm|leg"),
    ):
        if re.search(pat, text, re.IGNORECASE):
            return region
    return "other"


def _all_anchor_phrases(character: CharacterBible) -> list[str]:
    a = character.appearance
    # 顺序即默认优先级：distinguishing 是专门为锚点准备的字段，排最前。
    out: list[str] = []
    for field_text in (a.distinguishing, a.face, a.hair, a.skin, a.body, a.wardrobe):
        out.extend(_split_anchor_phrases(field_text))
    return out


def consistency_anchors(character: CharacterBible, *, k: int = 6) -> list[str]:
    """提取最强的几个一致性锚点词，供 prompt_os 优先拼入。

    排序逻辑：LoRA 触发词 > 不可移除体征 > 有位置有限定的结构特征 > 其它。
    触发词排第一是因为它在挂了 LoRA 的模型上是**最短且最强**的身份指针，
    一个 token 顶一整句外形描述，token 预算应该先花在它身上。

    返回的是短语而不是整句：prompt_os 要把它们插在提示词最前面，
    整句外形描述会把后面的动作/环境层挤出注意力窗口。
    """
    out: list[str] = []
    if character.lora and character.lora.trigger_word:
        out.append(character.lora.trigger_word)

    scored = [score_anchor(p) for p in _all_anchor_phrases(character)]
    # 稳定排序：先按强度降序，同强度保持原字段顺序（distinguishing 在前）
    ranked = sorted(
        [s for s in scored if not s.generic],
        key=lambda s: -s.rank,
    )
    seen: set[str] = set()
    for s in ranked:
        t = s.text.strip()
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------- 参考图质量

# 定妆图短边阈值。分两档而不是一刀切：512 以下是「多镜之间必漂」，
# 1024 以下只是「建议提高」—— 把两者混成一个警告，人就会两个都不理。
PORTRAIT_PX_CRITICAL = 512
PORTRAIT_PX_ADVISED = 1024
# 宽高比区间。超出这个范围的图裁不出各种景别：太扁的裁不出竖构图的全身，
# 太长的裁特写会只剩半张脸。
PORTRAIT_ASPECT_RANGE = (0.6, 1.7)


@dataclass(frozen=True)
class ImageQuality:
    """一张参考图的体检结果。problems 为空即可用。"""

    uri: str
    exists: bool
    width: int = 0
    height: int = 0
    problems: tuple[str, ...] = ()

    @property
    def short_side(self) -> int:
        return min(self.width, self.height) if self.width and self.height else 0


def _axis_aligned_overlay(gray) -> list[str]:
    """找「轴对齐的长直边」——UI 叠加最特异的特征。

    播放器控件、水印底板、角标都是矩形，边界严格水平或垂直；而头发、
    肩线、衣褶这些自然轮廓是任意角度的，不会让某一整行像素同时成为
    「垂直梯度强、水平梯度弱」的边。实测在带播放器的图上，底部那一行
    有 99% 的像素命中，是全图中位数的 85 倍；干净的图一条都没有。

    只看外围 25%：画面中间的长直边可能是真实场景（桌沿、窗框），
    而 UI 几乎总是贴边放的。
    """
    import numpy as np

    a = np.asarray(gray).astype(float)
    h, w = a.shape
    if h < 32 or w < 32:
        return []
    gy, gx = np.gradient(a)
    horiz = (np.abs(gy) > 10) & (np.abs(gx) < np.abs(gy) * 0.35)
    vert = (np.abs(gx) > 10) & (np.abs(gy) < np.abs(gx) * 0.35)
    row_p, col_p = horiz.sum(1) / w, vert.sum(0) / h

    out: list[str] = []
    for p, n, axis, unit in ((row_p, h, "水平", "行"), (col_p, w, "垂直", "列")):
        med = float(np.median(p)) + 1e-6
        edge_zone = [i for i in range(n) if i < n * 0.25 or i > n * 0.75]
        hits = [i for i in edge_zone if p[i] > 0.28 and p[i] > med * 4]
        if hits:
            where = "上/左" if hits[0] < n * 0.5 else "下/右"
            out.append(
                f"疑似 UI 叠加：靠近{where}边缘的第 {hits[0]}–{hits[-1]} {unit}有一条"
                f"{axis}长直边（占该{unit} {p[hits[0]]:.0%} 像素，是全图中位数的 "
                f"{p[hits[0]]/med:.0f} 倍）。播放器控件、水印底板、角标都长这样，"
                f"而头发和肩线不会 —— 模型会把它当成画面内容学进去"
            )
    return out


def inspect_portrait(uri: str, *, root: str | Path | None = None) -> ImageQuality:
    """体检一张参考图。文件不在就只返回 exists=False，不算问题。

    为什么不算问题：uri 可能是 s3:// 这类远端地址，也可能是还没出图的占位。
    真正该拦的是「图在，但不能用」。
    """
    path = Path(uri)
    if not path.is_absolute() and root is not None:
        path = Path(root) / uri
    if not path.is_file():
        return ImageQuality(uri=uri, exists=False)

    try:
        from PIL import Image
    except ImportError:
        return ImageQuality(uri=uri, exists=True)

    probs: list[str] = []
    with Image.open(path) as im:
        w, h = im.size
        short = min(w, h)
        if short < PORTRAIT_PX_CRITICAL:
            probs.append(
                f"分辨率过低（{w}x{h}，短边 {short} < {PORTRAIT_PX_CRITICAL}）："
                "参考图携带的身份信息量直接取决于像素量，这个尺寸下引擎取不到"
                "足够的面部细节，多镜之间必然漂移"
            )
        elif short < PORTRAIT_PX_ADVISED:
            probs.append(
                f"分辨率偏低（{w}x{h}，短边 {short} < {PORTRAIT_PX_ADVISED}）："
                "近景镜（ECU/CU）会明显发软，建议重出或超分到 1024 以上"
            )
        aspect = w / h if h else 0.0
        lo, hi = PORTRAIT_ASPECT_RANGE
        if aspect and not (lo <= aspect <= hi):
            probs.append(
                f"宽高比 {aspect:.2f} 超出 {lo}–{hi}：裁不出全部景别"
                "（太扁裁不出竖构图全身，太长裁特写只剩半张脸）"
            )
        if im.mode in ("RGBA", "LA"):
            import numpy as np

            alpha = np.asarray(im.convert("RGBA"))[..., 3]
            transparent = float((alpha < 16).mean())
            if transparent > 0.05:
                probs.append(
                    f"含 {transparent:.0%} 透明像素：引擎不认 alpha，透明区会被当成"
                    "黑色或随机填充，请先合成到确定的背景色再用"
                )
        gray = im.convert("L")
        W = 256
        gray = gray.resize((W, max(32, int(h * W / w))), Image.LANCZOS)
    probs.extend(_axis_aligned_overlay(gray))
    return ImageQuality(uri=uri, exists=True, width=w, height=h, problems=tuple(probs))


def validate_bible(character: CharacterBible, *, check_images: bool = True, asset_root: str | Path | None = None) -> list[str]:
    """角色圣经完整性检查。返回问题清单，空列表 = 可以进产线。

    不抛异常：问题要一次列全。抛第一个异常会让人改一条跑一次，
    改到第五条才发现第六条 —— 与 `Storyboard.validate_continuity()` 同样的理由。
    """
    problems: list[str] = []
    cid = character.id

    # ---- 1) 成年声明（复用 compliance 的判据，不另造平行实现）
    try:
        from .compliance import _age_problem          # 延迟导入：compliance 拉 ffmpeg 封装
        p = _age_problem(cid, character.age_statement)
        if p:
            problems.append(p)
    except ImportError as e:                           # pragma: no cover - 依赖缺失时的兜底
        problems.append(f"角色 {cid}: 无法加载 compliance 做年龄校验（{e}），必须人工复核")

    # ---- 2) 必填外形字段
    a = character.appearance
    for fname, label in (("face", "面部"), ("hair", "发型"),
                         ("wardrobe", "服装"), ("distinguishing", "特征锚点")):
        if not getattr(a, fname, "").strip():
            problems.append(
                f"角色 {cid}: appearance.{fname}（{label}）为空 —— "
                f"缺失的维度会由模型自由发挥，每一镜发挥得都不一样"
            )

    # ---- 3) 锚点独特性（这是防跨镜漂移的核心判据）
    scored = [score_anchor(p) for p in _all_anchor_phrases(character)]
    strong = [s for s in scored if s.strong]
    generic = [s.text for s in scored if s.generic]
    immutable = [s for s in strong if s.immutable]
    regions = {_region_of(s.text) for s in strong} - {"other"}

    if len(strong) < MIN_STRONG_ANCHORS:
        problems.append(
            f"角色 {cid}: 强锚点只有 {len(strong)} 个（要求 ≥{MIN_STRONG_ANCHORS}）。"
            "判据：短语必须不是泛词，且『含不可移除体征』或『同时有位置限定和形状/颜色/数量限定』。"
            f"当前强锚点：{[s.text for s in strong] or '无'}。"
            "少于 3 个时，任意一个被服装/发型/光位遮住，这一镜就没有身份约束了。"
        )
    if len(immutable) < MIN_IMMUTABLE_ANCHORS:
        problems.append(
            f"角色 {cid}: 没有不可移除体征（疤/痣/纹身/异色瞳/断眉/接缝等）。"
            "全靠发色、服装这类可变特征的角色，一换装就会被模型当成另一个人。"
        )
    if len(regions) < MIN_ANCHOR_REGIONS:
        problems.append(
            f"角色 {cid}: 锚点全部集中在 {regions or {'未知'}} 区域。"
            "全身镜、背影镜看不到脸，需要至少一个非面部锚（发型轮廓/颈部/服装标识/手部）。"
        )
    if generic:
        problems.append(
            f"角色 {cid}: 描述里含泛词 {generic[:4]} —— "
            "这些词在模型隐空间里指向一大片人，占着提示词位置却不提供约束，建议替换为可判定的具体描述。"
        )

    # ---- 4) 定妆图覆盖度
    specs_by_key = {s.key: s for s in _CORE_MATRIX}
    have_keys: set[str] = set()
    for ref in (*character.portraits, *character.turnaround):
        k = _parse_slot_key(ref.note)
        if k:
            have_keys.add(k)

    if not character.portraits:
        problems.append(f"角色 {cid}: 没有任何定妆图，refpack 无法给镜头挂 identity 锚")
    elif not have_keys:
        # 手写角色圣经的常见情况：有图但 note 不带矩阵坐标，只能降级成粗粒度检查。
        views_txt = " ".join(r.note for r in (*character.portraits, *character.turnaround))
        for need, label, why in (
            (r"正面|正脸|front", "正面", "身份底图，所有光效与表情都在它之上形变"),
            (r"3/4|三分之|侧 ?45|three.quarter|tq", "3/4 侧", "对话戏最常见机位（过肩镜）"),
            (r"正侧|侧脸|profile|90", "正侧", "鼻梁-下颌-后脑的轮廓线，正面图没有这个信息"),
        ):
            if not re.search(need, views_txt, re.IGNORECASE):
                problems.append(
                    f"角色 {cid}: 定妆图疑似缺「{label}」视角（note 里没有矩阵坐标，"
                    f"只能按关键词粗判）—— {why}"
                )
    else:
        missing = [k for k in specs_by_key if k not in have_keys]
        if missing:
            problems.append(
                f"角色 {cid}: 核心定妆矩阵缺 {len(missing)}/{len(specs_by_key)} 格："
                + "；".join(specs_by_key[k].label() for k in missing[:4])
                + ("…" if len(missing) > 4 else "")
            )
        have_views = {specs_by_key[k].view for k in have_keys if k in specs_by_key}
        have_lights = {specs_by_key[k].light for k in have_keys if k in specs_by_key}
        have_exprs = {specs_by_key[k].expression for k in have_keys if k in specs_by_key}
        if len(have_views) < 3:
            problems.append(f"角色 {cid}: 视角只覆盖 {len(have_views)} 种（至少要正面/3-4 侧/正侧）")
        if len(have_lights) < 2:
            problems.append(f"角色 {cid}: 光位只覆盖 {len(have_lights)} 种（平光之外至少再要一个方向光）")
        if len(have_exprs) < 2:
            problems.append(f"角色 {cid}: 表情只覆盖 {len(have_exprs)} 种（中性之外至少再要一个情绪）")

    if len(character.turnaround) < 3:
        problems.append(
            f"角色 {cid}: 转身序列只有 {len(character.turnaround)} 张（建议 ≥3）。"
            "i2v 的起幅机位由分镜决定，转身序列是覆盖任意机位最便宜的办法。"
        )

    # ---- 5) 语音
    v = character.voice
    if not (v.timbre_ref_uri or v.tts_voice_id):
        problems.append(
            f"角色 {cid}: 既没有 timbre_ref_uri 也没有 tts_voice_id，"
            "音色无法跨集复现 —— 观众对声音变化比对脸变化更敏感"
        )

    # ---- 6) 负面词是否在挡「近似错误」
    if not character.negative_prompt.strip():
        problems.append(
            f"角色 {cid}: negative_prompt 为空。角色级负面词的作用是挡住"
            "『差一点就对』的错误（短发角色被画成长发、无须角色被加上胡子），"
            "这类错误全局负面词挡不住。"
        )

    # ---- 7) LoRA 触发词不能是常见词
    if character.lora:
        tw = character.lora.trigger_word
        if len(tw) < 4 or tw.lower() in {"man", "woman", "person", "girl", "boy", "char"}:
            problems.append(
                f"角色 {cid}: LoRA 触发词 {tw!r} 太常见 —— "
                "触发词必须是训练集外几乎不出现的罕见 token，否则会被基模的既有语义盖过"
            )

    # ---- 7) 参考图本身的质量（不是描述，是文件）
    # 前面几项查的都是「你怎么描述这个角色」，这一项查「你给的图能不能用」。
    # 分开的理由：描述写得再好，图是 310px 带播放器 UI 的截图，照样出不来。
    if check_images:
        seen: set[str] = set()
        for ref in (*character.portraits, *character.turnaround):
            if ref.uri in seen:
                continue
            seen.add(ref.uri)
            q = inspect_portrait(ref.uri, root=asset_root)
            if not q.exists:
                continue          # 远端地址或尚未出图的占位，不算问题
            for pr in q.problems:
                problems.append(f"角色 {cid}: 定妆图 {Path(ref.uri).name} {pr}")

    return problems


def coverage_report(character: CharacterBible, *, level: str = "core") -> str:
    """定妆覆盖率报告：还缺哪些格、缺的那些会影响哪类镜头。"""
    jobs = emit_portrait_prompts(character, level=level)
    done = [j for j in jobs if j.covered_by]
    todo = [j for j in jobs if not j.covered_by]
    L = [
        f"定妆覆盖 · {character.name}（{character.id}）· 等级 {level}",
        f"  {len(done)}/{len(jobs)} 已有，{len(todo)} 待拍",
    ]
    if todo:
        L.append("  待拍（按优先级）：")
        for j in sorted(todo, key=lambda x: list(Tier).index(x.tier)):
            L.append(f"    [{j.tier.value:<11}] {j.label:<26} {j.purpose}")
    anchors = consistency_anchors(character)
    L.append(f"  一致性锚点（prompt_os 优先拼入）：{' | '.join(anchors)}")
    probs = validate_bible(character)
    L.append(f"  完整性检查：{'通过' if not probs else f'{len(probs)} 条问题'}")
    L += [f"    - {p}" for p in probs]
    return "\n".join(L)


# ================================================================= 示例角色

def demo_cast() -> list[CharacterBible]:
    """三个完整的虚构成年角色示例。全部能通过 `validate_bible()`。

    这三个角色刻意做成「锚点结构不同」的三类，用来验证判据不是只对一种写法有效：
      沈牧  —— 锚在体征 + 发色（面部 + 头发 + 手部三个区域）
      贺沅  —— 锚在骨骼结构 + 职业痕迹（面部 + 手部 + 服装）
      祁诺  —— 锚在虹膜异色 + 金属饰品（面部 + 颈部 + 耳部）
    """
    shen = (
        CharacterBibleBuilder(
            "shen_mu", "沈牧", age=34, occupation="深海声学工程师",
            persona="话少，习惯用录音而不是笔记记录；对声音的记忆远好于对脸的记忆。",
        )
        .describe(
            face="方圆脸，下颌线清晰，左眉尾有一道 2cm 弧形旧疤，右侧虎牙有一个小缺角",
            hair="黑色齐肩直发常扎成低马尾，发尾挑染一缕钴蓝",
            body="中等偏高，右肩比左肩略低（长期背设备），走路步幅大",
            skin="冷调中性肤色，鼻梁和颧骨有细密雀斑",
            wardrobe="灰蓝色防水工装夹克，左袖口缝着褪色的橙色编号布贴，腰间挂一圈线缆",
            distinguishing="左眉尾弧形旧疤，右虎牙缺角，发尾钴蓝挑染，右手虎口有一道被缆绳磨出的横向老茧",
        )
        .with_voice(tts_engine="indextts2", tts_voice_id="v_shen_mu",
                    timbre_ref_uri="assets/voice/shen_mu_ref.wav",
                    emotion_default="calm", language="zh", speed=0.96)
        .with_lora(base_model="wan2.2-i2v-a14b", trigger_word="shenmu_char", strength=0.82)
        .with_negative("长卷发, 披发, 刘海遮眉, 全黑发无挑染, 浓妆, 大眼滤镜")
        .with_motion_ref("assets/cast/shen_mu/walk_gear.mp4", "背设备行走，右肩略低")
        .build()
    )

    he = (
        CharacterBibleBuilder(
            "he_yuan", "贺沅", age=41, occupation="离岛灯塔看守",
            persona="礼貌但话里留白；习惯在说重要事情之前先看一眼窗外。",
        )
        .describe(
            face="长脸，颧骨高，左眉骨中段有一道白色断眉痕，鼻梁中部有一处陈年折痕微微偏右",
            hair="花白短寸，右侧鬓角有一小片不长毛的白斑",
            body="瘦高，肩背略驼，站立时习惯性双手插在外套口袋",
            skin="偏暖的黄褐色，长期风吹日晒，眼角有深刻的放射状细纹",
            wardrobe="深绿色厚呢子外套，右肘打了一块深棕色皮补丁，颈间挂一枚铜制哨子",
            distinguishing="左眉骨白色断眉痕，鼻梁陈年折痕偏右，右鬓角白斑，颈间铜哨",
        )
        .with_voice(tts_engine="indextts2", tts_voice_id="v_he_yuan",
                    timbre_ref_uri="assets/voice/he_yuan_ref.wav",
                    emotion_default="neutral", language="zh", speed=0.92, pitch_shift=-1.5)
        .with_negative("络腮胡, 年轻面孔, 光滑皮肤, 全黑头发, 戴眼镜")
        .build()
    )

    qi = (
        CharacterBibleBuilder(
            "qi_nuo", "祁诺", age=29, occupation="旧物修复师",
            persona="嘴上刻薄手上温柔；对任何坏掉的东西都先问『它原来是什么样』。",
        )
        .describe(
            face="尖下巴，圆眼，左眼虹膜琥珀色而右眼灰蓝色（异色瞳），左嘴角有一颗小痣",
            hair="栗棕色短卷发，右侧比左侧短约两指，额前一缕总是翘起",
            body="偏瘦，坐着时习惯把一条腿盘起来",
            skin="暖调浅褐，颈侧右下方有三颗排成一条斜线的小痣",
            wardrobe="米色工作围裙套在洗旧的墨绿衬衫外，围裙口袋边缘沾着洗不掉的靛蓝染料",
            distinguishing="左琥珀右灰蓝的异色瞳，左嘴角小痣，颈侧右下三颗斜线排列的痣，右耳三枚银色耳骨钉",
        )
        .with_voice(tts_engine="indextts2", tts_voice_id="v_qi_nuo",
                    timbre_ref_uri="assets/voice/qi_nuo_ref.wav",
                    emotion_default="wry", language="zh", speed=1.04)
        .with_lora(base_model="wan2.2-i2v-a14b", trigger_word="qinuo_char", strength=0.80)
        .with_negative("双眼同色, 长直发, 无耳饰, 浓妆, 西装")
        .build()
    )

    return [shen, he, qi]


# ================================================================= 自测

def _selftest() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # ---- 1) 构建器与拍摄清单
    b = (CharacterBibleBuilder("shen_mu", "沈牧", age=34, occupation="深海声学工程师")
         .describe(face="方圆脸，左眉尾有一道 2cm 弧形旧疤",
                   hair="黑色齐肩直发，发尾挑染一缕钴蓝",
                   wardrobe="灰蓝色防水工装夹克",
                   distinguishing="左眉尾弧形旧疤，右虎口横向老茧"))
    print(b.matrix_rationale("core"))

    core = b.shot_list("core")
    assert len(core) == MIN_CORE_SHOTS == 15, len(core)
    for lvl, specs in MATRIX_LEVELS.items():
        keys = [s.key for s in specs]
        assert len(keys) == len(set(keys)), f"{lvl} 矩阵里有重复格子"
        uris = [b.uri_for(s) for s in specs]
        assert len(uris) == len(set(uris)), f"{lvl} 矩阵里有两格指向同一个文件"
    views = {s.view for s in core if not s.turnaround}
    lights = {s.light for s in core}
    exprs = {s.expression for s in core}
    assert {View.FRONT, View.TQ_RIGHT, View.PROFILE} <= views, views
    assert len(lights) >= 3 and len(exprs) >= 3, (lights, exprs)
    assert sum(1 for s in core if s.turnaround) == 5
    assert any(s.framing is Framing.EYES for s in core)
    assert any(s.role == "wardrobe" for s in core)
    assert all(s.purpose for s in core), "每一格都必须写明为什么需要它"
    assert len(b.shot_list("recommended")) == 23
    assert len(b.shot_list("lora")) == 31          # ≥ lora.recommend_character_lora 的 18 张下限

    # 未成年设定必须在构建阶段就被拒，不能指望下游门禁
    try:
        CharacterBibleBuilder("x", "X", age=16)
        raise AssertionError("未成年设定居然构建成功了")
    except ValueError as e:
        assert "成年" in str(e)

    # ---- 2) demo_cast 全部通过校验
    cast = demo_cast()
    assert len(cast) == 3
    print("\n=== demo_cast 校验 ===")
    for c in cast:
        probs = validate_bible(c)
        print(f"  {c.id:<9} {c.name}  定妆 {len(c.portraits)} + 转身 {len(c.turnaround)}  "
              f"问题 {len(probs)}")
        for p in probs:
            print(f"      ! {p}")
        assert not probs, f"{c.id} 未通过校验"
        assert c.is_fictional is True
        assert "虚构" in c.age_statement
        assert re.search(r"(\d{2})\s*岁", c.age_statement)
        assert int(re.search(r"(\d{2})\s*岁", c.age_statement).group(1)) >= 18

    # ---- 3) 锚点判据真的能分辨泛描述
    vague = CharacterBible(
        id="vague", name="路人甲",
        age_statement="虚构角色，设定年龄 25 岁",
        appearance=Appearance(face="漂亮的年轻女性，五官精致", hair="长发",
                              wardrobe="连衣裙", distinguishing="气质很好"),
    )
    vp = validate_bible(vague)
    assert any("强锚点" in p for p in vp), vp
    assert any("不可移除体征" in p for p in vp), vp
    assert any("泛词" in p for p in vp), vp
    assert any("定妆图" in p for p in vp), vp
    assert any("timbre_ref_uri" in p for p in vp), vp
    print(f"\n=== 泛描述角色的检查结果（应有 {len(vp)} 条）===")
    for p in vp:
        print(f"  ! {p}")

    # 逐条短语判据
    assert score_anchor("左眉尾一道 2cm 弧形旧疤").strong
    assert score_anchor("颈侧右下三颗斜线排列的痣").strong
    assert not score_anchor("漂亮").strong
    assert not score_anchor("长发").strong, "只有部位没有限定，不该算强锚点"
    assert score_anchor("黑色齐肩直发").strong
    assert score_anchor("异色瞳").immutable
    assert _region_of("左眉尾旧疤") == "head" and _region_of("右虎口老茧") == "limb"

    # ---- 4) 一致性锚点排序
    anchors = consistency_anchors(cast[0])
    print(f"\n=== 一致性锚点 · {cast[0].name} ===\n  " + " | ".join(anchors))
    assert anchors[0] == "shenmu_char", "挂了 LoRA 时触发词必须排第一"
    assert any("疤" in a for a in anchors), "不可移除体征没有进前几位"
    assert all("漂亮" not in a for a in anchors)
    assert len(anchors) == len(set(anchors)), "锚点有重复"
    no_lora = consistency_anchors(cast[1])
    assert no_lora and "断眉" in no_lora[0], no_lora   # 没有 LoRA 时体征排第一

    # ---- 5) 出图工单
    jobs = emit_portrait_prompts(cast[0], level="core",
                                 extra_style="cinematic teal and amber palette")
    assert len(jobs) == MIN_CORE_SHOTS
    assert all(j.covered_by for j in jobs), "builder 产出的角色应全部命中矩阵坐标"
    assert len({j.seed for j in jobs}) == len(jobs), "同一角色的各格种子必须互不相同"
    j2 = emit_portrait_prompts(cast[0], level="core",
                               extra_style="cinematic teal and amber palette")
    assert [j.seed for j in jobs] == [j.seed for j in j2], "种子不可复现"
    assert [j.prompt for j in jobs] == [j.prompt for j in j2], "提示词不幂等"

    head = jobs[0]
    assert head.prompt.startswith("shenmu_char"), head.prompt[:60]
    assert "frontal view" in head.prompt and "flat even frontal lighting" in head.prompt
    assert head.prompt.rstrip().endswith("high detail skin texture")
    assert "sunglasses" in head.negative_prompt and "长卷发" in head.negative_prompt
    eyes = next(j for j in jobs if "eyes" in j.key.lower())
    assert eyes.width == eyes.height == 1024
    full = next(j for j in jobs if "FULL_BODY" in j.key)
    assert full.height > full.width
    print(f"\n=== 出图工单样例（共 {len(jobs)} 条）===")
    for j in (jobs[0], jobs[5], jobs[9]):
        print("  " + j.render().replace("\n", "\n  "))

    # 手写角色圣经（note 无矩阵坐标）也要能出清单，只是全部算待拍
    hand = CharacterBible(
        id="hand", name="手写", age_statement="虚构角色，设定年龄 30 岁",
        appearance=Appearance(face="左眉尾浅疤", hair="黑色短发"),
        portraits=[ImageRef(role="identity", uri="a.png", note="正面 平光 中性",
                            subject_id="hand")],
    )
    hjobs = emit_portrait_prompts(hand)
    assert len(hjobs) == MIN_CORE_SHOTS and not any(j.covered_by for j in hjobs)
    hprobs = validate_bible(hand)
    assert any("疑似缺「3/4 侧」" in p for p in hprobs), hprobs   # 粗粒度关键词降级判断生效
    # 用带书名号的完整模式匹配，别用「正面」裸子串 —— 其它条目的解释文字里
    # 就有「正面图没有这个信息」，裸子串会把一条正确的提示误判成缺失告警。
    assert not any("疑似缺「正面」" in p for p in hprobs), hprobs

    # ---- 6) 覆盖率报告
    # ---- 7) 参考图体检（用程序生成的图，不依赖外部资产）
    import tempfile

    from PIL import Image, ImageDraw

    with tempfile.TemporaryDirectory(prefix="charbible_img_") as td:
        td = Path(td)
        # a) 干净的大图：不该报任何问题
        clean = td / "clean.png"
        im = Image.new("RGB", (1200, 1200), (150, 150, 155))
        d = ImageDraw.Draw(im)
        d.ellipse([400, 300, 800, 800], fill=(190, 170, 160))     # 一个「人头」，边缘是曲线
        im.save(clean)
        q = inspect_portrait(str(clean))
        assert q.exists and not q.problems, f"干净大图不该有问题：{q.problems}"

        # b) 小图：必须报分辨率过低
        small = td / "small.png"
        im.resize((300, 300)).save(small)
        q = inspect_portrait(str(small))
        assert any("分辨率过低" in x for x in q.problems), q.problems

        # c) 带 UI 底板的大图：必须报疑似 UI。
        #    画一个贴着底边的纯色矩形 —— 播放器控件/水印底板就是这个形状。
        ui = td / "ui.png"
        im2 = im.copy()
        ImageDraw.Draw(im2).rectangle([0, 1120, 380, 1200], fill=(20, 20, 20))
        im2.save(ui)
        q = inspect_portrait(str(ui))
        assert any("疑似 UI 叠加" in x for x in q.problems), f"没抓到 UI 底板：{q.problems}"
        # 同一张图的曲线边缘（人头）不该被误报成 UI
        assert not any("疑似 UI 叠加" in x for x in inspect_portrait(str(clean)).problems)

        # d) 极端宽高比
        wide = td / "wide.png"
        Image.new("RGB", (2400, 800), (150, 150, 155)).save(wide)
        assert any("宽高比" in x for x in inspect_portrait(str(wide)).problems)

        # e) 大面积透明
        alpha = td / "alpha.png"
        Image.new("RGBA", (1200, 1200), (150, 150, 155, 0)).save(alpha)
        assert any("透明像素" in x for x in inspect_portrait(str(alpha)).problems)

        # f) 文件不存在：静默跳过，不算问题（uri 可能是 s3:// 或未出图的占位）
        q = inspect_portrait(str(td / "nope.png"))
        assert not q.exists and not q.problems

    print("[7] 参考图体检：分辨率分档 / UI 底板 / 宽高比 / 透明 / 缺图静默 全部符合预期")

    print("\n=== 覆盖率报告 ===")
    print(coverage_report(cast[2]))

    # ---- 7) 与 storyboard_gen / refpack 的接口对齐
    from .refpack import pick_identity_anchors
    from .schema import ShotSize
    picked = pick_identity_anchors(cast[0], ShotSize.CU, 2)
    assert picked, "refpack 取不到 identity 锚"
    assert all(r.subject_id == cast[0].id for r in picked)
    print(f"\nrefpack 在 CU 镜为 {cast[0].name} 选出的锚：" +
          " | ".join(r.note.split(" [")[0] for r in picked))

    print("\ncharbible 自测通过")


if __name__ == "__main__":
    _selftest()
