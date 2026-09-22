"""剧本节拍 → 分镜生成 —— 把导演知识写成可执行的分配规则与镜头模板库。

这一层要回答的是「一段文字该拆成几个镜、每个镜多长、用什么机位」。
它决定了后面所有工位（参考位 / 提示词 / 路由 / 拼接）的工作量和成片节奏，
所以规则必须**显式、可解释、可复现**，而不是让 LLM 自由发挥再人工返工。

三条核心设计：

1. **时长分配不是均匀切。** 把 300 秒平均分成 N 个 8 秒镜是最省事也最难看的做法：
   建置段和高潮段的镜长需求相差两倍以上。本模块把分配拆成两个**互相独立**的量：
   - 「信息密度 density」决定这段拿多少**屏幕时间**（要交代的信息多就要多给时间）；
   - 「情绪强度 intensity」决定这段时间**切多少刀**（强度高靠多切，不靠拉长）。
   这两个量正交，是整套分配规则的地基。详见 `plan_beats()`。

2. **镜头模板库是导演知识的代码化。** `SHOT_PATTERNS` 里每个模板都是一段可复用
   的镜头语法骨架（正反打、建置递进、动作节奏型……），带景别 / 运镜 / 时长比例。
   模板让「这场戏怎么拍」变成查表 + 参数化，而不是每次重新想。

3. **轴线按角色追踪，不按相邻镜比较。** `schema.Storyboard.validate_continuity()` 的
   判据是「同一个角色的银幕方向在后续镜头里翻转」——正反打里两人方向相反是**正确**
   的对切。所以本模块给每个角色分配一个全片固定的 `screen_direction`，
   多人同框镜一律 `neutral`（同一镜里两个方向冲突，写哪个都会污染追踪状态）。

生成路径有两条，共用同一套校验：
  - 离线确定性路径：`beat_sheet_to_scenes()` + `apply_pattern()`，无需模型，CI 可跑；
  - LLM 路径：`llm_prompt_for_storyboard()` 出提示词，`validate_and_repair()` 收残局。
LLM 出来的分镜几乎必然有毛病（时长超限、引用不存在的角色、跳轴、枚举值瞎编），
`validate_and_repair()` 的职责是**修掉能修的、如实报告修不了的**，绝不静默吞掉。
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field, ValidationError

from .schema import (
    CameraMove,
    CharacterBible,
    Continuity,
    ContentRating,
    DialogueLine,
    EngineHint,
    Grade,
    Scene,
    Shot,
    ShotSize,
    Storyboard,
    Transition,
    json_schema,
)

logger = logging.getLogger(__name__)


# ================================================================= 节拍表

class BeatFunction(str, Enum):
    """节拍在叙事里承担的功能。它决定默认模板和镜长修正系数。

    值写成英文术语，方便直接拼进 LLM 提示词。
    """

    ESTABLISH = "establish"       # 建置：交代空间、时间、人物关系
    DIALOGUE = "dialogue"         # 对话：信息靠台词走
    ACTION = "action"             # 动作：信息靠动作走
    REVEAL = "reveal"             # 揭示：一个之前藏着的事实被亮出来
    REACTION = "reaction"         # 反应：对上一拍的情绪回应
    TRANSITION = "transition"     # 过场：空间/时间位移
    CLIMAX = "climax"             # 高潮
    RESOLUTION = "resolution"     # 收束


class Beat(BaseModel):
    """一个剧本节拍。这是 `beat_sheet_to_scenes()` 的唯一输入单位。

    `intensity` 与 `density` 必须分别给：把两者混成一个「重要性」会让
    「安静但信息量极大的解释段」和「吵闹但没信息的打戏」拿到同样的镜头方案，
    而这两段的正确拍法恰好相反（前者要长镜少切，后者要短镜多切）。
    """

    id: str
    title: str = ""
    synopsis: str = ""
    location: str = ""
    time_of_day: str = "day"
    function: BeatFunction = BeatFunction.DIALOGUE

    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="情绪强度：决定切多快")
    density: float = Field(default=0.5, ge=0.0, le=1.0, description="信息密度：决定给多少屏幕时间")
    weight: float = Field(default=1.0, gt=0.0, description="人工权重，用于强行给某拍加/减戏")

    characters: list[str] = Field(default_factory=list, description="出场角色 id，顺序即 A/B 位")
    dialogue: list[DialogueLine] = Field(default_factory=list)
    pattern: str | None = Field(default=None, description="指定模板名；None 走 function 的默认模板")

    lighting: str = ""
    mood: str = ""
    sfx: list[str] = Field(default_factory=list)
    base_grade: Grade = Field(default_factory=Grade)
    content_rating: ContentRating = ContentRating.G


# ---------------------------------------------------------------- 分配规则常量

# 情绪强度 → 镜长的映射指数。<1 表示曲线在低强度段更平：
# 强度 0.2 和 0.4 的段落镜长差别不该太大（都是「文戏」），
# 但一旦冲到 0.8 以上就要明显变快（高潮段的加速必须能被观众感觉到）。
_INTENSITY_CURVE = 0.85

# 情绪强度对**屏幕时间**的影响系数。有意压得很小（0.25）：
# 高潮段确实值得多待一会儿，但它主要应该表现为「同样的时间里切更多刀」，
# 而不是「把戏拉长」。系数给大了会让高潮段又长又碎，总时长立刻失控。
_INTENSITY_TIME_COEF = 0.25

# 节拍功能 → 镜长修正系数。乘在强度算出的基准镜长上。
# 依据是各功能对「单镜内可读信息量」的要求：
#   建置/收束要让观众读完整个画面（空间关系、谁在哪），必须给长镜；
#   动作/反应的信息在瞬间完成，长镜只会让观众等；
#   揭示要「停住」让观众消化，所以比对话还略长。
_FUNCTION_LENGTH_FACTOR: dict[BeatFunction, float] = {
    BeatFunction.ESTABLISH: 1.20,
    BeatFunction.DIALOGUE: 1.00,
    BeatFunction.ACTION: 0.85,
    BeatFunction.REVEAL: 1.10,
    BeatFunction.REACTION: 0.80,
    BeatFunction.TRANSITION: 1.15,
    BeatFunction.CLIMAX: 0.80,
    BeatFunction.RESOLUTION: 1.25,
}

# 每个功能的最少镜头数。理由是镜头语法本身的下限：
# 对话没有两个机位就不成其为对话（只有一个机位那叫「录像」）；
# 动作段少于 3 个镜就没有节奏可言（起-承-落）。
_FUNCTION_MIN_SHOTS: dict[BeatFunction, int] = {
    BeatFunction.ESTABLISH: 2,
    BeatFunction.DIALOGUE: 2,
    BeatFunction.ACTION: 3,
    BeatFunction.REVEAL: 2,
    BeatFunction.REACTION: 1,
    BeatFunction.TRANSITION: 1,
    BeatFunction.CLIMAX: 3,
    BeatFunction.RESOLUTION: 1,
}

# 生成引擎的原子窗口硬边界。取自 providers/base.Capabilities 的默认值
# （min 4.0 / max 15.0）与各家实测档位（dashscope wan 固定 5s 档、
# 聚合站多数 5–10s）。低于 4s 的镜大部分引擎按最低档计费，等于白烧额度；
# 高于 15s 的镜没有任何在售引擎能一次出完，必须交给 chain.py 做续写链。
HARD_MIN_SHOT_S = 4.0
HARD_MAX_SHOT_S = 15.0

# 镜长量化步长。对齐到 0.5s 是为了让同一条产线上的 duration 值收敛成有限集合，
# 提高 Shot.fingerprint() 的缓存命中率 —— 8.03s 和 8.07s 是两个不同的指纹，
# 但它们在成片里没有任何可感知差别，白白多烧一次生成。
_QUANT_S = 0.5


def _quantize(v: float) -> float:
    return round(v / _QUANT_S) * _QUANT_S


def shot_length_window(atomic_shot_s: float) -> tuple[float, float]:
    """由标称原子镜长推出本片允许的镜长区间。

    不直接让调用方传 (min, max) 的原因：这两个数必须相对标称值成比例，
    否则「标称 6s 但允许 4–15s」会让分配器算出一堆贴着上限的 15s 镜，
    整片节奏与标称值完全脱节。
    """
    lo = max(HARD_MIN_SHOT_S, _quantize(atomic_shot_s * 0.55))
    hi = min(HARD_MAX_SHOT_S, _quantize(atomic_shot_s * 1.55))
    if hi <= lo:                      # 标称值贴着硬边界时的兜底
        hi = min(HARD_MAX_SHOT_S, lo + _QUANT_S * 2)
    return lo, hi


def _base_shot_length(beat: Beat, lo: float, hi: float) -> float:
    """单拍的基准镜长：强度决定位置，功能做修正，再夹回窗口。"""
    span = hi - lo
    raw = hi - span * (beat.intensity ** _INTENSITY_CURVE)
    raw *= _FUNCTION_LENGTH_FACTOR.get(beat.function, 1.0)
    return max(lo, min(hi, raw))


@dataclass
class BeatAllocation:
    """一拍分到的镜头预算，以及**为什么是这个数**。

    rationale 不是装饰。分镜方案要给人看、要被推翻重来，
    只给一个「12 秒 3 个镜」的结论，没人能判断它对不对。
    """

    beat: Beat
    duration_s: float
    n_shots: int
    avg_shot_s: float
    time_share: float                 # 占全片时长的比例
    rationale: str = ""

    def render(self) -> str:
        return (
            f"{self.beat.id} [{self.beat.function.value}] "
            f"{self.duration_s:>6.1f}s × {self.n_shots:>2d} 镜 "
            f"(均 {self.avg_shot_s:.1f}s, 占比 {self.time_share:.1%})  {self.rationale}"
        )


def plan_beats(
    beats: Sequence[Beat],
    *,
    target_duration_s: float,
    atomic_shot_s: float = 8.0,
) -> list[BeatAllocation]:
    """把总时长按节拍分配成「时长 + 镜头数」，并给出每一步的理由。

    分配规则（两个正交维度）：

    第一步 —— **屏幕时间**按信息密度分。
        time_weight = weight × (density + 0.25 × intensity)
        密度主导：一拍要交代的信息越多，观众需要的时间越长。强度只做 0.25 的小幅
        加权（高潮值得多待一会儿），刻意压小是因为强度的正确表达方式是下一步的
        「切得更碎」，而不是把戏拉长 —— 系数给大了高潮段会又长又碎，总时长失控。

    第二步 —— **切多少刀**按情绪强度算。
        基准镜长 L = hi − (hi − lo) × intensity^0.85，再乘功能修正系数。
        强度 0 拿窗口上限（长镜、让观众看），强度 1 拿下限（短镜、推着走）。
        功能修正是因为「建置」和「反应」在同样强度下镜长需求差 1.5 倍：
        建置要让观众读完空间，反应的信息一瞬间就完成了。

    第三步 —— **对齐总时长**。
        n = round(time / L)，夹到功能下限（对话至少 2 个机位，否则不叫对话），
        然后按 n × L 重算时长并整体缩放到 target，再把每镜均长夹回 [lo, hi]。
        缩放而不是重分配，是为了保住第一、二步算出的**相对**关系。

    返回的每条 BeatAllocation 都带 rationale，说明它这三步各走到了哪。
    """
    if not beats:
        return []
    lo, hi = shot_length_window(atomic_shot_s)

    # --- 第一步：屏幕时间权重
    weights = [b.weight * (b.density + _INTENSITY_TIME_COEF * b.intensity) for b in beats]
    # 全 0 权重（所有拍 density=intensity=0）时退化为均分，而不是除零。
    total_w = sum(weights)
    if total_w <= 0:
        weights = [1.0] * len(beats)
        total_w = float(len(beats))
    times = [target_duration_s * w / total_w for w in weights]

    # --- 第二步：基准镜长与镜头数
    lengths = [_base_shot_length(b, lo, hi) for b in beats]
    counts: list[int] = []
    for b, t, L in zip(beats, times, lengths):
        n = int(round(t / L)) if L > 0 else 1
        n = max(n, _FUNCTION_MIN_SHOTS.get(b.function, 1))
        counts.append(max(1, n))

    # --- 第三步：按 n × L 重算并整体缩放对齐 target
    durations = [n * L for n, L in zip(counts, lengths)]
    total = sum(durations)
    scale = target_duration_s / total if total > 0 else 1.0
    allocs: list[BeatAllocation] = []
    for b, n, L, d in zip(beats, counts, lengths, durations):
        d2 = d * scale
        avg = max(lo, min(hi, d2 / n))    # 缩放后均长可能出窗，夹回去
        d2 = _quantize(avg) * n
        allocs.append(
            BeatAllocation(
                beat=b,
                duration_s=round(d2, 2),
                n_shots=n,
                avg_shot_s=round(d2 / n, 2),
                time_share=0.0,           # 下面统一回填，避免用缩放前的分母
                rationale="",
            )
        )

    grand = sum(a.duration_s for a in allocs) or 1.0
    for a, L in zip(allocs, lengths):
        a.time_share = a.duration_s / grand
        b = a.beat
        a.rationale = (
            f"密度 {b.density:.2f}/强度 {b.intensity:.2f} → 时间权重 "
            f"{b.weight:.2f}×({b.density:.2f}+{_INTENSITY_TIME_COEF}×{b.intensity:.2f})"
            f"={b.weight * (b.density + _INTENSITY_TIME_COEF * b.intensity):.2f}；"
            f"基准镜长 {L:.1f}s（窗口 {lo:.1f}–{hi:.1f}s，"
            f"{b.function.value} 修正 ×{_FUNCTION_LENGTH_FACTOR.get(b.function, 1.0):.2f}）"
            f" → {a.n_shots} 镜"
        )

    drift = grand - target_duration_s
    logger.info(
        "节拍分配完成：%d 拍 / %d 镜 / %.1fs（目标 %.1fs，偏差 %+.1fs = %+.1f%%）",
        len(allocs), sum(a.n_shots for a in allocs), grand,
        target_duration_s, drift, 100 * drift / max(target_duration_s, 1e-6),
    )
    return allocs


# ================================================================= 镜头模板库

# 模板槽里的主体角色位。用符号而不是具体 id，模板才能复用到任何一场戏。
#   "A" / "B" —— 本场的第一、第二主体（Beat.characters 的顺序）
#   "ALL"     —— 全部出场角色（双人镜、群镜）
#   "NONE"    —— 空镜 / 道具 / 环境（insert、建置大远景）
SlotRole = str


@dataclass(frozen=True)
class PatternSlot:
    """模板里的一个槽位：一条不带具体内容的镜头骨架。"""

    shot_size: ShotSize
    camera_move: CameraMove = CameraMove.STATIC
    duration_ratio: float = 1.0       # 相对本模板内其它槽的时长比例（不必归一）
    lens_mm: int = 35
    role: SlotRole = "ALL"
    transition_in: Transition = Transition.CUT
    inherit_last_frame: bool = False  # 本镜首帧接上一镜尾帧（连续动作/同机位续接）
    match_on: str = ""                # MATCH_CUT 的匹配元素
    priority: int = 5                 # 裁剪时保留优先级，越大越不可裁
    repeatable: bool = False          # 镜头数超出模板时，从可重复槽里循环补
    note: str = ""


@dataclass(frozen=True)
class ShotPattern:
    """一段镜头语法骨架。这是导演知识的代码化形态。"""

    name: str
    label: str
    rationale: str                    # 为什么这样排（给人看的，也拼进 LLM 提示词）
    slots: tuple[PatternSlot, ...]
    min_subjects: int = 0

    def total_ratio(self) -> float:
        return sum(s.duration_ratio for s in self.slots) or 1.0


# 依据说明：以下模板是经典剪辑语法的常规写法（正反打的 30° 规则与轴线、
# 建置段的「远→中→近」递进、动作段的「宽-快-点-宽」节奏），属于行业通行知识，
# 没有可引用的单一权威出处；各槽的 duration_ratio 是本产线的经验初值，
# 需要用实际成片回归标定（见 pacing_report 的建议输出）。
SHOT_PATTERNS: dict[str, ShotPattern] = {
    "shot_reverse_shot": ShotPattern(
        name="shot_reverse_shot",
        label="对话正反打",
        rationale=(
            "先用双人镜建立两人的空间关系与轴线，之后的过肩/特写才有参照系；"
            "过肩镜负责『在场』，特写负责『情绪』，两者交替推进；"
            "结尾回到双人镜收束，让观众重新确认空间，避免长段对切后失去方位感。"
            "A、B 的银幕方向天然相反 —— 这是正反打成立的条件，不是跳轴。"
        ),
        min_subjects=2,
        slots=(
            PatternSlot(ShotSize.TWO, CameraMove.STATIC, 1.35, 35, "ALL",
                        priority=9, note="建立轴线与两人关系"),
            PatternSlot(ShotSize.OTS, CameraMove.STATIC, 1.15, 50, "A",
                        priority=8, repeatable=True, note="越过 B 肩看 A"),
            PatternSlot(ShotSize.OTS, CameraMove.STATIC, 1.15, 50, "B",
                        priority=8, repeatable=True, note="越过 A 肩看 B"),
            PatternSlot(ShotSize.MCU, CameraMove.STATIC, 0.95, 85, "A",
                        priority=7, repeatable=True, note="A 情绪推进"),
            PatternSlot(ShotSize.MCU, CameraMove.DOLLY_IN, 0.95, 85, "B",
                        priority=7, repeatable=True, note="B 情绪推进"),
            PatternSlot(ShotSize.CU, CameraMove.STATIC, 0.85, 85, "A",
                        priority=6, note="最近的一次，留给最重的一句"),
            PatternSlot(ShotSize.TWO, CameraMove.DOLLY_OUT, 1.30, 35, "ALL",
                        priority=8, note="退回双人镜收束"),
        ),
    ),
    "establish_progression": ShotPattern(
        name="establish_progression",
        label="建置递进（远→中→近）",
        rationale=(
            "观众读一个新空间需要三层信息：在哪（大远景）、谁在里面（全景/中景）、"
            "他在想什么（近景）。跳过任何一层，后面的近景就失去坐标 —— "
            "这也是为什么建置段的镜长必须给足：远景的信息量最大，读完它要时间。"
            "推进方向单向收紧，中途回宽会让『进入』的感觉被打断。"
        ),
        slots=(
            PatternSlot(ShotSize.ELS, CameraMove.DOLLY_IN, 1.55, 24, "NONE",
                        transition_in=Transition.FADE_IN, priority=9, note="在哪"),
            PatternSlot(ShotSize.LS, CameraMove.PAN_R, 1.30, 24, "ALL",
                        priority=8, repeatable=True, note="谁在里面"),
            PatternSlot(ShotSize.MS, CameraMove.STEADICAM, 1.10, 35, "A",
                        priority=7, repeatable=True, inherit_last_frame=True, note="跟进"),
            PatternSlot(ShotSize.MCU, CameraMove.STATIC, 0.95, 50, "A",
                        priority=6, inherit_last_frame=True, note="他在看什么"),
            PatternSlot(ShotSize.INSERT, CameraMove.DOLLY_IN, 0.70, 85, "NONE",
                        transition_in=Transition.MATCH_CUT, match_on="视线落点",
                        priority=5, note="他看到的东西"),
            PatternSlot(ShotSize.CU, CameraMove.STATIC, 0.85, 85, "A",
                        priority=6, note="他的反应"),
        ),
    ),
    "action_rhythm": ShotPattern(
        name="action_rhythm",
        label="动作节奏型（宽-快-点-宽）",
        rationale=(
            "动作段的可读性取决于观众知不知道『谁在哪、朝哪打』，所以每一轮快切之前"
            "必须有一个交代空间的宽镜。节奏做成加速：宽镜最长，中间的动作镜逐个变短，"
            "撞击点给最短的插入镜（观众的注意力在撞击瞬间只能处理一个信息），"
            "然后立刻回宽镜让观众喘一口并重新定位。全程 handheld / whip 制造不稳定感，"
            "但宽镜保持相对稳定 —— 连宽镜都晃，观众会彻底丢失空间。"
        ),
        slots=(
            PatternSlot(ShotSize.FS, CameraMove.STATIC, 1.40, 28, "ALL",
                        priority=9, note="交代空间：谁在哪"),
            PatternSlot(ShotSize.MS, CameraMove.HANDHELD, 1.00, 35, "A",
                        priority=7, repeatable=True, inherit_last_frame=True, note="发起"),
            PatternSlot(ShotSize.MCU, CameraMove.WHIP, 0.70, 50, "B",
                        priority=7, repeatable=True, note="承接"),
            PatternSlot(ShotSize.INSERT, CameraMove.STATIC, 0.55, 85, "NONE",
                        priority=6, note="撞击点：只给一个信息"),
            PatternSlot(ShotSize.CU, CameraMove.HANDHELD, 0.65, 85, "B",
                        priority=6, repeatable=True, note="受力反应"),
            PatternSlot(ShotSize.FS, CameraMove.DOLLY_OUT, 1.35, 28, "ALL",
                        priority=8, note="回宽镜，重新定位"),
        ),
    ),
    "reveal_pullback": ShotPattern(
        name="reveal_pullback",
        label="揭示（近→远拉开）",
        rationale=(
            "揭示的力量来自『观众先以为自己知道，然后发现画外还有东西』。"
            "所以顺序必须与建置相反：先给一个信息受限的近景，再拉开暴露全貌。"
            "拉开那一镜要给足时长 —— 观众需要时间重新解释他刚才看到的一切。"
        ),
        slots=(
            PatternSlot(ShotSize.ECU, CameraMove.STATIC, 0.75, 100, "NONE",
                        priority=8, note="信息受限的细节"),
            PatternSlot(ShotSize.CU, CameraMove.STATIC, 0.85, 85, "A",
                        priority=7, repeatable=True, note="人物尚未察觉"),
            PatternSlot(ShotSize.MS, CameraMove.DOLLY_OUT, 1.15, 35, "A",
                        priority=7, inherit_last_frame=True, note="开始拉开"),
            PatternSlot(ShotSize.ELS, CameraMove.CRANE, 1.75, 24, "ALL",
                        priority=9, note="全貌：给足时间重新解释"),
        ),
    ),
    "intercut_tension": ShotPattern(
        name="intercut_tension",
        label="交叉加速（两条线收紧）",
        rationale=(
            "平行剪辑的张力来自切换频率本身：只要每一轮都比上一轮短，"
            "观众就会预期『要撞上了』，哪怕画面内容没有任何新信息。"
            "所以 duration_ratio 单调递减，最后用一个最短的镜停在撞击前。"
        ),
        min_subjects=2,
        slots=(
            PatternSlot(ShotSize.MS, CameraMove.DOLLY_IN, 1.25, 35, "A",
                        priority=8, repeatable=True, note="线一"),
            PatternSlot(ShotSize.MS, CameraMove.DOLLY_IN, 1.10, 35, "B",
                        priority=8, repeatable=True, note="线二"),
            PatternSlot(ShotSize.MCU, CameraMove.DOLLY_IN, 0.90, 50, "A",
                        priority=7, repeatable=True, note="线一收紧"),
            PatternSlot(ShotSize.MCU, CameraMove.DOLLY_IN, 0.80, 50, "B",
                        priority=7, repeatable=True, note="线二收紧"),
            PatternSlot(ShotSize.ECU, CameraMove.STATIC, 0.60, 100, "A",
                        priority=9, note="停在撞击前"),
        ),
    ),
    "single_take_breath": ShotPattern(
        name="single_take_breath",
        label="长镜喘息",
        rationale=(
            "高强度段之间必须有喘息，否则观众会疲劳到对高潮无感。"
            "喘息段用最少的切和最慢的运镜，把镜长顶到引擎窗口上限 —— "
            "这也是把生成预算从数量转向单镜质量的地方（适合走 HERO 引擎）。"
        ),
        slots=(
            PatternSlot(ShotSize.MLS, CameraMove.STEADICAM, 1.60, 28, "ALL",
                        priority=9, repeatable=True, note="不切，跟着走"),
            PatternSlot(ShotSize.MCU, CameraMove.STATIC, 1.00, 50, "A",
                        priority=6, inherit_last_frame=True, note="停一下"),
        ),
    ),
    "transition_move": ShotPattern(
        name="transition_move",
        label="过场位移",
        rationale=(
            "过场镜的唯一职责是让观众接受『时间/地点变了』。"
            "用一个中性的空间镜 + 一个进入新空间的镜就够，多给一秒都是浪费预算。"
        ),
        slots=(
            PatternSlot(ShotSize.LS, CameraMove.PAN_R, 1.20, 24, "NONE",
                        transition_in=Transition.DISSOLVE, priority=8, note="旧空间离开"),
            PatternSlot(ShotSize.MLS, CameraMove.STEADICAM, 1.00, 28, "ALL",
                        priority=7, repeatable=True, note="进入新空间"),
        ),
    ),
}

# 节拍功能 → 默认模板。没指定 Beat.pattern 时走这张表。
DEFAULT_PATTERN_FOR: dict[BeatFunction, str] = {
    BeatFunction.ESTABLISH: "establish_progression",
    BeatFunction.DIALOGUE: "shot_reverse_shot",
    BeatFunction.ACTION: "action_rhythm",
    BeatFunction.REVEAL: "reveal_pullback",
    BeatFunction.REACTION: "shot_reverse_shot",
    BeatFunction.TRANSITION: "transition_move",
    BeatFunction.CLIMAX: "intercut_tension",
    BeatFunction.RESOLUTION: "single_take_breath",
}


# ================================================================= 轴线

def assign_screen_directions(
    characters: Sequence[CharacterBible] | Sequence[str],
) -> dict[str, str]:
    """给每个角色分配一个**全片固定**的银幕方向。

    为什么固定：`Storyboard.validate_continuity()` 按角色追踪方向，
    同一角色在同一场里翻转就是跳轴。与其在每场戏里现场推方向（推错一次
    后面全错），不如在角色表上一次定死 —— 观众对「这个人总是朝右」的
    空间记忆本来就是跨场建立的。

    分配方式：按角色表顺序交替 l2r / r2l。第一主体朝右（l2r）是惯例，
    因为观众的阅读方向决定了朝右等于「向前/推进」。
    """
    ids = [c.id if isinstance(c, CharacterBible) else str(c) for c in characters]
    return {cid: ("l2r" if i % 2 == 0 else "r2l") for i, cid in enumerate(ids)}


def _direction_for(subject_ids: Sequence[str], directions: Mapping[str, str]) -> str:
    """单主体镜取该角色的固定方向；多主体或空镜一律 neutral。

    多主体镜必须 neutral：一个镜里两个角色方向相反，`screen_direction` 只有一个值，
    写成任何一方都会污染另一方的追踪状态，让下一镜被误判成跳轴。
    """
    if len(subject_ids) == 1:
        return directions.get(subject_ids[0], "neutral")
    return "neutral"


# ================================================================= 套模板

def _expand_slots(pattern: ShotPattern, n: int) -> list[PatternSlot]:
    """把模板的槽位拉伸/压缩到 n 个。

    拉伸走「可重复槽循环」而不是等比复制全部槽：正反打加镜应该加对切，
    不应该把开场的建立镜也复制一份（观众不需要看两次同一个双人镜）。
    压缩按 priority 保高价值槽，并保持模板内的原始顺序 —— 顺序就是镜头语法本身。
    """
    slots = list(pattern.slots)
    if n <= 0:
        return []
    if n == len(slots):
        return slots
    if n < len(slots):
        keep = sorted(
            sorted(range(len(slots)), key=lambda i: slots[i].priority, reverse=True)[:n]
        )
        return [slots[i] for i in keep]

    loopable = [s for s in slots if s.repeatable] or slots
    out = list(slots)
    i = 0
    # 插在倒数第一个槽之前：模板的最后一个槽通常是收束镜，必须留在末尾。
    tail = out.pop() if len(out) > 1 else None
    while len(out) + (1 if tail else 0) < n:
        out.append(loopable[i % len(loopable)])
        i += 1
    if tail is not None:
        out.append(tail)
    return out


def _distribute_durations(
    slots: Sequence[PatternSlot], total_s: float, lo: float, hi: float
) -> list[float]:
    """按槽位比例切分总时长，夹进引擎窗口后把残差补回去。

    夹窗口会破坏总和（被夹的镜吐出或吃掉时间），所以必须做一次残差再分配，
    否则 `Scene.duration_s()` 会系统性偏离 `plan_beats` 给的预算，
    多个场景累积起来就是整片时长跑偏。
    """
    if not slots:
        return []
    ratios = [s.duration_ratio for s in slots]
    rsum = sum(ratios) or 1.0
    raw = [total_s * r / rsum for r in ratios]
    out = [max(lo, min(hi, _quantize(v))) for v in raw]

    # 残差再分配：每轮把差额摊到还有余量的镜上，最多 8 轮避免病态输入下死循环。
    for _ in range(8):
        diff = total_s - sum(out)
        if abs(diff) < _QUANT_S / 2:
            break
        room = [
            (hi - out[i]) if diff > 0 else (out[i] - lo)
            for i in range(len(out))
        ]
        if sum(room) < _QUANT_S / 2:
            break
        step = math.copysign(_QUANT_S, diff)
        for i in sorted(range(len(out)), key=lambda k: room[k], reverse=True):
            if abs(total_s - sum(out)) < _QUANT_S / 2:
                break
            if lo <= out[i] + step <= hi:
                out[i] = round(out[i] + step, 3)
    return [round(v, 2) for v in out]


def _resolve_subjects(role: SlotRole, cast_ids: Sequence[str]) -> list[str]:
    """把模板里的符号角色位解析成具体角色 id。"""
    if role == "NONE" or not cast_ids:
        return []
    if role == "ALL":
        return list(cast_ids)
    if role == "A":
        return [cast_ids[0]]
    if role == "B":
        return [cast_ids[1] if len(cast_ids) > 1 else cast_ids[0]]
    return [role] if role in cast_ids else []


def apply_pattern(
    scene: Scene,
    pattern: ShotPattern | str,
    characters: Sequence[CharacterBible] | Sequence[str],
    *,
    total_duration_s: float | None = None,
    n_shots: int | None = None,
    atomic_shot_s: float = 8.0,
    directions: Mapping[str, str] | None = None,
    beat: Beat | None = None,
    prev_shot_id: str | None = None,
    start_index: int = 1,
    seed_base: int = 1000,
    style: str = "",
) -> list[Shot]:
    """按模板给一场戏生成镜头列表。

    自动处理三件容易出错的事：

    1. **轴线**：单主体镜取该角色全片固定方向，多主体镜 neutral（见
       `_direction_for` 的注释）。正反打里 A、B 方向相反是对的，
       `Storyboard.validate_continuity()` 按角色追踪，不会误报。
    2. **连续性**：prev/next 双向指针补全；模板声明 `inherit_last_frame` 的槽
       只有在真的有前镜时才生效（声明继承却没有前镜会被 validate_continuity 判错）；
       OVERLAP_BLEND 转场自动给 overlap_frames，MATCH_CUT 自动补 match_on
       （没有匹配元素的 match cut 在拼接器那边等于硬切，还会骗过跳轴检查）。
    3. **时长**：按槽位比例切分并夹进引擎窗口，残差再分配保住总预算。

    `scene` 只读取元信息（id/location/time_of_day/base_grade）；返回的镜头列表
    由调用方决定是否写回 `scene.shots`，这样同一场戏可以试多个模板做 A/B。
    """
    pat = SHOT_PATTERNS[pattern] if isinstance(pattern, str) else pattern
    cast_ids = [c.id if isinstance(c, CharacterBible) else str(c) for c in characters]
    dirs = dict(directions) if directions is not None else assign_screen_directions(cast_ids)
    lo, hi = shot_length_window(atomic_shot_s)

    if n_shots is None:
        n_shots = len(pat.slots)
    if total_duration_s is None:
        total_duration_s = (scene.duration_s()
                            or n_shots * _base_shot_length(
                                beat or Beat(id=scene.id), lo, hi))

    slots = _expand_slots(pat, n_shots)
    durations = _distribute_durations(slots, total_duration_s, lo, hi)

    if pat.min_subjects and len(cast_ids) < pat.min_subjects:
        # 不抛异常：模板不匹配是常见情况（两人对话模板套到独角戏上），
        # 降级成单主体仍能出片，但必须留痕，否则没人知道成片里少了反打。
        logger.warning(
            "模板 %s 需要 %d 个主体，本场只有 %d 个；A/B 位将退化到同一角色",
            pat.name, pat.min_subjects, len(cast_ids),
        )

    dialogue_queue = list(beat.dialogue) if beat else []
    shots: list[Shot] = []
    for k, (slot, dur) in enumerate(zip(slots, durations)):
        sid = f"{scene.id}_s{start_index + k:02d}"
        subs = _resolve_subjects(slot.role, cast_ids)

        # 对白按「该角色的第一个在场镜」派发：让台词落在能看到说话人的镜上，
        # 音频先行（audio_first）算口型时才有画面可对。派不出去的留到下一镜。
        line: list[DialogueLine] = []
        for j, dl in enumerate(dialogue_queue):
            if dl.speaker_id in subs:
                line = [dialogue_queue.pop(j)]
                break

        trans = slot.transition_in
        # 场内第一镜不该用 FADE_IN 以外的入场转场覆盖调用方的意图；
        # 但模板里声明的 FADE_IN 只在整场开头有意义，中途淡入是失误。
        if k > 0 and trans in (Transition.FADE_IN, Transition.FADE_OUT):
            trans = Transition.CUT

        prev = shots[-1].id if shots else prev_shot_id
        inherit = slot.inherit_last_frame and prev is not None

        match_on = slot.match_on
        if trans is Transition.MATCH_CUT and not match_on:
            match_on = "构图/动作匹配（待美术确认）"

        shots.append(
            Shot(
                id=sid,
                scene_id=scene.id,
                index=start_index + k,
                duration_s=dur,
                shot_size=slot.shot_size,
                camera_move=slot.camera_move,
                lens_mm=slot.lens_mm,
                aperture="f/2.0" if slot.lens_mm >= 50 else "f/2.8",
                subject_ids=subs,
                action=(beat.synopsis if beat else scene.synopsis) or slot.note,
                environment=scene.location,
                lighting=(beat.lighting if beat else "") or "",
                mood=(beat.mood if beat else "") or "",
                style=style,
                dialogue=line,
                sfx=list(beat.sfx) if beat and k == 0 else [],
                transition_in=trans,
                content_rating=beat.content_rating if beat else ContentRating.G,
                engine_hint=_engine_hint_for(slot, beat),
                seed=seed_base + start_index + k,
                grade=scene.base_grade.model_copy(),
                continuity=Continuity(
                    prev_shot_id=prev,
                    inherit_last_frame=inherit,
                    emit_last_frame=True,
                    overlap_frames=6 if trans is Transition.OVERLAP_BLEND else 0,
                    screen_direction=_direction_for(subs, dirs),
                    match_on=match_on,
                ),
            )
        )

    # 回填 next 指针（双向指针让 chain.py 不必反查）
    for a, b in zip(shots, shots[1:]):
        a.continuity.next_shot_id = b.id

    if dialogue_queue:
        logger.warning(
            "场 %s 有 %d 条对白没有派发到镜头（说话人不在任何镜的 subject_ids 里）",
            scene.id, len(dialogue_queue),
        )
    return shots


def _engine_hint_for(slot: PatternSlot, beat: Beat | None) -> EngineHint:
    """引擎倾向：把贵的算力花在观众真正会盯着看的镜上。

    判据是「该镜的缺陷是否可修」：大远景和特写一旦崩了后期救不回来
    （远景崩的是结构，特写崩的是脸），所以走 HERO；
    插入镜/空镜内容简单且可重抽，走 OPEN 自建引擎省钱。
    """
    if beat and beat.function is BeatFunction.CLIMAX:
        return EngineHint.HERO
    if slot.shot_size in (ShotSize.ELS, ShotSize.ECU):
        return EngineHint.HERO
    if slot.shot_size is ShotSize.INSERT:
        return EngineHint.OPEN
    if slot.priority >= 8:
        return EngineHint.OFFICIAL
    return EngineHint.AUTO


def beat_sheet_to_scenes(
    beats: Sequence[Beat],
    *,
    target_duration_s: float,
    atomic_shot_s: float = 8.0,
    characters: Sequence[CharacterBible] | Sequence[str] = (),
    directions: Mapping[str, str] | None = None,
    style: str = "",
    allocations_out: list[BeatAllocation] | None = None,
) -> list[Scene]:
    """节拍表 → 场景与镜头。一拍一场，镜头由该拍的默认/指定模板生成。

    时长与镜头数**不是均匀切**，分配规则见 `plan_beats()` 的 docstring：
    屏幕时间按信息密度分，切多少刀按情绪强度算，最后整体缩放对齐目标时长。

    `allocations_out` 传进来可以拿到每一拍的分配理由（用于打分镜方案说明）。
    """
    allocs = plan_beats(beats, target_duration_s=target_duration_s,
                        atomic_shot_s=atomic_shot_s)
    if allocations_out is not None:
        allocations_out.extend(allocs)

    all_ids = [c.id if isinstance(c, CharacterBible) else str(c) for c in characters]
    dirs = dict(directions) if directions is not None else assign_screen_directions(all_ids)

    scenes: list[Scene] = []
    prev_id: str | None = None
    for a in allocs:
        b = a.beat
        scene = Scene(
            id=b.id,
            title=b.title,
            synopsis=b.synopsis,
            location=b.location,
            time_of_day=b.time_of_day,
            base_grade=b.base_grade.model_copy(),
        )
        # 本场的 cast 顺序决定 A/B 位；未在 characters 里出现的 id 也照收，
        # 由 validate_continuity 去报「角色不在角色圣经里」，不在这里吞掉。
        cast = b.characters or all_ids
        pat_name = b.pattern or DEFAULT_PATTERN_FOR.get(b.function, "shot_reverse_shot")
        scene.shots = apply_pattern(
            scene,
            SHOT_PATTERNS[pat_name],
            cast,
            total_duration_s=a.duration_s,
            n_shots=a.n_shots,
            atomic_shot_s=atomic_shot_s,
            directions=dirs,
            beat=b,
            prev_shot_id=prev_id,
            seed_base=1000 + 100 * len(scenes),
            style=style,
        )
        if scene.shots:
            prev_id = scene.shots[-1].id
        scenes.append(scene)

    # 跨场指针：上一场尾镜的 next 指向下一场首镜，chain.py 才能连着续写。
    for s1, s2 in zip(scenes, scenes[1:]):
        if s1.shots and s2.shots:
            s1.shots[-1].continuity.next_shot_id = s2.shots[0].id
    return scenes


# ================================================================= LLM 提示词

@dataclass
class StoryboardConstraints:
    """喂给 LLM 的硬约束。全部会被写成提示词里的显式条款。"""

    target_duration_s: float = 300.0
    atomic_shot_s: float = 8.0
    n_scenes: int = 3
    max_shots: int = 40
    language: str = "zh"
    style_bible: str = ""
    global_negative: str = ""
    allowed_ratings: tuple[str, ...] = ("g", "pg13")
    resolution: tuple[int, int] = (1280, 720)
    fps: int = 24
    extra_notes: tuple[str, ...] = ()


def _enum_vocab(enum_cls: type[Enum]) -> str:
    """把枚举打成 `NAME=value` 的词表。

    两边都给：LLM 更容易吐出 NAME（ECU、DOLLY_IN 这类术语它更熟），
    而 value 才是 JSON 里的合法值；`validate_and_repair` 两种都认。
    """
    return "、".join(f"{m.name}={m.value}" for m in enum_cls)


def _pruned_schema() -> dict[str, Any]:
    """裁剪版 JSON Schema：只留 LLM 真正要填的结构。

    完整 `schema.json_schema()` 里有大量产线回填字段（provider_used / job_id /
    render_uri / qc_score）和参考位结构 —— 这些由后续工位写入，
    让 LLM 看见只会诱导它编造 URI。整份塞进提示词还要烧掉几千 token，
    而 token 预算应该花在剧本内容上。需要完整版时传 full_schema=True。
    """
    full = json_schema()
    defs = full.get("$defs", {})
    keep = ("Shot", "Scene", "Continuity", "DialogueLine", "Grade")
    drop_shot_fields = {
        "refs", "provider_used", "job_id", "render_uri", "qc_score", "takes",
    }
    pruned: dict[str, Any] = {}
    for name in keep:
        node = defs.get(name)
        if not node:
            continue
        node = json.loads(json.dumps(node))     # 深拷贝，别改到 pydantic 的缓存
        props = node.get("properties", {})
        if name == "Shot":
            for f in drop_shot_fields:
                props.pop(f, None)
        pruned[name] = node
    return pruned


def llm_prompt_for_storyboard(
    logline: str,
    characters: Sequence[CharacterBible],
    constraints: StoryboardConstraints | None = None,
    *,
    full_schema: bool = False,
    suggest_patterns: bool = True,
) -> str:
    """生成可直接喂给 LLM 的结构化分镜提示词。

    提示词里把三样东西写死，因为这三样是 LLM 最爱自由发挥的地方：
    枚举值（它会编 "medium-wide"）、时长窗口（它会写 30s 一个镜）、
    轴线规则（它完全不知道 screen_direction 按角色追踪）。
    """
    c = constraints or StoryboardConstraints()
    lo, hi = shot_length_window(c.atomic_shot_s)
    est_shots = int(round(c.target_duration_s / c.atomic_shot_s))

    cast_lines = []
    for ch in characters:
        cast_lines.append(
            f"  - id={ch.id} 名字={ch.name} 年龄声明={ch.age_statement}\n"
            f"    人设：{ch.persona or '（未填）'}\n"
            f"    外形锚点：{ch.appearance.to_prompt() or '（未填）'}"
        )

    pattern_lines = []
    if suggest_patterns:
        for p in SHOT_PATTERNS.values():
            sizes = "→".join(s.shot_size.name for s in p.slots)
            pattern_lines.append(f"  - {p.name}（{p.label}）：{sizes}\n    为什么：{p.rationale}")

    schema_obj = json_schema() if full_schema else _pruned_schema()
    schema_txt = json.dumps(schema_obj, ensure_ascii=False, indent=2)

    parts: list[str] = [
        "你是一位分镜导演。请把下面的故事梗概拆成可直接投产的分镜 JSON。",
        "",
        "# 内容策略（不可协商）",
        "- 只写虚构的成年角色。每个角色的 age_statement 必须显式写明「虚构」且年龄 ≥ 18。",
        "- 不写任何露骨性内容；不写规避平台审核的表达。",
        f"- 每个镜的 content_rating 只能取：{', '.join(c.allowed_ratings)}。",
        "",
        "# 故事",
        f"logline：{logline}",
        "",
        "# 角色（只能用这些 id，禁止新增）",
        *cast_lines,
        "",
        "# 硬约束",
        f"- 目标总时长 {c.target_duration_s:.0f}s；场景数约 {c.n_scenes}；镜头总数 ≤ {c.max_shots}（参考值 {est_shots}）。",
        f"- 单镜 duration_s 必须落在 [{lo:.1f}, {hi:.1f}] 秒。这是视频生成引擎的原子窗口，"
        f"超出的镜无法一次生成完（硬边界 {HARD_MIN_SHOT_S:.0f}–{HARD_MAX_SHOT_S:.0f}s）。",
        f"- 交付 {c.resolution[0]}x{c.resolution[1]} @ {c.fps}fps。",
        f"- 对白语言：{c.language}。每条对白 ≤ 该镜时长 × 4 个字（中文约 4 字/秒）。",
        f"- 全片画风：{c.style_bible or '（由你提议，但必须全片统一，写进 style_bible）'}",
        f"- 全局负面词：{c.global_negative or '（可留空）'}",
        *[f"- {n}" for n in c.extra_notes],
        "",
        "# 镜头语法枚举（只能用这些值；写 NAME 或 value 都可以，别自创）",
        f"- shot_size：{_enum_vocab(ShotSize)}",
        f"- camera_move：{_enum_vocab(CameraMove)}",
        f"- transition_in：{_enum_vocab(Transition)}",
        f"- content_rating：{_enum_vocab(ContentRating)}",
        f"- engine_hint：{_enum_vocab(EngineHint)}",
        "",
        "# 节奏要求（时长不要平均分）",
        "- 建置与收束段用长镜（靠近窗口上限），动作与高潮段用短镜（靠近下限）。",
        "- 一段戏里连续三个镜用同一个景别 = 节奏死板，必须避免。",
        "- 每场至少一个宽镜（LS/ELS/FS）交代空间，至少一个近景（CU/MCU）承载情绪。",
        "",
        "# 轴线规则（最常见的错误，请逐条照做）",
        "- continuity.screen_direction 只能是 l2r / r2l / neutral。",
        "- 它是**按角色追踪**的：同一个角色在同一场里方向不能翻转。",
        "- 正反打里 A 朝右、B 朝左是**正确**的，不是跳轴。",
        "- 一个镜里出现两个及以上角色时，screen_direction 必须写 neutral"
        "（一个字段装不下两个方向，写谁都会让另一个角色被判跳轴）。",
        "- 确实需要翻转时，必须在 continuity.match_on 里写明匹配元素（形状/动作/颜色）。",
        "",
        "# 连续性字段",
        "- continuity.prev_shot_id / next_shot_id 必须指向真实存在的镜头 id。",
        "- inherit_last_frame=true 表示本镜首帧接上一镜尾帧，只能在有 prev_shot_id 时用；"
        "适合同一动作的连续段，不适合跨场景。",
        "- transition_in=match_cut 时必须填 match_on，否则拼接器按硬切处理。",
        "",
    ]

    if pattern_lines:
        parts += [
            "# 可用的镜头组合模板（推荐直接套用，也可以自己组）",
            *pattern_lines,
            "",
        ]

    parts += [
        "# 输出格式",
        "只输出一个 JSON 对象，不要 Markdown 代码围栏，不要任何解释文字。",
        "顶层字段：project, episode, logline, target_duration_s, characters, scenes, "
        "style_bible, global_negative。",
        "characters 直接复用上面给出的角色（原样抄 id / name / age_statement 即可）。",
        "",
        "# JSON Schema（scenes / shots 部分，按此填写）",
        "```json",
        schema_txt,
        "```",
    ]
    return "\n".join(parts)


# ================================================================= 校验与修复

def _alias_table(enum_cls: type[Enum], extra: Mapping[str, Enum] | None = None
                 ) -> dict[str, Enum]:
    """为一个枚举建「宽松字符串 → 成员」映射。

    注意 Transition.FADE_IN 与 FADE_OUT 的 value 都是 "fadeblack"：
    按 value 建表会撞键。这里先按 name 建（name 唯一），value 只在没撞时写入，
    所以 "fadeblack" 解析成 FADE_IN —— 对拼接器而言两者行为一致，
    真正的区分在于它出现在片头还是片尾，那是语义不是取值。
    """
    table: dict[str, Enum] = {}
    for m in enum_cls:
        for key in (m.name, m.name.replace("_", " "), m.name.replace("_", "")):
            table.setdefault(key.lower(), m)
    for m in enum_cls:
        v = str(m.value)
        for key in (v, v.replace("-", " "), v.replace("-", ""), v.replace(" ", "")):
            table.setdefault(key.lower(), m)
    for k, v in (extra or {}).items():
        table[k.lower()] = v
    return table


_SHOT_SIZE_ALIASES = _alias_table(ShotSize, {
    "closeup": ShotSize.CU, "close up": ShotSize.CU, "特写": ShotSize.CU,
    "大特写": ShotSize.ECU, "extreme closeup": ShotSize.ECU,
    "近景": ShotSize.MCU, "中近景": ShotSize.MCU,
    "中景": ShotSize.MS, "medium": ShotSize.MS, "mid": ShotSize.MS,
    "中远景": ShotSize.MLS, "medium wide": ShotSize.MLS, "mw": ShotSize.MLS,
    "全景": ShotSize.FS, "wide": ShotSize.FS, "full": ShotSize.FS,
    "远景": ShotSize.LS, "long": ShotSize.LS,
    "大远景": ShotSize.ELS, "extreme wide": ShotSize.ELS, "establishing": ShotSize.ELS,
    "过肩": ShotSize.OTS, "over the shoulder": ShotSize.OTS,
    "主观": ShotSize.POV, "pov shot": ShotSize.POV,
    "插入": ShotSize.INSERT, "cutaway": ShotSize.INSERT, "detail": ShotSize.INSERT,
    "双人": ShotSize.TWO, "两人": ShotSize.TWO,
})

_CAMERA_ALIASES = _alias_table(CameraMove, {
    "static": CameraMove.STATIC, "fixed": CameraMove.STATIC,
    "locked": CameraMove.STATIC, "locked off": CameraMove.STATIC, "固定": CameraMove.STATIC,
    "pan left": CameraMove.PAN_L, "pan right": CameraMove.PAN_R,
    "tilt up": CameraMove.TILT_U, "tilt down": CameraMove.TILT_D,
    "dolly in": CameraMove.DOLLY_IN, "push in": CameraMove.DOLLY_IN, "推": CameraMove.DOLLY_IN,
    "dolly out": CameraMove.DOLLY_OUT, "pull out": CameraMove.DOLLY_OUT,
    "pull back": CameraMove.DOLLY_OUT, "拉": CameraMove.DOLLY_OUT,
    "truck left": CameraMove.TRUCK_L, "truck right": CameraMove.TRUCK_R,
    "track left": CameraMove.TRUCK_L, "track right": CameraMove.TRUCK_R,
    "orbit": CameraMove.ORBIT_L, "arc": CameraMove.ORBIT_L,
    "zoom in": CameraMove.ZOOM_IN, "zoom out": CameraMove.ZOOM_OUT,
    "handheld": CameraMove.HANDHELD, "手持": CameraMove.HANDHELD,
    "steadicam": CameraMove.STEADICAM, "follow": CameraMove.STEADICAM,
    "crane": CameraMove.CRANE, "jib": CameraMove.CRANE,
    "whip pan": CameraMove.WHIP, "甩": CameraMove.WHIP,
    "dolly zoom": CameraMove.PUSH_PULL, "vertigo": CameraMove.PUSH_PULL,
})

_TRANSITION_ALIASES = _alias_table(Transition, {
    "hard cut": Transition.CUT, "硬切": Transition.CUT, "切": Transition.CUT,
    "cross dissolve": Transition.DISSOLVE, "叠化": Transition.DISSOLVE,
    "fade": Transition.FADE_IN, "fade in": Transition.FADE_IN,
    "fade out": Transition.FADE_OUT, "fade to black": Transition.FADE_OUT,
    "淡入": Transition.FADE_IN, "淡出": Transition.FADE_OUT,
    "wipe": Transition.WIPE_L, "slide": Transition.SLIDE_L,
    "match cut": Transition.MATCH_CUT, "匹配剪辑": Transition.MATCH_CUT,
    "overlap": Transition.OVERLAP_BLEND, "blend": Transition.OVERLAP_BLEND,
})

_RATING_ALIASES = _alias_table(ContentRating, {
    "pg-13": ContentRating.PG13, "pg": ContentRating.PG13,
    "r": ContentRating.R_VIOLENCE, "violence": ContentRating.R_VIOLENCE,
    "suggestive": ContentRating.R_SUGGESTIVE,
})

_HINT_ALIASES = _alias_table(EngineHint, {
    "any": EngineHint.AUTO, "default": EngineHint.AUTO,
    "closed": EngineHint.OFFICIAL, "api": EngineHint.OFFICIAL,
    "opensource": EngineHint.OPEN, "local": EngineHint.OPEN,
})

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
_DIRECTIONS = ("l2r", "r2l", "neutral")

# 未声明年龄时填入的占位串。故意写成会被 compliance._age_problem 判不合格的形式：
# 这里**不能**替 LLM 编一个「成年」声明 —— age_statement 是合规材料，
# 由产线凭空生成等于伪造，必须停在人工补录这一步。
_AGE_PLACEHOLDER = "未声明（需人工补录：必须写明虚构角色且年龄 ≥ 18）"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = [ln for ln in t.splitlines() if not _FENCE_RE.match(ln) or ln.strip().strip("`")]
        t = "\n".join(ln for ln in t.splitlines()[1:] if not ln.strip().startswith("```"))
    # 容错：模型常在 JSON 前后带一句话，截取第一个 { 到最后一个 }
    i, j = t.find("{"), t.rfind("}")
    return t[i:j + 1] if i >= 0 and j > i else t


def _coerce_enum(raw: Any, table: Mapping[str, Enum], default: Enum,
                 label: str, where: str, report: list[str]) -> Enum:
    if raw is None:
        return default
    if isinstance(raw, Enum):
        return raw
    key = str(raw).strip().lower()
    hit = table.get(key) or table.get(key.replace("-", " ")) or table.get(key.replace("_", " "))
    if hit is None:
        report.append(f"[修复] {where}: {label} 值 {raw!r} 不是合法枚举，回落到 {default.name}")
        return default
    if str(hit.value) != str(raw) and hit.name != str(raw):
        report.append(f"[修复] {where}: {label} {raw!r} → {hit.name}")
    return hit


def validate_and_repair(
    raw_json: str | bytes | Mapping[str, Any],
    *,
    atomic_shot_s: float = 8.0,
    known_characters: Sequence[CharacterBible] = (),
) -> tuple[Storyboard, list[str]]:
    """把 LLM 吐出的分镜收拾成合法 Storyboard，并如实报告改了什么。

    修复清单（按发生频率排序，都是实测踩过的）：
      1. Markdown 围栏 / JSON 前后的解释文字 → 截取
      2. 缺 id / scene_id / index → 按位置补
      3. 枚举值自创（"medium-wide"、"push in"、"淡入"）→ 别名表映射
      4. duration_s 超出引擎窗口 → 夹回 [lo, hi]
      5. subject_ids 引用不存在的角色 → 剔除
      6. prev/next 指针指向不存在的镜 / 与实际顺序不符 → 按实际顺序重建
      7. inherit_last_frame 但没有前镜 → 关掉
      8. 跳轴：同角色在同一场里方向翻转 → 统一回该角色首次出现的方向
      9. 多主体镜写了非 neutral 方向 → 改 neutral（一个字段装不下两个方向）

    **不修**的两类（只报告）：
      - content_rating = blocked：把它降级等于替内容策略做决定，必须人工看；
      - 缺失的 age_statement：那是合规材料，产线凭空生成等于伪造。
    两类都会让 `compliance.compliance_check()` 在发片前拦下。
    """
    report: list[str] = []
    lo, hi = shot_length_window(atomic_shot_s)

    # --- 解析
    if isinstance(raw_json, Mapping):
        data: dict[str, Any] = json.loads(json.dumps(dict(raw_json), default=str))
    else:
        text = raw_json.decode("utf-8") if isinstance(raw_json, bytes) else raw_json
        cleaned = _strip_fences(text)
        if cleaned != text.strip():
            report.append("[修复] 输入含 Markdown 围栏或额外文字，已截取 JSON 主体")
        data = json.loads(cleaned)

    # --- 角色
    known = {c.id: c for c in known_characters}
    chars_raw = data.get("characters") or []
    characters: list[CharacterBible] = []
    for i, cr in enumerate(chars_raw):
        if not isinstance(cr, Mapping):
            report.append(f"[丢弃] characters[{i}] 不是对象")
            continue
        cid = str(cr.get("id") or f"char_{i+1:02d}")
        if cid in known:
            # 已有角色圣经以本地为准：LLM 抄回来的外形描述常常漂，
            # 用它覆盖角色圣经会让所有镜的身份锚一起漂。
            characters.append(known[cid])
            continue
        payload = dict(cr)
        payload["id"] = cid
        payload["is_fictional"] = True
        if not str(payload.get("age_statement") or "").strip():
            payload["age_statement"] = _AGE_PLACEHOLDER
            report.append(f"[拒单] 角色 {cid} 缺 age_statement，已填占位串；发片前必须人工补录")
        try:
            characters.append(CharacterBible.model_validate(payload))
        except ValidationError as e:
            report.append(f"[丢弃] 角色 {cid} 结构非法：{e.errors()[0].get('msg', '')}")
    for cid, ch in known.items():
        if not any(c.id == cid for c in characters):
            characters.append(ch)
            report.append(f"[修复] 角色 {cid} 未出现在 LLM 输出里，已从本地角色圣经补入")
    char_ids = {c.id for c in characters}

    # --- 场景与镜头
    scenes: list[Scene] = []
    for si, sr in enumerate(data.get("scenes") or [], 1):
        if not isinstance(sr, Mapping):
            report.append(f"[丢弃] scenes[{si-1}] 不是对象")
            continue
        sc_id = str(sr.get("id") or f"sc{si:02d}")
        shots: list[Shot] = []
        for ki, shr in enumerate(sr.get("shots") or [], 1):
            if not isinstance(shr, Mapping):
                report.append(f"[丢弃] {sc_id}.shots[{ki-1}] 不是对象")
                continue
            where = f"{sc_id}_s{ki:02d}"
            sh = dict(shr)
            sh_id = str(sh.get("id") or where)
            if not sh.get("id"):
                report.append(f"[修复] {where}: 缺 id，按位置补为 {sh_id}")
            if str(sh.get("scene_id") or "") != sc_id:
                sh["scene_id"] = sc_id
            sh["id"] = sh_id
            sh["index"] = ki

            # 时长
            try:
                dur = float(sh.get("duration_s", atomic_shot_s))
            except (TypeError, ValueError):
                dur = atomic_shot_s
                report.append(f"[修复] {sh_id}: duration_s 非数值，回落到 {dur:.1f}s")
            if not (lo - 1e-9 <= dur <= hi + 1e-9):
                fixed = max(lo, min(hi, dur))
                report.append(
                    f"[修复] {sh_id}: duration_s {dur:.1f}s 超出引擎窗口 "
                    f"[{lo:.1f}, {hi:.1f}]，夹到 {fixed:.1f}s"
                )
                dur = fixed
            sh["duration_s"] = _quantize(dur)

            # 枚举
            sh["shot_size"] = _coerce_enum(sh.get("shot_size"), _SHOT_SIZE_ALIASES,
                                           ShotSize.MS, "shot_size", sh_id, report)
            sh["camera_move"] = _coerce_enum(sh.get("camera_move"), _CAMERA_ALIASES,
                                             CameraMove.STATIC, "camera_move", sh_id, report)
            sh["transition_in"] = _coerce_enum(sh.get("transition_in"), _TRANSITION_ALIASES,
                                               Transition.CUT, "transition_in", sh_id, report)
            sh["content_rating"] = _coerce_enum(sh.get("content_rating"), _RATING_ALIASES,
                                                ContentRating.G, "content_rating", sh_id, report)
            sh["engine_hint"] = _coerce_enum(sh.get("engine_hint"), _HINT_ALIASES,
                                             EngineHint.AUTO, "engine_hint", sh_id, report)
            if sh["content_rating"] is ContentRating.BLOCKED:
                report.append(
                    f"[拒单] {sh_id}: content_rating=blocked。不自动降级 —— "
                    "改分级等于替内容策略做决定，必须人工复核后改剧本或删镜。"
                )

            # 角色引用
            subs = [str(x) for x in (sh.get("subject_ids") or [])]
            bad = [x for x in subs if x not in char_ids]
            if bad:
                report.append(f"[修复] {sh_id}: 剔除不存在的角色 {bad}")
                subs = [x for x in subs if x in char_ids]
            sh["subject_ids"] = subs

            # 对白说话人
            dlg = []
            for dl in (sh.get("dialogue") or []):
                if not isinstance(dl, Mapping):
                    continue
                spk = str(dl.get("speaker_id") or "")
                if spk not in char_ids:
                    report.append(f"[丢弃] {sh_id}: 对白说话人 {spk!r} 不在角色圣经里")
                    continue
                if spk not in subs:
                    report.append(f"[告警] {sh_id}: 说话人 {spk} 不在画面里（画外音？）")
                dlg.append({"speaker_id": spk, "text": str(dl.get("text") or ""),
                            "emotion": str(dl.get("emotion") or "neutral")})
            sh["dialogue"] = dlg

            # 连续性容器
            cont = dict(sh.get("continuity") or {})
            d = str(cont.get("screen_direction") or "neutral").lower()
            if d not in _DIRECTIONS:
                report.append(f"[修复] {sh_id}: screen_direction {d!r} 非法，改 neutral")
                d = "neutral"
            if len(subs) != 1 and d != "neutral":
                report.append(
                    f"[修复] {sh_id}: {len(subs)} 个主体的镜不能声明方向 {d}，改 neutral"
                )
                d = "neutral"
            cont["screen_direction"] = d
            sh["continuity"] = cont

            try:
                shots.append(Shot.model_validate(sh))
            except ValidationError as e:
                err = e.errors()[0]
                report.append(
                    f"[丢弃] {sh_id}: 结构非法 {'.'.join(str(x) for x in err.get('loc', ()))}"
                    f" {err.get('msg', '')}"
                )

        if not shots:
            report.append(f"[丢弃] 场景 {sc_id} 没有任何合法镜头")
            continue
        scenes.append(Scene(
            id=sc_id,
            title=str(sr.get("title") or ""),
            synopsis=str(sr.get("synopsis") or ""),
            location=str(sr.get("location") or ""),
            time_of_day=str(sr.get("time_of_day") or "day"),
            base_grade=Grade.model_validate(sr.get("base_grade") or {}),
            shots=shots,
        ))

    # --- 全片指针重建（放在镜头全部落地之后，否则 id 集合还不完整）
    ordered = [s for sc in scenes for s in sc.shots]
    ids = {s.id for s in ordered}
    for a, b in zip(ordered, ordered[1:]):
        if a.continuity.next_shot_id not in (None, b.id):
            report.append(f"[修复] {a.id}: next_shot_id 与实际顺序不符，重建为 {b.id}")
        a.continuity.next_shot_id = b.id
    for prev, cur in zip([None, *ordered], ordered):
        want = prev.id if prev else None
        if cur.continuity.prev_shot_id not in (None, want):
            old = cur.continuity.prev_shot_id
            tag = "指向不存在的镜" if old not in ids else "与实际顺序不符"
            report.append(f"[修复] {cur.id}: prev_shot_id {old!r} {tag}，重建为 {want!r}")
        cur.continuity.prev_shot_id = want
        if cur.continuity.inherit_last_frame and want is None:
            report.append(f"[修复] {cur.id}: 声明继承尾帧但它是第一个镜，已关闭")
            cur.continuity.inherit_last_frame = False
    if ordered:
        ordered[-1].continuity.next_shot_id = None

    # --- 跳轴修复：按角色追踪，场内首次出现的方向是基准
    for sc in scenes:
        first_dir: dict[str, str] = {}
        for shot in sc.shots:
            d = shot.continuity.screen_direction
            if d == "neutral" or len(shot.subject_ids) != 1:
                continue
            cid = shot.subject_ids[0]
            base = first_dir.setdefault(cid, d)
            if d != base:
                if shot.continuity.match_on:
                    report.append(
                        f"[告警] {shot.id}: 角色 {cid} 方向翻转（{base}→{d}），"
                        f"已声明匹配元素「{shot.continuity.match_on}」，按有意翻转放行"
                    )
                    first_dir[cid] = d
                else:
                    report.append(
                        f"[修复] {shot.id}: 角色 {cid} 跳轴（{base}→{d}），统一回 {base}"
                    )
                    shot.continuity.screen_direction = base
        # match_cut 缺匹配元素：拼接器会退化成硬切，还会绕过上面的跳轴判据
        for shot in sc.shots:
            if shot.transition_in is Transition.MATCH_CUT and not shot.continuity.match_on:
                shot.continuity.match_on = "构图/动作匹配（待美术确认）"
                report.append(f"[修复] {shot.id}: match_cut 缺 match_on，已补占位说明")

    sb = Storyboard(
        project=str(data.get("project") or "untitled"),
        episode=str(data.get("episode") or "ep01"),
        logline=str(data.get("logline") or ""),
        target_duration_s=float(data.get("target_duration_s") or 300.0),
        characters=characters,
        scenes=scenes,
        style_bible=str(data.get("style_bible") or ""),
        global_negative=str(data.get("global_negative") or ""),
    )

    # --- 时长偏差
    total = sb.duration_s()
    if sb.target_duration_s > 0:
        drift = (total - sb.target_duration_s) / sb.target_duration_s
        if abs(drift) > 0.15:
            report.append(
                f"[告警] 总时长 {total:.1f}s 偏离目标 {sb.target_duration_s:.0f}s "
                f"{drift:+.0%}（不自动缩放：改时长会改节奏，应增删镜头）"
            )

    # --- 交给契约自己再查一遍，确保修完真的干净
    for p in sb.validate_continuity():
        report.append(f"[遗留] {p}")

    logger.info("分镜修复完成：%d 场 / %d 镜 / %.1fs，报告 %d 条",
                len(sb.scenes), len(sb.all_shots()), total, len(report))
    return sb, report


# ================================================================= 节奏报告

# 叙事片平均镜长（ASL）经验区间。查证日期 2026-09-20：
#   - David Bordwell 的统计：1960 年前主流好莱坞 ASL 约 8–11s，近年收敛到 4–6s；
#   - James Cutting（150 部 1935–2010 影片样本）：1930 年代约 12s → 当代约 2.5s；
#   - Cinemetrics / Stephen Follows 的类型均值：动作 4.0s、冒险 5.1s、科幻 6.2s。
# 来源：
#   https://stephenfollows.com/p/many-shots-average-movie
#   https://www.filmmakersacademy.com/glossary/average-shot-length-asl/
#   https://jov.arvojournals.org/article.aspx?articleid=2138011
#
# **但这些数字不能直接照抄到 AI 产线上**：生成引擎的原子窗口下限是 4–5s
# （providers/base 默认 min 4.0s，dashscope wan 只有固定 5s 档），
# 所以 ASL 低于 5s 意味着每个镜都在生成完就被剪掉一大半 —— 花钱买了不用的帧。
# AI 原生的合理区间因此整体上移，见 AI_NATIVE_ASL。
GENRE_ASL_BANDS: dict[str, tuple[float, float]] = {
    "action": (3.5, 4.5),
    "adventure": (4.5, 5.5),
    "scifi": (5.5, 7.0),
    "drama_contemporary": (4.0, 6.0),
    "drama_classic": (8.0, 11.0),
}
AI_NATIVE_ASL: tuple[float, float] = (6.0, 10.0)

# 以下三组是本产线的**经验区间**，查证日期 2026-09-20 未找到可引用的权威统计，
# 属于自定判据，请用实际成片回归标定后再调整。
WIDE_SIZES = frozenset({ShotSize.ELS, ShotSize.LS, ShotSize.MLS, ShotSize.FS})
MID_SIZES = frozenset({ShotSize.MS, ShotSize.MCU, ShotSize.TWO, ShotSize.OTS, ShotSize.POV})
CLOSE_SIZES = frozenset({ShotSize.CU, ShotSize.ECU, ShotSize.INSERT})
SIZE_MIX_BANDS: dict[str, tuple[float, float]] = {
    "wide": (0.18, 0.40),
    "mid": (0.30, 0.55),
    "close": (0.18, 0.40),
}
STATIC_SHARE_BAND: tuple[float, float] = (0.25, 0.55)
MAX_SAME_SIZE_RUN = 2


def _bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def pacing_report(storyboard: Storyboard, *, genre: str = "scifi") -> str:
    """打一份节奏体检报告：镜长、景别分布、运镜分布 + 对比经验区间的建议。

    报告是给人做决定用的，所以每条建议都带「为什么」和「差多少」，
    不写「节奏偏慢」这种没法照着改的结论。
    """
    shots = storyboard.all_shots()
    if not shots:
        return "（空分镜，无可分析）"

    durs = [s.duration_s for s in shots]
    total = sum(durs)
    asl = total / len(durs)
    msl = statistics.median(durs)
    sd = statistics.pstdev(durs) if len(durs) > 1 else 0.0

    L: list[str] = [
        f"节奏报告 · {storyboard.project}/{storyboard.episode}",
        f"  {len(storyboard.scenes)} 场 / {len(shots)} 镜 / {total:.1f}s"
        f"（{total/60:.2f} 分钟，目标 {storyboard.target_duration_s:.0f}s）",
        f"  平均镜长 ASL {asl:.2f}s   中位 MSL {msl:.2f}s   标准差 {sd:.2f}s"
        f"   区间 {min(durs):.1f}–{max(durs):.1f}s",
        "",
    ]

    # 景别分布
    size_count: dict[ShotSize, int] = {}
    for s in shots:
        size_count[s.shot_size] = size_count.get(s.shot_size, 0) + 1
    L.append("景别分布：")
    for size, n in sorted(size_count.items(), key=lambda kv: -kv[1]):
        p = n / len(shots)
        L.append(f"  {size.name:<7}{n:>3} 镜 {p:>6.1%} {_bar(p)}  {size.value}")
    groups = {
        "wide": sum(n for k, n in size_count.items() if k in WIDE_SIZES),
        "mid": sum(n for k, n in size_count.items() if k in MID_SIZES),
        "close": sum(n for k, n in size_count.items() if k in CLOSE_SIZES),
    }
    L.append("  归组：" + "  ".join(
        f"{k} {v}({v/len(shots):.0%})" for k, v in groups.items()))
    L.append("")

    # 运镜分布
    move_count: dict[CameraMove, int] = {}
    for s in shots:
        move_count[s.camera_move] = move_count.get(s.camera_move, 0) + 1
    L.append("运镜分布：")
    for mv, n in sorted(move_count.items(), key=lambda kv: -kv[1]):
        p = n / len(shots)
        L.append(f"  {mv.name:<11}{n:>3} 镜 {p:>6.1%} {_bar(p)}")
    static_share = move_count.get(CameraMove.STATIC, 0) / len(shots)
    L.append("")

    # 每场镜长
    L.append("分场节奏：")
    for sc in storyboard.scenes:
        if not sc.shots:
            continue
        d = sc.duration_s()
        L.append(
            f"  {sc.id:<10}{len(sc.shots):>3} 镜 {d:>6.1f}s  "
            f"均 {d/len(sc.shots):.1f}s  {sc.title or sc.location}"
        )
    L.append("")

    # ---- 建议
    tips: list[str] = []
    band = GENRE_ASL_BANDS.get(genre, GENRE_ASL_BANDS["drama_contemporary"])
    lo_ai, hi_ai = AI_NATIVE_ASL
    L.append(
        f"对比基准：{genre} 类型真人片 ASL {band[0]:.1f}–{band[1]:.1f}s"
        f"（Bordwell / Cinemetrics 统计，查证 2026-09-20）；"
        f"AI 产线因引擎原子窗口下限 {HARD_MIN_SHOT_S:.0f}s，合理区间上移到 "
        f"{lo_ai:.1f}–{hi_ai:.1f}s"
    )
    if asl < lo_ai:
        tips.append(
            f"ASL {asl:.2f}s 低于 AI 原生下限 {lo_ai:.1f}s：引擎最短也要出 "
            f"{HARD_MIN_SHOT_S:.0f}s，剪掉的部分是白烧的生成额度。"
            "要么合并相邻同机位镜，要么接受这笔浪费换节奏。"
        )
    elif asl > hi_ai:
        tips.append(
            f"ASL {asl:.2f}s 高于 AI 原生上限 {hi_ai:.1f}s：长镜考验引擎的时序稳定性，"
            "超过 10s 的镜容易出现身份漂移和运动累积误差，建议拆成续写链（chain.py）"
            "或在高强度段增加切点。"
        )
    else:
        tips.append(f"ASL {asl:.2f}s 落在 AI 原生区间内。")

    if sd < asl * 0.18:
        tips.append(
            f"镜长标准差 {sd:.2f}s 只有 ASL 的 {sd/asl:.0%}：镜长过于均匀，"
            "说明时长是平均切的而不是按节拍分的。观众感知到的『节奏』本质上是"
            "镜长的**变化率**，全片等长等于没有节奏。"
        )

    for k, (blo, bhi) in SIZE_MIX_BANDS.items():
        p = groups[k] / len(shots)
        if p < blo:
            missing = {"wide": "缺宽镜：观众不知道人在哪，后面的近景失去空间坐标",
                       "mid": "缺中景：宽近直切缺少过渡层，观众每次都要重新定位",
                       "close": "缺近景：情绪没有承载面，对白戏会变成『远远看着两个人说话』"}[k]
            tips.append(f"{k} 占比 {p:.0%} 低于经验下限 {blo:.0%} —— {missing}。")
        elif p > bhi:
            over = {"wide": "宽镜过多：信息稀，观众容易走神，且宽镜是最贵的（细节多、易崩）",
                    "mid": "中景过多：全片像电视剧，缺少景别落差带来的强调",
                    "close": "近景过多：观众失去空间感，且近景对身份一致性最敏感，返工率最高"}[k]
            tips.append(f"{k} 占比 {p:.0%} 高于经验上限 {bhi:.0%} —— {over}。")

    slo, shi = STATIC_SHARE_BAND
    if static_share < slo:
        tips.append(
            f"固定机位仅 {static_share:.0%}（经验下限 {slo:.0%}）：全片都在动，"
            "观众没有静止参照会疲劳；固定机位也是生成最稳、最省重抽的镜型。"
        )
    elif static_share > shi:
        tips.append(
            f"固定机位高达 {static_share:.0%}（经验上限 {shi:.0%}）：画面偏静，"
            "可在建置和高潮段各加一个推/拉，运镜本身就是免费的节奏。"
        )

    # 连续同景别
    for sc in storyboard.scenes:
        run, run_size, start = 0, None, ""
        for s in sc.shots:
            if s.shot_size is run_size:
                run += 1
            else:
                run, run_size, start = 1, s.shot_size, s.id
            if run > MAX_SAME_SIZE_RUN:
                tips.append(
                    f"{sc.id}: 从 {start} 起连续 {run} 个 {run_size.name} —— "
                    "同景别连切超过 2 个，观众会觉得『镜头没变』，节奏读感消失。"
                )
                run = 0

    # 对白覆盖
    dlg_shots = sum(1 for s in shots if s.dialogue)
    if dlg_shots and dlg_shots / len(shots) > 0.75:
        tips.append(
            f"{dlg_shots}/{len(shots)} 个镜带对白：信息几乎全靠台词，"
            "画面沦为配图；给动作和空镜留出位置能显著降低生成失败的代价。"
        )
    over_talk = [
        s.id for s in shots
        if s.dialogue and sum(len(d.text) for d in s.dialogue) > s.duration_s * 4.5
    ]
    if over_talk:
        tips.append(
            f"{len(over_talk)} 个镜的台词字数超过 4.5 字/秒（{', '.join(over_talk[:5])}"
            f"{' …' if len(over_talk) > 5 else ''}）："
            "音频先行产线会按 TTS 实测时长回填，这些镜的画面会不够长。"
        )

    # 超窗口
    out_of_window = [s.id for s in shots
                     if not (HARD_MIN_SHOT_S <= s.duration_s <= HARD_MAX_SHOT_S)]
    if out_of_window:
        tips.append(
            f"{len(out_of_window)} 个镜超出引擎硬窗口 "
            f"{HARD_MIN_SHOT_S:.0f}–{HARD_MAX_SHOT_S:.0f}s："
            f"{', '.join(out_of_window[:5])} —— 必须拆镜或走 chain.py 续写链。"
        )

    L.append("")
    L.append("建议：")
    L += [f"  {i+1}. {t}" for i, t in enumerate(tips)] or ["  （无）"]
    return "\n".join(L)


# ================================================================= 自测

def demo_beat_sheet() -> list[Beat]:
    """一份 6 拍的示例节拍表，强度/密度刻意拉开差距以验证「不是均匀切」。"""
    return [
        Beat(id="sc01", title="接舷", function=BeatFunction.ESTABLISH,
             synopsis="考古学家与仿生助理登上失联的深空站，走廊失重、碎屑漂浮",
             location="废弃空间站 · 失重走廊", time_of_day="night",
             intensity=0.20, density=0.85, characters=["lin_lan", "kai_k7"],
             lighting="practical rim light, volumetric haze", mood="cold curiosity",
             base_grade=Grade(palette="teal shadows, amber practicals",
                              contrast=1.08, saturation=0.92, temperature=-0.25)),
        Beat(id="sc02", title="符号墙", function=BeatFunction.DIALOGUE,
             synopsis="两人在刻满符号的舱壁前争论要不要触碰",
             location="废弃空间站 · 符号墙", time_of_day="night",
             intensity=0.40, density=0.70, characters=["lin_lan", "kai_k7"],
             mood="tense",
             dialogue=[
                 DialogueLine(speaker_id="lin_lan", text="它在回应光。", emotion="curious"),
                 DialogueLine(speaker_id="kai_k7", text="建议不要触碰。上一支队的记录在这里中断。",
                              emotion="neutral"),
                 DialogueLine(speaker_id="lin_lan", text="上一支队没带你。", emotion="dry"),
             ]),
        Beat(id="sc03", title="唤醒", function=BeatFunction.ACTION,
             synopsis="整面舱壁亮起，符号重排，隔离门逐层开启",
             location="废弃空间站 · 隔离门", time_of_day="night",
             intensity=0.90, density=0.45, characters=["lin_lan", "kai_k7"],
             mood="alarm", sfx=["metal groan", "pressure hiss"]),
        Beat(id="sc04", title="环形舱", function=BeatFunction.REVEAL,
             synopsis="镜头拉开，环形舱顶部悬浮着一个缓慢旋转的几何体",
             location="环形舱", time_of_day="night",
             intensity=0.35, density=0.95, characters=["lin_lan", "kai_k7"],
             mood="awe"),
        Beat(id="sc05", title="倒计时", function=BeatFunction.CLIMAX,
             synopsis="助理电量只剩 41 分钟，两人同时走向几何体",
             location="环形舱 · 几何体下方", time_of_day="night",
             intensity=0.95, density=0.55, characters=["lin_lan", "kai_k7"],
             mood="urgent"),
        Beat(id="sc06", title="接触", function=BeatFunction.RESOLUTION,
             synopsis="她独自走向几何体，光覆盖她的脸，站体随后归于黑暗",
             location="环形舱", time_of_day="night",
             intensity=0.25, density=0.40, characters=["lin_lan"],
             mood="quiet"),
    ]


def _selftest() -> None:
    from pathlib import Path

    from .charbible import demo_cast

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cast = demo_cast()[:2]
    # 让示例节拍表的角色 id 与 demo_cast 对齐（自测里角色表是权威）
    beats = demo_beat_sheet()
    id_map = {"lin_lan": cast[0].id, "kai_k7": cast[1].id}
    for b in beats:
        b.characters = [id_map.get(c, c) for c in b.characters]
        for d in b.dialogue:
            d.speaker_id = id_map.get(d.speaker_id, d.speaker_id)

    # ---- 1) 分配不是均匀切
    allocs: list[BeatAllocation] = []
    scenes = beat_sheet_to_scenes(
        beats, target_duration_s=260.0, atomic_shot_s=8.0,
        characters=cast, style="cinematic sci-fi, teal and amber, 35mm grain",
        allocations_out=allocs,
    )
    print("=== 节拍分配 ===")
    for a in allocs:
        print("  " + a.render())

    by_id = {a.beat.id: a for a in allocs}
    assert by_id["sc03"].avg_shot_s < by_id["sc01"].avg_shot_s, "高强度段的镜长必须更短"
    assert by_id["sc05"].avg_shot_s < by_id["sc06"].avg_shot_s, "高潮镜长必须短于收束"
    assert by_id["sc04"].duration_s > by_id["sc03"].duration_s, "高密度段必须拿到更多屏幕时间"
    avgs = [a.avg_shot_s for a in allocs]
    assert max(avgs) - min(avgs) > 1.5, f"镜长没拉开差距，退化成均匀切：{avgs}"

    sb = Storyboard(
        project="deep-archive", episode="ep02",
        logline="两名探索者登上失联的深空考古站，发现符号不是留给后人看的，而是在等人回应。",
        target_duration_s=260.0, characters=cast, scenes=scenes,
        style_bible="cinematic sci-fi, teal and amber, 35mm grain",
        global_negative="extra fingers, warped face, text, watermark",
    )
    dur = sb.duration_s()
    assert abs(dur - 260.0) / 260.0 < 0.10, f"总时长 {dur:.1f}s 偏离目标超过 10%"
    lo, hi = shot_length_window(8.0)
    assert all(lo <= s.duration_s <= hi for s in sb.all_shots()), "有镜头超出引擎窗口"

    # ---- 2) 轴线：正反打两人方向相反，但每个角色全片不翻转
    problems = sb.validate_continuity()
    assert not problems, f"自动生成的分镜居然跳轴了：{problems[:3]}"
    dirs = assign_screen_directions(cast)
    assert dirs[cast[0].id] != dirs[cast[1].id], "正反打两人方向必须相反"
    seen: dict[str, set[str]] = {}
    for s in sb.all_shots():
        if len(s.subject_ids) == 1 and s.continuity.screen_direction != "neutral":
            seen.setdefault(s.subject_ids[0], set()).add(s.continuity.screen_direction)
    assert all(len(v) == 1 for v in seen.values()), f"同一角色出现了多个方向：{seen}"
    multi = [s.id for s in sb.all_shots()
             if len(s.subject_ids) > 1 and s.continuity.screen_direction != "neutral"]
    assert not multi, f"多主体镜不该声明方向：{multi}"

    # 继承尾帧必须有前镜；首镜不得继承
    for s in sb.all_shots():
        assert not (s.continuity.inherit_last_frame and not s.continuity.prev_shot_id)
    assert sb.all_shots()[0].continuity.prev_shot_id is None

    print(f"\n=== 生成结果 ===\n  {len(sb.scenes)} 场 / {len(sb.all_shots())} 镜 / "
          f"{dur:.1f}s；连续性检查通过；指纹样例 {sb.all_shots()[0].fingerprint()}")

    # ---- 3) 模板库自洽
    for name, p in SHOT_PATTERNS.items():
        assert p.name == name and p.slots and p.rationale
        assert p.total_ratio() > 0
        assert any(s.repeatable for s in p.slots), f"{name} 没有可重复槽，无法扩镜"
    sc = Scene(id="x", location="走廊")
    long_shots = apply_pattern(sc, "shot_reverse_shot", cast,
                               total_duration_s=90.0, n_shots=12, atomic_shot_s=8.0)
    assert len(long_shots) == 12
    assert abs(sum(s.duration_s for s in long_shots) - 90.0) < 1.0
    assert long_shots[-1].shot_size is ShotSize.TWO, "扩镜后收束镜必须仍在末尾"
    short_shots = apply_pattern(sc, "shot_reverse_shot", cast, total_duration_s=20.0, n_shots=3)
    assert len(short_shots) == 3 and short_shots[0].shot_size is ShotSize.TWO

    # ---- 4) LLM 提示词
    prompt = llm_prompt_for_storyboard(
        sb.logline, cast,
        StoryboardConstraints(target_duration_s=260.0, atomic_shot_s=8.0, n_scenes=6,
                              style_bible=sb.style_bible, global_negative=sb.global_negative),
    )
    for token in ("extreme close-up", "dolly zoom (vertigo effect)", "match_cut",
                  "screen_direction", "虚构", "JSON Schema", cast[0].id,
                  f"[{lo:.1f}, {hi:.1f}]"):
        assert token in prompt, f"提示词缺少 {token!r}"
    assert "shot_reverse_shot" in prompt and "action_rhythm" in prompt
    assert '"duration_s"' in prompt, "提示词里必须带 schema 字段"
    print(f"\n=== LLM 提示词 ===  {len(prompt)} 字符，约 {len(prompt)//3} token")
    print("\n".join(prompt.splitlines()[:14]))
    print("  …（略）")

    # ---- 5) 坏 JSON 修复
    bad = {
        "project": "broken", "target_duration_s": 60,
        "characters": [{"id": cast[0].id, "name": "冒牌", "age_statement": ""},
                       {"id": "ghost", "name": "幽灵"}],
        "scenes": [{
            "id": "sc01", "title": "坏例子", "location": "走廊",
            "shots": [
                {"duration_s": 40, "shot_size": "medium-wide", "camera_move": "push in",
                 "subject_ids": [cast[0].id], "transition_in": "淡入",
                 "continuity": {"screen_direction": "l2r", "prev_shot_id": "nope",
                                "inherit_last_frame": True}},
                {"id": "sc01_s02", "duration_s": 0.5, "shot_size": "特写",
                 "camera_move": "手持", "subject_ids": [cast[0].id, "nobody"],
                 "transition_in": "match cut",
                 "continuity": {"screen_direction": "r2l"},
                 "dialogue": [{"speaker_id": "nobody", "text": "我不存在"},
                              {"speaker_id": cast[0].id, "text": "这句非常非常长以至于四秒钟根本说不完真的说不完",
                               "emotion": "flat"}]},
                {"id": "sc01_s03", "duration_s": 7, "shot_size": "TWO",
                 "camera_move": "静止", "subject_ids": [cast[0].id, cast[1].id],
                 "content_rating": "blocked",
                 "continuity": {"screen_direction": "l2r", "next_shot_id": "nowhere"}},
            ],
        }],
    }
    raw = "这是我生成的分镜：\n```json\n" + json.dumps(bad, ensure_ascii=False) + "\n```\n希望有用！"
    fixed, rep = validate_and_repair(raw, atomic_shot_s=8.0, known_characters=cast)
    print("\n=== 修复报告 ===")
    for r in rep:
        print("  " + r)

    kinds = [r.split("]")[0] + "]" for r in rep]
    assert "[修复]" in kinds and "[拒单]" in kinds, kinds
    assert any("围栏" in r for r in rep)
    assert any("超出引擎窗口" in r for r in rep)
    assert any("剔除不存在的角色" in r for r in rep)
    assert any("prev_shot_id" in r and "不存在" in r for r in rep)
    assert any("跳轴" in r or "不能声明方向" in r for r in rep)
    assert any("blocked" in r for r in rep)
    assert any("age_statement" in r for r in rep)
    # 本地角色圣经必须压过 LLM 抄回来的版本
    assert fixed.character(cast[0].id) is not None
    assert fixed.character(cast[0].id).name == cast[0].name, "LLM 的角色描述覆盖了角色圣经"
    assert fixed.character(cast[1].id) is not None, "缺席角色未从本地补入"
    for s in fixed.all_shots():
        assert lo <= s.duration_s <= hi
        assert all(cid in {c.id for c in fixed.characters} for cid in s.subject_ids)
        if len(s.subject_ids) != 1:
            assert s.continuity.screen_direction == "neutral"
    assert fixed.all_shots()[0].continuity.prev_shot_id is None
    assert not fixed.all_shots()[0].continuity.inherit_last_frame
    assert fixed.all_shots()[-1].continuity.next_shot_id is None
    # blocked 不被静默降级
    assert any(s.content_rating is ContentRating.BLOCKED for s in fixed.all_shots())

    # 幂等：修好的分镜再修一次不应再产生 [修复]
    again, rep2 = validate_and_repair(
        json.dumps(fixed.model_dump(mode="json"), ensure_ascii=False),
        atomic_shot_s=8.0, known_characters=cast)
    assert not any(r.startswith("[修复]") for r in rep2), f"修复不幂等：{rep2}"
    assert again.duration_s() == fixed.duration_s()

    # ---- 6) 节奏报告 + 与手写 demo 分镜的兼容性
    print("\n=== 节奏报告（本模块生成的分镜）===")
    rpt = pacing_report(sb, genre="scifi")
    print(rpt)
    assert "平均镜长 ASL" in rpt and "景别分布" in rpt and "运镜分布" in rpt
    assert "建议：" in rpt

    demo_path = Path(__file__).resolve().parents[2] / "configs" / "demo_episode.json"
    if demo_path.exists():
        hand = Storyboard.load(demo_path)
        hand_rpt = pacing_report(hand, genre="scifi")
        print("\n=== 节奏报告（手写 demo_episode.json，验证结构兼容）===")
        print("\n".join(hand_rpt.splitlines()[:6]))
        assert len(hand.all_shots()) == 30
        assert not hand.validate_continuity()
        # 本模块生成的结构必须能被同一套消费者吃下：字段集合一致
        gen_keys = set(sb.all_shots()[0].model_dump().keys())
        hand_keys = set(hand.all_shots()[0].model_dump().keys())
        assert gen_keys == hand_keys, gen_keys ^ hand_keys
        # 反过来：手写分镜也能走本模块的修复器且不被改坏
        rt, rrep = validate_and_repair(
            json.dumps(hand.model_dump(mode="json"), ensure_ascii=False),
            atomic_shot_s=8.7, known_characters=hand.characters)
        assert len(rt.all_shots()) == 30, "修复器吃掉了手写分镜的镜头"
        assert not [r for r in rrep if r.startswith("[丢弃]")], rrep
        print(f"  手写分镜往返修复：30 镜保持，报告 {len(rrep)} 条 "
              f"（{sum(1 for r in rrep if r.startswith('[修复]'))} 条修复）")
    else:
        print(f"\n（跳过兼容性检查：{demo_path} 不存在）")

    print("\nstoryboard_gen 自测通过")


if __name__ == "__main__":
    _selftest()
