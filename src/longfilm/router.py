"""引擎路由 —— 双引擎（官方闭源 / 自建开源）调度大脑。

三件事：
1. **内容门禁**（content_gate）：先判这一镜该不该拍、能投给哪类引擎。
   R 级镜头只走自建开源，不是为了绕审核，而是因为官方通道的条款本来就不接这类单，
   投过去必然被拒还连累账号信誉 —— 所以在路由层直接拒单，不做任何规避尝试。
2. **候选打分**（plan）：只读 Capabilities 打分排序。router 里不允许出现
   `if provider.name == "xxx"`，换供应商时只改 Capabilities 声明，不改这里一行。
3. **执行与降级**（execute）：提交 → 轮询 → 按 FailureKind 分流。
   可重试的退避重试同一家，审核/配额/鉴权类立刻换下一家，参数错直接报工单。

成本账本（CostLedger）与决策解释（RoutingPlan.explain）是给运营看的：
一集片跑完要能回答「这一镜为什么花了这么多钱、为什么走的是这家」。
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping

from .providers.base import (
    Capabilities,
    FailureKind,
    GenRequest,
    GenResult,
    JobStatus,
    ProviderError,
    ProviderRegistry,
    VideoProvider,
)
from .schema import (
    CameraMove,
    ContentRating,
    EngineHint,
    ImageRef,
    RefPack,
    Shot,
    Storyboard,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- 策略


class QualityBias(str, Enum):
    """质量偏好。决定打分权重，不决定候选集合。"""

    QUALITY = "quality"   # 画质优先：英雄镜、成片
    COST = "cost"         # 成本优先：量产集数
    SPEED = "speed"       # 速度优先：预演/给编剧看的粗剪


# 打分权重：画质 / 成本 / 速度 / 契合度。
# 契合度（fit）在任何偏好下都不低于 0.25 —— 一个便宜但要把 8s 砍成 4s 的引擎，
# 省下的钱会在补镜和重拍里加倍还回来。
_WEIGHTS: dict[QualityBias, dict[str, float]] = {
    QualityBias.QUALITY: {"quality": 0.45, "cost": 0.10, "speed": 0.10, "fit": 0.35},
    QualityBias.COST: {"quality": 0.15, "cost": 0.50, "speed": 0.10, "fit": 0.25},
    QualityBias.SPEED: {"quality": 0.15, "cost": 0.10, "speed": 0.50, "fit": 0.25},
}

# 参考图位优先级：数字越小越先保留。身份锚永远第一个进、最后一个被砍。
_ROLE_PRIORITY: dict[str, int] = {
    "identity": 0,
    "first_frame": 1,
    "last_frame": 2,
    "wardrobe": 3,
    "environment": 4,
    "style": 5,
    "lighting": 6,
    "prop": 7,
    "composition": 8,
}


@dataclass(frozen=True)
class ProviderQuota:
    """单家供应商的闸门。None = 不额外设限，沿用 Capabilities 里的声明。"""

    max_concurrency: int | None = None
    max_calls: int | None = None
    max_usd: float | None = None


def _default_rating_kinds() -> dict[ContentRating, frozenset[str]]:
    everything = frozenset({"official", "open", "aggregator"})
    return {
        ContentRating.G: everything,
        ContentRating.PG13: everything,
        # R 级：只投自建开源。聚合商（aggregator）背后仍是官方通道，同样排除。
        ContentRating.R_VIOLENCE: frozenset({"open"}),
        ContentRating.R_SUGGESTIVE: frozenset({"open"}),
        ContentRating.BLOCKED: frozenset(),
    }


@dataclass(frozen=True)
class RoutingPolicy:
    """路由策略：分级 → 引擎类型、质量偏好、预算、配额。

    策略是**数据**不是代码：一集片换一套 policy 就能从「画质优先的成片」
    切成「成本优先的量产」，不用碰 Router。
    """

    rating_kinds: Mapping[ContentRating, frozenset[str]] = field(
        default_factory=_default_rating_kinds
    )
    # 这些分级必须走无审核通道。与 rating_kinds 正交：前者限「哪类引擎」，
    # 后者限「该引擎是否带审核」—— 自建引擎前面也可能挂了审核中间件。
    unmoderated_ratings: frozenset[ContentRating] = frozenset(
        {ContentRating.R_VIOLENCE, ContentRating.R_SUGGESTIVE}
    )
    bias: QualityBias = QualityBias.QUALITY
    episode_budget_usd: float = 120.0
    quotas: Mapping[str, ProviderQuota] = field(default_factory=dict)

    hero_min_tier: int = 4              # 英雄镜的画质门槛
    max_fallbacks: int = 3              # 备选链长度上限，再长不如报工单人工介入
    duration_tolerance_s: float = 2.0   # 时长被 clamp 超过这个值算严重失配

    def allowed_kinds(self, rating: ContentRating) -> frozenset[str]:
        return self.rating_kinds.get(rating, frozenset())

    def requires_unmoderated(self, rating: ContentRating) -> bool:
        return rating in self.unmoderated_ratings

    def quota(self, name: str) -> ProviderQuota:
        return self.quotas.get(name, ProviderQuota())

    def bias_for(self, hint: EngineHint) -> QualityBias:
        if hint is EngineHint.HERO:
            return QualityBias.QUALITY
        if hint is EngineHint.DRAFT:
            return QualityBias.COST
        return self.bias


DEFAULT_POLICY = RoutingPolicy()


# ---------------------------------------------------------------- 内容门禁


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    """门禁结论。allowed=False 就是拒单，不存在「换个说法再试」的分支。"""

    shot_id: str
    rating: ContentRating
    allowed: bool
    allowed_kinds: frozenset[str]
    require_unmoderated: bool
    checks: tuple[GateCheck, ...]

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(c.detail for c in self.checks if not c.passed)

    @property
    def failure_kind(self) -> FailureKind:
        # 分级/策略冲突归 MODERATION，资料缺失归 BAD_REQUEST（要回去改工单）。
        if self.rating is ContentRating.BLOCKED or not self.allowed_kinds:
            return FailureKind.MODERATION
        return FailureKind.BAD_REQUEST

    def check_provider(self, caps: Capabilities) -> str | None:
        """该 provider 是否被门禁放行。返回拒绝原因，None = 放行。"""
        if not self.allowed:
            return f"门禁未放行：{'；'.join(self.blockers)}"
        if caps.kind not in self.allowed_kinds:
            return (
                f"分级 {self.rating.value} 只允许 {sorted(self.allowed_kinds)} 类引擎，"
                f"{caps.name} 是 {caps.kind}"
            )
        if self.require_unmoderated and caps.moderated:
            return (
                f"分级 {self.rating.value} 不得投递到带内容审核的通道"
                f"（{caps.name}.moderated=True）：拒单，不做任何规避"
            )
        return None

    def explain(self) -> str:
        lines = [
            f"内容门禁 {self.shot_id}：{'放行' if self.allowed else '拒单'}"
            f"（分级 {self.rating.value}）"
        ]
        for c in self.checks:
            lines.append(f"  [{'ok' if c.passed else 'NG'}] {c.name}：{c.detail}")
        lines.append(
            f"  允许的引擎类型：{sorted(self.allowed_kinds) or '（无）'}；"
            f"要求无审核通道：{self.require_unmoderated}"
        )
        return "\n".join(lines)


def content_gate(
    shot: Shot, storyboard: Storyboard, policy: RoutingPolicy = DEFAULT_POLICY
) -> GateDecision:
    """内容门禁：分级合法性 + 角色成年声明 + 分级与通道属性的冲突检查。

    本产线只做虚构成年角色，所以「每个出镜角色都能在角色圣经里查到并且写明了
    成年声明」是硬门槛 —— 这条不过，镜头连打分都不进。
    """
    checks: list[GateCheck] = []
    rating = shot.content_rating

    blocked = rating is ContentRating.BLOCKED
    checks.append(
        GateCheck(
            "分级非 blocked",
            not blocked,
            "分级为 blocked，策略禁止生产" if blocked else f"分级 {rating.value}",
        )
    )

    missing: list[str] = []
    for sid in shot.subject_ids:
        ch = storyboard.character(sid)
        if ch is None:
            missing.append(f"{sid}（不在角色圣经里）")
        elif not ch.age_statement.strip():
            missing.append(f"{sid}（age_statement 为空）")
    checks.append(
        GateCheck(
            "角色成年声明",
            not missing,
            "缺失：" + "、".join(missing) if missing else
            f"{len(shot.subject_ids)} 个出镜角色均已声明为虚构成年角色",
        )
    )

    kinds = policy.allowed_kinds(rating)
    checks.append(
        GateCheck(
            "分级有可投引擎类型",
            bool(kinds),
            f"允许 {sorted(kinds)}" if kinds else f"分级 {rating.value} 无任何允许的引擎类型",
        )
    )

    need_unmod = policy.requires_unmoderated(rating)
    checks.append(
        GateCheck(
            "通道审核属性",
            True,
            "该分级必须走无审核的自建通道；投给官方审核通道会被拒单"
            if need_unmod else "该分级可走带审核的官方通道",
        )
    )

    allowed = all(c.passed for c in checks)
    if not allowed:
        log.warning("内容门禁拒单 %s：%s", shot.id, "；".join(
            c.detail for c in checks if not c.passed))
    return GateDecision(
        shot_id=shot.id,
        rating=rating,
        allowed=allowed,
        allowed_kinds=kinds,
        require_unmoderated=need_unmod,
        checks=tuple(checks),
    )


# ---------------------------------------------------------------- 成本账本


class BudgetExceeded(RuntimeError):
    def __init__(self, spent: float, budget: float, attempted: float):
        super().__init__(
            f"预算超限：已花 ${spent:.4f} / 上限 ${budget:.2f}，本次 ${attempted:.4f}"
        )
        self.spent = spent
        self.budget = budget
        self.attempted = attempted


@dataclass
class CostEntry:
    shot_id: str
    provider: str
    duration_s: float
    cost_usd: float
    status: str
    job_id: str = ""
    attempt: int = 1
    ts: float = 0.0


class CostLedger:
    """成本账本。失败的调用也记账 —— 被拒的单同样烧了额度和时间。"""

    def __init__(self, budget_usd: float | None = None) -> None:
        self.budget_usd = budget_usd
        self.entries: list[CostEntry] = []

    def ensure_affordable(self, cost_usd: float) -> None:
        """下单前的闸门。预估超预算就别提交，提交完再抛异常钱已经花了。"""
        if self.budget_usd is None:
            return
        if self.total() + cost_usd > self.budget_usd + 1e-9:
            raise BudgetExceeded(self.total(), self.budget_usd, cost_usd)

    def record(self, entry: CostEntry) -> CostEntry:
        """如实入账，然后才检查是否超支。

        顺序很重要：钱已经花出去了，不能因为要抛异常就把这笔记录丢掉，
        否则账本和对账单永远对不上。
        """
        if not entry.ts:
            entry.ts = time.time()
        self.entries.append(entry)
        if self.budget_usd is not None and self.total() > self.budget_usd + 1e-9:
            raise BudgetExceeded(self.total(), self.budget_usd, entry.cost_usd)
        return entry

    def total(self) -> float:
        return round(sum(e.cost_usd for e in self.entries), 6)

    def by_shot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries:
            out[e.shot_id] = round(out.get(e.shot_id, 0.0) + e.cost_usd, 6)
        return out

    def by_provider(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.entries:
            out[e.provider] = round(out.get(e.provider, 0.0) + e.cost_usd, 6)
        return out

    def calls_by_provider(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            out[e.provider] = out.get(e.provider, 0) + 1
        return out

    def remaining(self) -> float | None:
        return None if self.budget_usd is None else round(self.budget_usd - self.total(), 6)

    def report(self) -> str:
        lines = [f"成本账本：共 {len(self.entries)} 次调用，合计 ${self.total():.4f}"]
        if self.budget_usd is not None:
            lines[0] += f" / 预算 ${self.budget_usd:.2f}（剩余 ${self.remaining():.4f}）"
        for p, c in sorted(self.by_provider().items(), key=lambda kv: -kv[1]):
            lines.append(f"  供应商 {p}: ${c:.4f}（{self.calls_by_provider()[p]} 次）")
        for s, c in sorted(self.by_shot().items()):
            lines.append(f"  镜头 {s}: ${c:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 需求与候选


@dataclass(frozen=True)
class ShotNeeds:
    """从 Shot 抽出来的、与供应商无关的能力需求。打分与编译都读它。"""

    duration_s: float
    fps: int
    resolution: tuple[int, int]
    want_first_frame: bool
    want_last_frame: bool
    want_job_continue: bool
    want_lora: bool
    want_native_audio: bool
    want_camera: bool
    want_seed: bool
    want_negative: bool
    n_images: int
    n_videos: int
    n_audios: int
    image_roles: frozenset[str]


def shot_needs(shot: Shot, storyboard: Storyboard) -> ShotNeeds:
    roles = {r.role for r in shot.refs.images}
    has_lora = any(
        (c := storyboard.character(sid)) is not None and c.lora is not None
        for sid in shot.subject_ids
    )
    return ShotNeeds(
        duration_s=shot.duration_s,
        fps=shot.fps,
        resolution=storyboard.delivery.resolution,
        want_first_frame=shot.continuity.inherit_last_frame or "first_frame" in roles,
        want_last_frame="last_frame" in roles,
        want_job_continue=shot.continuity.extend_from_job is not None,
        want_lora=has_lora,
        # 音频先行时我们自己出对白轨，反而不要引擎自带声；只有非音频先行才需要原生音。
        want_native_audio=bool(shot.dialogue) and not storyboard.audio.audio_first,
        want_camera=shot.camera_move is not CameraMove.STATIC,
        want_seed=shot.seed is not None,
        want_negative=bool(shot.negative_prompt or storyboard.global_negative),
        n_images=len(shot.refs.images),
        n_videos=len(shot.refs.videos),
        n_audios=len(shot.refs.audios),
        image_roles=frozenset(roles),
    )


@dataclass(frozen=True)
class CandidateScore:
    name: str
    kind: str
    total: float
    parts: Mapping[str, float]
    penalties: tuple[str, ...]
    clamped_duration_s: float
    est_cost_usd: float
    est_latency_s: float

    def line(self) -> str:
        p = "，".join(f"{k}={v:.2f}" for k, v in self.parts.items())
        s = f"  {self.name}({self.kind}) 总分 {self.total:.3f} [{p}] " \
            f"时长 {self.clamped_duration_s:g}s 预估 ${self.est_cost_usd:.4f}/" \
            f"{self.est_latency_s:.0f}s"
        if self.penalties:
            s += "\n      扣分：" + "；".join(self.penalties)
        return s


@dataclass
class RoutingPlan:
    """一次路由决策的完整快照。运营拿它对账，工程拿它复盘。"""

    shot_id: str
    gate: GateDecision
    bias: QualityBias
    engine_hint: EngineHint
    primary: str | None
    fallbacks: tuple[str, ...]
    request: GenRequest | None
    scores: tuple[CandidateScore, ...]
    rejected: tuple[tuple[str, str], ...]
    estimated_cost_usd: float
    estimated_latency_s: float
    degradations: tuple[str, ...]

    @property
    def chain(self) -> tuple[str, ...]:
        return ((self.primary,) if self.primary else ()) + self.fallbacks

    @property
    def routable(self) -> bool:
        return self.primary is not None

    def explain(self) -> str:
        lines = [
            f"=== 路由决策 {self.shot_id} ===",
            self.gate.explain(),
            f"偏好：{self.bias.value}（engine_hint={self.engine_hint.value}）",
        ]
        if self.rejected:
            lines.append("出局候选：")
            lines += [f"  - {n}：{why}" for n, why in self.rejected]
        if self.scores:
            lines.append("候选打分（越高越优）：")
            lines += [s.line() for s in self.scores]
        if self.primary:
            top = self.scores[0]
            lines.append(
                f"选中：{self.primary}，因为在 {self.bias.value} 权重下总分最高"
                f"（{top.total:.3f}）"
            )
            lines.append(f"备选链：{' -> '.join(self.fallbacks) or '（无，失败即报工单）'}")
            lines.append(
                f"预估：${self.estimated_cost_usd:.4f} / {self.estimated_latency_s:.0f}s"
            )
        else:
            lines.append("选中：无可用引擎，拒单")
        if self.degradations:
            lines.append("降级记录：")
            lines += [f"  * {d}" for d in self.degradations]
        return "\n".join(lines)


# ---------------------------------------------------------------- Router


def _compose_prompt(shot: Shot, storyboard: Storyboard) -> str:
    """兜底提示词拼装。正式产线由 prompt_os 注入（Router(prompt_fn=...)）。"""
    bits: list[str] = [storyboard.style_bible, shot.style]
    for sid in shot.subject_ids:
        ch = storyboard.character(sid)
        if ch is not None:
            bits.append(ch.identity_prompt())
    bits += [
        shot.action,
        shot.environment,
        shot.lighting,
        shot.mood,
        shot.shot_size.value,
        shot.camera_move.value,
        f"{shot.lens_mm}mm lens, {shot.aperture}",
    ]
    return ", ".join(b.strip() for b in bits if b and b.strip())


def _negative_prompt(shot: Shot, storyboard: Storyboard) -> str:
    bits = [shot.negative_prompt, storyboard.global_negative]
    for sid in shot.subject_ids:
        ch = storyboard.character(sid)
        if ch is not None and ch.negative_prompt:
            bits.append(ch.negative_prompt)
    return ", ".join(b.strip() for b in bits if b and b.strip())


def _fit_refs(refs: RefPack, caps: Capabilities) -> tuple[RefPack, list[str]]:
    """把参考包裁到该家上限。优先调 refpack.fit_to_budget，缺席时用内置优先级裁剪。

    refpack 由并行工位实现，可能还不在树上；这里的兜底保证 router 单独可跑，
    而不是让整条产线卡在一个尚未落地的模块上。
    """
    try:
        from .refpack import fit_to_budget  # type: ignore[attr-defined]

        return (
            fit_to_budget(
                refs,
                max_images=caps.max_ref_images,
                max_videos=caps.max_ref_videos,
                max_audios=caps.max_ref_audios,
                allowed_roles=caps.supported_image_roles or None,
            ),
            [],
        )
    except (ImportError, TypeError) as exc:  # 模块未落地 / 签名尚未对齐
        log.debug("refpack.fit_to_budget 不可用（%s），用 router 内置裁剪器", exc)

    notes: list[str] = []
    imgs = list(refs.images)
    if caps.supported_image_roles:
        kept = [r for r in imgs if r.role in caps.supported_image_roles]
        if len(kept) != len(imgs):
            dropped = sorted({r.role for r in imgs} - set(caps.supported_image_roles))
            notes.append(f"{caps.name} 不支持参考位角色 {dropped}，已丢弃")
        imgs = kept
    imgs.sort(key=lambda r: (_ROLE_PRIORITY.get(r.role, 99), -r.weight))
    if len(imgs) > caps.max_ref_images:
        notes.append(
            f"参考图 {len(imgs)} 张超过 {caps.name} 上限 {caps.max_ref_images} 张，"
            f"按角色优先级砍掉 {[r.role for r in imgs[caps.max_ref_images:]]}"
        )
        imgs = imgs[: caps.max_ref_images]
    vids = list(refs.videos)[: caps.max_ref_videos]
    auds = list(refs.audios)[: caps.max_ref_audios]
    if len(refs.videos) > caps.max_ref_videos:
        notes.append(f"参考视频裁到 {caps.max_ref_videos} 段")
    if len(refs.audios) > caps.max_ref_audios:
        notes.append(f"参考音频裁到 {caps.max_ref_audios} 段")
    return RefPack(images=imgs, videos=vids, audios=auds), notes


def _norm_high_is_good(v: float, lo: float, hi: float) -> float:
    return 1.0 if hi - lo < 1e-9 else (v - lo) / (hi - lo)


def _norm_low_is_good(v: float, lo: float, hi: float) -> float:
    return 1.0 if hi - lo < 1e-9 else 1.0 - (v - lo) / (hi - lo)


class Router:
    """引擎路由器。决策只读 Capabilities，执行只认 FailureKind。"""

    def __init__(
        self,
        registry: ProviderRegistry,
        policy: RoutingPolicy = DEFAULT_POLICY,
        *,
        ledger: CostLedger | None = None,
        prompt_fn: Callable[[Shot, Storyboard], str] = _compose_prompt,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_interval_s: float = 5.0,
        job_timeout_s: float = 900.0,
        backoff_base_s: float = 4.0,
        rng: random.Random | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.ledger = ledger if ledger is not None else CostLedger(policy.episode_budget_usd)
        self.prompt_fn = prompt_fn
        self._sleep = sleeper
        self._clock = clock
        self.poll_interval_s = poll_interval_s
        self.job_timeout_s = job_timeout_s
        self.backoff_base_s = backoff_base_s
        self._rng = rng or random.Random(20260919)
        self._inflight: dict[str, int] = {}

    # ---------------- 门禁

    def content_gate(self, shot: Shot, storyboard: Storyboard) -> GateDecision:
        return content_gate(shot, storyboard, self.policy)

    # ---------------- 打分

    def _effective_concurrency(self, caps: Capabilities) -> int:
        q = self.policy.quota(caps.name)
        return min(caps.max_concurrency, q.max_concurrency or caps.max_concurrency)

    def _hard_reject(
        self, p: VideoProvider, shot: Shot, gate: GateDecision, needs: ShotNeeds
    ) -> str | None:
        caps = p.caps
        reason = gate.check_provider(caps)
        if reason:
            return reason
        hint = shot.engine_hint
        if hint is EngineHint.OFFICIAL and caps.kind != "official":
            return f"engine_hint=official，{caps.name} 是 {caps.kind}"
        if hint is EngineHint.OPEN and caps.kind != "open":
            return f"engine_hint=open，{caps.name} 是 {caps.kind}"
        if hint is EngineHint.HERO and caps.quality_tier < self.policy.hero_min_tier:
            return (
                f"英雄镜要求画质档 >= {self.policy.hero_min_tier}，"
                f"{caps.name} 只有 {caps.quality_tier}"
            )
        q = self.policy.quota(caps.name)
        spent = self.ledger.by_provider().get(caps.name, 0.0)
        calls = self.ledger.calls_by_provider().get(caps.name, 0)
        if q.max_usd is not None and spent >= q.max_usd - 1e-9:
            return f"该供应商配额用尽（已花 ${spent:.4f} / ${q.max_usd:.2f}）"
        if q.max_calls is not None and calls >= q.max_calls:
            return f"该供应商调用次数用尽（{calls}/{q.max_calls}）"
        limit = self._effective_concurrency(caps)
        if self._inflight.get(caps.name, 0) >= limit:
            return f"并发已满（{self._inflight.get(caps.name, 0)}/{limit}）"
        return None

    def _penalties(self, caps: Capabilities, needs: ShotNeeds) -> tuple[float, list[str]]:
        """能力缺口 → 扣分。返回 (扣分总和, 人话说明)。"""
        pen = 0.0
        why: list[str] = []
        clamped = caps.clamp_duration(needs.duration_s)
        delta = abs(clamped - needs.duration_s)
        if delta > 1e-6:
            hit = min(0.5, delta / max(needs.duration_s, 1.0))
            if delta > self.policy.duration_tolerance_s:
                hit = min(0.6, hit + 0.2)
            pen += hit
            why.append(f"时长 {needs.duration_s:g}s 被压到 {clamped:g}s（-{hit:.2f}）")
        if needs.want_first_frame and not caps.supports_first_frame:
            pen += 0.30
            why.append("不支持首帧锁戏，续接要退回纯 I2V（-0.30）")
        if needs.want_last_frame and not caps.supports_last_frame:
            pen += 0.20
            why.append("不支持尾帧锁戏（-0.20）")
        if needs.want_job_continue and not (caps.supports_job_continue or caps.supports_extend):
            pen += 0.25
            why.append("不支持 job 续写/官方延长（-0.25）")
        if needs.want_lora and not caps.supports_lora:
            pen += 0.20
            why.append("不能挂角色 LoRA，身份一致性只能靠参考图（-0.20）")
        if needs.want_native_audio and not caps.supports_native_audio:
            pen += 0.15
            why.append("无原生对白声（-0.15）")
        if needs.want_camera and not caps.supports_camera_control:
            pen += 0.10
            why.append("无运镜参数，只能靠提示词描述（-0.10）")
        if needs.want_seed and not caps.supports_seed:
            pen += 0.05
            why.append("不支持 seed，复现只能靠缓存（-0.05）")
        if needs.want_negative and not caps.supports_negative_prompt:
            pen += 0.05
            why.append("不支持负向提示词（-0.05）")
        if needs.n_images:
            usable = min(needs.n_images, caps.max_ref_images)
            if caps.supported_image_roles:
                usable = min(
                    usable, len(needs.image_roles & caps.supported_image_roles)
                )
            lost = (needs.n_images - usable) / needs.n_images
            if lost > 0:
                hit = 0.35 * lost
                pen += hit
                why.append(
                    f"{needs.n_images} 个参考位只吃得下 {usable} 个（-{hit:.2f}）"
                )
        res = caps.nearest_resolution(needs.resolution)
        if res[0] * res[1] < needs.resolution[0] * needs.resolution[1]:
            pen += 0.10
            why.append(f"最高只到 {res[0]}x{res[1]}，低于交付 {needs.resolution}（-0.10）")
        return pen, why

    def plan(self, shot: Shot, storyboard: Storyboard) -> RoutingPlan:
        """决定这一镜走谁、为什么、备选是谁、预计花多少。"""
        gate = self.content_gate(shot, storyboard)
        needs = shot_needs(shot, storyboard)
        bias = self.policy.bias_for(shot.engine_hint)

        rejected: list[tuple[str, str]] = []
        alive: list[VideoProvider] = []
        for p in self.registry.all():
            reason = self._hard_reject(p, shot, gate, needs)
            if reason:
                rejected.append((p.name, reason))
            else:
                alive.append(p)

        if not alive:
            return RoutingPlan(
                shot_id=shot.id, gate=gate, bias=bias, engine_hint=shot.engine_hint,
                primary=None, fallbacks=(), request=None, scores=(),
                rejected=tuple(rejected), estimated_cost_usd=0.0,
                estimated_latency_s=0.0, degradations=(),
            )

        raw: list[tuple[VideoProvider, float, list[str], float, float, float]] = []
        for p in alive:
            c = p.caps
            clamped = c.clamp_duration(needs.duration_s)
            pen, why = self._penalties(c, needs)
            cost = round(c.cost_per_second_usd * clamped, 6)
            raw.append((p, clamped, why, pen, cost, c.typical_latency_s))

        lo_c = min(r[4] for r in raw)
        hi_c = max(r[4] for r in raw)
        lo_l = min(r[5] for r in raw)
        hi_l = max(r[5] for r in raw)
        w = _WEIGHTS[bias]

        scores: list[CandidateScore] = []
        for p, clamped, why, pen, cost, lat in raw:
            parts = {
                "quality": p.caps.quality_tier / 5.0,
                "cost": _norm_low_is_good(cost, lo_c, hi_c),
                "speed": _norm_low_is_good(lat, lo_l, hi_l),
                "fit": max(0.0, 1.0 - pen),
            }
            total = sum(w[k] * v for k, v in parts.items())
            scores.append(
                CandidateScore(
                    name=p.name, kind=p.caps.kind, total=round(total, 6),
                    parts={k: round(v, 4) for k, v in parts.items()},
                    penalties=tuple(why), clamped_duration_s=clamped,
                    est_cost_usd=cost, est_latency_s=lat,
                )
            )
        # 同分时按成本、再按名字排 —— 排序必须确定，否则重跑的账单对不上。
        scores.sort(key=lambda s: (-s.total, s.est_cost_usd, s.name))

        primary = self.registry.get(scores[0].name)
        req = self.compile_request(shot, storyboard, primary)
        return RoutingPlan(
            shot_id=shot.id, gate=gate, bias=bias, engine_hint=shot.engine_hint,
            primary=primary.name,
            fallbacks=tuple(s.name for s in scores[1 : 1 + self.policy.max_fallbacks]),
            request=req, scores=tuple(scores), rejected=tuple(rejected),
            estimated_cost_usd=primary.estimate_cost(req),
            estimated_latency_s=primary.caps.typical_latency_s,
            degradations=tuple(req.extra.get("degradations", ())),
        )

    # ---------------- 编译

    def compile_request(
        self, shot: Shot, storyboard: Storyboard, provider: VideoProvider
    ) -> GenRequest:
        """把 Shot 裁成这一家吃得下的 GenRequest，降级原因写进 extra。"""
        caps = provider.caps
        degr: list[str] = []

        duration = caps.clamp_duration(shot.duration_s)
        if abs(duration - shot.duration_s) > 1e-6:
            degr.append(
                f"时长 {shot.duration_s:g}s → {duration:g}s（{caps.name} 档位 "
                f"{caps.duration_steps or f'{caps.min_duration_s:g}-{caps.max_duration_s:g}s'}）"
            )
        resolution = caps.nearest_resolution(storyboard.delivery.resolution)
        if resolution != storyboard.delivery.resolution:
            degr.append(f"分辨率 {storyboard.delivery.resolution} → {resolution}，后期靠超分补")
        fps = min(caps.fps_options, key=lambda f: abs(f - shot.fps))
        if fps != shot.fps:
            degr.append(f"帧率 {shot.fps} → {fps}，交付前补帧")

        # 首尾帧走独立字段，不占参考位：先摘出来再裁参考包，能多留一个身份锚。
        images = list(shot.refs.images)
        first_uri = next((r.uri for r in images if r.role == "first_frame"), None)
        last_uri = next((r.uri for r in images if r.role == "last_frame"), None)
        keep: list[ImageRef] = []
        for r in images:
            if r.role == "first_frame":
                if not caps.supports_first_frame:
                    degr.append(
                        f"{caps.name} 不支持首帧锁戏 → 退回纯 I2V，首帧参考降级为构图参考；"
                        "跨镜连续性交给 chain 的重锚策略"
                    )
                    keep.append(ImageRef(role="composition", uri=r.uri,
                                         weight=r.weight, note="降级自 first_frame",
                                         subject_id=r.subject_id))
                continue
            if r.role == "last_frame":
                if not caps.supports_last_frame:
                    degr.append(f"{caps.name} 不支持尾帧锁戏 → 丢弃尾帧参考，退回纯 I2V")
                continue
            keep.append(r)
        refs, notes = _fit_refs(RefPack(images=keep, videos=list(shot.refs.videos),
                                        audios=list(shot.refs.audios)), caps)
        degr += notes

        extend_job = shot.continuity.extend_from_job
        if extend_job and not (caps.supports_job_continue or caps.supports_extend):
            degr.append(
                f"{caps.name} 不支持 job 续写 → 改用尾帧续接；漂移预算按尾帧链计"
            )
            extend_job = None

        negative = _negative_prompt(shot, storyboard)
        if negative and not caps.supports_negative_prompt:
            degr.append(f"{caps.name} 不支持负向提示词 → 负向约束并入正向描述")
            negative = ""

        seed = shot.seed
        if seed is not None and not caps.supports_seed:
            degr.append(f"{caps.name} 不支持 seed → 复现只能靠 fingerprint 缓存")
            seed = None

        lora_cfg: dict[str, object] | None = None
        for sid in shot.subject_ids:
            ch = storyboard.character(sid)
            if ch is None or ch.lora is None:
                continue
            if caps.supports_lora:
                lora_cfg = {
                    "path": ch.lora.path,
                    "trigger_word": ch.lora.trigger_word,
                    "strength": ch.lora.strength,
                    "base_model": ch.lora.base_model,
                }
            else:
                degr.append(
                    f"{caps.name} 不能挂 {ch.id} 的 LoRA → 身份只靠 identity 参考图，"
                    "漂移预算需上浮"
                )
            break

        audio_uri = next((a.uri for a in refs.audios if a.role == "dialogue"), None)
        if shot.dialogue and not storyboard.audio.audio_first \
                and not caps.supports_native_audio:
            degr.append(f"{caps.name} 无原生对白声 → 回落到音频先行 + 口型对齐")

        req = GenRequest(
            prompt=self.prompt_fn(shot, storyboard),
            duration_s=duration,
            resolution=resolution,
            fps=fps,
            negative_prompt=negative,
            seed=seed,
            refs=refs,
            first_frame_uri=first_uri if caps.supports_first_frame else None,
            last_frame_uri=last_uri if caps.supports_last_frame else None,
            extend_from_job=extend_job,
            audio_uri=audio_uri,
            camera_move=shot.camera_move.value if caps.supports_camera_control else None,
            lora=lora_cfg,
            shot_id=shot.id,
            content_rating=shot.content_rating.value,
            idempotency_key=shot.fingerprint(),
            extra={"degradations": tuple(degr), "provider": caps.name},
        )
        if degr:
            log.info("镜头 %s 编译到 %s 发生 %d 处降级", shot.id, caps.name, len(degr))
        return req

    # ---------------- 执行

    def _await_job(self, provider: VideoProvider, job_id: str) -> GenResult:
        """轮询到终态。超时按 TIMEOUT 处理（可重试），不无限等。"""
        deadline = self._clock() + self.job_timeout_s
        while True:
            r = provider.poll(job_id)
            if r.status in {JobStatus.SUCCEEDED, JobStatus.FAILED,
                            JobStatus.REJECTED, JobStatus.CANCELLED}:
                return r
            if self._clock() > deadline:
                provider.cancel(job_id)
                return GenResult(
                    job_id=job_id, status=JobStatus.FAILED, provider=provider.name,
                    failure=FailureKind.TIMEOUT,
                    message=f"轮询 {self.job_timeout_s:g}s 未出终态，已取消",
                )
            self._sleep(self.poll_interval_s)

    def _backoff(self, attempt: int) -> float:
        # 指数退避 + 抖动：同一批镜头同时撞限流时，不要再同时重试一次。
        return self.backoff_base_s * (2 ** (attempt - 1)) * (1.0 + self._rng.random() * 0.3)

    def execute(
        self, shot: Shot, storyboard: Storyboard, *, max_attempts: int = 3
    ) -> GenResult:
        """跑完整条降级链。返回最后一次结果；BAD_REQUEST 直接抛出去报工单。"""
        plan = self.plan(shot, storyboard)
        if not plan.gate.allowed:
            raise ProviderError(
                plan.gate.failure_kind,
                f"镜头 {shot.id} 未通过内容门禁：{'；'.join(plan.gate.blockers)}",
                {"gate": plan.gate.explain()},
            )
        if not plan.routable:
            raise ProviderError(
                FailureKind.QUOTA,
                f"镜头 {shot.id} 没有可用引擎："
                + "；".join(f"{n}={why}" for n, why in plan.rejected),
                {"plan": plan.explain()},
            )

        trail: list[str] = []
        last: GenResult | None = None
        for name in plan.chain:
            provider = self.registry.get(name)
            if not provider.health():
                trail.append(f"{name}: 探活失败，跳过")
                continue
            req = plan.request if name == plan.primary else \
                self.compile_request(shot, storyboard, provider)

            for attempt in range(1, max_attempts + 1):
                est = provider.estimate_cost(req)
                self.ledger.ensure_affordable(est)
                self._inflight[name] = self._inflight.get(name, 0) + 1
                try:
                    job_id = (
                        provider.extend(req)
                        if req.extend_from_job and provider.caps.supports_extend
                        else provider.submit(req)
                    )
                    result = self._await_job(provider, job_id)
                except ProviderError as exc:
                    result = GenResult(
                        job_id="", status=JobStatus.FAILED, provider=name,
                        failure=exc.kind, message=str(exc),
                    )
                finally:
                    self._inflight[name] -= 1

                shot.takes += 1
                last = result
                self.ledger.record(
                    CostEntry(
                        shot_id=shot.id, provider=name, duration_s=req.duration_s,
                        cost_usd=result.cost_usd if result.ok else 0.0,
                        status=result.status.value, job_id=result.job_id, attempt=attempt,
                    )
                )
                if result.ok:
                    shot.provider_used = name
                    shot.job_id = result.job_id
                    shot.render_uri = result.video_uri
                    trail.append(f"{name}#{attempt}: 成功 ${result.cost_usd:.4f}")
                    log.info("镜头 %s 由 %s 出片（第 %d 次尝试）", shot.id, name, attempt)
                    result.raw.setdefault("routing_trail", trail)
                    return result

                kind = result.failure
                trail.append(f"{name}#{attempt}: {kind.value} {result.message}")
                if kind is FailureKind.BAD_REQUEST:
                    raise ProviderError(
                        kind,
                        f"镜头 {shot.id} 参数被 {name} 判为非法，不重试：{result.message}；"
                        "请回工单修正（时长/分辨率/参考位/提示词）",
                        {"trail": trail, "request": req.extra},
                    )
                if kind is FailureKind.MODERATION:
                    # 审核拒绝在同一家重试只会再被拒一次，还白烧一次额度。
                    log.warning("镜头 %s 被 %s 审核拒绝，立刻换引擎", shot.id, name)
                    break
                if not kind.retryable:
                    break
                if attempt < max_attempts:
                    self._sleep(self._backoff(attempt))

        msg = f"镜头 {shot.id} 全链路失败：" + "；".join(trail)
        log.error(msg)
        if last is None:
            return GenResult(job_id="", status=JobStatus.FAILED, provider="",
                             failure=FailureKind.UNKNOWN, message=msg,
                             raw={"routing_trail": trail})
        last.message = msg
        last.raw.setdefault("routing_trail", trail)
        return last


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    import io

    logging.basicConfig(level=logging.WARNING, stream=io.StringIO())
    from .schema import (
        Appearance, CharacterBible, DeliverySpec, Scene, Storyboard as SB,
    )

    class _Fake(VideoProvider):
        """自测用最小 provider：mock.py 由并行工位实现，这里不依赖它。"""

        def __init__(self, caps: Capabilities, *, fail: list[FailureKind] | None = None):
            super().__init__(caps)
            self.fail = list(fail or [])
            self.submits: list[GenRequest] = []
            self._jobs: dict[str, GenRequest] = {}

        def submit(self, req: GenRequest) -> str:
            self.submits.append(req)
            jid = f"{self.name}-job-{len(self.submits)}"
            self._jobs[jid] = req
            return jid

        def extend(self, req: GenRequest) -> str:
            return self.submit(req)

        def poll(self, job_id: str) -> GenResult:
            req = self._jobs[job_id]
            if self.fail:
                k = self.fail.pop(0)
                return GenResult(
                    job_id=job_id, status=JobStatus.REJECTED
                    if k is FailureKind.MODERATION else JobStatus.FAILED,
                    provider=self.name, failure=k, message=f"注入失败 {k.value}",
                )
            return GenResult(
                job_id=job_id, status=JobStatus.SUCCEEDED, provider=self.name,
                video_uri=f"/out/{job_id}.mp4", last_frame_uri=f"/out/{job_id}.png",
                duration_s=req.duration_s, cost_usd=self.estimate_cost(req),
            )

    official = Capabilities(
        name="alpha_official", kind="official", min_duration_s=5.0, max_duration_s=10.0,
        duration_steps=(5.0, 10.0), resolutions=((1920, 1080), (1280, 720)),
        max_ref_images=4, max_ref_videos=1, max_ref_audios=1,
        supported_image_roles=frozenset({"identity", "first_frame", "style", "environment"}),
        supports_first_frame=True, supports_extend=True, supports_job_continue=True,
        supports_camera_control=True, moderated=True, cost_per_second_usd=0.30,
        typical_latency_s=180.0, quality_tier=5, max_concurrency=2,
    )
    openeng = Capabilities(
        name="beta_open", kind="open", min_duration_s=2.0, max_duration_s=8.0,
        resolutions=((1280, 720),), max_ref_images=9, max_ref_videos=3, max_ref_audios=3,
        supports_first_frame=True, supports_last_frame=True, supports_lora=True,
        moderated=False, cost_per_second_usd=0.02, typical_latency_s=420.0,
        quality_tier=3,
    )
    cheap = Capabilities(
        name="gamma_agg", kind="aggregator", min_duration_s=4.0, max_duration_s=12.0,
        resolutions=((854, 480),), max_ref_images=2, supports_seed=False,
        moderated=True, cost_per_second_usd=0.01, typical_latency_s=60.0, quality_tier=2,
    )

    def build_reg(**fails: list[FailureKind]) -> ProviderRegistry:
        reg = ProviderRegistry()
        for caps in (official, openeng, cheap):
            reg.register(_Fake(caps, fail=fails.get(caps.name)))
        return reg

    hero = CharacterBible(
        id="ch_lin", name="林默", age_statement="虚构角色，设定年龄 29 岁，成年",
        appearance=Appearance(face="瘦削", hair="黑色短发", distinguishing="左眉疤"),
    )
    sb = SB(
        project="demo", characters=[hero], style_bible="cinematic anamorphic, 35mm film",
        delivery=DeliverySpec(resolution=(1920, 1080), fps=24),
        scenes=[Scene(id="sc01", shots=[])],
    )

    def mk(**kw) -> Shot:
        base = dict(id="sh001", scene_id="sc01", index=0, duration_s=8.0,
                    subject_ids=["ch_lin"], action="推门而入", seed=7)
        base.update(kw)
        s = Shot(**base)
        sb.scenes[0].shots = [s]
        return s

    # 1) 门禁：正常镜放行
    shot = mk()
    reg = build_reg()
    r = Router(reg, RoutingPolicy(episode_budget_usd=50.0),
               sleeper=lambda _: None, poll_interval_s=0.0)
    g = r.content_gate(shot, sb)
    assert g.allowed and g.allowed_kinds == frozenset({"official", "open", "aggregator"})

    # 2) 门禁：角色不在圣经里 → 拒单，且是 BAD_REQUEST（改工单）
    bad = mk(subject_ids=["ch_ghost"])
    g2 = content_gate(bad, sb)
    assert not g2.allowed and g2.failure_kind is FailureKind.BAD_REQUEST
    assert "ch_ghost" in g2.blockers[0]

    # 3) 门禁：BLOCKED 分级拒单
    g3 = content_gate(mk(content_rating=ContentRating.BLOCKED), sb)
    assert not g3.allowed and g3.failure_kind is FailureKind.MODERATION

    # 4) 门禁：R 级 + 官方审核通道 → 拒绝该通道，只剩开源
    rshot = mk(content_rating=ContentRating.R_VIOLENCE)
    plan_r = r.plan(rshot, sb)
    assert plan_r.primary == "beta_open", plan_r.explain()
    names = {n for n, _ in plan_r.rejected}
    assert names == {"alpha_official", "gamma_agg"}, plan_r.rejected
    assert any("不得投递到带内容审核" in w or "只允许" in w for _, w in plan_r.rejected)

    # 5) 画质优先 → 选 tier5 官方；成本优先 → 选最便宜
    shot = mk()
    assert Router(build_reg(), RoutingPolicy(bias=QualityBias.QUALITY),
                  sleeper=lambda _: None).plan(shot, sb).primary == "alpha_official"
    cost_plan = Router(build_reg(), RoutingPolicy(bias=QualityBias.COST),
                       sleeper=lambda _: None).plan(shot, sb)
    assert cost_plan.primary == "gamma_agg", cost_plan.explain()
    speed_plan = Router(build_reg(), RoutingPolicy(bias=QualityBias.SPEED),
                        sleeper=lambda _: None).plan(shot, sb)
    assert speed_plan.primary == "gamma_agg"

    # 6) engine_hint=hero 强制高画质档，draft 转成本优先
    hero_plan = Router(build_reg(), sleeper=lambda _: None).plan(
        mk(engine_hint=EngineHint.HERO), sb)
    assert hero_plan.primary == "alpha_official"
    assert {n for n, _ in hero_plan.rejected} == {"beta_open", "gamma_agg"}
    draft_plan = Router(build_reg(), sleeper=lambda _: None).plan(
        mk(engine_hint=EngineHint.DRAFT), sb)
    assert draft_plan.bias is QualityBias.COST

    # 7) compile_request：时长档位、分辨率、参考位裁剪、能力降级
    many = RefPack(images=[
        ImageRef(role="identity", uri="/id1.png", subject_id="ch_lin"),
        ImageRef(role="identity", uri="/id2.png", subject_id="ch_lin"),
        ImageRef(role="wardrobe", uri="/w.png"),
        ImageRef(role="environment", uri="/env.png"),
        ImageRef(role="prop", uri="/p.png"),
        ImageRef(role="last_frame", uri="/last.png"),
    ])
    shot = mk(duration_s=8.0, refs=many, camera_move=CameraMove.DOLLY_IN)
    reg = build_reg()
    r = Router(reg, sleeper=lambda _: None)
    req = r.compile_request(shot, sb, reg.get("alpha_official"))
    assert req.duration_s == 10.0, req.duration_s        # 8s 落到最近档位 10s
    assert req.resolution == (1920, 1080)
    assert len(req.refs.images) <= 4
    assert [i.role for i in req.refs.images][0] == "identity"   # 身份锚保底
    assert all(i.role != "prop" for i in req.refs.images)       # 不支持的角色被丢
    assert req.last_frame_uri is None                           # 官方家不支持尾帧
    assert req.camera_move == CameraMove.DOLLY_IN.value
    assert req.idempotency_key == shot.fingerprint()
    degr = req.extra["degradations"]
    assert any("尾帧" in d for d in degr) and any("时长" in d for d in degr), degr

    req_cheap = r.compile_request(shot, sb, reg.get("gamma_agg"))
    assert req_cheap.seed is None and any("seed" in d for d in req_cheap.extra["degradations"])
    assert req_cheap.resolution == (854, 480)

    # 8) MODERATION：立刻换引擎，同一家绝不重试
    reg = build_reg(alpha_official=[FailureKind.MODERATION, FailureKind.MODERATION])
    r = Router(reg, RoutingPolicy(episode_budget_usd=50.0),
               sleeper=lambda _: None, poll_interval_s=0.0)
    shot = mk()
    res = r.execute(shot, sb)
    assert res.ok and res.provider == "beta_open", res
    assert len(reg.get("alpha_official").submits) == 1, "审核拒绝后不得重试同一家"
    assert shot.provider_used == "beta_open" and shot.takes == 2

    # 9) RATE_LIMIT：同一家退避重试
    reg = build_reg(alpha_official=[FailureKind.RATE_LIMIT, FailureKind.RATE_LIMIT])
    r = Router(reg, sleeper=lambda _: None, poll_interval_s=0.0)
    shot = mk()
    res = r.execute(shot, sb, max_attempts=3)
    assert res.ok and res.provider == "alpha_official"
    assert len(reg.get("alpha_official").submits) == 3, "限流应重试同一家"

    # 10) BAD_REQUEST：不重试，直接抛出报工单
    reg = build_reg(alpha_official=[FailureKind.BAD_REQUEST])
    r = Router(reg, sleeper=lambda _: None, poll_interval_s=0.0)
    try:
        r.execute(mk(), sb)
        raise AssertionError("BAD_REQUEST 必须抛出")
    except ProviderError as exc:
        assert exc.kind is FailureKind.BAD_REQUEST and "回工单" in str(exc)
    assert len(reg.get("alpha_official").submits) == 1

    # 11) 账本与预算
    reg = build_reg()
    ledger = CostLedger(budget_usd=10.0)
    r = Router(reg, ledger=ledger, sleeper=lambda _: None, poll_interval_s=0.0)
    for i in range(3):
        r.execute(mk(id=f"sh10{i}"), sb)
    assert len(ledger.by_shot()) == 3
    assert abs(ledger.total() - 3 * 3.0) < 1e-6, ledger.report()   # 10s * $0.30
    assert ledger.by_provider()["alpha_official"] == ledger.total()
    try:
        r.execute(mk(id="sh999"), sb)
        raise AssertionError("超预算必须抛 BudgetExceeded")
    except BudgetExceeded as exc:
        assert exc.budget == 10.0

    # 12) 并发闸门：并发占满的家不进候选
    reg = build_reg()
    r = Router(reg, RoutingPolicy(quotas={"alpha_official": ProviderQuota(max_concurrency=1)}),
               sleeper=lambda _: None)
    r._inflight["alpha_official"] = 1
    p = r.plan(mk(), sb)
    assert p.primary != "alpha_official"
    assert any("并发已满" in why for n, why in p.rejected if n == "alpha_official")

    # 13) 配额闸门
    q_ledger = CostLedger()
    q_ledger.record(CostEntry(shot_id="x", provider="alpha_official", duration_s=10,
                              cost_usd=5.0, status="succeeded"))
    r = Router(build_reg(), RoutingPolicy(quotas={"alpha_official": ProviderQuota(max_usd=4.0)}),
               ledger=q_ledger, sleeper=lambda _: None)
    assert r.plan(mk(), sb).primary != "alpha_official"

    # 14) 无可用引擎 → 拒单
    empty = ProviderRegistry()
    empty.register(_Fake(official))
    r = Router(empty, sleeper=lambda _: None)
    try:
        r.execute(mk(content_rating=ContentRating.R_SUGGESTIVE), sb)
        raise AssertionError("R 级无开源引擎时必须拒单")
    except ProviderError as exc:
        assert exc.kind is FailureKind.QUOTA

    # 15) explain 可读
    text = Router(build_reg(), sleeper=lambda _: None).plan(mk(), sb).explain()
    assert "路由决策" in text and "候选打分" in text and "选中" in text
    print(text)
    print()
    print(ledger.report())
    print("\nrouter 自测通过：门禁/打分/编译降级/失败分流/账本/配额 全部符合预期")


if __name__ == "__main__":
    _selftest()
