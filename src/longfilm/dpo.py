"""偏好对齐与数据飞轮 —— 把出片结果回流成偏好对，并监控飞轮是否在坍缩。

这个模块处在产线的**回流环**上：qc.py 判过没过，distill.py 把过检镜收成教师集，
本模块再往前一步，把「同一条镜头的多个 take 谁更好」这件事固化成 DPO 训练对。

为什么偏好对比教师集更值钱：SFT/蒸馏只能告诉模型「照着这个拍」，
学不到「不要拍成那样」。而生成模型的废片往往不是缺了什么，是**多**了什么
（闪烁、结构崩、色漂）。负样本是唯一能把这些「多出来的东西」写进梯度的载体，
而产线每天本来就在批量制造负样本 —— 不用白不用。

--------------------------------------------------------------------------
外部做法查证（2026-09-20，WebSearch/WebFetch）
--------------------------------------------------------------------------

**VideoDPO**（Liu et al., CVPR 2025, arXiv:2412.14167，https://videodpo.github.io/）
- 每条提示词生成 **N = 4** 条视频，用 OmniScore 打分后取**最高分作 win、最低分作 lose**，
  严格 best-vs-worst，论文未设分差门槛。
- OmniScore = 七个子维度的加权：motion smoothness（帧插值模型）、temporal flickering、
  dynamic degree（RAFT 光流）、subject consistency（DINO 特征相似度）、
  imaging quality（MUSIQ）、aesthetic quality（LAION 美学预测器）、
  text-video alignment（ViCLIP）。**注意它里面有 dynamic degree** —— 见下面的偏差讨论。
- 按分数做重加权：w_pair = (β / prob(s_W, s_L))^α，其中
  prob(s_W, s_L) = sqrt(p(s_W)·p(s_L))，p 来自分数直方图的频率，
  β 取「最高频样本的概率」。作用是给「罕见的分数组合」更高权重。
- 数据集：VidProm 里 K = 10000 条人写提示词 → 10000 对。

**Flow-DPO**（"Improving Video Generation with Human Feedback", arXiv:2501.13918, NeurIPS 2025）
- 把 Diffusion-DPO 直接搬到 rectified flow 上会得到 β_t = β(1-t)^2 的**时间相关 KL 系数**，
  t→1 时惩罚消失，模型于是只在高噪声档上对齐，表现为 reward hacking、文图对齐反而变差。
  Flow-DPO 的改动就是**把 β 拉成常数**，去掉时间依赖。
- 奖励模型 VideoReward 三维打分：VQ(视觉质量) / MQ(运动质量) / TA(文图对齐)，
  用 Bradley-Terry-with-Ties 训练；人工数据 16000 条提示词 × 12 个模型 → 108000 条视频、
  182000 个标注三元组，标注形式是「A 胜 / 平 / B 胜」**逐维度**给，外加每条 1-5 的 Likert 分。
  本模块的人工回流 CSV 就照这个口径设计（逐维度 + 允许平局）。

**Diffusion-DPO**（arXiv:2311.12908, https://github.com/SalesforceAIResearch/DiffusionDPO）
- 训练脚本吃 `yuvalkirstain/pickapic_v2`，列名是 `caption` / `jpg_0` / `jpg_1` /
  `label_0` / `label_1`；label_0 = 1 表示 0 号是 win。
- 参考超参 `--beta_dpo 5000`（扩散侧的 β 量纲和 LLM 侧完全不同，别照搬 0.1）。

**TRL DPOTrainer**（https://huggingface.co/docs/trl/dpo_trainer）
- 标准偏好格式就三个键：`prompt` / `chosen` / `rejected`，JSONL 每行一条；
  列名必须精确，改名会直接 KeyError。

**坍缩判据**
- Shumailov et al., "AI models collapse when trained on recursively generated data",
  Nature 631 (2024) —— early collapse 是**尾部先消失**，late collapse 是**方差塌缩到点估计**；
  关键结论：只要不断用自己的输出替换真实数据，误差逐代累积；**累加**而不是**替换**数据
  才能把误差界住。这条直接决定了本模块 FlywheelMonitor 的默认策略建议。
- Vendi Score（arXiv:2210.02410）：多样性 = 相似度矩阵特征值的 Shannon 熵取指数，
  不需要参考集。本模块在离散标签轴上用它的退化特例 exp(H)（相似度取 one-hot），
  语义就是「有效模式数」，比归一化熵更好向人解释。
- LPIPS / IC-LPIPS 类内距离与 precision-recall：mode collapse 的典型签名是
  **precision 高而 recall 低**（像得很、但只会那几样）。本模块的
  「通过率升 + 多样性降」剪刀差就是这个签名在产线上的廉价代理。
  TODO(2026-09-20)：像素级的 LPIPS 类内距离需要 torch + AlexNet 权重，
  本机 3GB 无 GPU 跑不动，故未实现；上线到有卡的机器后应补成 FlywheelMonitor
  的可选维度（接口已预留 extra_diversity）。

--------------------------------------------------------------------------
内容策略
--------------------------------------------------------------------------
偏好对沿用 distill.DEFAULT_TEACHER_RATINGS（只收 G / PG13 的虚构成年角色镜头）。
偏好数据比教师数据更危险：负样本也会进梯度，分级信息在权重里同样会丢失，
所以这里不放宽、也不提供放宽的便捷开关。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, Field

from .distill import (
    DEFAULT_TEACHER_RATINGS,
    DistillDataset,
    DiversityMetrics,
    RenderLike,
    TeacherSample,
    diversity_metrics,
    harvest,
)
from .qc import QCReport, Verdict
from .schema import CameraMove, ContentRating, ShotSize

log = logging.getLogger(__name__)

#: 本模块所有外部规格的查证日期。改动默认阈值时同步更新。
CHECKED_ON = "2026-09-20"


# ================================================================ 1. 偏好对类型


class PairSource(str, Enum):
    """这一对是谁标的。**必须落盘**：自动标注和人工标注的偏差方向完全不同，
    混在一起训练而不记来源，事后连归因都做不了。"""

    QC_AUTO = "qc_auto"      # 质检分自动构对 —— 便宜、量大、有系统性偏差
    HUMAN = "human"          # 人工打分回流 —— 贵、量小、是唯一的偏差校准源
    HYBRID = "hybrid"        # 质检初筛 + 人工复核


class ScoreKind(str, Enum):
    """win_score / lose_score 的量纲来源。"""

    QC = "qc"                # qc.QCReport.score，0..1 加权总分
    HUMAN_LIKERT = "human_likert"   # 人工 1-5 分（已归一到 0..1）
    HUMAN_PAIRWISE = "human_pairwise"  # 人工直接给的胜负，分数是占位的 1.0/0.0


class PreferencePair(BaseModel):
    """同一条提示词下的 win/lose 对。

    两侧都用 distill.TeacherSample 而不是裸 uri：偏好对要按镜头语法分桶
    做偏差审计（「赢家是不是系统性地更静止」），只存路径就审不了。
    """

    pair_id: str
    prompt: str
    negative_prompt: str = ""

    win: TeacherSample
    lose: TeacherSample

    win_score: float
    lose_score: float
    score_kind: ScoreKind = ScoreKind.QC
    source: PairSource = PairSource.QC_AUTO

    #: VideoDPO 的 w_pair。默认 1.0，调用 PreferenceDataset.reweight_videodpo() 后回填。
    weight: float = Field(default=1.0, ge=0.0)
    #: 标注可信度 0..1。自动标注取归一分差，人工取评委一致率。
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def margin(self) -> float:
        """分差。DPO 的梯度本身不看分差，但**构对时**看：
        分差小于标注噪声的对，等于给模型喂随机标签。"""
        return self.win_score - self.lose_score

    @property
    def shot_id(self) -> str:
        return self.win.shot_id

    def flipped(self) -> PreferencePair:
        """把 win/lose 对调。只用于自测和「标签污染」消融实验，不要在产线上调。"""
        return self.model_copy(update={
            "win": self.lose, "lose": self.win,
            "win_score": self.lose_score, "lose_score": self.win_score,
            "meta": {**self.meta, "flipped": True},
        })


class PreferenceDataset(BaseModel):
    """一批偏好对。可直接 JSON 落盘，dataset_hash 回填进 LoRASpec 做复现。"""

    name: str = "pref"
    pairs: list[PreferencePair] = Field(default_factory=list)
    built_on: str = CHECKED_ON
    #: 构对时用的最小分差门槛，写进数据集自身，免得导出后无从追溯。
    min_margin: float = 0.0
    notes: list[str] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    def by_source(self) -> dict[str, int]:
        return dict(Counter(p.source.value for p in self.pairs))

    def filter_margin(self, min_margin: float) -> PreferenceDataset:
        kept = [p for p in self.pairs if p.margin >= min_margin]
        return PreferenceDataset(
            name=f"{self.name}-m{min_margin:g}", pairs=kept,
            built_on=self.built_on, min_margin=min_margin,
            notes=[*self.notes, f"按分差 >= {min_margin} 过滤：{len(self.pairs)} -> {len(kept)}"],
        )

    def merge(self, other: PreferenceDataset, *, name: str | None = None) -> PreferenceDataset:
        """合并两批对。同 pair_id 以 **后者** 为准 —— 人工复核就是要覆盖自动标注。"""
        idx = {p.pair_id: p for p in self.pairs}
        overridden = sum(1 for p in other.pairs if p.pair_id in idx)
        idx.update({p.pair_id: p for p in other.pairs})
        return PreferenceDataset(
            name=name or f"{self.name}+{other.name}",
            pairs=list(idx.values()),
            built_on=max(self.built_on, other.built_on),
            min_margin=max(self.min_margin, other.min_margin),
            notes=[*self.notes, *other.notes,
                   f"合并：{len(self.pairs)} + {len(other.pairs)} -> {len(idx)}"
                   f"（其中 {overridden} 对被后者覆盖）"],
        )

    # ---------------------------------------------------------- 重加权

    def reweight_videodpo(self, *, alpha: float = 1.0, bins: int = 20) -> PreferenceDataset:
        """按 VideoDPO 的频率重加权回填 weight（就地改，返回 self 便于链式调用）。

            w_pair = (β / sqrt(p(s_W) · p(s_L)))^α ，β = max_s p(s)

        直觉：分数直方图里扎堆的那些对（大家都 0.9 对 0.7）信息量低，
        罕见组合（0.99 对 0.35）信息量高。β 取最高频概率使得最常见的对权重≈1，
        其余 ≥1，避免整体尺度漂移把学习率变相放大。
        来源：arXiv:2412.14167 §3.2（2026-09-20 查证）。
        """
        if not self.pairs:
            return self
        hist: Counter[int] = Counter()
        for p in self.pairs:
            hist[self._bin(p.win_score, bins)] += 1
            hist[self._bin(p.lose_score, bins)] += 1
        total = sum(hist.values())
        prob = {k: v / total for k, v in hist.items()}
        beta = max(prob.values())
        for p in self.pairs:
            pw = prob[self._bin(p.win_score, bins)]
            pl = prob[self._bin(p.lose_score, bins)]
            p.weight = float((beta / math.sqrt(pw * pl)) ** alpha)
        self.notes.append(
            f"VideoDPO 重加权 alpha={alpha} bins={bins}："
            f"weight ∈ [{min(p.weight for p in self.pairs):.3f}, "
            f"{max(p.weight for p in self.pairs):.3f}]"
        )
        return self

    @staticmethod
    def _bin(score: float, bins: int) -> int:
        return min(bins - 1, max(0, int(score * bins)))

    # ---------------------------------------------------------- 统计与落盘

    def score_gap_stats(self) -> dict[str, float]:
        if not self.pairs:
            return {"n": 0.0}
        m = sorted(p.margin for p in self.pairs)
        n = len(m)
        return {
            "n": float(n),
            "min": round(m[0], 4),
            "p25": round(m[n // 4], 4),
            "median": round(m[n // 2], 4),
            "p75": round(m[(3 * n) // 4], 4),
            "max": round(m[-1], 4),
            "mean": round(sum(m) / n, 4),
        }

    def dataset_hash(self) -> str:
        """同指纹 = 同一批对（含胜负方向）。喂给 LoRASpec.dataset_hash。"""
        blob = "|".join(sorted(
            f"{p.pair_id}:{p.win.video_uri}>{p.lose.video_uri}" for p in self.pairs
        ))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_distill_dataset(self, *, winners_only: bool = True) -> DistillDataset:
        """转成 distill.DistillDataset，好复用那边现成的多样性体检与训练集导出。

        winners_only 默认 True：飞轮监控看的是「**被选中**的那一半长什么样」，
        把输家也混进去会把坍缩信号稀释掉一半。
        """
        samples = [p.win for p in self.pairs]
        if not winners_only:
            samples += [p.lose for p in self.pairs]
        return DistillDataset(name=f"{self.name}-win", samples=samples, built_on=self.built_on)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> PreferenceDataset:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ================================================================ 2. 质检自动构对


def qc_score_and_pass(report: Any) -> tuple[float, bool | None]:
    """把质检结论归一成 (score, passed)。

    比 distill.qc_score_of 多认一样东西：qc.QCReport 身上**没有** passed 字段，
    只有三态 verdict。直接用那边的实现会把每份 QCReport 都读成 passed=None，
    于是 min_score 以外的门槛全部失效 —— 这是接两个模块时最容易踩的坑。
    """
    if report is None:
        return 0.0, None
    if isinstance(report, QCReport):
        return float(report.score), report.verdict is Verdict.PASS
    if isinstance(report, (int, float)):
        return float(report), None
    if isinstance(report, Mapping):
        score = report.get("score", report.get("qc_score", 0.0))
        verdict = report.get("verdict")
        passed = report.get("passed", report.get("ok"))
    else:
        score = getattr(report, "score", getattr(report, "qc_score", 0.0))
        verdict = getattr(report, "verdict", None)
        passed = getattr(report, "passed", getattr(report, "ok", None))
    if passed is None and verdict is not None:
        v = verdict.value if isinstance(verdict, Enum) else str(verdict)
        passed = v == Verdict.PASS.value
    return float(score or 0.0), (None if passed is None else bool(passed))


def metric_value_of(report: Any, name: str) -> float | None:
    """从质检报告里取某一项指标的原始数值（取不到返回 None）。

    偏差审计要看「赢家是不是更静止」，就必须能读到 motion_sanity 的原始值，
    而不是被归一过的 score —— score 已经把「太静」和「太狂」折叠成同一个低分了。
    """
    if report is None:
        return None
    metrics: Any = None
    if isinstance(report, QCReport):
        metrics = report.metrics
    elif isinstance(report, Mapping):
        metrics = report.get("metrics")
    else:
        metrics = getattr(report, "metrics", None)
    if not metrics:
        return None
    for m in metrics:
        mname = m.get("name") if isinstance(m, Mapping) else getattr(m, "name", None)
        if mname != name:
            continue
        val = m.get("value") if isinstance(m, Mapping) else getattr(m, "value", None)
        if val is None:
            return None
        v = float(val)
        return None if math.isnan(v) else v
    return None


def build_pairs_from_qc(
    renders: Iterable[RenderLike],
    qc_reports: Mapping[str, Any] | None = None,
    *,
    min_margin: float = 0.08,
    min_win_score: float = 0.70,
    require_win_passed: bool = True,
    strategy: Literal["best_worst", "all_pairs"] = "best_worst",
    allowed_ratings: Iterable[ContentRating] = DEFAULT_TEACHER_RATINGS,
    prompt_of: Mapping[str, str] | None = None,
    name: str = "pref-qc",
) -> PreferenceDataset:
    """从质检分数自动构造偏好对：同一条 shot 的多个 take 比分数。

    默认 best_worst，对齐 VideoDPO（N=4 条里取最高分 vs 最低分，arXiv:2412.14167）。
    与论文的差别有两处，都是产线现实逼出来的：

    1. **加了分差门槛 min_margin**。论文里 OmniScore 是连续奖励模型，分差天然拉得开；
       本产线的 qc.QCGate.score 是七项归一分的加权平均，两条都正常的 take 分差常在
       0.02 以内 —— 那是解码抖动，不是偏好。低于门槛的对不构，宁缺毋滥。
    2. **加了 min_win_score / require_win_passed**。论文的 win 只需相对更好；
       本产线里「两条都废，只是废得程度不同」的情形非常多，把这种对喂进去
       等于教模型「废成这样就行」。所以赢家必须自己先过检。

    ============================================================
    ⚠ 自动标注相对人工标注的偏差风险（这是本函数最重要的一段文档）
    ============================================================

    **质检指标衡量的是「没崩」，不是「好看」。** qc.py 的七项指标里，
    temporal_flicker / color_drift / freeze / black / conformance 五项是
    「异常检测」，motion_sanity 是个双边区间，seam 只在拼接产物上有意义。
    没有任何一项在回答「这个镜头拍得好不好」。用它当偏好信号，会引入三重偏差：

    **(a) 保守化偏差 —— 最危险的一条。**
    闪烁指标是逐帧亮度一阶差分的标准差，色漂指标是首尾均值差。这两项都随
    **画面变化量**单调增长。于是在「同一条镜头、只有运镜幅度不同」的多个 take 里，
    镜头动得越少的那条分越高。本模块自测用 ffmpeg 真跑过这条曲线：
    同一源素材、crop 平移振幅 20 / 60 / 140 px，QC 总分是
    0.947 / 0.786 / 0.721 —— **单调递减**，而 motion_sanity 原始值
    4.14 / 6.48 / 11.52 单调递增。也就是说自动标注会稳定地把
    「幅度最小的那条」选成赢家。几轮飞轮下来，模型学到的是「少动就不会被判废」，
    产出向保守静止的画面坍缩。VideoDPO 的 OmniScore 里专门放了一项
    dynamic degree（RAFT 光流）来对冲这件事，本产线的质检**没有**这一项，
    所以必须靠 audit_static_bias() + FlywheelMonitor 在外面兜。

    **(b) 可测偏差。** 能自动算的维度会被过度优化，算不了的维度（表演、节奏、
    构图、情绪准确性）在偏好信号里权重为零。DPO 的梯度不会对没出现在标签里的
    维度做任何保护，只会为了可测维度的收益牺牲它们 —— Flow-DPO 论文里
    「时间相关 β 导致 reward hacking、文图对齐反而变差」是同一类现象。

    **(c) 同源偏差。** 分数和视频出自同一套生成/评估管线，管线自己的系统性偏好
    （某引擎偏暖、某分辨率更容易过检）会被当成「人类偏好」写进权重，且无法自纠。

    **所以产线上的正确用法**：自动对用来跑量、做冷启动；每轮必须掺**至少 10%**
    的人工对做校准（见 build_pairs_from_human），并且每轮都跑 audit_static_bias()。
    审计里保守化 z > 2 就说明自动标签已经在系统性地奖励静止 —— 这时候要么
    给质检补一项动态度指标，要么把该批自动对降权，不能直接上训练。

    ============================================================

    Args:
        renders: 出片记录，同 distill.harvest（RenderRecord 或 (Shot, GenResult) 二元组）。
        qc_reports: 按 job_id（退回 shot_id）索引的质检结论。QCReport / dict / float 都认。
        min_margin: 最小分差，低于此不构对。
        min_win_score: 赢家的分数下限。
        require_win_passed: 赢家必须 verdict=PASS（报告里给不出判定时此条不生效）。
        strategy: best_worst 每 shot 一对；all_pairs 枚举所有满足门槛的组合（量大、相关性高）。
        prompt_of: 编译后的提示词，按 job_id / shot_id 索引。
    """
    # 复用 harvest 做「成片过滤 + 分级门禁 + 许可证登记 + 提示词回填」，
    # 但把 min_score 放到 0：偏好对**需要**输家，harvest 默认那套会把输家全筛掉。
    pool = harvest(
        renders, qc_reports, min_score=0.0, allowed_ratings=allowed_ratings,
        prompt_of=prompt_of, name=f"{name}-pool",
    )
    reports = qc_reports or {}

    def report_for(s: TeacherSample) -> Any:
        return reports.get(str(s.meta.get("job_id") or ""), reports.get(s.shot_id))

    by_shot: dict[str, list[TeacherSample]] = defaultdict(list)
    for s in pool.samples:
        by_shot[s.shot_id].append(s)

    pairs: list[PreferencePair] = []
    dropped: Counter[str] = Counter()

    for shot_id, takes in sorted(by_shot.items()):
        if len(takes) < 2:
            dropped["只有 1 个 take，构不了对"] += 1
            continue
        ranked = sorted(takes, key=lambda s: s.qc_score, reverse=True)
        combos: list[tuple[TeacherSample, TeacherSample]]
        if strategy == "best_worst":
            combos = [(ranked[0], ranked[-1])]
        else:
            combos = [(ranked[i], ranked[j])
                      for i in range(len(ranked)) for j in range(i + 1, len(ranked))]

        for win, lose in combos:
            margin = win.qc_score - lose.qc_score
            if margin < min_margin:
                dropped[f"分差 {margin:.3f} < {min_margin}（落在标注噪声里）"] += 1
                continue
            if win.qc_score < min_win_score:
                dropped[f"赢家自己也没到 {min_win_score}（两条都废）"] += 1
                continue
            _, win_passed = qc_score_and_pass(report_for(win))
            if require_win_passed and win_passed is False:
                dropped["赢家未过检"] += 1
                continue

            w_motion = metric_value_of(report_for(win), "motion_sanity")
            l_motion = metric_value_of(report_for(lose), "motion_sanity")
            pairs.append(PreferencePair(
                pair_id=f"{shot_id}__{win.key}_vs_{lose.key}",
                prompt=win.prompt,
                negative_prompt=win.negative_prompt,
                win=win, lose=lose,
                win_score=win.qc_score, lose_score=lose.qc_score,
                score_kind=ScoreKind.QC, source=PairSource.QC_AUTO,
                # 分差归一到 0..1 当可信度：0.3 以上的分差在本产线上基本不会翻盘。
                confidence=float(min(1.0, margin / 0.30)),
                meta={
                    "win_job_id": win.meta.get("job_id"),
                    "lose_job_id": lose.meta.get("job_id"),
                    "win_motion": w_motion, "lose_motion": l_motion,
                    "n_takes": len(takes), "strategy": strategy,
                },
            ))

    if dropped:
        log.info("build_pairs_from_qc 丢弃统计：%s", dict(dropped))
    ds = PreferenceDataset(
        name=name, pairs=pairs, min_margin=min_margin,
        notes=[
            f"自动标注（{strategy}），门槛 margin>={min_margin} win>={min_win_score}",
            "⚠ 质检指标衡量的是「没崩」而非「好看」，本批对带保守化偏差，"
            "上训练前必须跑 audit_static_bias() 并掺人工对校准。",
        ],
    )
    audit = audit_static_bias(ds)
    if audit.suspicious:
        log.warning("自动构对存在保守化偏差：%s", audit.summary())
    return ds


# ---------------------------------------------------------------- 偏差审计


@dataclass(frozen=True)
class StaticBiasAudit:
    """「赢家是不是系统性地更静止」的统计审计。

    判据是**符号检验**而不是均值比较：motion 的量纲（YDIF 中位数）在不同镜头间
    差一个数量级，均值会被少数大运动镜主导。符号检验只问「赢家更静的对占几成」，
    零假设下该比例 = 0.5，标准差 = 0.5/sqrt(n)，于是 z = (ratio-0.5)*2*sqrt(n)。
    """

    n_pairs: int
    n_comparable: int          # 两侧都拿到 motion 数值的对
    n_win_more_static: int
    ratio: float               # 赢家更静止的占比
    z: float                   # 符号检验 z 值，>0 表示偏向静止
    mean_delta: float          # mean(motion_win - motion_lose)，负数 = 赢家更静
    z_threshold: float = 2.0

    @property
    def suspicious(self) -> bool:
        """z > 2 ≈ 单侧 p < 0.023，即「赢家更静」不太可能是巧合。

        为什么用 2 而不是 1.96：z 是在**同一批出片**上算的，同一条镜头的多个 take
        不独立，实际自由度低于 n，所以门槛比教科书的 1.96 略紧一点。
        """
        return self.n_comparable >= 8 and self.z > self.z_threshold

    def summary(self) -> str:
        if self.n_comparable < 8:
            return f"可比对仅 {self.n_comparable} 条（<8），样本不足，不下结论"
        verdict = "⚠ 赢家系统性更静止" if self.suspicious else "未检出显著偏向"
        return (f"{verdict}：{self.n_win_more_static}/{self.n_comparable} "
                f"({self.ratio:.0%}) 的对里赢家运动量更低，z={self.z:+.2f}，"
                f"平均运动量差 {self.mean_delta:+.3f}")


def audit_static_bias(dataset: PreferenceDataset, *, z_threshold: float = 2.0) -> StaticBiasAudit:
    """审计自动标注的保守化偏差。

    数据来源是构对时随手记进 meta 的 win_motion / lose_motion；
    没有质检报告（比如纯人工对）时 n_comparable = 0，审计自动弃权而不是报警。
    """
    deltas: list[float] = []
    for p in dataset.pairs:
        w, l = p.meta.get("win_motion"), p.meta.get("lose_motion")
        if w is None or l is None:
            continue
        deltas.append(float(w) - float(l))
    n = len(deltas)
    if n == 0:
        return StaticBiasAudit(len(dataset.pairs), 0, 0, 0.0, 0.0, 0.0, z_threshold)
    more_static = sum(1 for d in deltas if d < 0)
    ratio = more_static / n
    z = (ratio - 0.5) * 2.0 * math.sqrt(n)
    return StaticBiasAudit(
        n_pairs=len(dataset.pairs), n_comparable=n, n_win_more_static=more_static,
        ratio=round(ratio, 4), z=round(z, 4),
        mean_delta=round(sum(deltas) / n, 4), z_threshold=z_threshold,
    )


# ================================================================ 3. 人工打分回流


#: 人工 CSV 认的两种行格式。逐维度打分对齐 Flow-DPO 的 VideoReward 标注口径
#: （VQ 视觉质量 / MQ 运动质量 / TA 文图对齐，外加 1-5 Likert），
#: 但把「必填」压到最小 —— 标注成本每多一列就掉一截完成率。
_RATING_COLS = ("rating", "score", "likert")
_DIM_COLS = ("vq", "mq", "ta")


def _norm_likert(x: float, lo: float, hi: float) -> float:
    """Likert 归一到 0..1。lo/hi 从数据里实测而不是写死 1/5：
    有的标注表用 0-10，有的用 1-5，写死会让 min_margin 的语义随表变化。"""
    if hi <= lo:
        return 0.5
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def build_pairs_from_human(
    ratings_csv: str | Path,
    *,
    prompts: Mapping[str, str] | None = None,
    min_margin: float = 0.15,
    min_raters: int = 1,
    require_agreement: float = 0.0,
    name: str = "pref-human",
) -> PreferenceDataset:
    """人工打分回流。

    支持两种 CSV，按表头自动识别 —— 标注团队用哪种取决于他们的工具，
    强行统一只会让数据进不来：

    **A. 逐条打分（推荐）**，每行一个 take：
        ``shot_id,take,video_uri,rating[,prompt,rater,vq,mq,ta,duration_s,notes]``
        同 shot 内按平均分排序，最高 vs 最低构一对。rating 可以是任意区间的
        Likert（1-5 / 0-10 都行），实测 min/max 后归一。

    **B. 直接给胜负**，每行一对：
        ``shot_id,win_uri,lose_uri[,prompt,rater,notes]``
        多个评委对同一 shot 投票，按多数票定胜负；平票丢弃。

    Args:
        prompts: 提示词表（按 shot_id 索引）。CSV 里带 prompt 列时以 CSV 为准。
        min_margin: 归一后的最小分差。默认 0.15 比自动对的 0.08 松得多，
            因为人工分是离散的（5 档里差 1 档 = 0.25），门槛设太细会把整批筛没。
        min_raters: 每个 take 至少几个评委给过分。
        require_agreement: 格式 B 下的最低多数票比例，0 表示只要不是平票就收。

    为什么人工对不可替代：自动对的偏差是**系统性**的（见 build_pairs_from_qc 的
    偏差讨论），系统性偏差加再多数据也不会被平均掉，只会被更牢地学进去。
    人工对是唯一和质检管线**不同源**的信号，它的作用不是「更准」，是「不同源」。
    """
    path = Path(ratings_csv)
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        log.warning("人工评分表 %s 是空的", path)
        return PreferenceDataset(name=name, min_margin=min_margin,
                                 notes=[f"{path} 为空"])

    cols = {c.strip().lower() for c in (rows[0].keys() or [])}
    rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in rows]

    if {"win_uri", "lose_uri"} <= cols:
        pairs = _human_pairwise(rows, prompts, require_agreement)
        kind = ScoreKind.HUMAN_PAIRWISE
        fmt = "B(直接胜负)"
    else:
        rating_col = next((c for c in _RATING_COLS if c in cols), None)
        if rating_col is None:
            raise ValueError(
                f"{path} 的表头 {sorted(cols)} 既没有 win_uri/lose_uri，"
                f"也没有 {'/'.join(_RATING_COLS)} 列，无法识别格式"
            )
        pairs = _human_likert(rows, rating_col, prompts, min_margin, min_raters)
        kind = ScoreKind.HUMAN_LIKERT
        fmt = f"A(逐条打分, 列={rating_col})"

    for p in pairs:
        p.score_kind = kind
    ds = PreferenceDataset(
        name=name, pairs=pairs, min_margin=min_margin,
        notes=[f"人工回流 {path.name} 格式 {fmt}：{len(rows)} 行 -> {len(pairs)} 对",
               "人工对与质检管线不同源，是自动对系统性偏差的唯一校准源，不要因为量小就降权。"],
    )
    log.info("人工回流 %s：%d 行 -> %d 对", path.name, len(rows), len(pairs))
    return ds


def _mk_sample(shot_id: str, take: int, uri: str, prompt: str, score: float,
               row: Mapping[str, str]) -> TeacherSample:
    """把 CSV 行包成 TeacherSample。

    镜头语法字段留默认值并在 meta 里标明 —— 人工表通常只有 uri 和分数，
    **不要**从文件名猜景别/运镜：猜错会直接污染多样性体检的分布。
    """
    def num(key: str, default: float) -> float:
        try:
            return float(row.get(key) or default)
        except ValueError:
            return default

    return TeacherSample(
        shot_id=shot_id, take=take, video_uri=uri,
        duration_s=max(0.0, num("duration_s", 0.0)),
        prompt=prompt, qc_score=float(min(1.0, max(0.0, score))),
        meta={"from": "human_csv", "labels_unknown": True,
              **{k: row[k] for k in _DIM_COLS if row.get(k)},
              **({"rater": row["rater"]} if row.get("rater") else {}),
              **({"notes": row["notes"]} if row.get("notes") else {})},
    )


def _human_likert(rows: Sequence[Mapping[str, str]], rating_col: str,
                  prompts: Mapping[str, str] | None,
                  min_margin: float, min_raters: int) -> list[PreferencePair]:
    # 先实测分值区间再归一，避免把 1-5 表和 0-10 表按同一把尺子量。
    vals: list[float] = []
    for r in rows:
        try:
            vals.append(float(r[rating_col]))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return []
    lo, hi = min(vals), max(vals)

    # (shot_id, take) -> [归一分...]；同一 take 多个评委就在这里聚合
    agg: dict[tuple[str, int], list[float]] = defaultdict(list)
    info: dict[tuple[str, int], tuple[str, str, Mapping[str, str]]] = {}
    for r in rows:
        sid = r.get("shot_id") or ""
        uri = r.get("video_uri") or r.get("uri") or ""
        if not sid or not uri:
            continue
        try:
            take = int(r.get("take") or 0)
            raw = float(r[rating_col])
        except (KeyError, TypeError, ValueError):
            continue
        key = (sid, take)
        agg[key].append(_norm_likert(raw, lo, hi))
        info[key] = (uri, r.get("prompt") or (prompts or {}).get(sid, ""), r)

    by_shot: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for (sid, take), scores in agg.items():
        if len(scores) < min_raters:
            continue
        by_shot[sid].append((take, sum(scores) / len(scores), len(scores)))

    out: list[PreferencePair] = []
    for sid, takes in sorted(by_shot.items()):
        if len(takes) < 2:
            continue
        takes.sort(key=lambda t: t[1], reverse=True)
        (wt, ws, wn), (lt, ls, ln) = takes[0], takes[-1]
        if ws - ls < min_margin:
            continue
        w_uri, w_prompt, w_row = info[(sid, wt)]
        l_uri, _, l_row = info[(sid, lt)]
        out.append(PreferencePair(
            pair_id=f"{sid}__h{wt}_vs_h{lt}",
            prompt=w_prompt or (prompts or {}).get(sid, ""),
            win=_mk_sample(sid, wt, w_uri, w_prompt, ws, w_row),
            lose=_mk_sample(sid, lt, l_uri, w_prompt, ls, l_row),
            win_score=ws, lose_score=ls,
            source=PairSource.HUMAN,
            confidence=float(min(1.0, (ws - ls) / 0.5)),
            meta={"raters_win": wn, "raters_lose": ln,
                  "likert_range": [lo, hi], "n_takes": len(takes)},
        ))
    return out


def _human_pairwise(rows: Sequence[Mapping[str, str]],
                    prompts: Mapping[str, str] | None,
                    require_agreement: float) -> list[PreferencePair]:
    # (shot_id, frozenset{两条uri}) -> Counter{胜者uri: 票数}
    votes: dict[tuple[str, frozenset[str]], Counter[str]] = defaultdict(Counter)
    ctx: dict[tuple[str, frozenset[str]], Mapping[str, str]] = {}
    for r in rows:
        sid = r.get("shot_id") or ""
        w, l = r.get("win_uri") or "", r.get("lose_uri") or ""
        if not sid or not w or not l or w == l:
            continue
        key = (sid, frozenset({w, l}))
        votes[key][w] += 1
        votes[key].setdefault(l, 0)
        ctx[key] = r

    out: list[PreferencePair] = []
    for (sid, uris), c in sorted(votes.items(), key=lambda kv: (kv[0][0], sorted(kv[0][1]))):
        total = sum(c.values())
        (top_uri, top_n), (bot_uri, _) = c.most_common()[0], c.most_common()[-1]
        share = top_n / total if total else 0.0
        if top_uri == bot_uri or share <= 0.5:
            continue    # 平票：人都没达成一致，不该让模型替人做决定
        if share < require_agreement:
            continue
        r = ctx[(sid, uris)]
        prompt = r.get("prompt") or (prompts or {}).get(sid, "")
        out.append(PreferencePair(
            pair_id=f"{sid}__p{hashlib.sha1('|'.join(sorted(uris)).encode()).hexdigest()[:8]}",
            prompt=prompt,
            win=_mk_sample(sid, 0, top_uri, prompt, 1.0, r),
            lose=_mk_sample(sid, 1, bot_uri, prompt, 0.0, r),
            win_score=1.0, lose_score=0.0,
            source=PairSource.HUMAN,
            confidence=float(share),
            meta={"votes": dict(c), "agreement": round(share, 3)},
        ))
    return out


# ================================================================ 4. 训练格式导出


ExportFormat = Literal["trl", "pickapic", "videodpo"]

#: 各格式的来源与字段口径（2026-09-20 查证）。导出时一并写进 manifest，
#: 免得半年后没人记得 label_0 到底是「0 号赢」还是「0 号的分数」。
EXPORT_FORMATS: dict[str, dict[str, str]] = {
    "trl": {
        "file": "train.jsonl",
        "fields": "prompt / chosen / rejected",
        "source": "https://huggingface.co/docs/trl/dpo_trainer",
        "note": "TRL 标准偏好格式，列名必须精确，改名即 KeyError。"
                "本产线的 chosen/rejected 放的是视频路径而非文本 —— "
                "TRL 原生只吃文本，视频侧需要自定义 data collator。",
    },
    "pickapic": {
        "file": "train.jsonl",
        "fields": "caption / video_0 / video_1 / label_0 / label_1",
        "source": "https://github.com/SalesforceAIResearch/DiffusionDPO ；"
                  "yuvalkirstain/pickapic_v2",
        "note": "Diffusion-DPO 训练脚本的列名，原版是 jpg_0/jpg_1（图像字节）。"
                "视频侧把 jpg_* 换成 video_*（路径），label_0=1 表示 0 号是赢家。"
                "参考超参 --beta_dpo 5000（扩散侧 β 量纲与 LLM 侧不同，勿照搬 0.1）。",
    },
    "videodpo": {
        "file": "pairs.json",
        "fields": "prompt / win / lose / score_win / score_lose / weight",
        "source": "arXiv:2412.14167 (CVPR 2025) ；https://github.com/CIntellifusion/VideoDPO",
        "note": "官方仓库只发了 tar 包和 configs/dpo/vidpro/train_data.yaml，"
                "未公开列名规范。TODO(2026-09-20)：接入前需对照该 yaml 校准字段名。"
                "weight 是论文的 w_pair=(β/sqrt(p_W·p_L))^α，已随对导出。",
    },
}


def export_for_training(
    dataset: PreferenceDataset,
    out_dir: str | Path,
    fmt: ExportFormat = "trl",
    *,
    materialize: bool = False,
    relative: bool = True,
) -> Path:
    """导出成常见 DPO 训练格式，返回写好的数据文件路径。

    无论哪种格式都额外写一份 ``manifest.json``，里面有 dataset_hash、
    来源分布、分差统计和**保守化偏差审计结果**。偏差审计跟着数据走是硬要求：
    训练脚本读不到警告没关系，人在 review 数据集时必须一眼看见。

    Args:
        materialize: True 时把视频拷进 ``videos/``，并把清单里的路径改成相对路径。
            默认 False —— 偏好集动辄几百 GB，拷贝是 CI 里最容易打爆磁盘的一步。
        relative: materialize 时路径写成相对 out_dir 的形式，便于整目录搬运。
    """
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"未知导出格式 {fmt!r}，可选 {sorted(EXPORT_FORMATS)}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    uri_map: dict[str, str] = {}
    if materialize:
        vdir = out / "videos"
        vdir.mkdir(exist_ok=True)
        for p in dataset.pairs:
            for side in (p.win, p.lose):
                src = Path(side.video_uri)
                if side.video_uri in uri_map:
                    continue
                if not src.exists():
                    # 远端 uri（s3:// 之类）不拷，原样保留 —— 静默失败比报错更坏。
                    log.warning("materialize 跳过不存在的素材：%s", side.video_uri)
                    uri_map[side.video_uri] = side.video_uri
                    continue
                dst = vdir / f"{side.key}{src.suffix or '.mp4'}"
                shutil.copy2(src, dst)
                uri_map[side.video_uri] = (
                    str(dst.relative_to(out)) if relative else str(dst.resolve())
                )

    def uri(s: TeacherSample) -> str:
        return uri_map.get(s.video_uri, s.video_uri)

    spec = EXPORT_FORMATS[fmt]
    data_path = out / spec["file"]

    if fmt == "trl":
        with data_path.open("w", encoding="utf-8") as fh:
            for p in dataset.pairs:
                fh.write(json.dumps({
                    "prompt": p.prompt,
                    "chosen": uri(p.win),
                    "rejected": uri(p.lose),
                    # 非标准列，TRL 会忽略；留着是为了训练时能做加权采样与事后归因。
                    "weight": round(p.weight, 6),
                    "margin": round(p.margin, 6),
                    "source": p.source.value,
                    "pair_id": p.pair_id,
                }, ensure_ascii=False) + "\n")

    elif fmt == "pickapic":
        with data_path.open("w", encoding="utf-8") as fh:
            for i, p in enumerate(dataset.pairs):
                # 交替把赢家放在 0 / 1 号位：Diffusion-DPO 的 collator 不打乱顺序，
                # 全让赢家占 0 号会让模型学到「0 号更好」这个纯粹的位置先验。
                win_first = (i % 2 == 0)
                a, b = (p.win, p.lose) if win_first else (p.lose, p.win)
                fh.write(json.dumps({
                    "caption": p.prompt,
                    "video_0": uri(a), "video_1": uri(b),
                    "label_0": 1.0 if win_first else 0.0,
                    "label_1": 0.0 if win_first else 1.0,
                    "weight": round(p.weight, 6),
                    "pair_id": p.pair_id,
                }, ensure_ascii=False) + "\n")

    else:  # videodpo
        payload = {
            "_format": "videodpo",
            "_source": spec["source"],
            "_todo": spec["note"],
            "negative_prompt_global": "",
            "pairs": [{
                "prompt": p.prompt,
                "negative_prompt": p.negative_prompt,
                "win": uri(p.win), "lose": uri(p.lose),
                "score_win": round(p.win_score, 6), "score_lose": round(p.lose_score, 6),
                "weight": round(p.weight, 6),
                "confidence": round(p.confidence, 4),
                "source": p.source.value, "pair_id": p.pair_id,
                "shot_size": p.win.shot_size.value, "camera_move": p.win.camera_move.value,
            } for p in dataset.pairs],
        }
        data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = audit_static_bias(dataset)
    manifest = {
        "name": dataset.name,
        "format": fmt,
        "format_spec": spec,
        "built_on": dataset.built_on,
        "exported_on": CHECKED_ON,
        "n_pairs": len(dataset),
        "dataset_hash": dataset.dataset_hash(),
        "sources": dataset.by_source(),
        "score_gap": dataset.score_gap_stats(),
        "min_margin": dataset.min_margin,
        "materialized": materialize,
        "static_bias_audit": {
            "n_comparable": audit.n_comparable,
            "win_more_static_ratio": audit.ratio,
            "z": audit.z,
            "mean_motion_delta": audit.mean_delta,
            "suspicious": audit.suspicious,
            "summary": audit.summary(),
        },
        "notes": dataset.notes,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("导出 %d 对到 %s（格式 %s）", len(dataset), data_path, fmt)
    return data_path


# ================================================================ 5. 数据飞轮监控


class AlertLevel(str, Enum):
    OK = "ok"
    WARN = "warn"            # 记一笔，继续跑
    ALERT = "alert"          # 该轮数据降权或掺真实数据
    CRITICAL = "critical"    # 停止只回收过检镜，本轮不要上训练


_LEVEL_ORDER = {AlertLevel.OK: 0, AlertLevel.WARN: 1, AlertLevel.ALERT: 2, AlertLevel.CRITICAL: 3}


@dataclass(frozen=True)
class CollapseAlert:
    rule: str
    level: AlertLevel
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    action: str = ""

    def __str__(self) -> str:
        mark = {"ok": "  ", "warn": "⚠ ", "alert": "‼ ", "critical": "⛔"}[self.level.value]
        return f"{mark}[{self.rule}] {self.message}" + (f"\n      → {self.action}" if self.action else "")


def effective_modes(counter: Counter[str]) -> float:
    """有效模式数 = exp(Shannon 熵)。

    这是 Vendi Score（arXiv:2210.02410）在相似度取 one-hot 时的退化特例，
    语义是「等价于几个等概率的类别」。比归一化熵好用的地方在于它**有量纲**：
    「景别的有效模式数从 6.2 掉到 2.1」人能直接判断严不严重，
    而「熵从 0.81 掉到 0.55」不能。
    """
    total = sum(counter.values())
    if total == 0:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counter.values() if c)
    return round(math.exp(h), 3)


def _slope(ys: Sequence[float]) -> float:
    """最小二乘斜率（x 取 0,1,2,...）。轮次少，不值得引入 scipy。"""
    n = len(ys)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(ys) / n
    den = sum((i - mx) ** 2 for i in range(n))
    return round(sum((i - mx) * (y - my) for i, y in enumerate(ys)) / den, 6) if den else 0.0


@dataclass
class FlywheelRound:
    """一轮飞轮的体检快照。"""

    round_idx: int
    label: str = ""
    n_generated: int = 0        # 本轮总出片数
    n_passed: int = 0           # 过检数
    pass_rate: float = 0.0
    diversity: DiversityMetrics | None = None
    bias: StaticBiasAudit | None = None
    extra: dict[str, float] = field(default_factory=dict)   # 预留：LPIPS 类内距离等

    @property
    def overall_diversity(self) -> float:
        return self.diversity.overall() if self.diversity else 0.0

    @property
    def modes_shot_size(self) -> float:
        return effective_modes(self.diversity.shot_size) if self.diversity else 0.0

    @property
    def modes_camera_move(self) -> float:
        return effective_modes(self.diversity.camera_move) if self.diversity else 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "round": self.round_idx, "label": self.label,
            "n_generated": self.n_generated, "n_passed": self.n_passed,
            "pass_rate": round(self.pass_rate, 4),
            "diversity_overall": self.overall_diversity,
            "modes_shot_size": self.modes_shot_size,
            "modes_camera_move": self.modes_camera_move,
        }
        if self.diversity:
            d.update({
                "distinct_3": self.diversity.distinct_3,
                "max_class_share": self.diversity.max_class_share,
                "shot_size_coverage": self.diversity.shot_size_coverage,
                "camera_move_coverage": self.diversity.camera_move_coverage,
                "n_samples": self.diversity.n_samples,
            })
        if self.bias:
            d["static_bias_z"] = self.bias.z
        d.update(self.extra)
        return d


class FlywheelMonitor:
    """数据飞轮健康监控 —— 「只回收过检镜」这条策略的安全阀。

    ============================================================
    为什么必须同时看两个数
    ============================================================
    飞轮的回路是：出片 → 质检 → **只把过检的**收进训练集 → 训练 → 出片。
    这个回路里每一步都在做「选择」，而选择的方向由质检指标定义。
    质检指标衡量的是「没崩」，所以回路的不动点是**最不容易崩的那类画面**：
    固定机位、小幅动作、均匀打光、中景。到达这个不动点时，
    通过率会非常漂亮地爬到 95%+，而模型已经不会拍别的了。

    **单看通过率，坍缩长得和成功一模一样。** Shumailov et al.（Nature 2024）
    在语言模型上给出的正是这个结论：递归自训练的 early collapse 表现为
    尾部先消失，而主流指标不降反升；到 late collapse 时方差塌成点估计，
    这时已经救不回来了。所以本监控的核心判据不是任何单一指标的绝对值，
    而是**两条曲线的剪刀差**：通过率上行 + 多样性下行。

    ============================================================
    默认阈值与依据
    ============================================================
    R1 剪刀差（核心）：单轮 pass_rate 上升 ≥ +5pp **且** overall 多样性
       相对下降 ≥ 5% → ALERT。
       依据：按每轮 200 条出片、基准通过率 0.8 估，通过率的标准误
       sqrt(0.8·0.2/200) ≈ 2.8pp，5pp ≈ 1.8σ，单轮误报概率 < 5%；
       多样性侧取同量级的 5% 以免一边宽一边紧。两个条件是「与」关系，
       因为通过率单独上升是好事（工艺进步），多样性单独下降可能只是
       这一集的戏就窄（全是室内对话戏）—— 只有两者同时发生才是坍缩签名。
       这一条对应文献里 mode collapse 的经典签名「precision 高 / recall 低」。

    R2 累计多样性跌幅（相对基线轮）：-15% → ALERT，-30% → CRITICAL。
       依据：overall 是六个分量的几何平均，-30% 大致等于**有效模式数少掉三分之一**
       （exp(H) 意义上，见 effective_modes）。到这一步已经不是「风格收敛」，
       是镜头语法表真的缺项了，靠继续训练不会自己长回来。

    R3 单类占比 > 50% → WARN。沿用 distill.diversity_report 的同一条线，
       两个模块口径必须一致，否则同一批数据在两处给出不同结论。

    R4 提示词模板化，三条任一命中 → WARN：
       (a) 完全重复提示词占比 > 20%（沿用 distill 的线，**与语料规模无关**）；
       (b) distinct-3 相对基线轮下降 ≥ 25%；
       (c) distinct-3 < 0.05 这条硬地板。
       依据与一条实测更正：distinct-N 是**语料规模的单调减函数**，不能跨轮用绝对阈值。
       用 distill._tokens（中文按字切）在同一套提示词生成器上实测（2026-09-20）：
       完全不重复的 60 条提示词 distinct-3 = 0.27，同一生成器出 120 条降到 0.18，
       而全部用同一模板的 120 条是 0.008。也就是说 distill.diversity_report 里
       那条绝对 0.30 的线只在**小批量**下成立，直接搬到飞轮的跨轮比较上会恒报警。
       所以这里主判据取相对跌幅，绝对地板压到 0.05 只兜「真·一个模板打天下」。

    R5 覆盖率：shot_size < 40% 或 camera_move < 30% → ALERT。同样沿用 distill。

    R6 保守化偏差 z > 2 → ALERT。自动标注特有，见 audit_static_bias。
       R1 抓的是「已经塌了」，R6 抓的是「正在往塌的方向推」，R6 比 R1 早一到两轮。

    R7 通过率 > 95% 且多样性未同步上升 → WARN。
       依据：扩散视频引擎的随机性决定了 5% 左右的废片率是物理下限
       （见 qc.py 的口径讨论）。通过率逼近 1 通常不代表工艺变好，
       代表提示词分布已经缩进模型的舒适区 —— 难的镜头压根没被提出来。

    R8 多轮趋势：≥3 轮时，多样性最小二乘斜率 < 0 且通过率斜率 > 0 → ALERT。
       单轮剪刀差可能是噪声，连续三轮同向就不是了。

    ============================================================
    触发之后该做什么（按 Nature 2024 的结论）
    ============================================================
    关键不是「停下」，是**累加而不是替换**：保留每一轮的人工对与早期真实数据，
    新一轮只做增量，不要用本轮自产数据整体覆盖训练集 —— 论文证明了
    累加策略能把误差界住，而替换策略的误差随代数线性/指数增长。
    具体到本产线：①提高人工对配比；②给质检补一项动态度指标（对冲 R6）；
    ③强制在提示词侧注入被冷落的景别/运镜配额；④把本轮自动对的 weight 打折。
    """

    # ---- 阈值（可按项目覆盖；改动请同步上面的依据段落）
    pass_rate_jump: float = 0.05
    diversity_drop_rel: float = 0.05
    cum_drop_alert: float = 0.15
    cum_drop_critical: float = 0.30
    max_class_share: float = 0.50
    distinct3_floor: float = 0.05          # 硬地板；绝对值随语料规模变，故压得很低
    distinct3_drop_rel: float = 0.25       # 相对基线轮的跌幅，这才是主判据
    dup_prompt_ratio_max: float = 0.20     # 完全重复提示词占比，与规模无关
    shot_size_coverage_floor: float = 0.40
    camera_move_coverage_floor: float = 0.30
    static_bias_z: float = 2.0
    saturated_pass_rate: float = 0.95
    min_rounds_for_trend: int = 3

    def __init__(self, name: str = "flywheel", **thresholds: float) -> None:
        self.name = name
        self.rounds: list[FlywheelRound] = []
        for k, v in thresholds.items():
            if not hasattr(type(self), k):
                raise ValueError(f"未知阈值 {k!r}；可覆盖 {sorted(self._threshold_names())}")
            setattr(self, k, float(v))

    @classmethod
    def _threshold_names(cls) -> list[str]:
        return [k for k, v in vars(cls).items()
                if isinstance(v, (int, float)) and not k.startswith("_")]

    # ---------------------------------------------------------- 观测

    def observe(
        self,
        dataset: DistillDataset | PreferenceDataset,
        *,
        pass_rate: float | None = None,
        n_generated: int = 0,
        n_passed: int = 0,
        bias: StaticBiasAudit | None = None,
        label: str = "",
        extra: Mapping[str, float] | None = None,
    ) -> FlywheelRound:
        """记一轮。

        dataset 给 PreferenceDataset 时自动取**赢家侧**做多样性统计，
        并自动跑保守化审计 —— 飞轮真正回流的是赢家，输家只参与一次梯度就丢了。
        """
        if isinstance(dataset, PreferenceDataset):
            bias = bias or audit_static_bias(dataset, z_threshold=self.static_bias_z)
            dist = dataset.to_distill_dataset(winners_only=True)
        else:
            dist = dataset

        if pass_rate is None:
            if n_generated > 0:
                pass_rate = n_passed / n_generated
            else:
                raise ValueError("必须给 pass_rate，或给 n_generated/n_passed 让监控自己算")
        if not n_generated and n_passed:
            n_generated = n_passed

        r = FlywheelRound(
            round_idx=len(self.rounds), label=label,
            n_generated=n_generated, n_passed=n_passed,
            pass_rate=float(pass_rate),
            diversity=diversity_metrics(dist),
            bias=bias, extra=dict(extra or {}),
        )
        self.rounds.append(r)
        for a in self.check(r.round_idx):
            log.log(logging.WARNING if a.level is not AlertLevel.WARN else logging.INFO,
                    "飞轮 %s 第 %d 轮 %s", self.name, r.round_idx, a.message)
        return r

    def observe_from_qc(
        self,
        dataset: DistillDataset | PreferenceDataset,
        qc_reports: Iterable[Any],
        **kw: Any,
    ) -> FlywheelRound:
        """直接喂一批 qc.QCReport 算通过率，省得调用方自己数。"""
        reps = list(qc_reports)
        passed = sum(1 for r in reps if qc_score_and_pass(r)[1] is True)
        return self.observe(dataset, n_generated=len(reps), n_passed=passed, **kw)

    # ---------------------------------------------------------- 判定

    def check(self, round_idx: int | None = None) -> list[CollapseAlert]:
        """对某一轮（默认最后一轮）跑全部规则。按严重度降序返回。"""
        if not self.rounds:
            return []
        i = len(self.rounds) - 1 if round_idx is None else round_idx
        cur = self.rounds[i]
        base = self.rounds[0]
        prev = self.rounds[i - 1] if i > 0 else None
        out: list[CollapseAlert] = []
        d = cur.diversity

        # ---- R1 剪刀差
        if prev is not None and prev.overall_diversity > 0:
            d_pass = cur.pass_rate - prev.pass_rate
            rel_div = (cur.overall_diversity - prev.overall_diversity) / prev.overall_diversity
            if d_pass >= self.pass_rate_jump and rel_div <= -self.diversity_drop_rel:
                out.append(CollapseAlert(
                    "R1_剪刀差", AlertLevel.ALERT,
                    f"通过率 {prev.pass_rate:.1%} → {cur.pass_rate:.1%}（{d_pass:+.1%}）"
                    f"的同时，多样性 {prev.overall_diversity:.3f} → {cur.overall_diversity:.3f}"
                    f"（{rel_div:+.1%}）—— 指标涨、能力窄，典型坍缩签名。",
                    {"d_pass_rate": round(d_pass, 4), "rel_diversity": round(rel_div, 4)},
                    "本轮自动对降权到 0.5，并把人工对配比提到 20%；"
                    "训练集**累加**而不是替换（Nature 2024 结论）。",
                ))

        # ---- R2 累计跌幅
        if i > 0 and base.overall_diversity > 0:
            cum = (cur.overall_diversity - base.overall_diversity) / base.overall_diversity
            if cum <= -self.cum_drop_critical:
                out.append(CollapseAlert(
                    "R2_累计多样性", AlertLevel.CRITICAL,
                    f"相对第 0 轮多样性累计下降 {abs(cum):.1%}（≥{self.cum_drop_critical:.0%}）："
                    f"{base.overall_diversity:.3f} → {cur.overall_diversity:.3f}；"
                    f"景别有效模式数 {base.modes_shot_size} → {cur.modes_shot_size}，"
                    f"运镜 {base.modes_camera_move} → {cur.modes_camera_move}。",
                    {"cum_rel": round(cum, 4)},
                    "停止只回收过检镜。本轮不要上训练；先补齐缺失的景别/运镜配额，"
                    "并把第 0 轮的真实数据重新混回训练集。",
                ))
            elif cum <= -self.cum_drop_alert:
                out.append(CollapseAlert(
                    "R2_累计多样性", AlertLevel.ALERT,
                    f"相对第 0 轮多样性累计下降 {abs(cum):.1%}"
                    f"（≥{self.cum_drop_alert:.0%}）：{base.overall_diversity:.3f}"
                    f" → {cur.overall_diversity:.3f}。",
                    {"cum_rel": round(cum, 4)},
                    "在提示词侧强制注入被冷落的景别/运镜配额，本轮增量训练而非全量重训。",
                ))

        if d is not None and d.n_samples > 0:
            # ---- R3 单类占比
            if d.max_class_share > self.max_class_share:
                out.append(CollapseAlert(
                    "R3_单类占比", AlertLevel.WARN,
                    f"单一镜头语法占比 {d.max_class_share:.0%} > {self.max_class_share:.0%}，"
                    "分布已经偏斜。",
                    {"max_class_share": d.max_class_share},
                    "查 distill.diversity_report 看是哪一类吃掉了配额。",
                ))
            # ---- R4 提示词模板化（绝对地板 + 相对跌幅 + 完全重复率）
            reasons: list[str] = []
            base_d3 = base.diversity.distinct_3 if base.diversity else 0.0
            drop_d3 = ((d.distinct_3 - base_d3) / base_d3) if base_d3 > 0 else 0.0
            if d.dup_prompt_ratio > self.dup_prompt_ratio_max:
                reasons.append(f"完全重复提示词占比 {d.dup_prompt_ratio:.0%} "
                               f"> {self.dup_prompt_ratio_max:.0%}")
            if i > 0 and drop_d3 <= -self.distinct3_drop_rel:
                reasons.append(f"distinct-3 相对第 0 轮下降 {abs(drop_d3):.0%} "
                               f"（{base_d3:.3f} → {d.distinct_3:.3f}）")
            if d.distinct_3 < self.distinct3_floor:
                reasons.append(f"distinct-3 {d.distinct_3:.3f} 跌破硬地板 "
                               f"{self.distinct3_floor}")
            if reasons:
                out.append(CollapseAlert(
                    "R4_提示词模板化", AlertLevel.WARN,
                    "；".join(reasons) + f"。高频 3-gram：{d.top_3gram[:3]}",
                    {"distinct_3": d.distinct_3, "distinct_3_rel": round(drop_d3, 4),
                     "dup_prompt_ratio": d.dup_prompt_ratio},
                    "prompt_os 的模板句式要按镜头随机化，否则 LoRA 会把模板背下来。"
                    "注意 distinct-N 随语料规模单调下降，跨轮比较只看相对跌幅。",
                ))
            # ---- R5 覆盖率
            if (d.shot_size_coverage < self.shot_size_coverage_floor
                    or d.camera_move_coverage < self.camera_move_coverage_floor):
                out.append(CollapseAlert(
                    "R5_覆盖率", AlertLevel.ALERT,
                    f"景别覆盖 {d.shot_size_coverage:.0%}"
                    f"（底线 {self.shot_size_coverage_floor:.0%}，{len(d.shot_size)}/{len(ShotSize)}）、"
                    f"运镜覆盖 {d.camera_move_coverage:.0%}"
                    f"（底线 {self.camera_move_coverage_floor:.0%}，"
                    f"{len(d.camera_move)}/{len(CameraMove)}）。",
                    {"shot_size_coverage": d.shot_size_coverage,
                     "camera_move_coverage": d.camera_move_coverage},
                    "训练集撑不起完整镜头语法；补拍缺失类别，或本轮不训运镜相关的 LoRA。",
                ))

        # ---- R6 保守化偏差
        if cur.bias is not None and cur.bias.suspicious:
            out.append(CollapseAlert(
                "R6_保守化偏差", AlertLevel.ALERT,
                f"自动标注正在系统性奖励静止画面：{cur.bias.summary()}",
                {"z": cur.bias.z, "ratio": cur.bias.ratio},
                "给 qc.py 补一项动态度指标（对齐 VideoDPO 的 dynamic degree）再构对；"
                "在那之前把本轮自动对的 weight 打折，或只保留人工对。",
            ))

        # ---- R7 通过率饱和
        if cur.pass_rate > self.saturated_pass_rate:
            rising = prev is not None and cur.overall_diversity > prev.overall_diversity
            if not rising:
                out.append(CollapseAlert(
                    "R7_通过率饱和", AlertLevel.WARN,
                    f"通过率 {cur.pass_rate:.1%} > {self.saturated_pass_rate:.0%} 而多样性未同步上升。"
                    "扩散引擎的随机性决定了 5% 左右废片率是物理下限，"
                    "通过率逼近 1 通常意味着难镜头压根没被提出来。",
                    {"pass_rate": round(cur.pass_rate, 4)},
                    "往分镜里加难度：大幅运镜、多人同框、强动作、极端光比，看通过率是否回落。",
                ))

        # ---- R8 多轮趋势
        if i + 1 >= self.min_rounds_for_trend:
            hist = self.rounds[: i + 1]
            s_div = _slope([r.overall_diversity for r in hist])
            s_pass = _slope([r.pass_rate for r in hist])
            if s_div < 0 and s_pass > 0:
                out.append(CollapseAlert(
                    "R8_趋势", AlertLevel.ALERT,
                    f"连续 {len(hist)} 轮趋势背离：多样性斜率 {s_div:+.4f}/轮，"
                    f"通过率斜率 {s_pass:+.4f}/轮。单轮可能是噪声，连续同向不是。",
                    {"slope_diversity": s_div, "slope_pass_rate": s_pass},
                    "按 Nature 2024 的结论：训练集改为**累加**而非替换，"
                    "并保证每轮都有不同源（人工）的偏好信号进来。",
                ))

        out.sort(key=lambda a: -_LEVEL_ORDER[a.level])
        return out

    @property
    def level(self) -> AlertLevel:
        alerts = self.check()
        return max((a.level for a in alerts), key=lambda l: _LEVEL_ORDER[l], default=AlertLevel.OK)

    # ---------------------------------------------------------- 报表

    def report(self) -> str:
        """飞轮仪表盘（给人看的）。两条曲线必须并排，单看任何一条都会被骗。"""
        if not self.rounds:
            return f"飞轮 {self.name}：还没有任何一轮数据。"
        # 中文列名在终端里是双宽，所以列宽按「显示宽度」手调，并用两空格显式分隔，
        # 免得 format 的字符计数把表头和数据错开一格。
        head = ("轮次  标签            样本    通过率   多样性   Δ多样性"
                "  景别模式  运镜模式  distinct3   偏差z")
        width = max(len(head) + 8, 96)
        lines = [f"数据飞轮健康度 · {self.name} · {len(self.rounds)} 轮", "=" * width, head,
                 "-" * width]
        base = self.rounds[0].overall_diversity
        for r in self.rounds:
            rel = (r.overall_diversity - base) / base if base > 0 else 0.0
            d3 = f"{r.diversity.distinct_3:.3f}" if r.diversity else "-"
            z = f"{r.bias.z:+.2f}" if r.bias and r.bias.n_comparable else "-"
            lines.append(
                f"{r.round_idx:>4}  {r.label[:14]:<14}"
                f"{(r.diversity.n_samples if r.diversity else 0):>6}"
                f"{r.pass_rate:>9.1%}{r.overall_diversity:>9.3f}{rel:>+9.1%}"
                f"{r.modes_shot_size:>10.2f}{r.modes_camera_move:>10.2f}{d3:>11}{z:>8}"
            )
        lines.append("-" * width)
        alerts = self.check()
        if alerts:
            lines.append(f"最新一轮告警（最高级别 {self.level.value.upper()}）：")
            lines += [f"  {a}" for a in alerts]
        else:
            lines.append("最新一轮：未触发任何坍缩预警。")
        lines.append("=" * width)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.value,
            "thresholds": {k: getattr(self, k) for k in sorted(self._threshold_names())},
            "rounds": [r.to_dict() for r in self.rounds],
            "alerts": [{"rule": a.rule, "level": a.level.value, "message": a.message,
                        "evidence": a.evidence, "action": a.action} for a in self.check()],
        }


# ================================================================ 6. 自测


def _selftest() -> None:
    import tempfile

    from .providers.base import GenResult, JobStatus
    from .qc import QCContext, QCGate
    from .schema import Shot
    from .stitch import default_ffmpeg

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(name)s: %(message)s")
    work = Path(tempfile.mkdtemp(prefix="longfilm_dpo_"))
    try:
        # ============================================================
        # [1] 真跑 ffmpeg + qc.QCGate：证明「质检分随运镜幅度单调下降」
        #     这是本模块偏差论证的实证基础，不是文档里的断言。
        # ============================================================
        ff = default_ffmpeg()

        def make_take(amp: int) -> str:
            """同一源素材、同一时长，只改 crop 平移振幅 —— 纯粹的运镜幅度差。"""
            p = work / f"take_amp{amp}.mp4"
            ff.run([
                "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=24:d=3",
                "-vf", f"crop=320:180:x='160+{amp}*sin(2*PI*t/3)':y='90+{amp * 0.4}*cos(2*PI*t/3)'",
                "-pix_fmt", "yuv420p", str(p),
            ])
            return str(p)

        gate = QCGate.default(include_plugins=False)
        amps = [20, 60, 140]
        vids = [make_take(a) for a in amps]
        shot_probe = Shot(id="probe", scene_id="sc00", index=0, duration_s=3.0, fps=24)
        probe_reports = [gate.evaluate(v, shot_probe, context=QCContext(shot=shot_probe))
                         for v in vids]
        scores = [r.score for r in probe_reports]
        motions = [metric_value_of(r, "motion_sanity") for r in probe_reports]

        assert scores == sorted(scores, reverse=True), scores
        assert motions == sorted(motions), motions
        assert all(m is not None for m in motions)
        print("[1] 实证：同一镜头只改运镜幅度，质检分与运动量反向")
        for a, s, m in zip(amps, scores, motions):
            print(f"      平移振幅 {a:>3}px  QC总分 {s:.4f}  motion_sanity {m:.3f}")
        print(f"      -> 自动标注必然把振幅最小的 {amps[0]}px 选成赢家（这就是保守化偏差）")

        # ============================================================
        # [2] build_pairs_from_qc：真 QCReport 喂进来构对
        # ============================================================
        def rec(shot_id: str, take: int, video: str, *, size: ShotSize, move: CameraMove,
                action: str, env: str, scene: str, subj: str) -> Any:
            from .distill import RenderRecord
            sh = Shot(id=shot_id, scene_id=scene, index=take, duration_s=3.0, fps=24,
                      shot_size=size, camera_move=move, subject_ids=[subj],
                      action=action, environment=env, lighting="侧逆光", mood="克制",
                      style="胶片颗粒，冷调", content_rating=ContentRating.PG13, takes=take)
            return RenderRecord(
                shot=sh,
                result=GenResult(job_id=f"{shot_id}-t{take}", status=JobStatus.SUCCEEDED,
                                 provider="mock", video_uri=video, duration_s=3.0, cost_usd=0.1),
                take=take,
            )

        # 五条镜头，每条 3 个 take，分别对应三档运镜幅度（真实视频 + 真实质检报告）。
        # 取 5 条是为了让 all_pairs 出到 10 对，够 audit_static_bias 的 n>=8 起判门槛。
        sizes = [ShotSize.MS, ShotSize.CU, ShotSize.FS, ShotSize.LS, ShotSize.OTS]
        moves = [CameraMove.PAN_R, CameraMove.DOLLY_IN, CameraMove.ORBIT_L,
                 CameraMove.HANDHELD, CameraMove.CRANE]
        renders, qc_map = [], {}
        for si in range(5):
            for ti, v in enumerate(vids):
                r = rec(f"sh{si:02d}", ti, v, size=sizes[si], move=moves[si],
                        action=f"动作{si}", env=f"环境{si}", scene=f"sc{si:02d}",
                        subj=["lin", "qi"][si % 2])
                renders.append(r)
                qc_map[r.result.job_id] = probe_reports[ti]

        auto = build_pairs_from_qc(renders, qc_map, name="ep01-auto")
        assert len(auto) == 5, len(auto)
        for p in auto.pairs:
            assert p.source is PairSource.QC_AUTO and p.score_kind is ScoreKind.QC
            assert p.win_score > p.lose_score and p.margin >= auto.min_margin
            # 赢家 = 振幅最小的 take0，输家 = 振幅最大的 take2（best_worst）
            assert p.win.take == 0 and p.lose.take == 2, (p.win.take, p.lose.take)
        gap = auto.score_gap_stats()
        print(f"\n[2] build_pairs_from_qc: {len(renders)} 次出片 -> {len(auto)} 对；"
              f"分差 median={gap['median']} max={gap['max']}")

        # 分差门槛必须真的拦得住：把门槛抬到 0.5（实际分差约 0.23）
        strict = build_pairs_from_qc(renders, qc_map, min_margin=0.5, name="strict")
        assert len(strict) == 0, len(strict)
        # 赢家分数门槛：抬到 0.99 后没有赢家够格
        picky = build_pairs_from_qc(renders, qc_map, min_win_score=0.99, name="picky")
        assert len(picky) == 0, len(picky)
        # all_pairs 策略：3 个 take 最多 3 组合，扣掉分差不足的
        allp = build_pairs_from_qc(renders, qc_map, strategy="all_pairs", name="allp")
        assert len(allp) > len(auto), (len(allp), len(auto))
        print(f"    门槛有效：margin>=0.5 -> 0 对；win>=0.99 -> 0 对；"
              f"all_pairs -> {len(allp)} 对")

        # ============================================================
        # [3] 保守化偏差审计：必须检出 100% 的对里赢家更静止
        # ============================================================
        bias = audit_static_bias(allp)
        assert bias.n_comparable == len(allp)
        assert bias.ratio == 1.0, bias
        assert bias.mean_delta < 0, bias
        assert bias.suspicious, bias
        print(f"\n[3] audit_static_bias: {bias.summary()}")

        # 反例：人工对没有 motion 记录 -> 审计弃权而不是误报
        empty = PreferenceDataset(name="x", pairs=[p.model_copy(update={"meta": {}})
                                                   for p in allp.pairs])
        assert not audit_static_bias(empty).suspicious
        print("    无 motion 数据时审计弃权（不误报）")

        # ============================================================
        # [4] 人工回流：两种 CSV 格式
        # ============================================================
        csv_a = work / "ratings.csv"
        csv_a.write_text(
            "shot_id,take,video_uri,rating,rater,prompt,vq,mq,ta\n"
            # sh00：两个评委都更喜欢大运镜的 take2 —— 和自动标注**相反**
            f"sh00,0,{vids[0]},2,ann,雨夜天台上她转身走向栏杆,3,1,4\n"
            f"sh00,0,{vids[0]},3,bob,雨夜天台上她转身走向栏杆,3,2,4\n"
            f"sh00,2,{vids[2]},5,ann,雨夜天台上她转身走向栏杆,4,5,5\n"
            f"sh00,2,{vids[2]},5,bob,雨夜天台上她转身走向栏杆,4,5,5\n"
            # sh01：分差只有 1 档（0.25 归一）> 默认 0.15，应收
            f"sh01,0,{vids[0]},3,ann,旧书店二楼他抽出一本书,3,3,3\n"
            f"sh01,1,{vids[1]},4,ann,旧书店二楼他抽出一本书,4,4,4\n"
            # sh02：两个 take 同分，应被 min_margin 拦掉
            f"sh02,0,{vids[0]},4,ann,便利店后厨,4,4,4\n"
            f"sh02,1,{vids[1]},4,ann,便利店后厨,4,4,4\n",
            encoding="utf-8")
        human = build_pairs_from_human(csv_a, name="ep01-human")
        got = {p.shot_id: (p.win.take, p.lose.take) for p in human.pairs}
        assert set(got) == {"sh00", "sh01"}, got          # sh02 被平分拦掉
        assert got["sh00"] == (2, 0), got                  # 人工选了大运镜，和自动相反
        assert all(p.source is PairSource.HUMAN for p in human.pairs)
        assert all(p.score_kind is ScoreKind.HUMAN_LIKERT for p in human.pairs)
        sh00 = next(p for p in human.pairs if p.shot_id == "sh00")
        assert sh00.meta["raters_win"] == 2 and sh00.meta["likert_range"] == [2.0, 5.0]
        print(f"\n[4] 人工回流 格式A：8 行 -> {len(human)} 对（同分的 sh02 被 min_margin 拦掉）")
        print(f"    sh00 人工选 take2(大运镜)，自动选 take0(小运镜) —— "
              f"两种标注方向相反，正是掺人工对的意义")

        csv_b = work / "pairwise.csv"
        csv_b.write_text(
            "shot_id,win_uri,lose_uri,rater,prompt\n"
            f"sh10,{vids[2]},{vids[0]},ann,地铁末班车厢\n"
            f"sh10,{vids[2]},{vids[0]},bob,地铁末班车厢\n"
            f"sh10,{vids[0]},{vids[2]},cat,地铁末班车厢\n"     # 少数派
            f"sh11,{vids[0]},{vids[1]},ann,废弃泳池边\n"
            f"sh11,{vids[1]},{vids[0]},bob,废弃泳池边\n",      # 1:1 平票 -> 丢弃
            encoding="utf-8")
        hp = build_pairs_from_human(csv_b, name="pairwise")
        assert len(hp) == 1 and hp.pairs[0].shot_id == "sh10", [p.shot_id for p in hp.pairs]
        assert hp.pairs[0].win.video_uri == vids[2]
        assert abs(hp.pairs[0].confidence - 2 / 3) < 1e-6, hp.pairs[0].confidence
        assert hp.pairs[0].score_kind is ScoreKind.HUMAN_PAIRWISE
        print(f"    格式B：5 行 -> {len(hp)} 对（2:1 多数票收下，1:1 平票丢弃，"
              f"confidence={hp.pairs[0].confidence:.2f}）")

        # 合并：人工覆盖不了自动的 pair_id（不同 shot），所以是纯相加
        mixed = auto.merge(human, name="ep01-mixed")
        assert len(mixed) == len(auto) + len(human)
        assert mixed.by_source() == {"qc_auto": 5, "human": 2}, mixed.by_source()
        print(f"    合并后来源分布 {mixed.by_source()}")

        # ============================================================
        # [5] VideoDPO 频率重加权
        # ============================================================
        assert all(p.weight == 1.0 for p in mixed.pairs)
        mixed.reweight_videodpo(alpha=1.0, bins=20)
        ws = [p.weight for p in mixed.pairs]
        # β = 最高频分数的概率 => 落在最高频档位上的对权重 ≈ 1，其余被抬高
        assert min(ws) >= 1.0 - 1e-9, ws
        assert max(ws) > min(ws) * 1.5, ws
        # 罕见分数组合（人工的 1.0 vs 0.0）权重必须高于扎堆的自动对
        w_auto = [p.weight for p in mixed.pairs if p.source is PairSource.QC_AUTO]
        w_human = [p.weight for p in mixed.pairs if p.source is PairSource.HUMAN]
        assert max(w_human) > max(w_auto), (w_auto, w_human)
        print(f"\n[5] VideoDPO 重加权 w_pair=(β/√(p_W·p_L))^α："
              f"weight ∈ [{min(ws):.3f}, {max(ws):.3f}]；"
              f"自动对 max={max(w_auto):.3f} < 罕见的人工对 max={max(w_human):.3f}")

        # ============================================================
        # [6] 三种训练格式导出
        # ============================================================
        for fmt in ("trl", "pickapic", "videodpo"):
            d = work / f"export_{fmt}"
            f = export_for_training(mixed, d, fmt)  # type: ignore[arg-type]
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            assert man["n_pairs"] == len(mixed) and man["format"] == fmt
            assert "static_bias_audit" in man and man["dataset_hash"]
            if fmt == "videodpo":
                obj = json.loads(f.read_text(encoding="utf-8"))
                assert len(obj["pairs"]) == len(mixed)
                assert obj["pairs"][0]["win"] and obj["pairs"][0]["weight"]
            else:
                rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
                assert len(rows) == len(mixed)
                if fmt == "trl":
                    assert set(rows[0]) >= {"prompt", "chosen", "rejected"}
                    assert rows[0]["chosen"] != rows[0]["rejected"]
                else:
                    assert set(rows[0]) >= {"caption", "video_0", "video_1",
                                            "label_0", "label_1"}
                    # 赢家位置必须交替，否则模型能学到「0 号总是赢」的位置先验
                    labels = [r["label_0"] for r in rows]
                    assert set(labels) == {0.0, 1.0}, labels
                    for r in rows:
                        assert r["label_0"] + r["label_1"] == 1.0
            print(f"[6] 导出 {fmt:<9} -> {f.name} + manifest.json  ({len(mixed)} 对)")

        # materialize：真把 mp4 拷进 videos/
        mdir = work / "export_mat"
        export_for_training(mixed, mdir, "trl", materialize=True)
        mp4s = list((mdir / "videos").glob("*.mp4"))
        assert mp4s, "materialize 没拷出任何视频"
        rows = [json.loads(x) for x in (mdir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(not Path(r["chosen"]).is_absolute() for r in rows), rows[0]
        assert (mdir / rows[0]["chosen"]).exists()
        print(f"    materialize=True 拷出 {len(mp4s)} 个 mp4，清单路径为相对路径且可解析")

        # 落盘 / 读回 / 指纹
        saved = mixed.save(work / "pref.json")
        back = PreferenceDataset.load(saved)
        assert back.dataset_hash() == mixed.dataset_hash()
        assert len(back) == len(mixed)
        # 翻转胜负必须换指纹，否则标签污染查不出来
        flipped = PreferenceDataset(name="f", pairs=[p.flipped() for p in mixed.pairs])
        assert flipped.dataset_hash() != mixed.dataset_hash()
        print(f"    dataset_hash={mixed.dataset_hash()}；翻转胜负后指纹改变（标签污染可检出）")

        # ============================================================
        # [7] FlywheelMonitor：模拟五轮飞轮，坍缩必须被抓到
        # ============================================================
        all_sizes = list(ShotSize)
        all_moves = list(CameraMove)

        acts = ["她推门而入抖落肩上的雪", "他回头望向对面楼亮着的窗",
                "钥匙被慢慢攥进掌心", "蹲下系鞋带时瞥见身后的人",
                "端起凉透的茶又放下", "把外套披到对方肩上",
                "抽屉最深处翻出一张旧车票", "扶梯上被人流推着向前",
                "烟被按灭在窗台的积水里", "听见门铃后僵在原地",
                "拆开信封却没有读", "沿着护栏走到尽头停住",
                "抬手挡住突然打过来的车灯", "在镜子前撕掉贴了半天的创可贴",
                "把伞递出去又收回来", "数完第三遍零钱才抬头"]
        envs = ["雨夜天台", "老式电梯间", "旧书店二楼", "深夜便利店后厨",
                "废弃泳池边", "地铁末班车厢", "沿海公路收费站",
                "医院走廊尽头", "拆到一半的旧礼堂", "冬天的露天球场"]
        lights = ["侧逆光霓虹反射", "顶光压住眉骨", "街灯是唯一光源",
                  "日光灯全开的平光", "车灯扫过的移动光斑", "火光从下方打上来",
                  "百叶窗切碎的条形光", "清晨蓝调时刻"]
        styles = ["胶片颗粒冷调", "高对比黑白", "柔焦暖调浅景深", "数字锐利高饱和",
                  "低饱和纪录片质感", "手持轻微晃动", "宽银幕遮幅", "颗粒粗砺的翻拍感"]

        def synth_round(n: int, n_size: int, n_move: int, n_scene: int,
                        vocab: int) -> DistillDataset:
            """造一轮赢家样本。

            n_size/n_move/n_scene 越小 = 镜头语法越窄；vocab 越小 = 提示词越模板化。
            四个槽位用互质的步长取值（i、i//3、i//5、i//7），避免各槽同步变化
            导致「看起来有 n 种组合、其实只有 n 种」。
            """
            out = []
            for i in range(n):
                k = max(1, vocab)
                prompt = "，".join([
                    acts[i % min(k, len(acts))],
                    envs[(i // 3) % min(k, len(envs))],
                    lights[(i // 5) % min(k, len(lights))],
                    styles[(i // 7) % min(k, len(styles))],
                ])
                out.append(TeacherSample(
                    shot_id=f"r{i:03d}", video_uri=f"s3://w/{i}.mp4", duration_s=3.0,
                    prompt=prompt,
                    shot_size=all_sizes[i % n_size], camera_move=all_moves[i % n_move],
                    scene_id=f"sc{i % n_scene:02d}",
                    subject_ids=[["lin", "qi", "mo"][i % 3]],
                    qc_score=0.9,
                ))
            return DistillDataset(name="w", samples=out)

        mon = FlywheelMonitor("ep01")
        # 第 0 轮：健康 —— 宽镜头语法、通过率一般
        mon.observe(synth_round(120, 10, 14, 8, 8), pass_rate=0.72, label="r0-baseline")
        assert mon.level is AlertLevel.OK, [str(a) for a in mon.check()]
        print(f"\n[7] 飞轮第 0 轮（健康）：level={mon.level.value}")

        # 第 1 轮：轻微收窄，通过率小涨 —— 还不该报 R1（没到 5pp）
        mon.observe(synth_round(120, 9, 12, 7, 7), pass_rate=0.75, label="r1")
        rules1 = {a.rule for a in mon.check()}
        assert "R1_剪刀差" not in rules1, rules1
        print(f"    第 1 轮（+3pp，未达 5pp 门槛）：不报 R1，告警 {sorted(rules1) or '无'}")

        # 第 2 轮：剪刀差 —— 通过率 +9pp 同时语法明显收窄
        mon.observe(synth_round(120, 5, 6, 4, 4), pass_rate=0.84, label="r2-剪刀差")
        rules2 = {a.rule for a in mon.check()}
        assert "R1_剪刀差" in rules2, rules2
        assert mon.level in (AlertLevel.ALERT, AlertLevel.CRITICAL), mon.level
        print(f"    第 2 轮（+9pp & 多样性下滑）：触发 {sorted(rules2)}")

        # 第 3 轮：继续收窄，覆盖率与累计跌幅都该炸
        mon.observe(synth_round(120, 3, 3, 2, 2), pass_rate=0.93, label="r3")
        rules3 = {a.rule for a in mon.check()}
        assert "R5_覆盖率" in rules3 and "R2_累计多样性" in rules3, rules3
        assert "R8_趋势" in rules3, rules3
        print(f"    第 3 轮：触发 {sorted(rules3)}")

        # 第 4 轮：彻底坍缩 + 带上保守化审计 -> CRITICAL
        collapsed = synth_round(120, 2, 1, 1, 1)
        mon.observe(collapsed, pass_rate=0.99, label="r4-坍缩",
                    bias=audit_static_bias(allp))
        rules4 = {a.rule for a in mon.check()}
        assert mon.level is AlertLevel.CRITICAL, (mon.level, rules4)
        for r in ("R1_剪刀差", "R2_累计多样性", "R3_单类占比", "R4_提示词模板化",
                  "R5_覆盖率", "R6_保守化偏差", "R7_通过率饱和", "R8_趋势"):
            assert r in rules4, (r, sorted(rules4))
        print(f"    第 4 轮：八条规则全中 -> level={mon.level.value.upper()}")
        print()
        print(mon.report())

        # 有效模式数确实在掉（Vendi Score 的 one-hot 特例）
        m0, m4 = mon.rounds[0], mon.rounds[-1]
        assert m0.modes_shot_size > m4.modes_shot_size > 0
        assert m0.modes_camera_move > m4.modes_camera_move > 0
        print(f"\n    有效模式数 exp(H)：景别 {m0.modes_shot_size} -> {m4.modes_shot_size}；"
              f"运镜 {m0.modes_camera_move} -> {m4.modes_camera_move}")

        # 通过率单独上升（多样性同步上升）不该报 R1
        ok_mon = FlywheelMonitor("healthy")
        ok_mon.observe(synth_round(120, 6, 8, 5, 5), pass_rate=0.70, label="h0")
        ok_mon.observe(synth_round(120, 11, 16, 8, 8), pass_rate=0.85, label="h1")
        assert "R1_剪刀差" not in {a.rule for a in ok_mon.check()}, ok_mon.check()
        assert ok_mon.level is AlertLevel.OK, [str(a) for a in ok_mon.check()]
        print(f"    对照组：通过率 +15pp 但多样性同步上升 -> level={ok_mon.level.value}（不误报）")

        # observe_from_qc：直接数 QCReport
        qmon = FlywheelMonitor("fromqc")
        r = qmon.observe_from_qc(auto, probe_reports * 5, label="q0")
        n_pass = sum(1 for x in probe_reports * 5 if x.verdict is Verdict.PASS)
        assert r.n_generated == 15 and r.n_passed == n_pass
        assert abs(r.pass_rate - n_pass / 15) < 1e-9
        print(f"    observe_from_qc：15 份 QCReport -> 通过 {n_pass}，"
              f"通过率 {r.pass_rate:.1%}（直接吃 qc.QCReport 的三态 verdict）")

        # 阈值可覆盖，写错名字必须报错
        tight = FlywheelMonitor("tight", pass_rate_jump=0.01, diversity_drop_rel=0.01)
        assert tight.pass_rate_jump == 0.01
        try:
            FlywheelMonitor("bad", no_such_threshold=1.0)
        except ValueError as e:
            assert "no_such_threshold" in str(e)
        else:
            raise AssertionError("未知阈值没有报错")

        # 序列化
        d = mon.to_dict()
        assert d["level"] == "critical" and len(d["rounds"]) == 5
        assert d["thresholds"]["cum_drop_critical"] == 0.30
        json.dumps(d, ensure_ascii=False)

        # ============================================================
        # [8] 接口兼容：qc_score_and_pass 必须认 QCReport 的三态 verdict
        # ============================================================
        from .distill import qc_score_of
        rep = probe_reports[0]
        s1, p1 = qc_score_and_pass(rep)
        s2, p2 = qc_score_of(rep)
        assert s1 == s2 == rep.score
        assert p1 is True and p2 is None, (p1, p2)   # 这正是本模块要补的那个洞
        assert qc_score_and_pass({"score": 0.8, "verdict": "retake"}) == (0.8, False)
        assert qc_score_and_pass(0.5) == (0.5, None)
        assert qc_score_and_pass(None) == (0.0, None)
        print(f"\n[8] qc_score_and_pass 认 QCReport.verdict 三态："
              f"distill.qc_score_of 读成 passed=None，本模块读成 passed={p1}")

        print("\n[selftest] dpo.py 全部断言通过")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    _selftest()
