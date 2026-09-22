#!/usr/bin/env python3
"""长片产线成本计算器 —— 全 API 路线 vs 自建 GPU 路线的对比表。

为什么要有这个东西：
「自建便宜还是买 API 便宜」不是一个可以拍脑袋的问题，答案随四个输入翻转 ——
总时长、重拍率、画质档位、以及你愿不愿意吃抢占式机器。产线排期前跑一次这个，
比事后看账单便宜得多。

设计约束：
1. **价格数据一律外置在 prices.yaml**，代码里不写任何数字。价格每个月都在变，
   改价格不该触发代码 review。
2. 估计值一律给区间，并在表尾列出每条假设的可信度。给单点数字会被当成承诺。
3. 与 ``longfilm.lora.GPU_HOURLY_USD`` 交叉校验：同一份机时价在两处漂移，
   会让 router 的成本打分和排期表互相打架，所以导入得到就比一次，不一致就告警。

用法：
    python3 deploy/cost_calculator.py --episodes 12 --shots 80 --seconds 6 --retake 0.35
    python3 deploy/cost_calculator.py --preset draft_480p_4step --tier community --json
    python3 deploy/cost_calculator.py --selftest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

log = logging.getLogger("cost_calculator")

HERE = Path(__file__).resolve().parent
DEFAULT_PRICES = HERE / "prices.yaml"

Tier = Literal["community", "secure", "serverless", "vast"]
_TIER_FIELD: dict[str, str] = {
    "community": "runpod_community",
    "secure": "runpod_secure",
    "serverless": "runpod_serverless",
    "vast": "vast_typical",
    "paperspace": "paperspace",
}

#: prices.yaml 的 gpus[].key -> longfilm.lora.GPU_HOURLY_USD 的键。
#: 两边命名不同是历史原因（lora.py 先落盘），这里做映射而不是改任何一边 ——
#: 改 lora.py 的键会打断已经引用它的训练脚本。
_LORA_GPU_ALIAS: dict[str, str] = {
    "H100_SXM": "H100", "H200": "H200", "A100_SXM": "A100",
    "RTX_4090": "4090", "RTX_5090": "5090", "L40S": "L40S",
}


# ---------------------------------------------------------------- 文本排版


def _cell_width(s: str) -> int:
    """按终端显示宽度算字符串长度。

    中文表头和英文数字混排时，len() 算出来的宽度差一倍，列就会错开。
    East Asian Width 为 W/F 的字符占两列，这是对齐 CJK 表格的唯一正确做法。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int, align: str = "l") -> str:
    gap = max(0, width - _cell_width(s))
    if align == "r":
        return " " * gap + s
    if align == "c":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def render_table(headers: list[str], rows: list[list[str]], aligns: str = "") -> str:
    """渲染一张定宽表。不用第三方库 —— 云机器上第一次跑时还没装依赖。"""
    aligns = aligns or "l" * len(headers)
    widths = [_cell_width(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], _cell_width(c))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [sep, "| " + " | ".join(_pad(h, widths[i], "c") for i, h in enumerate(headers)) + " |", sep]
    for r in rows:
        out.append("| " + " | ".join(_pad(c, widths[i], aligns[i]) for i, c in enumerate(r)) + " |")
    out.append(sep)
    return "\n".join(out)


def _usd(x: float) -> str:
    """金额格式。跨 4 个数量级，固定小数位会要么全是 0 要么看不清。"""
    if x >= 10000:
        return f"${x:,.0f}"
    if x >= 100:
        return f"${x:,.1f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.4f}"


def _rng(lo: float, hi: float, fmt=_usd) -> str:
    return fmt(lo) if abs(hi - lo) < 1e-9 else f"{fmt(lo)} ~ {fmt(hi)}"


# ---------------------------------------------------------------- 输入


@dataclass
class Scenario:
    """一次排期的输入。字段名与制片口径对齐，不用工程黑话。"""

    episodes: int = 12
    shots_per_episode: int = 80
    seconds_per_shot: float = 6.0
    retake_rate: float = 0.35          # 0.35 = 平均每个镜头要多生成 0.35 次
    preset: str = "normal_720p_12step"
    tier: str = "community"
    sessions: int = 0                  # 开机次数；0 = 按集数算（一集一开）
    width: int = 0                     # 0 = 用 preset 的分辨率
    height: int = 0
    fps: int = 0

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.shots_per_episode < 1:
            raise ValueError("集数与每集镜头数必须 >= 1")
        if self.seconds_per_shot <= 0:
            raise ValueError("每镜秒数必须 > 0")
        if not 0.0 <= self.retake_rate <= 5.0:
            raise ValueError("重拍率取值 0~5（0.35 表示 35%）")
        if self.tier not in _TIER_FIELD:
            raise ValueError(f"未知机型档位 {self.tier!r}，可选 {sorted(_TIER_FIELD)}")

    @property
    def delivered_shots(self) -> int:
        """成片镜头数。"""
        return self.episodes * self.shots_per_episode

    @property
    def delivered_seconds(self) -> float:
        return self.delivered_shots * self.seconds_per_shot

    @property
    def generated_seconds(self) -> float:
        """实际要生成的秒数（含重拍）。成本全部按这个算，不是按成片长度。"""
        return self.delivered_seconds * (1.0 + self.retake_rate)

    @property
    def n_sessions(self) -> int:
        return self.sessions or self.episodes


# ---------------------------------------------------------------- 价目表


@dataclass
class PriceBook:
    """prices.yaml 的内存形态。只做查表与校验，不做任何计算。"""

    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PRICES) -> PriceBook:
        # 延迟导入：_miniyaml 与本文件同目录，但脚本可能从任意 cwd 被调用
        sys.path.insert(0, str(HERE))
        import _miniyaml

        p = Path(path)
        data = _miniyaml.load(str(p))
        if not isinstance(data, dict) or "gpus" not in data:
            raise ValueError(f"{p} 不像一份价目表：缺 gpus 段")
        return cls(raw=data, path=p)

    @property
    def verified_at(self) -> str:
        return str(self.raw.get("meta", {}).get("verified_at", "未标注"))

    @property
    def usd_cny(self) -> float:
        return float(self.raw["meta"]["fx"]["usd_cny"])

    def gpus(self) -> list[dict[str, Any]]:
        return list(self.raw["gpus"])

    def gpu(self, key: str) -> dict[str, Any]:
        for g in self.gpus():
            if g["key"] == key:
                return g
        raise KeyError(f"未登记的 GPU {key!r}；已登记 {[g['key'] for g in self.gpus()]}")

    def preset(self, key: str) -> dict[str, Any]:
        for p in self.raw["presets"]:
            if p["key"] == key:
                return p
        raise KeyError(f"未登记的画质档 {key!r}；已登记 {[p['key'] for p in self.raw['presets']]}")

    def api_models(self) -> list[dict[str, Any]]:
        return list(self.raw["api_models"])

    def overheads(self) -> dict[str, Any]:
        return dict(self.raw["selfhost_overheads"])

    def storage(self) -> dict[str, Any]:
        return dict(self.raw["storage_usd_per_gb_month"])

    def hourly(self, gpu_key: str, tier: str) -> float:
        """某卡某档位的小时价。0 表示该档位不提供这张卡。"""
        return float(self.gpu(gpu_key).get(_TIER_FIELD[tier], 0.0) or 0.0)

    def cross_check_with_lora(self) -> list[str]:
        """与 longfilm.lora 的机时表对账。返回不一致项的描述列表。

        为什么要对账：router 用 lora.GPU_HOURLY_USD 做「训 LoRA 值不值」的判断，
        这里用 prices.yaml 做「自建值不值」的判断。两份价目表一旦漂移，
        同一台机器在两个模块里是两个价，排期表就会自相矛盾。
        """
        try:
            sys.path.insert(0, str(HERE.parent / "src"))
            from longfilm.lora import GPU_HOURLY_USD  # 延迟导入：脱离仓库也要能跑
        except Exception as exc:  # noqa: BLE001 —— 导入失败不是错误，只是少一层校验
            log.debug("未能导入 longfilm.lora（%s），跳过价目交叉校验", exc)
            return []
        issues: list[str] = []
        for g in self.gpus():
            alias = _LORA_GPU_ALIAS.get(g["key"])
            if alias not in GPU_HOURLY_USD:
                continue
            lc, ls = GPU_HOURLY_USD[alias]
            if abs(lc - float(g["runpod_community"])) > 0.005:
                issues.append(f"{g['key']} community: prices.yaml={g['runpod_community']} "
                              f"vs lora.py={lc}")
            if abs(ls - float(g["runpod_secure"])) > 0.005:
                issues.append(f"{g['key']} secure: prices.yaml={g['runpod_secure']} "
                              f"vs lora.py={ls}")
        return issues


# ---------------------------------------------------------------- 计算


@dataclass
class ApiQuote:
    key: str
    label: str
    usd_per_second: float
    total_usd: float
    per_episode_usd: float
    quality_tier: int
    verified: bool
    note: str = ""


@dataclass
class SelfHostQuote:
    gpu: str
    label: str
    vram_gb: int
    hourly_usd: float
    gpu_hours_low: float
    gpu_hours_high: float
    compute_usd_low: float
    compute_usd_high: float
    overhead_usd: float
    total_usd_low: float
    total_usd_high: float
    per_episode_usd_low: float
    per_episode_usd_high: float
    effective_usd_per_second_low: float
    effective_usd_per_second_high: float
    fits_vram: bool
    warnings: list[str] = field(default_factory=list)


def quote_api(book: PriceBook, sc: Scenario) -> list[ApiQuote]:
    """全 API 路线：每家模型一行。

    口径：API 失败（审核拒绝/超时）一般不计费或可申诉，所以重拍只按
    generated_seconds 线性放大，不额外加损耗。
    """
    preset = book.preset(sc.preset)
    w = sc.width or int(preset["resolution"][0])
    h = sc.height or int(preset["resolution"][1])
    fps = sc.fps or int(preset["fps"])
    fx = book.usd_cny

    out: list[ApiQuote] = []
    for m in book.api_models():
        billing = m["billing"]
        if billing == "per_second":
            ups = float(m["usd_per_second"])
        elif billing == "token":
            # 方舟公式：token = 宽 × 高 × 帧率 × 时长 / 1024（每秒即去掉时长）
            tok_per_s = w * h * fps / 1024.0
            ups = tok_per_s / 1e6 * float(m["price_cny_per_mtoken"]) / fx
        else:
            log.warning("模型 %s 的计费方式 %r 未实现，跳过", m["key"], billing)
            continue
        total = ups * sc.generated_seconds
        out.append(ApiQuote(
            key=m["key"], label=m["label"], usd_per_second=ups, total_usd=total,
            per_episode_usd=total / sc.episodes,
            quality_tier=int(m.get("quality_tier", 3)),
            verified=bool(m.get("verified", False)),
            note=str(m.get("note", "") or ""),
        ))
    out.sort(key=lambda q: q.total_usd)
    return out


def quote_selfhost(book: PriceBook, sc: Scenario,
                   calibration: dict[str, float] | None = None) -> list[SelfHostQuote]:
    """自建路线：每张卡一行。

    与 API 路线的关键口径差异：**自建的失败镜头照样烧机时**。
    所以除了重拍本身的 generated_seconds，还要加一份「崩在中途」的浪费，
    比例取 selfhost_overheads.failed_shot_gpu_ratio。
    """
    preset = book.preset(sc.preset)
    ov = book.overheads()
    st = book.storage()
    cal = calibration or {}

    lo_base = float(cal.get("low", preset["h100_gpu_seconds_per_output_second_low"]))
    hi_base = float(cal.get("high", preset["h100_gpu_seconds_per_output_second_high"]))
    min_vram = int(preset.get("min_vram_gb", 24))

    # 崩掉的镜头只烧掉一部分机时（崩在第几步不确定），不是整镜重烧
    waste_s = sc.delivered_seconds * sc.retake_rate * float(ov["failed_shot_gpu_ratio"])
    billable_output_s = sc.generated_seconds + waste_s

    out: list[SelfHostQuote] = []
    for g in book.gpus():
        hourly = book.hourly(g["key"], sc.tier)
        if hourly <= 0:
            log.debug("%s 在 %s 档位无报价，跳过", g["key"], sc.tier)
            continue
        rel = float(g["rel_throughput"])
        gpu_s_lo = billable_output_s * lo_base / rel
        gpu_s_hi = billable_output_s * hi_base / rel

        # 固定开销：每次开机的暖机 + 一次性权重下载 + 权重盘的月费（按占用时长摊）
        warm_h = float(ov["warmup_minutes_per_session"]) / 60.0 * sc.n_sessions
        cold_h = float(ov["cold_download_minutes"]) / 60.0
        disk_gb = float(ov["model_disk_gb"])
        busy_h_mid = (gpu_s_lo + gpu_s_hi) / 2 / 3600.0 + warm_h + cold_h
        storage_usd = disk_gb * float(st["network_standard"]) * (busy_h_mid / 720.0)
        overhead_usd = (warm_h + cold_h) * hourly + storage_usd

        c_lo = gpu_s_lo / 3600.0 * hourly
        c_hi = gpu_s_hi / 3600.0 * hourly
        t_lo, t_hi = c_lo + overhead_usd, c_hi + overhead_usd

        warns: list[str] = []
        fits = int(g["vram_gb"]) >= min_vram
        if not fits:
            warns.append(f"显存 {g['vram_gb']}GB < 该档位需要的 {min_vram}GB")
        if int(g["vram_gb"]) < 40:
            warns.append("需 blocks_to_swap，实际步时可能比估计更差")
        if sc.tier == "paperspace":
            # Paperspace 的高端卡要先买订阅才解锁。这笔钱与跑多少小时无关，按月固定，
            # 所以不能摊进每秒单价 —— 只训一个 LoRA（几十小时）时它能占总成本一半以上。
            # 不足一个月按一个月算：订阅没有按天退款这回事。
            subs = book.raw.get("subscriptions", {}) or {}
            need = max((float(pl.get("usd_per_month", 0.0))
                        for pl in subs.values()
                        if g["key"] in (pl.get("unlocks") or [])), default=0.0)
            if need:
                months = max(1.0, busy_h_mid / 720.0)
                t_lo += need * months
                t_hi += need * months
                warns.append(f"需 ${need:.0f}/月订阅才解锁该卡（已按 {months:.1f} 个月计入总价）")
        if sc.tier == "community":
            warns.append("Community 可被抢占，长任务要能断点续跑")
        if sc.tier == "serverless":
            warns.append("Serverless 冷启动拉权重那段也计费，实际单价高于表列")

        out.append(SelfHostQuote(
            gpu=g["key"], label=g["label"], vram_gb=int(g["vram_gb"]), hourly_usd=hourly,
            gpu_hours_low=gpu_s_lo / 3600.0, gpu_hours_high=gpu_s_hi / 3600.0,
            compute_usd_low=c_lo, compute_usd_high=c_hi, overhead_usd=overhead_usd,
            total_usd_low=t_lo, total_usd_high=t_hi,
            per_episode_usd_low=t_lo / sc.episodes, per_episode_usd_high=t_hi / sc.episodes,
            effective_usd_per_second_low=t_lo / sc.delivered_seconds,
            effective_usd_per_second_high=t_hi / sc.delivered_seconds,
            fits_vram=fits, warnings=warns,
        ))
    out.sort(key=lambda q: (not q.fits_vram, q.total_usd_low))
    return out


def breakeven_seconds(book: PriceBook, sc: Scenario, api: ApiQuote,
                      self_q: SelfHostQuote) -> float | None:
    """自建追平某家 API 所需的成片秒数。

    模型：自建 = 固定开销 + 每秒边际成本；API = 纯每秒。
    边际成本高于 API 单价时永远追不平，返回 None —— 这种情况下「规模上去就便宜了」
    是错的，必须说清楚。
    """
    marginal = (self_q.total_usd_low - self_q.overhead_usd) / max(sc.delivered_seconds, 1e-9)
    if marginal >= api.usd_per_second:
        return None
    return self_q.overhead_usd / (api.usd_per_second - marginal)


# ---------------------------------------------------------------- 渲染


def render_report(book: PriceBook, sc: Scenario,
                  calibration: dict[str, float] | None = None) -> str:
    preset = book.preset(sc.preset)
    apis = quote_api(book, sc)
    selfs = quote_selfhost(book, sc, calibration)
    lines: list[str] = []

    w = sc.width or int(preset["resolution"][0])
    h = sc.height or int(preset["resolution"][1])
    fps = sc.fps or int(preset["fps"])

    lines.append("=" * 78)
    lines.append("longfilm 成本对比 —— 全 API 路线 vs 自建 GPU 路线")
    lines.append(f"价目表 {book.path.name}，查证日期 {book.verified_at}；汇率 1 USD = {book.usd_cny} CNY")
    lines.append("=" * 78)
    lines.append("")
    lines.append(render_table(
        ["输入", "值"],
        [["集数", f"{sc.episodes}"],
         ["每集镜头数", f"{sc.shots_per_episode}"],
         ["每镜秒数", f"{sc.seconds_per_shot:g}s"],
         ["重拍率", f"{sc.retake_rate:.0%}"],
         ["画质档", f"{preset['label']}"],
         ["出片规格", f"{w}x{h} @ {fps}fps"],
         ["机型档位", sc.tier],
         ["成片总长", f"{sc.delivered_seconds:,.0f}s（{sc.delivered_seconds / 60:.1f} 分钟）"],
         ["实际生成总长", f"{sc.generated_seconds:,.0f}s（含重拍）"]],
        aligns="lr"))
    lines.append("")

    lines.append("── 路线 A：全走官方/第三方 API " + "─" * 44)
    rows = []
    for q in apis:
        rows.append([
            q.label,
            _usd(q.usd_per_second) + "/s",
            _usd(q.per_episode_usd),
            _usd(q.total_usd),
            "★" * q.quality_tier,
            "已核" if q.verified else "二手",
        ])
    lines.append(render_table(
        ["模型", "单价", "每集", f"{sc.episodes} 集合计", "画质", "价格来源"],
        rows, aligns="lrrrcc"))
    lines.append("")

    lines.append(f"── 路线 B：自建 GPU（RunPod {sc.tier} 档） " + "─" * 36)
    rows = []
    for q in selfs:
        rows.append([
            q.label,
            _usd(q.hourly_usd) + "/h",
            _rng(q.gpu_hours_low, q.gpu_hours_high, lambda x: f"{x:,.1f}h"),
            _usd(q.overhead_usd),
            _rng(q.total_usd_low, q.total_usd_high),
            _rng(q.effective_usd_per_second_low, q.effective_usd_per_second_high),
            "OK" if q.fits_vram else "显存不足",
        ])
    lines.append(render_table(
        ["GPU", "机时价", "GPU 小时", "固定开销", f"{sc.episodes} 集合计", "折合每秒", "可行性"],
        rows, aligns="lrrrrrc"))
    lines.append("")

    # 结论
    best_api = apis[0]
    viable = [q for q in selfs if q.fits_vram]
    best_self = viable[0] if viable else None
    lines.append("── 结论 " + "─" * 68)
    lines.append(f"最便宜 API：{best_api.label} {_usd(best_api.total_usd)}"
                 f"（{_usd(best_api.usd_per_second)}/s）")
    if best_self is None:
        lines.append("没有一张登记的卡能吃下这个画质档 —— 先降档或换 profile。")
    else:
        lines.append(f"最便宜自建：{best_self.label} "
                     f"{_rng(best_self.total_usd_low, best_self.total_usd_high)}")
        delta_lo = best_self.total_usd_low - best_api.total_usd
        delta_hi = best_self.total_usd_high - best_api.total_usd
        if delta_hi < 0:
            lines.append(f"→ 自建在乐观与悲观两端都更便宜，省 {_usd(-delta_hi)} ~ {_usd(-delta_lo)}。")
        elif delta_lo > 0:
            lines.append(f"→ API 在两端都更便宜，自建要多花 {_usd(delta_lo)} ~ {_usd(delta_hi)}。"
                         "自建的理由只能是「可控/可微调/无审核」，不是省钱。")
        else:
            lines.append("→ 区间跨零：吞吐估计的不确定性大于两条路线的差价。"
                         "**先跑一次实测校准再定**，现在做决定等于掷硬币。")
        # 墙钟必须单独说：GPU 小时便宜不等于排得进档期。
        # 一张 4090 的 400 小时是 17 天连轴转，这条信息比单价更容易决定选型。
        d_lo = best_self.gpu_hours_low / 24.0
        d_hi = best_self.gpu_hours_high / 24.0
        lines.append(f"→ 单卡墙钟：{d_lo:,.1f} ~ {d_hi:,.1f} 天（24h 连跑）。"
                     f"要压到 7 天内需并行 {max(1, int(-(-d_hi // 7)))} 张卡 —— "
                     "并行不改总机时，只改档期与并发上限。")
        be = breakeven_seconds(book, sc, best_api, best_self)
        if be is None:
            lines.append("→ 自建的边际成本已高于该 API 单价：规模再大也追不平。")
        else:
            lines.append(f"→ 盈亏平衡点（乐观端）：成片 {be:,.0f} 秒 "
                         f"≈ {be / 60:,.1f} 分钟；本次排期 {sc.delivered_seconds:,.0f} 秒。")
    lines.append("")

    lines.append("── 假设与可信度 " + "─" * 60)
    lines.append(f"  [估计] 吞吐：H100 每出 1 秒成品烧 "
                 f"{preset['h100_gpu_seconds_per_output_second_low']}~"
                 f"{preset['h100_gpu_seconds_per_output_second_high']} 秒 GPU；"
                 "其余卡按 rel_throughput 线性换算。")
    lines.append("         这是按硬件规格外推的量级估计，**不是实测**。"
                 "跑一次真实渲染后用 --calibrate LOW HIGH 覆盖。")
    ov = book.overheads()
    lines.append(f"  [估计] 自建失败镜头按成功镜头机时的 "
                 f"{float(ov['failed_shot_gpu_ratio']):.0%} 计入浪费；API 失败按不计费处理。")
    lines.append(f"  [估计] 每次开机暖机 {ov['warmup_minutes_per_session']} 分钟 × "
                 f"{sc.n_sessions} 次；首次拉权重 {ov['cold_download_minutes']} 分钟。")
    lines.append(f"  [已核] 机时价与存储价：{book.raw['meta']['sources']['runpod']}（{book.verified_at}）")
    unverified = [q.label for q in apis if not q.verified]
    if unverified:
        lines.append(f"  [二手] 以下 API 单价来自测评/聚合站，排期前必须复核："
                     f"{'、'.join(unverified)}")
    lines.append("  [未计] 存储长期留存、出网流量、调色/超分/音频的额外机时、人力。")

    issues = book.cross_check_with_lora()
    if issues:
        lines.append("")
        lines.append("!! 价目交叉校验不一致（prices.yaml 与 longfilm.lora.GPU_HOURLY_USD）：")
        for i in issues:
            lines.append(f"   - {i}")
    return "\n".join(lines)


def report_json(book: PriceBook, sc: Scenario,
                calibration: dict[str, float] | None = None) -> dict[str, Any]:
    from dataclasses import asdict

    return {
        "prices_verified_at": book.verified_at,
        "scenario": asdict(sc),
        "delivered_seconds": sc.delivered_seconds,
        "generated_seconds": sc.generated_seconds,
        "api": [asdict(q) for q in quote_api(book, sc)],
        "selfhost": [asdict(q) for q in quote_selfhost(book, sc, calibration)],
    }


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cost_calculator.py",
        description="长片产线成本对比：全 API 路线 vs 自建 GPU 路线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--episodes", type=int, default=12, help="集数（默认 12）")
    p.add_argument("--shots", type=int, default=80, dest="shots_per_episode",
                   help="每集镜头数（默认 80）")
    p.add_argument("--seconds", type=float, default=6.0, dest="seconds_per_shot",
                   help="每镜秒数（默认 6）")
    p.add_argument("--retake", type=float, default=0.35, dest="retake_rate",
                   help="重拍率，0.35 = 35%%（默认 0.35）")
    p.add_argument("--preset", default="normal_720p_12step", help="画质档，见 prices.yaml presets")
    p.add_argument("--tier", default="community", choices=sorted(_TIER_FIELD),
                   help="机型档位（默认 community）")
    p.add_argument("--sessions", type=int, default=0, help="开机次数；0 = 按集数")
    p.add_argument("--width", type=int, default=0, help="覆盖 preset 分辨率宽")
    p.add_argument("--height", type=int, default=0, help="覆盖 preset 分辨率高")
    p.add_argument("--fps", type=int, default=0, help="覆盖 preset 帧率")
    p.add_argument("--prices", default=str(DEFAULT_PRICES), help="价目表路径")
    p.add_argument("--calibrate", nargs=2, type=float, metavar=("LOW", "HIGH"),
                   help="用实测的「H100 每秒成品所需 GPU 秒」覆盖 preset 估计值")
    p.add_argument("--json", action="store_true", help="输出 JSON 而不是表格")
    p.add_argument("--list-presets", action="store_true", help="列出可用画质档与 GPU 后退出")
    p.add_argument("--selftest", action="store_true", help="跑模块自测")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    if args.selftest:
        _selftest()
        return 0
    book = PriceBook.load(args.prices)
    if args.list_presets:
        print(render_table(["画质档", "说明", "最低显存"],
                           [[p["key"], p["label"], f"{p['min_vram_gb']}GB"]
                            for p in book.raw["presets"]]))
        print(render_table(["GPU", "显存", "community", "secure", "serverless", "vast(典型)"],
                           [[g["key"], f"{g['vram_gb']}GB", f"{g['runpod_community']}",
                             f"{g['runpod_secure']}", f"{g['runpod_serverless']}",
                             f"{g['vast_typical']}"] for g in book.gpus()], aligns="lrrrrr"))
        return 0

    sc = Scenario(
        episodes=args.episodes, shots_per_episode=args.shots_per_episode,
        seconds_per_shot=args.seconds_per_shot, retake_rate=args.retake_rate,
        preset=args.preset, tier=args.tier, sessions=args.sessions,
        width=args.width, height=args.height, fps=args.fps,
    )
    cal = {"low": args.calibrate[0], "high": args.calibrate[1]} if args.calibrate else None
    if args.json:
        print(json.dumps(report_json(book, sc, cal), ensure_ascii=False, indent=2))
    else:
        print(render_report(book, sc, cal))
    return 0


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    book = PriceBook.load()

    # 1) 价目表结构完整
    assert book.gpus(), "gpus 段为空"
    assert book.api_models(), "api_models 段为空"
    assert 5.0 < book.usd_cny < 9.0, f"汇率离谱：{book.usd_cny}"

    sc = Scenario(episodes=2, shots_per_episode=10, seconds_per_shot=5.0, retake_rate=0.5)
    assert sc.delivered_seconds == 100.0
    assert abs(sc.generated_seconds - 150.0) < 1e-9

    # 2) 成本对总量单调：镜头翻倍，API 成本翻倍（API 是纯线性的）
    a1 = quote_api(book, sc)
    sc2 = Scenario(episodes=4, shots_per_episode=10, seconds_per_shot=5.0, retake_rate=0.5)
    a2 = quote_api(book, sc2)
    by_key = {q.key: q for q in a1}
    for q in a2:
        assert abs(q.total_usd - by_key[q.key].total_usd * 2) < 1e-6, q.key

    # 3) 自建有固定开销 => 单位成本随规模下降（严格递减）
    s1 = quote_selfhost(book, sc)
    s2 = quote_selfhost(book, sc2)
    m1 = {q.gpu: q for q in s1}
    for q in s2:
        assert q.effective_usd_per_second_low < m1[q.gpu].effective_usd_per_second_low, q.gpu

    # 4) 更快的卡在同档位下 GPU 小时更少
    h100 = next(q for q in s1 if q.gpu == "H100_SXM")
    a6000 = next(q for q in s1 if q.gpu == "RTX_A6000")
    assert h100.gpu_hours_low < a6000.gpu_hours_low

    # 5) 方舟 token 计费随分辨率二次增长：1080p 应约为 720p 的 2.25 倍
    sc720 = Scenario(width=1280, height=720, fps=24)
    sc1080 = Scenario(width=1920, height=1080, fps=24)
    ark720 = next(q for q in quote_api(book, sc720) if q.key == "ark_seedance_1_0_pro")
    ark1080 = next(q for q in quote_api(book, sc1080) if q.key == "ark_seedance_1_0_pro")
    ratio = ark1080.usd_per_second / ark720.usd_per_second
    assert abs(ratio - 2.25) < 1e-6, f"分辨率缩放错了：{ratio}"

    # 6) 显存不足的卡被排在后面且标记出来
    s_hero = quote_selfhost(book, Scenario(preset="hero_720p_20step"))
    assert all(q.fits_vram for q in s_hero[:1]) or not any(q.fits_vram for q in s_hero)

    # 7) 表格对 CJK 对齐：每行显示宽度必须一致
    tbl = render_table(["中文表头", "x"], [["值", "1"], ["很长的中文值", "22"]])
    widths = {_cell_width(ln) for ln in tbl.splitlines()}
    assert len(widths) == 1, f"表格宽度不一致：{widths}"

    # 8) 非法输入必须拒绝
    for bad in (dict(episodes=0), dict(retake_rate=-0.1), dict(tier="nope")):
        try:
            Scenario(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应当拒绝：{bad}")

    # 9) 与 lora.py 的机时表不应漂移
    issues = book.cross_check_with_lora()
    if issues:
        log.warning("价目交叉校验不一致：%s", issues)

    # 10) 报告能渲染出来且包含关键段落
    rep = render_report(book, Scenario())
    for seg in ("路线 A", "路线 B", "结论", "假设与可信度"):
        assert seg in rep, seg
    log.info("cost_calculator 自测通过（%d 家 API / %d 张卡 / 交叉校验 %s）",
             len(book.api_models()), len(book.gpus()), "一致" if not issues else "有差异")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
