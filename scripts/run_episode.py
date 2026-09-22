"""真实跑一集：路由 → 渲染 → 记账，产出 demo 需要的全部中间产物。

引擎用两个能力不同的 mock 实例模拟真实的双通道格局：
  mock-official  贵、有审核、画质 tier5、单段 ≤15s 且只有档位时长、不支持 LoRA
  mock-open      便宜、无审核、画质 tier2、单段 ≤60s 连续时长、支持 LoRA
这两组 Capabilities 是照着调研到的闭源 API / 自建开源引擎的真实差异填的，
所以路由做出的选择、降级和拒单都是真决策，不是演给 demo 看的。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from longfilm.providers import REGISTRY, ProviderError  # noqa: E402
from longfilm.providers.mock import MockProvider  # noqa: E402
from longfilm.router import (  # noqa: E402
    DEFAULT_POLICY,
    ProviderQuota,
    QualityBias,
    Router,
    RoutingPolicy,
)
from longfilm.schema import Storyboard  # noqa: E402

WORK = ROOT / "out" / "work" / "episode"


class PricedMock(MockProvider):
    """按 caps 单价如实报成本的 mock。

    MockProvider 本身 cost_usd 恒为 0（本地渲染确实不花钱），但当它扮演一个
    有定价的引擎时必须报出那个价，否则成本账本全是 0，路由的成本权重也失效。
    """

    def poll(self, job_id: str):
        r = super().poll(job_id)
        if r.ok and not r.cost_usd:
            secs = r.duration_s or 0.0
            r.cost_usd = round(self.caps.cost_per_second_usd * secs, 6)
        return r


def setup_engines(render_dir: Path) -> None:
    """注册双通道引擎。已注册的同名 provider 会被覆盖，重复跑不会叠加。"""
    REGISTRY._providers.clear()

    official = PricedMock("mock-official", out_dir=render_dir / "official",
                            quality_tier=5, native_audio=True)
    official.caps = dataclasses.replace(
        official.caps,
        kind="official",
        moderated=True,                       # 官方通道强制审核
        min_duration_s=4.0,
        max_duration_s=15.0,
        duration_steps=(4.0, 8.0, 12.0, 15.0),  # 只能取档位
        supports_lora=False,                  # 权重闭源，挂不了自训 LoRA
        supports_extend=True,
        supports_native_audio=True,
        cost_per_second_usd=0.12,
        typical_latency_s=180.0,
        quality_tier=5,
        max_concurrency=2,
        notes="模拟闭源 API：画质天花板，有审核，不可微调",
    )

    openp = PricedMock("mock-open", out_dir=render_dir / "open",
                         quality_tier=3, native_audio=False)
    openp.caps = dataclasses.replace(
        openp.caps,
        kind="open",
        moderated=False,
        min_duration_s=1.0,
        max_duration_s=60.0,
        duration_steps=(),                    # 连续时长
        supports_lora=True,                   # 开源侧唯一能自主微调的通道
        supports_native_audio=False,          # 声音要走独立 TTS 轨
        cost_per_second_usd=0.004,            # 自建 GPU 摊销
        typical_latency_s=240.0,
        quality_tier=3,
        max_concurrency=4,
        notes="模拟自建开源引擎：可 LoRA、无平台审核、画质略低",
    )

    flaky = PricedMock("mock-open-spot", out_dir=render_dir / "spot",
                         quality_tier=2, fail_rate=0.25, native_audio=False)
    flaky.caps = dataclasses.replace(
        flaky.caps,
        kind="open",
        moderated=False,
        max_duration_s=60.0,
        supports_lora=True,
        cost_per_second_usd=0.0015,           # 竞价实例：更便宜但会被抢占
        quality_tier=2,
        max_concurrency=6,
        notes="模拟竞价 GPU：最便宜，但 25% 概率被抢占",
    )

    for p in (official, openp, flaky):
        REGISTRY.register(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 镜（0=全部）")
    ap.add_argument("--scale", type=float, default=1.0, help="时长缩放，加速渲染")
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--out", default=str(WORK))
    args = ap.parse_args()

    work = Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    setup_engines(work / "renders")

    sb = Storyboard.load(ROOT / "configs" / "demo_episode.json")
    shots = sb.all_shots()
    if args.limit:
        shots = shots[: args.limit]
    if args.scale != 1.0:
        for s in shots:
            s.duration_s = round(max(1.0, s.duration_s * args.scale), 2)

    policy = RoutingPolicy(
        bias=QualityBias.QUALITY,
        episode_budget_usd=args.budget,
        quotas={
            "mock-official": ProviderQuota(max_concurrency=2, max_calls=14),
            "mock-open": ProviderQuota(max_concurrency=4),
            "mock-open-spot": ProviderQuota(max_concurrency=6),
        },
    )
    router = Router(REGISTRY, policy, poll_interval_s=0.2, job_timeout_s=300.0,
                    backoff_base_s=0.2)

    rows: list[dict] = []
    t0 = time.time()
    print(f"跑 {len(shots)} 镜（总时长 {sum(s.duration_s for s in shots):.0f}s）")
    print(f"引擎：{', '.join(f'{p.name}[{p.caps.kind}/tier{p.caps.quality_tier}]' for p in REGISTRY.all())}\n")

    for i, shot in enumerate(shots, 1):
        plan = router.plan(shot, sb)
        row: dict = {
            "shot_id": shot.id,
            "scene_id": shot.scene_id,
            "duration_s": shot.duration_s,
            "shot_size": shot.shot_size.value,
            "camera_move": shot.camera_move.value,
            "lens_mm": shot.lens_mm,
            "subjects": shot.subject_ids,
            "engine_hint": shot.engine_hint.value,
            "rating": shot.content_rating.value,
            "planned": plan.primary,
            "fallbacks": list(plan.fallbacks),
            "est_cost": round(plan.estimated_cost_usd, 4),
            "degradations": list(plan.degradations),
            "gate_ok": plan.gate.allowed,
            "scores": [
                {"name": s.name, "total": round(s.total, 3)} for s in plan.scores[:4]
            ],
            "inherit_last_frame": shot.continuity.inherit_last_frame,
            "dialogue": [d.text for d in shot.dialogue],
        }
        try:
            res = router.execute(shot, sb, max_attempts=3)
            row |= {
                "status": res.status.value,
                "provider": res.provider,
                "video": res.video_uri,
                "last_frame": res.last_frame_uri,
                "cost": round(res.cost_usd, 4),
                "elapsed_s": round(res.elapsed_s, 2),
            }
            shot.provider_used = res.provider
            shot.render_uri = res.video_uri
            flag = "ok " if res.ok else "FAIL"
        except ProviderError as e:
            row |= {"status": "error", "provider": None, "error": str(e), "cost": 0.0}
            flag = "ERR "
        rows.append(row)
        deg = f"  降级:{len(row['degradations'])}" if row["degradations"] else ""
        print(f"  [{i:2}/{len(shots)}] {flag} {shot.id:11} {shot.duration_s:5.1f}s "
              f"{str(row.get('provider') or row['planned']):16} ${row.get('cost',0):.4f}{deg}")

    ledger = router.ledger
    summary = {
        "project": f"{sb.project}/{sb.episode}",
        "shots": len(rows),
        "ok": sum(1 for r in rows if r.get("status") == "succeeded"),
        "total_cost_usd": round(ledger.total(), 4),
        "by_provider": {k: round(v, 4) for k, v in ledger.by_provider().items()},
        "wall_s": round(time.time() - t0, 1),
        "total_video_s": sum(r["duration_s"] for r in rows),
        "rows": rows,
    }
    (work / "episode_result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n成片素材 {summary['ok']}/{summary['shots']} 镜  "
          f"总成本 ${summary['total_cost_usd']:.4f}  墙钟 {summary['wall_s']:.0f}s")
    for k, v in summary["by_provider"].items():
        n = sum(1 for r in rows if r.get("provider") == k)
        print(f"  {k:18} ${v:8.4f}  {n:2} 镜")
    degs = [d for r in rows for d in r["degradations"]]
    if degs:
        print(f"  降级 {len(degs)} 次，样例：{degs[0]}")
    print(f"\n结果 → {work / 'episode_result.json'}")


if __name__ == "__main__":
    main()
