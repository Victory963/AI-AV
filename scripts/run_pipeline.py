"""端到端产线：音频先行 → 续接规划 → 参考位 → 路由渲染 → 质检重拍 → 调色 → 拼接 → 超分 → 合规交付。

这是平台的主流程，也是 demo 视频的数据来源。每个阶段都落一份真实产物和
一份 JSON 记录，demo 只是把这些记录画出来 —— 不存在"为了演示而编的数字"。

引擎用 run_episode.py 里的 PricedMock 双通道：本机没有 GPU 也没有厂商密钥，
但调度、降级、重拍、记账这些真正难的部分全是真跑。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from longfilm.providers import REGISTRY, ProviderError  # noqa: E402
from longfilm.router import ProviderQuota, QualityBias, Router, RoutingPolicy  # noqa: E402
from longfilm.schema import Scene, Shot, Storyboard  # noqa: E402
from run_episode import setup_engines  # noqa: E402

log = logging.getLogger("pipeline")

STAGES = ("audio", "render", "grade", "assemble", "post", "deliver")

# 阶段依赖：某些阶段需要前序阶段在**内存里**留下的状态，不只是磁盘产物。
# 典型的是 assemble —— 字幕时间来自 DialogueLine.start_s，那是 audio 阶段回填到
# Storyboard 对象上的；--resume 能从磁盘捡回视频文件，却捡不回这个回填。
# audio 有 TTS 缓存，补跑只要一秒，所以自动前置比让用户记住依赖靠谱。
STAGE_DEPS: dict[str, tuple[str, ...]] = {
    "assemble": ("audio",),
    "deliver": ("audio",),
}


@dataclass
class StageRecord:
    name: str
    ok: bool = False
    elapsed_s: float = 0.0
    detail: dict = field(default_factory=dict)
    error: str = ""


class Pipeline:
    def __init__(self, sb: Storyboard, work: Path, *, budget: float = 120.0,
                 retake_limit: int = 2, quick: bool = False, resume: bool = False):
        self.sb = sb
        self.work = work
        self.quick = quick
        self.retake_limit = retake_limit
        for d in ("renders", "graded", "chain", "dialogue", "deliver", "qc"):
            (work / d).mkdir(parents=True, exist_ok=True)

        setup_engines(work / "renders")
        policy = RoutingPolicy(
            bias=QualityBias.QUALITY,
            episode_budget_usd=budget,
            quotas={
                "mock-official": ProviderQuota(max_concurrency=2),
                "mock-open": ProviderQuota(max_concurrency=4),
                "mock-open-spot": ProviderQuota(max_concurrency=6),
            },
        )
        self.router = Router(REGISTRY, policy, poll_interval_s=0.2,
                             job_timeout_s=600.0, backoff_base_s=0.2)
        self.records: list[StageRecord] = []
        self.track = None            # DialogueTrack
        self.chain_steps: dict[str, list] = {}
        self.renders: dict[str, str] = {}      # shot_id -> 视频路径
        self.last_frames: dict[str, str] = {}
        self.qc_reports: dict[str, dict] = {}
        self.graded: dict[str, str] = {}
        self.episode_video: str | None = None
        self.final_video: str | None = None
        self.manifest_path: str | None = None
        if resume:
            self._resume()

    def _resume(self) -> bool:
        """从上次的 pipeline_result.json 捡回已渲染的片段。

        一集 30 镜的渲染是整条产线里最慢也最贵的一段；调后面的调色/拼接/交付
        参数时没有理由把它重跑一遍。产物按路径存在性校验，文件被删了就当没跑过。
        """
        p = self.work / "pipeline_result.json"
        if not p.exists():
            return False
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
        rec = next((s for s in d.get("stages", [])
                    if s.get("name") == "render" and s.get("ok")), None)
        if not rec:
            return False
        n = 0
        for row in rec.get("detail", {}).get("rows", []):
            v = row.get("video")
            if v and Path(v).exists():
                self.renders[row["shot_id"]] = v
                n += 1
                lf = row.get("last_frame")
                if lf and Path(lf).exists():
                    self.last_frames[row["shot_id"]] = lf
                if row.get("qc_score") is not None:
                    self.qc_reports[row["shot_id"]] = {
                        "score": row["qc_score"], "verdict": row.get("qc_verdict", ""),
                        "advice": row.get("qc_advice", []),
                    }
        # 调色产物同理
        g = next((s for s in d.get("stages", []) if s.get("name") == "grade" and s.get("ok")), None)
        if g:
            for sid in list(self.renders):
                gp = self.work / "graded" / f"{sid}.mp4"
                if gp.exists():
                    self.graded[sid] = str(gp)
        for s_ in d.get("stages", []):
            if s_.get("name") == "assemble" and s_.get("ok"):
                o = s_.get("detail", {}).get("out")
                if o and Path(o).exists():
                    self.episode_video = o
        log.info("断点续跑：捡回 %d 个渲染产物、%d 个调色产物", n, len(self.graded))
        print(f"  (resume) 捡回 {n} 个渲染产物、{len(self.graded)} 个调色产物")
        return n > 0

    # ------------------------------------------------------------ 阶段

    def stage_audio(self) -> dict:
        """音频先行：对白轨是全片的时间基准，必须在出画之前定下来。"""
        from longfilm.audio_first import (
            EdgeTTSBackend, SilentTTSBackend, build_audio_refs,
            retime_shots_to_audio, synthesize_dialogue,
        )

        out = self.work / "dialogue"
        try:
            backend = EdgeTTSBackend()
            if not backend.available():
                raise RuntimeError("edge-tts 不可用")
            self.track = synthesize_dialogue(self.sb, backend, out)
        except Exception as e:
            log.warning("在线 TTS 不可用（%s），回退静音后端：时长按文本长度估算", e)
            self.track = synthesize_dialogue(self.sb, SilentTTSBackend(), out)

        # max_s 取目标 provider 的单段上限：装不下的镜会在换气处自动拆分
        caps_max = max(p.caps.max_duration_s for p in REGISTRY.all())
        retime = retime_shots_to_audio(self.sb, self.track, min_s=3.0,
                                       max_s=min(caps_max, 15.0))
        refs = build_audio_refs(self.track, self.sb, stem_dir=out / "stems")
        return {
            "backend": self.track.backend,
            "cues": len(self.track.cues),
            "speech_s": round(self.track.speech_duration_s(), 2),
            "retime_changes": len(retime),
            "retime_sample": retime[:6],
            "audio_refs": len(refs),
            "shots_after_retime": len(self.sb.all_shots()),
            "episode_s": round(self.sb.duration_s(), 1),
        }

    def _plan_chain(self, scene: Scene) -> list:
        from longfilm.chain import ChainPlanner, DriftBudget

        planner = ChainPlanner(budget=DriftBudget(), workdir=self.work / "chain")
        steps = planner.plan(scene, self.sb, self.router)
        self.chain_steps[scene.id] = steps
        return steps

    def _prepare_link(self, shot: Shot, step, prev_shot: Shot | None,
                      prev_result, prev_provider: str | None) -> dict:
        """把上一段的产物真正喂进这一镜（首帧 / job_id / 重叠帧）。"""
        from longfilm.chain import ChainPlanner, LinkContext

        if step is None or not step.is_continuation or prev_result is None:
            return {}
        planner = ChainPlanner(workdir=self.work / "chain")
        try:
            strat = planner.strategy(step.strategy)
        except KeyError:
            return {}
        caps = REGISTRY.get(step.provider).caps if step.provider in REGISTRY else None
        if caps is None:
            return {}
        ctx = LinkContext(shot=shot, storyboard=self.sb, caps=caps, prev_shot=prev_shot,
                          prev_result=prev_result, prev_provider=prev_provider,
                          workdir=self.work / "chain")
        if not strat.applicable(ctx):
            return {}
        lp = strat.prepare(ctx)
        return {"strategy": lp.strategy, "inputs": lp.inputs, "notes": list(lp.notes)}

    def stage_render(self) -> dict:
        """逐镜：续接注入 → 参考位 → 路由执行 → 质检 → 不合格重拍。"""
        from longfilm.qc import QCContext, QCGate, Verdict
        from longfilm.refpack import RefBudget, build_refpack

        gate = QCGate.default()
        rows: list[dict] = []
        links = 0
        retakes = 0

        for scene in self.sb.scenes:
            steps = self._plan_chain(scene)
            by_id = {s.shot_id: s for s in steps}
            prev_shot: Shot | None = None
            prev_result = None
            prev_provider: str | None = None
            scene_clips: list[str] = []

            for shot in scene.shots:
                step = by_id.get(shot.id)
                link = self._prepare_link(shot, step, prev_shot, prev_result, prev_provider)
                if link:
                    links += 1

                # 参考位：按目标 provider 的上限编排，被裁的身份锚要能看见
                target = step.provider if step and step.provider in REGISTRY else None
                budget = RefBudget.from_capabilities(REGISTRY.get(target).caps) if target else None
                drops: list = []
                # 续接链刚抽出来的尾帧要传给 refpack，否则这一步重建参考位时会把它冲掉。
                # provider 直出尾帧（mock / 官方 API）走 self.last_frames；
                # 自建引擎不直出，链会从上一镜视频里抽一帧，只在 link 里，别漏了。
                chain_frame = (link or {}).get("inputs", {}).get("first_frame_uri")
                anchor = chain_frame or (self.last_frames.get(prev_shot.id) if prev_shot else None)
                shot.refs = build_refpack(
                    shot, self.sb, budget=budget,
                    prev_last_frame_uri=anchor,
                    drops_out=drops,
                )

                row: dict = {
                    "shot_id": shot.id, "scene_id": scene.id,
                    "duration_s": shot.duration_s,
                    "shot_size": shot.shot_size.value,
                    "camera_move": shot.camera_move.value,
                    "lens_mm": shot.lens_mm, "subjects": list(shot.subject_ids),
                    "engine_hint": shot.engine_hint.value,
                    "dialogue": [d.text for d in shot.dialogue],
                    "chain": link or None,
                    "chain_strategy": step.strategy if step else None,
                    "drift_after": round(step.drift_after, 3) if step else None,
                    "reanchor": bool(step.reanchor) if step else False,
                    "refs": shot.refs.slots_used(),
                    "refs_dropped": len(drops),
                    "refs_dropped_critical": sum(1 for d in drops if getattr(d, "critical", False)),
                }

                verdict = None
                report = None
                for take in range(self.retake_limit + 1):
                    if take:
                        shot.seed = (shot.seed or 0) + 7919 * take   # 换 seed 重摇
                        retakes += 1
                    # 先拿到路由计划：质检的期望值必须是**实际下发的工单**，
                    # 不是导演的原始意图。router 会按 caps.clamp_duration /
                    # nearest_resolution 归整，拿原始 Shot 去比对必然判规格不符 ——
                    # 那不是废片，那是降级被正确执行了。
                    plan = self.router.plan(shot, self.sb)
                    req = plan.request
                    if plan.degradations and "degradations" not in row:
                        row["degradations"] = list(plan.degradations)
                    try:
                        res = self.router.execute(shot, self.sb, max_attempts=3)
                    except ProviderError as e:
                        row |= {"status": "error", "error": str(e)[:200]}
                        break
                    if not res.ok:
                        row |= {"status": res.status.value, "error": res.message[:200]}
                        break
                    ctx = QCContext(
                        shot=shot, storyboard=self.sb,
                        expect_duration_s=req.duration_s if req else None,
                        expect_fps=req.fps if req else None,
                        expect_resolution=tuple(req.resolution) if req else None,
                    )
                    report = gate.evaluate(res.video_uri, shot, self.sb, context=ctx)
                    verdict = report.verdict
                    row |= {
                        "status": "succeeded", "provider": res.provider,
                        "video": res.video_uri, "last_frame": res.last_frame_uri,
                        "cost": round(res.cost_usd, 4), "take": take + 1,
                        "qc_score": round(report.score, 3),
                        "qc_verdict": verdict.value,
                        "qc_metrics": [
                            {"name": m.name, "value": round(float(m.value), 4),
                             "passed": bool(m.passed)}
                            for m in report.metrics[:8]
                        ],
                        "qc_advice": list(report.advice)[:3],
                    }
                    if verdict is not Verdict.RETAKE:
                        break

                if row.get("video"):
                    self.renders[shot.id] = row["video"]
                    if row.get("last_frame"):
                        self.last_frames[shot.id] = row["last_frame"]
                    scene_clips.append(row["video"])
                    shot.render_uri = row["video"]
                    shot.provider_used = row.get("provider")
                    shot.qc_score = row.get("qc_score")
                    prev_result = res
                    prev_provider = row.get("provider")
                    prev_shot = shot
                if report is not None:
                    self.qc_reports[shot.id] = {
                        "score": round(report.score, 3), "verdict": report.verdict.value,
                        "advice": list(report.advice)[:3],
                    }
                rows.append(row)
                print(f"    {len(rows):2}/{len(self.sb.all_shots())} {shot.id:11} "
                      f"{row.get('status','?'):10} {str(row.get('provider') or '-'):16} "
                      f"take={row.get('take','-')} qc={row.get('qc_score','-')}", flush=True)

        ok = [r for r in rows if r.get("status") == "succeeded"]
        passed = [r for r in ok if r.get("qc_verdict") == "pass"]
        return {
            "shots": len(rows), "ok": len(ok), "qc_pass": len(passed),
            "retakes": retakes, "chain_links": links,
            "cost_usd": round(self.router.ledger.total(), 4),
            "by_provider": {k: round(v, 4) for k, v in self.router.ledger.by_provider().items()},
            "calls_by_provider": self.router.ledger.calls_by_provider(),
            "rows": rows,
        }

    def stage_grade(self) -> dict:
        """跨镜色彩统一：不同引擎出来的片段色温会飘，直接拼一眼假。"""
        from longfilm.grade import match_to_reference, sample_stats, scene_reference

        out_dir = self.work / "graded"
        per_scene: list[dict] = []
        for scene in self.sb.scenes:
            clips = [self.renders[s.id] for s in scene.shots if s.id in self.renders]
            if len(clips) < 2:
                for c in clips:
                    self.graded[Path(c).parent.name] = c
                continue
            before = [sample_stats(c) for c in clips]
            ref = scene_reference(before)
            drift_before = max(abs(b.luma_mean - ref.luma_mean) for b in before)
            after_paths: list[str] = []
            for shot, clip, st in zip([s for s in scene.shots if s.id in self.renders], clips, before):
                dst = out_dir / f"{shot.id}.mp4"
                match_to_reference(clip, ref, dst, src_stats=st, strength=0.85)
                self.graded[shot.id] = str(dst)
                after_paths.append(str(dst))
            after = [sample_stats(p) for p in after_paths]
            drift_after = max(abs(a.luma_mean - ref.luma_mean) for a in after)
            per_scene.append({
                "scene": scene.id, "clips": len(clips),
                "ref_luma": round(ref.luma_mean, 2), "ref_cct": round(ref.cct_k, 0),
                "max_drift_before": round(drift_before, 2),
                "max_drift_after": round(drift_after, 2),
                "improvement": round(drift_before - drift_after, 2),
            })
        return {"scenes": per_scene,
                "graded": len(self.graded),
                "total_improvement": round(sum(s["improvement"] for s in per_scene), 2)}

    def stage_assemble(self) -> dict:
        """15s 原子镜 × N → 一集。转场、混音、响度归一一次 filter_complex 走完。"""
        from longfilm.stitch import assemble_episode, default_ffmpeg
        from longfilm.timeline import build_timeline, export_ass_subtitles, export_edl

        clips = [self.graded.get(s.id) or self.renders.get(s.id)
                 for s in self.sb.all_shots()]
        clips = [c for c in clips if c]
        if not clips:
            raise RuntimeError(
                "没有可拼的片段：先跑 render 阶段，或加 --resume 从上次结果捡回产物"
            )

        tl = build_timeline(self.sb, {s.id: (self.graded.get(s.id) or self.renders.get(s.id))
                                      for s in self.sb.all_shots() if s.id in self.renders},
                            dialogue=self.track)
        ass = export_ass_subtitles(tl, font="Microsoft YaHei")
        ass_p = self.work / "deliver" / "subs.ass"
        ass_p.write_text(ass, encoding="utf-8")
        edl_p = self.work / "deliver" / "episode.edl"
        edl_p.write_text(export_edl(tl), encoding="utf-8")

        out = self.work / "deliver" / "episode_720p.mp4"
        # 拼接只归一到**生成分辨率**，放大交给 post 的超分器。
        # stitch.VideoTarget.from_delivery 默认一次缩到 upscale_to（它的理由是
        # 避免二次插值），但那等于用 lanczos 放大冒充超分，post 阶段就没事干了。
        # 省 API 费的前提正是「生成 720p → 本地 AI 超分」，所以这里显式屏蔽 upscale_to。
        base_delivery = self.sb.delivery.model_copy(update={"upscale_to": None})
        batches = self._assemble_batched(
            clips, out, base_delivery,
            subtitles=ass_p if self.sb.delivery.burn_subtitles else None,
        )
        self.episode_video = str(out)
        info = default_ffmpeg().probe(out)
        return {
            "clips": len(clips), "batches": batches, "out": str(out),
            "duration_s": round(info.duration_s, 2),
            "resolution": f"{info.width}x{info.height}", "fps": info.fps,
            "size_mb": round(out.stat().st_size / 1e6, 2),
            "subtitles": str(ass_p), "edl": str(edl_p),
            "subtitle_lines": ass.count("\nDialogue:"),
        }

    def _assemble_batched(self, clips: list[str], out: Path, delivery,
                          *, subtitles=None, batch: int = 6) -> int:
        """分批归并拼接。返回批数（1 = 没分批）。

        stitch.assemble_episode 的设计是「一次 filter_complex 走完，中间不落盘」，
        理由正当：拼完再转一次会白白多一代压缩。但 30 路输入意味着 ffmpeg 要同时
        开 30 个解码器加整张滤镜图 —— 在 4GB 内存的机器上会被 OOM killer 干掉
        （ffmpeg 退出码 -9），而且是在跑了 13 分钟之后才死。

        所以这里做树形归并：先每 batch 个拼成中间段，再把中间段拼成成片。
        代价是中间段多一代 CRF 18 压缩，换的是「能跑完」。批大小按内存定：
        每路输入的解码缓冲大致与分辨率成正比，720p 下 6 路是实测安全值。
        """
        from longfilm.stitch import assemble_episode

        if len(clips) <= batch:
            assemble_episode(clips, out, delivery=delivery, audio_plan=self.sb.audio,
                             transition_s=0.4, subtitles=subtitles,
                             use_clip_audio=True, two_pass_loudnorm=False)
            return 1

        parts: list[str] = []
        tmp = self.work / "deliver" / "_parts"
        tmp.mkdir(parents=True, exist_ok=True)
        n_batch = (len(clips) + batch - 1) // batch
        # 中间段不烧字幕、不做响度归一 —— 那两件只在最后一次做，否则字幕会被
        # 二次编码糊掉、响度会被归一两次。burn_subtitles 必须在 delivery 上真的
        # 关掉：assemble_episode 会校验「声明要烧字幕却没给路径」并直接拒绝。
        mid_delivery = delivery.model_copy(update={"burn_subtitles": False})
        for i in range(0, len(clips), batch):
            chunk = clips[i : i + batch]
            pp = tmp / f"part_{i // batch:02d}.mp4"
            assemble_episode(chunk, pp, delivery=mid_delivery, audio_plan=self.sb.audio,
                             transition_s=0.4, use_clip_audio=True,
                             two_pass_loudnorm=False)
            parts.append(str(pp))
            print(f"    批次 {i // batch + 1}/{n_batch} ({len(chunk)} 镜) -> {pp.name}",
                  flush=True)

        assemble_episode(parts, out, delivery=delivery, audio_plan=self.sb.audio,
                         transition_s=0.4, subtitles=subtitles,
                         use_clip_audio=True, two_pass_loudnorm=False)
        return len(parts)

    def stage_post(self) -> dict:
        """720p 出片省额度，本地拉 1080 —— 本机只有 CPU，走 ffmpeg 基线。"""
        from longfilm.stitch import default_ffmpeg
        from longfilm.upscale import Hardware, plan_postprocess

        src = Path(self.episode_video)
        info = default_ffmpeg().probe(src)
        hw = Hardware.detect()
        plan = plan_postprocess(self.sb.delivery, hw,
                                source_wh=(info.width, info.height),
                                source_fps=int(round(info.fps)),
                                duration_s=info.duration_s)
        out = self.work / "deliver" / "episode_1080p.mp4"
        plan.run(src, out)
        got = default_ffmpeg().probe(out)
        self.final_video = str(out)
        return {
            "hardware": f"cuda={hw.has_cuda} vram={hw.vram_gb}GB cpu={hw.cpu_cores}c ram={hw.ram_gb:.1f}G comfy={hw.comfy_url or None}",
            "steps": [getattr(s, "name", str(s)) for s in plan.steps],
            "warnings": list(plan.warnings)[:4],
            "from": f"{info.width}x{info.height}", "to": f"{got.width}x{got.height}",
            "out": str(out), "size_mb": round(out.stat().st_size / 1e6, 2),
            "est_minutes": round(getattr(plan, "est_minutes", 0.0), 2),
        }

    def stage_deliver(self) -> dict:
        """合规封装：显式标识 + 隐式元数据 + 溯源清单。"""
        from longfilm.compliance import (
            ProvenanceManifest, add_visible_disclosure, compliance_check, write_metadata,
        )

        problems = compliance_check(self.sb)
        src = Path(self.final_video or self.episode_video)
        disc = self.work / "deliver" / "episode_disclosed.mp4"
        add_visible_disclosure(src, disc, text="本视频由 AI 生成")
        manifest = ProvenanceManifest.from_storyboard(
            self.sb,
            content_producer="longfilm demo pipeline",
            producer_code="LONGFILM-DEMO",
            model_versions={p.name: f"{p.caps.kind}/tier{p.caps.quality_tier}"
                            for p in REGISTRY.all()},
            asset_root=ROOT,
        )
        mp = self.work / "deliver" / "provenance.json"
        mp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.manifest_path = str(mp)
        final = self.work / "deliver" / "episode_final.mp4"
        write_metadata(disc, final, manifest)
        self.final_video = str(final)
        return {
            "compliance_problems": problems,
            "manifest": str(mp),
            "manifest_shots": len(manifest.shots),
            "out": str(final),
            "size_mb": round(final.stat().st_size / 1e6, 2),
        }

    # ------------------------------------------------------------ 编排

    def run(self, stages: tuple[str, ...]) -> dict:
        # 自动补齐依赖阶段（去重并保持 STAGES 的先后顺序）
        want = set(stages)
        for st in stages:
            want |= set(STAGE_DEPS.get(st, ()))
        if want - set(stages):
            added = sorted(want - set(stages), key=STAGES.index)
            print(f"自动前置依赖阶段: {', '.join(added)}（它们的产物有缓存，代价很低）")
        stages = tuple(n for n in STAGES if n in want)

        print(f"阶段: {' -> '.join(stages)}")
        for name in stages:
            if name == "render" and len(self.renders) >= len(self.sb.all_shots()):
                print(f"\n──── [{name}] ──── (resume: {len(self.renders)} 镜已就绪，跳过)")
                continue
            fn = getattr(self, f"stage_{name}")
            rec = StageRecord(name=name)
            t0 = time.time()
            print(f"\n──── [{name}] ────")
            try:
                rec.detail = fn() or {}
                rec.ok = True
            except Exception as e:
                rec.error = f"{type(e).__name__}: {e}"
                log.error("阶段 %s 失败: %s", name, rec.error)
                traceback.print_exc(limit=4)
            rec.elapsed_s = round(time.time() - t0, 1)
            self.records.append(rec)
            brief = {k: v for k, v in rec.detail.items() if k != "rows"}
            print(f"  {'✓' if rec.ok else '✗'} {rec.elapsed_s}s  "
                  f"{json.dumps(brief, ensure_ascii=False)[:420]}")
            self._flush()      # 每阶段落盘，进程被杀也能 --resume 接上
            if not rec.ok and name in ("render", "assemble"):
                break

        return self._flush()

    def _flush(self) -> dict:
        # 结果按阶段名**合并**写回，不整体覆盖：只跑 grade 时不能把上一轮的
        # render 记录抹掉，否则 --resume 下一次就捡不回任何产物了。
        p = self.work / "pipeline_result.json"
        prev: dict = {}
        if p.exists():
            try:
                prev = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        merged: dict[str, dict] = {s["name"]: s for s in prev.get("stages", [])}
        for r in self.records:
            merged[r.name] = asdict(r)
        order = {n: i for i, n in enumerate(STAGES)}
        result = {
            "project": f"{self.sb.project}/{self.sb.episode}",
            "stages": sorted(merged.values(), key=lambda s: order.get(s["name"], 99)),
            "episode_video": self.episode_video or prev.get("episode_video"),
            "final_video": self.final_video or prev.get("final_video"),
            "manifest": self.manifest_path or prev.get("manifest"),
            "total_s": round(sum(r.elapsed_s for r in self.records), 1),
            "ran_this_time": [r.name for r in self.records],
        }
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--work", default=str(ROOT / "out" / "work" / "pipeline"))
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--storyboard", default=str(ROOT / "configs" / "demo_episode.json"),
                    help="要拍的分镜 JSON")
    ap.add_argument("--resume", action="store_true", help="捡回上次已渲染的产物，只跑后续阶段")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname).1s %(name)s: %(message)s",
    )

    sb = Storyboard.load(args.storyboard)
    if args.limit:
        kept, n = [], 0
        for sc in sb.scenes:
            if n >= args.limit:
                break
            take = sc.shots[: args.limit - n]
            n += len(take)
            sc.shots = take
            kept.append(sc)
        sb.scenes = kept
    if args.scale != 1.0:
        for s in sb.all_shots():
            s.duration_s = round(max(1.0, s.duration_s * args.scale), 2)

    work = Path(args.work)
    print(f"分镜 {sb.project}/{sb.episode}：{len(sb.all_shots())} 镜 / {sb.duration_s():.0f}s")
    Pipeline(sb, work, budget=args.budget, resume=args.resume).run(
        tuple(args.stages.split(","))
    )


if __name__ == "__main__":
    main()
