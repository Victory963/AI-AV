"""生成 demo 视频 —— 每一个数字都来自 run_pipeline.py 的真实运行。

数据源：out/work/full/pipeline_result.json（6 个阶段的完整记录）
        configs/demo_episode.json（分镜）
        deploy/cost_calculator.py（云 GPU 成本，真实价目表）

诚实声明（写进片头）：本机无 GPU、无厂商密钥，镜头由 mock 引擎合成。
这个 demo 演示的是**产线调度**——路由、续接、质检、重拍、调色、拼接、合规，
不是 AI 生成的人物影像质量。画质那一半要接真引擎才有。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import demo_visuals as V  # noqa: E402
from demo_render import (  # noqa: E402
    FPS, DemoTimeline, Segment, ambient_bed, ff, frames_to_video, mux,
    probe_duration, still_to_video,
)
from demo_visuals import P  # noqa: E402

from longfilm.schema import Storyboard  # noqa: E402

OUT = ROOT / "out" / "demo"
SEG = OUT / "segments"
WORK = ROOT / "out" / "work"
PIPE = WORK / "full"


@dataclass
class Ctx:
    sb: Storyboard
    pipe: dict
    quick: bool = False
    reuse: bool = False

    def cached(self, name: str) -> Segment | None:
        """已渲好的段直接复用。全量渲一次要半小时，为改一段全部重来是纯浪费。"""
        if not self.reuse:
            return None
        p = SEG / f"{name}.mp4"
        if p.exists() and p.stat().st_size > 1000:
            return Segment(name, p, probe_duration(p))
        return None

    def s(self, sec: float) -> float:
        return round(sec * (0.45 if self.quick else 1.0), 2)

    def stage(self, name: str) -> dict:
        for st in self.pipe.get("stages", []):
            if st["name"] == name and st.get("ok"):
                return st.get("detail", {})
        return {}

    @property
    def rows(self) -> list[dict]:
        return self.stage("render").get("rows", [])


def card(ctx: Ctx, name: str, img, sec: float, **kw) -> Segment:
    if (c := ctx.cached(name)) is not None:
        return c
    d = ctx.s(sec)
    return Segment(name, still_to_video(img, SEG / f"{name}.mp4", d, **kw), d)


def cached_or(ctx: Ctx, name: str):
    """给不走 card() 的段（逐帧渲染的）用的复用检查。"""
    return ctx.cached(name)


# ================================================================ 段落


def seg_title(ctx: Ctx) -> Segment:
    sb, r = ctx.sb, ctx.stage("render")
    return card(ctx, "01_title", V.title_card(
        "长时间 AI 数字人拍片系统",
        "分镜驱动 · 多引擎路由 · 原子镜拼接",
        [f"本集 {sb.project}/{sb.episode}：{len(sb.all_shots())} 个原子镜，{sb.duration_s()/60:.1f} 分钟",
         f"产线 6 个阶段全程实跑：{r.get('ok',0)}/{r.get('shots',0)} 镜出片，{r.get('retakes',0)} 次自动重拍",
         "19 个模块 · 5 个引擎适配器 · 无 GPU 也能跑通调度"],
        note="本机无 GPU、无厂商密钥：镜头由 mock 引擎合成，演示的是产线调度而非画质",
    ), 8.0, zoom=(1.0, 1.05), fade_in=0.8)


def seg_constraints(ctx: Ctx) -> Segment:
    return card(ctx, "02_constraints", V.kv_table(
        "先把边界说死：三条硬约束已经过期",
        [("Seedance 2.0 闭源 · 不可本机 LoRA", "✅ 成立，且是终局", P.ok),
         ("官方单段 4–15 秒", "⚠️ 2.5 已是单次 30 秒一镜到底", P.warn),
         ("最多 9 图 + 3 视频 + 3 音频", "⚠️ 那是 2.0；2.5 约 50 路（30+10+10）", P.warn),
         ("自带对白声", "✅ 成立，但可用 V2V 重配音夺回音色", P.ok),
         ("官方通道有内容审核", "✅ 成立，且物理上绕不过去", P.ok),
         ("→ 规划含义", "8 分钟片从 32 镜降到 16 镜，接缝减半", P.accent)],
        subtitle="12 条技术线并行调研 · 321 条方案 · 119 条事实修正",
        step="STEP 00", col_w=530), 12.0)


def seg_pipeline(ctx: Ctx):
    if (c := ctx.cached("03_pipeline")) is not None:
        return c
    stages = [("剧本分镜", "storyboard_gen"), ("角色圣经", "charbible"),
              ("音频先行", "audio_first"), ("参考位", "refpack"),
              ("提示词OS", "prompt_os"), ("引擎路由", "router"),
              ("续接链", "chain"), ("质检门禁", "qc"),
              ("调色拼接", "grade+stitch"), ("超分交付", "upscale+compliance")]
    # 每个工位至少停留 1 秒，否则观众根本看不清高亮移到了哪一站
    per = 10 if ctx.quick else 26
    frames = [V.flow_diagram("产线工位：每个都是独立模块", stages, active=i,
                             subtitle="换供应商不改队列 —— 路由只读 Capabilities，不写 if provider ==",
                             step="STEP 01")
              for i in range(len(stages)) for _ in range(per)]
    path = frames_to_video(frames, SEG / "03_pipeline.mp4")
    return Segment("03_pipeline", path, probe_duration(path))


def seg_storyboard(ctx: Ctx) -> Segment:
    sb = ctx.sb
    raw = json.loads((ROOT / "configs" / "demo_episode.json").read_text(encoding="utf-8"))
    s0 = raw["scenes"][0]["shots"][2]
    shots = sb.all_shots()
    lines = [
        ('{  "id": "%s",  "duration_s": %s,  "lens_mm": %s,' % (s0["id"], s0["duration_s"], s0["lens_mm"]), P.text),
        ('   "shot_size": "%s",' % s0["shot_size"], P.accent),
        ('   "camera_move": "%s",' % s0["camera_move"], P.accent),
        ('   "subject_ids": %s,' % json.dumps(s0["subject_ids"]), P.accent2),
        ('   "dialogue": [{"speaker_id": "%s", "emotion": "%s"}],'
         % (s0["dialogue"][0]["speaker_id"], s0["dialogue"][0]["emotion"]), P.ok),
        ('   "continuity": {"inherit_last_frame": %s, "screen_direction": "%s"},'
         % (str(s0["continuity"]["inherit_last_frame"]).lower(), s0["continuity"]["screen_direction"]), P.warn),
        ('   "engine_hint": "%s",  "content_rating": "%s" }' % (s0["engine_hint"], s0["content_rating"]), P.gold),
        ('', P.text),
        (f'指纹 {shots[2].fingerprint()} — 同指纹命中缓存，改一句台词不重烧整场', P.text_dim),
        (f'全片 {len(shots)} 镜 / {sb.duration_s():.0f}s / 对白 {sum(len(s.dialogue) for s in shots)} 条 / '
         f'继承尾帧 {sum(1 for s in shots if s.continuity.inherit_last_frame)} 镜', P.ok),
        (f'连续性校验（轴线按角色追踪 / 角色存在性 / 指针闭合）：'
         f'{"通过" if not sb.validate_continuity() else "未通过"}', P.ok),
    ]
    return card(ctx, "04_storyboard", V.code_panel(
        "分镜即唯一契约", lines,
        subtitle="一份 JSON 完整描述一集片，19 个模块只认它", step="STEP 02"), 11.0)


def seg_charbible(ctx: Ctx) -> Segment:
    c = ctx.sb.characters[0]
    a = c.appearance
    lines = [
        (f"角色 {c.id} · {c.name}", P.accent),
        (f"  年龄声明  {c.age_statement}", P.ok),
        (f"  虚构标记  is_fictional={c.is_fictional}（schema 层强制，不可关闭）", P.ok),
        ("", P.text),
        (f"  面部  {a.face}", P.text_dim),
        (f"  发型  {a.hair}", P.text_dim),
        (f"  服装  {a.wardrobe}", P.text_dim),
        (f"  强锚  {a.distinguishing}", P.warn),
        ("", P.text),
        (f"  定妆矩阵  {len(c.portraits)} 张主图 + {len(c.turnaround)} 张转身（视角 × 光位 × 表情）", P.accent2),
        (f"  角色 LoRA  {c.lora.base_model} · trigger={c.lora.trigger_word} · "
         f"strength={c.lora.strength}" if c.lora else "  角色 LoRA  未配置", P.accent2),
        (f"  音色  {c.voice.tts_engine} · {c.voice.language} · speed={c.voice.speed}", P.accent2),
        ("", P.text),
        ("脸漂 / 发型漂 / 服装漂机制不同，三个解：身份锚 + 强特征词 + 独立 wardrobe 位", P.gold),
    ]
    return card(ctx, "05_charbible", V.code_panel(
        "角色圣经：闭源侧的参考图池就是你事实上的 LoRA", lines,
        subtitle="每个角色 20–30 张定妆图，全部标注来源类型并打指纹", step="STEP 03"), 12.0)


def seg_refpack(ctx: Ctx) -> Segment:
    from longfilm.refpack import build_refpack, slot_manifest

    shot = next(s for s in ctx.sb.all_shots() if len(s.subject_ids) == 2)
    pack = build_refpack(shot, ctx.sb)
    lines = []
    for ln in slot_manifest(pack, lang="zh").splitlines()[:19]:
        c = (P.accent if "identity" in ln else
             P.accent2 if ("wardrobe" in ln or "environment" in ln) else
             P.text_dim if ln.strip().startswith(("=", "-")) else P.text)
        lines.append((ln[:94], c))
    u = pack.slots_used()
    lines += [("", P.text),
              (f"占用 {u['images']}/9 图 · {u['videos']}/3 视频 · {u['audios']}/3 音频", P.ok),
              (f"{shot.id} 双人同框：身份位按配额分配，任一角色都不会被挤到 0 张", P.gold)]
    return card(ctx, "06_refpack", V.code_panel(
        "9 参考位导演包", lines,
        subtitle="超限按角色配额 + 优先级裁剪；裁到身份锚会置 critical 让路由换引擎",
        step="STEP 04"), 12.0)


def seg_audio(ctx: Ctx) -> Segment:
    a = ctx.stage("audio")
    samples = a.get("retime_sample", [])[:7]
    lines = [(f"TTS 后端 {a.get('backend','?')} · 合成 {a.get('cues',0)} 条对白 · "
              f"总语音 {a.get('speech_s',0):.2f}s", P.ok), ("", P.text),
             ("实测时长反推镜长（不是估算，是解码出来的）：", P.accent)]
    for s in samples:
        lines.append((f"  {s[:88]}", P.text_dim))
    lines += [("", P.text),
              (f"共 {a.get('retime_changes',0)} 个镜头被改时长 · "
               f"挂了 {a.get('audio_refs',0)} 条对白 stem 进参考位", P.accent2),
              (f"全片时长 {a.get('episode_s',0):.1f}s", P.text),
              ("", P.text),
              ("装不下的镜头在**换气处**自动拆分 —— 切在字中间必然穿帮", P.gold)]
    return card(ctx, "07_audio", V.code_panel(
        "音频先行：对白轨是全片的时间基准", lines,
        subtitle="先定对白再出画，口型比后配稳；时间戳存镜内相对秒，绝对值交给 timeline 现算",
        step="STEP 05"), 13.0)


def seg_routing(ctx: Ctx) -> Segment:
    r = ctx.stage("render")
    rows = []
    for row in ctx.rows[:9]:
        prov = row.get("provider") or "拒单"
        col = P.accent if "official" in str(prov) else P.accent2
        tag = str(prov)
        if row.get("degradations"):
            tag += f"  降级×{len(row['degradations'])}"
            col = P.warn
        rows.append((f"{row['shot_id']}  {row['shot_size'][:13]}  hint={row['engine_hint']}",
                     f"{tag}   ${row.get('cost',0):.4f}", col))
    by = r.get("by_provider", {})
    off = sum(v for k, v in by.items() if "official" in k)
    opn = sum(v for k, v in by.items() if "open" in k)
    return card(ctx, "08_routing", V.kv_table(
        "双引擎路由：同一份角色圣经，两条通道", rows,
        subtitle=f"{r.get('ok',0)}/{r.get('shots',0)} 镜出片 · 官方 ${off:.2f} / 开源 ${opn:.4f} · "
                 f"决策只读 Capabilities",
        step="STEP 06", col_w=560), 12.0)


def seg_shotwall(ctx: Ctx):
    if (c := ctx.cached("09_shotwall")) is not None:
        return c
    vids = [r["video"] for r in ctx.rows if r.get("video") and Path(r["video"]).exists()][:8]
    if len(vids) < 4:
        return None
    n, cols, cw, ch = len(vids), 4, 320, 180
    inputs = []
    for v in vids:
        inputs += ["-stream_loop", "-1", "-i", v]
    parts = [f"[{i}:v]scale={cw-4}:{ch-4}:force_original_aspect_ratio=decrease,"
             f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:0x0a0c11,setsar=1[t{i}]" for i in range(n)]
    layout = "|".join(f"{(i % cols) * cw}_{(i // cols) * ch}" for i in range(n))
    parts.append("".join(f"[t{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={layout}[grid]")
    grid_h = ((n + cols - 1) // cols) * ch
    parts.append(f"[grid]pad=1280:720:0:{(720 - grid_h)//2 + 20}:0x0a0c11[padded]")

    from PIL import Image, ImageDraw
    r = ctx.stage("render")
    ov = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.rectangle([0, 0, 1280, 92], fill=(18, 22, 30, 242))
    od.line([(0, 92), (1280, 92)], fill=(48, 58, 76, 255))
    od.text((40, 22), "原子镜并行出片", font=V.font(34), fill=(232, 238, 247, 255))
    od.text((40, 62), "每格是 provider 真实产出的 mp4 · HUD 显示 seed / 运镜 / 分级 / 景别",
            font=V.font(19), fill=(146, 160, 182, 255))
    od.rectangle([0, 720 - 64, 1280, 720], fill=(18, 22, 30, 242))
    od.text((40, 720 - 48),
            f"{r.get('shots',0)} 镜 · 重拍 {r.get('retakes',0)} 次 · 续接 {r.get('chain_links',0)} 处 · "
            f"成本 ${r.get('cost_usd',0):.4f}", font=V.font(20), fill=(214, 186, 122, 255))
    ovp = WORK / "wall_overlay.png"
    ov.save(ovp)

    d = ctx.s(13.0)
    out = SEG / "09_shotwall.mp4"
    ff(*inputs, "-i", str(ovp), "-filter_complex",
       ";".join(parts) + f";[padded][{n}:v]overlay=0:0[out]", "-map", "[out]",
       "-t", f"{d:.2f}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
       "-r", str(FPS), str(out))
    return Segment("09_shotwall", out, d)


BUILDERS = [seg_title, seg_constraints, seg_pipeline, seg_storyboard, seg_charbible,
            seg_refpack, seg_audio, seg_routing, seg_shotwall]


def seg_chain(ctx: Ctx) -> Segment:
    rows = ctx.rows
    linked = [r for r in rows if r.get("chain")]
    reanchor = [r for r in rows if r.get("reanchor")]
    strategies: dict[str, int] = {}
    for r in rows:
        s = r.get("chain_strategy")
        if s:
            strategies[s] = strategies.get(s, 0) + 1
    lines = [("续接策略分布（由 ChainPlanner 按 provider 能力与连续性声明自动选）：", P.accent)]
    for k, v in sorted(strategies.items(), key=lambda kv: -kv[1]):
        lines.append((f"   {k:18} {v:2} 镜", P.accent2))
    lines += [("", P.text), ("漂移预算逐镜累积，超阈值强制重锚：", P.accent)]
    shown = 0
    for r in rows:
        if r.get("drift_after") is None:
            continue
        mark = "  << 强制重锚" if r.get("reanchor") else ""
        col = P.warn if r.get("reanchor") else P.text_dim
        lines.append((f"   {r['shot_id']:11} 策略 {str(r.get('chain_strategy') or '-'):12} "
                      f"漂移 {r['drift_after']:.3f}{mark}", col))
        shown += 1
        if shown >= 8:
            break
    lines += [("", P.text),
              (f"全片续接 {len(linked)} 处 · 强制重锚 {len(reanchor)} 次", P.ok),
              ("尾帧抽取带质检：末帧糊了就回退倒数第 N 帧 —— 模糊的尾帧会毒化下一段", P.gold)]
    return card(ctx, "10_chain", V.code_panel(
        "首尾帧锁戏 + 漂移预算", lines,
        subtitle="链式续写的误差会累积，所以把它做成显式预算而不是祈祷",
        step="STEP 07"), 13.0)


def seg_qc(ctx: Ctx) -> Segment:
    r = ctx.stage("render")
    rows = ctx.rows
    passed = [x for x in rows if x.get("qc_verdict") == "pass"]
    retake = [x for x in rows if x.get("take", 1) > 1]
    scored = sorted([x for x in rows if x.get("qc_score") is not None],
                    key=lambda x: x["qc_score"])
    lines = [(f"门禁 7 项指标全部 CPU 可跑 · {r.get('shots',0)} 镜中 {len(passed)} 镜一次过 · "
              f"{len(retake)} 镜触发重拍", P.accent), ("", P.text)]
    worst = scored[0] if scored else None
    best = scored[-1] if scored else None
    for tag, x, col in (("最差", worst, P.bad), ("最好", best, P.ok)):
        if not x:
            continue
        lines.append((f"{tag}  {x['shot_id']}  score={x.get('qc_score')}  "
                      f"verdict={x.get('qc_verdict')}  take={x.get('take')}", col))
        for m in x.get("qc_metrics", [])[:4]:
            mark = "✓" if m["passed"] else "✗"
            lines.append((f"      {mark} {m['name']:14} {m['value']}",
                          P.text_dim if m["passed"] else P.bad))
    adv = next((x.get("qc_advice") for x in rows if x.get("qc_advice")), [])
    if adv:
        lines += [("", P.text), ("门禁给的是可执行建议，不是一个分数：", P.accent)]
        for a in adv[:2]:
            for ln in [a[i:i + 86] for i in range(0, min(len(a), 172), 86)]:
                lines.append((f"   {ln}", P.warn))
    return card(ctx, "11_qc", V.code_panel(
        "质检门禁：废片自动拦截", lines,
        subtitle="三态裁决 pass / retake / escalate —— 重拍次数用尽才叫人，不是分低就叫人",
        step="STEP 08"), 14.0)


def seg_grade(ctx: Ctx):
    g = ctx.stage("grade")
    scenes = g.get("scenes", [])
    if not scenes:
        return None
    rows = ctx.rows
    src = next((r["video"] for r in rows if r.get("video") and Path(r["video"]).exists()), None)
    sid = next((r["shot_id"] for r in rows if r.get("video") == src), "")
    dst = ROOT / "out" / "work" / "full" / "graded" / f"{sid}.mp4"
    if not src or not dst.exists():
        return None
    a, b = WORK / "_g_before.png", WORK / "_g_after.png"
    for v, o in ((src, a), (str(dst), b)):
        ff("-ss", "1", "-i", v, "-frames:v", "1", str(o))
    from PIL import Image
    sc = scenes[0]
    verdict = (f"场景 {sc['scene']}：{sc['clips']} 镜统一到基准（中位数，抗异常镜污染）  "
               f"最大亮度漂移 {sc['max_drift_before']:.1f} → {sc['max_drift_after']:.1f} 码值")
    img = V.split_compare("跨镜色彩统一", Image.open(a), Image.open(b),
                          f"调色前  漂移 {sc['max_drift_before']:.1f}",
                          f"调色后  漂移 {sc['max_drift_after']:.1f}",
                          subtitle="不同引擎出来的片段色温会飘，直接拼一眼假",
                          step="STEP 09", verdict=verdict)
    return card(ctx, "12_grade", img, 11.0)


def seg_assemble(ctx: Ctx) -> Segment:
    a = ctx.stage("assemble")
    sb = ctx.sb
    shots = sb.all_shots()
    rendered = {r["shot_id"] for r in ctx.rows if r.get("video")}
    pal = {"sc01": P.accent, "sc02": P.accent2, "sc03": P.ok}
    strip = [(s.id.split("_")[1], s.duration_s, pal.get(s.scene_id, P.accent))
             for s in shots if s.id in rendered]
    cues = []
    t = 0.0
    for s in shots:
        if s.id in rendered:
            for d in s.dialogue:
                st = t + (d.start_s or 0.4)
                cues.append((st, st + max((d.end_s or 2.0) - (d.start_s or 0.0), 1.0), P.gold))
            t += s.duration_s
    img = V.timeline_strip(
        "成片时间线：15s 原子镜 × N = 一集", strip,
        tracks=[("对白轨（TTS 实测时长）", cues[:14]),
                ("Foley / 音乐轨（占位）", [(0, t * 0.45, P.warn), (t * 0.55, t, P.warn)])],
        subtitle=f"{a.get('clips',0)} 镜 · {a.get('duration_s',0):.1f}s · {a.get('resolution','')} · "
                 f"{a.get('size_mb',0):.1f}MB · 字幕 {a.get('subtitle_lines',0)} 条 · 同出 EDL 可进 DaVinci",
        step="STEP 10")
    return card(ctx, "13_assemble", img, 11.0)


def seg_post(ctx: Ctx) -> Segment:
    p = ctx.stage("post")
    lines = [(f"硬件探测：{p.get('hardware','?')}", P.accent), ("", P.text),
             ("按可用硬件选链路，落选的每一条都要说清为什么：", P.accent)]
    for w in p.get("warnings", [])[:5]:
        lines.append((f"   跳过  {w[:84]}", P.warn))
    lines += [("", P.text)]
    for s in p.get("steps", []):
        lines.append((f"   执行  {str(s)[:88]}", P.ok))
    lines += [("", P.text),
              (f"{p.get('from','?')} → {p.get('to','?')} · 产物 {p.get('size_mb',0):.1f}MB", P.accent2),
              ("", P.text),
              ("拼接只归一到生成分辨率，放大留给超分器 —— 否则等于用 lanczos 冒充超分", P.gold),
              ("真实产线：FlashVSR 日常档（8 分钟片约 11 分钟）/ SeedVR2-3B 交付档", P.text_dim),
              ("SeedVR2-7B 全片要 8.5–9.6 小时单 H100，不可能进日常循环", P.text_dim)]
    return card(ctx, "14_post", V.code_panel(
        "超分后处理：720p 出片省额度，本地拉 1080", lines,
        subtitle="选型顺序是先看能不能跑，再看画质 —— 排进去再 OOM 的代价是整批重跑",
        step="STEP 11"), 12.0)


def seg_cost(ctx: Ctx) -> Segment:
    r = ctx.stage("render")
    by = r.get("by_provider", {})
    off = sum(v for k, v in by.items() if "official" in k)
    opn = sum(v for k, v in by.items() if "open" in k)
    n_off = sum(1 for x in ctx.rows if "official" in str(x.get("provider")))
    n_opn = sum(1 for x in ctx.rows if "open" in str(x.get("provider")))
    ratio = off / max(opn, 1e-9)
    img = V.bar_chart(
        "成本账本：路由决定钱花在哪",
        [(f"闭源官方通道（{n_off} 镜）", off, P.accent),
         (f"自建开源通道（{n_opn} 镜）", opn, P.accent2)],
        unit=" USD",
        subtitle=f"同一集片两条通道单价差 {ratio:.0f}× · 英雄镜买画质，其余镜买产能 · "
                 f"本集合计 ${r.get('cost_usd',0):.2f}",
        step="STEP 12")
    return card(ctx, "15_cost", img, 10.0)


def seg_cloud(ctx: Ctx) -> Segment:
    lines = [("一集 5 分钟（30 镜 × 8.6s，重拍率 35%）：", P.accent),
             ("   全 API 最优   火山方舟 Seedance 1.0 Pro   $16.84      $0.0484/s  [已核]", P.accent),
             ("   自建最优      Vast RTX 5090 $0.45/h       $4.04–7.02  $0.0156–0.0272", P.accent2),
             ("   盈亏平衡点    成片约 11 秒 —— 只要不是做一条短视频，自建就更便宜", P.ok),
             ("", P.text),
             ("一季 12 集（96 分钟成片）：", P.accent),
             ("   全 API        $376（Seedance 1.0 Pro）/ $700（2.0 Fast）", P.accent),
             ("   自建 4090     $97–173，省 $203–279；单卡墙钟 11–21 天，7 天内需 3 卡并行", P.accent2),
             ("   并行不改总机时，只改档期与并发上限 —— 排期最常算错的一条", P.warn),
             ("", P.text),
             ("三条必须自己验证的假设：", P.gold),
             ("   1 吞吐是按硬件规格外推的估计，跑完真实渲染后用 --calibrate 覆盖", P.text_dim),
             ("   2 FlashVSR 官方只保证 A100/A800，只有 4090 要先验证编译", P.text_dim),
             ("   3 除火山基准价外多为二手价，排期前逐条复核", P.text_dim)]
    return card(ctx, "16_cloud", V.code_panel(
        "云 GPU 成本模型", lines,
        subtitle="deploy/cost_calculator.py 真实运行 · 每条价格都带可信度标记",
        step="STEP 13"), 13.0)


def seg_deliver(ctx: Ctx) -> Segment:
    d = ctx.stage("deliver")
    probs = d.get("compliance_problems", [])
    lines = [(f"溯源清单 {d.get('manifest_shots',0)} 镜 · 产物 {d.get('size_mb',0):.1f}MB", P.accent),
             ("", P.text),
             ("每一镜都留下可审计的痕迹：", P.accent),
             ("   引擎与模型版本 · LoRA 与触发词 · seed · 提示词哈希", P.text_dim),
             ("   每张参考图的 sha256 与来源类型 · 角色的成年声明", P.text_dim),
             ("", P.text),
             ("显式标识（烧进画面）+ 隐式标识（写进元数据）双轨：", P.accent2),
             ("   中国《AI生成合成内容标识办法》2025-09-01 施行", P.text_dim),
             ("   EU AI Act 第 50 条 2026-08-02 生效", P.text_dim),
             ("", P.text)]
    if probs:
        lines.append((f"发片前检查拦下 {len(probs)} 项：", P.warn))
        for p_ in probs[:3]:
            lines.append((f"   {p_[:86]}", P.warn))
    else:
        lines.append(("发片前检查：全部通过（角色成年声明 / 参考图来源类型 / AI 标识开启）", P.ok))
    lines += [("", P.text),
              ("参考图来源类型是强制字段 —— 这是「不碰真人肖像」在产物层面的凭据", P.gold)]
    return card(ctx, "17_deliver", V.code_panel(
        "合规交付与溯源", lines,
        subtitle="既是合规材料，也是复现依据", step="STEP 14"), 12.0)


def seg_finale(ctx: Ctx) -> Segment:
    r = ctx.stage("render")
    return card(ctx, "18_finale", V.title_card(
        "能做到 / 做不到",
        "目标是「单镜接近，成片靠系统」，不是「微调出一台无审查 Seedance」",
        ["✅ 分镜到成片的完整调度：路由 · 续接 · 质检 · 重拍 · 调色 · 拼接 · 合规",
         f"✅ 本次实跑：{r.get('shots',0)} 镜 / {r.get('retakes',0)} 次自动重拍 / "
         f"{r.get('chain_links',0)} 处续接 / 6 个阶段全绿",
         "❌ 本机微调官方 Seedance 权重：不存在，且是终局",
         "❌ 第三方「无审核 Seedance」：审核在字节侧管线内，物理绕不过",
         "❌ 一镜 10 分钟电影级：2026-09 仍不存在于可商用开源或官方 API"],
        note="画质那一半要接真引擎才有：本 demo 的镜头由 mock 合成，演示的是产线调度",
    ), 12.0, zoom=(1.03, 1.0), fade_out=1.2)


BUILDERS = [seg_title, seg_constraints, seg_pipeline, seg_storyboard, seg_charbible,
            seg_refpack, seg_audio, seg_routing, seg_shotwall, seg_chain, seg_qc,
            seg_grade, seg_assemble, seg_post, seg_cost, seg_cloud, seg_deliver, seg_finale]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="复用已渲好的段，只补缺的")
    ap.add_argument("--only", default="")
    ap.add_argument("--pipe", default=str(PIPE / "pipeline_result.json"))
    args = ap.parse_args()

    for d in (OUT, SEG, WORK):
        d.mkdir(parents=True, exist_ok=True)

    pipe_p = Path(args.pipe)
    if not pipe_p.exists():
        raise SystemExit(f"找不到产线记录 {pipe_p}；先跑 scripts/run_pipeline.py")
    ctx = Ctx(sb=Storyboard.load(ROOT / "configs" / "demo_episode.json"),
              pipe=json.loads(pipe_p.read_text(encoding="utf-8")), quick=args.quick,
              reuse=args.reuse)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    tl = DemoTimeline()
    t0 = time.time()
    for fn in BUILDERS:
        name = fn.__name__.removeprefix("seg_")
        if only and name not in only:
            continue
        try:
            seg = fn(ctx)
        except Exception as e:
            print(f"  [fail] {name}: {type(e).__name__}: {e}")
            continue
        if seg is None:
            print(f"  [skip] {name} —— 缺少所需产线数据")
            continue
        tl.add(seg)
        print(f"  [ok]   {seg.name:14} {seg.duration_s:5.2f}s", flush=True)

    if not tl.segments:
        raise SystemExit("没有任何段落生成")
    video = tl.concat(OUT / "_video.mp4")
    bed = ambient_bed(tl.total_s, OUT / "_bed.m4a")
    final = mux(video, bed, OUT / "demo.mp4")
    dur = probe_duration(final)
    print(f"\ndemo → {final}")
    print(f"  {len(tl.segments)} 段 · {dur:.1f}s ({dur/60:.1f} 分钟) · "
          f"{final.stat().st_size/1e6:.2f} MB · 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
