"""用真引擎出几条数字人样片，落进 web/media/，再重建操作台页面。

mock 通道验证的是调度，出不了画质。这个脚本把画质那一半接上：走火山方舟
Seedance，按产线的正常路径跑（路由 → 生成 → 质检 → 烧标识），产物直接进页面。

为什么不是随手调个 API：样片要能代表产线，就得和产线走同一条路 ——
同样的提示词编译、同样的质检门禁、同样的合规标识。否则页面上的片子
和真出片时的结果对不上。

    # 先看计划与报价，不发请求、不花钱
    python scripts/make_showreel.py --dry-run

    # 真出片（密钥从 .env 读，不回显）
    python scripts/make_showreel.py --budget 5

角色一律虚构成年。提示词里写死「虚构角色」，不指涉任何真人。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from longfilm.compliance import add_visible_disclosure  # noqa: E402
from longfilm.providers.ark_seedance import MODEL_CATALOG, ArkSeedanceProvider  # noqa: E402
from longfilm.providers.base import GenRequest, ProviderError  # noqa: E402

MEDIA = ROOT / "web" / "media"
WORK = ROOT / "out" / "showreel"


def load_env_key(name: str = "ARK_API_KEY") -> str:
    """从环境变量或 .env 取密钥。**永远不回显**，只说有没有。"""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


@dataclass(frozen=True)
class ReelShot:
    """一条样片的工单。duration 会被 provider 的档位归整。"""

    key: str
    model: str
    label: str
    duration_s: float
    prompt: str
    dialogue: str = ""          # 只有原生音模型吃得下
    camera: str = ""
    why: str = ""               # 这条样片想证明什么


# 三条样片各自证明一件事：口型同步、剧情镜的运镜与光比、同题材的成本下限。
REEL: tuple[ReelShot, ...] = (
    ReelShot(
        key="presenter_talking",
        model="doubao-seedance-2-0-260128",
        label="虚构主播口播 · 原生对白声",
        duration_s=8.0,
        camera="镜头缓慢推近，手持微晃",
        dialogue="今天这条片子，从分镜到成片，全流程由产线自动完成。",
        prompt=(
            "虚构成年女性主播的中近景，三十岁左右，深蓝色西装外套配白色衬衫，"
            "站在现代新闻演播室里，背景是虚化的蓝色屏幕墙和暖色边光。"
            "她面向镜头自然讲话，口型与语音同步，表情从平静转为轻微微笑，"
            "有自然的眨眼和细微的头部动作。柔和的主光从左前方打来，右侧有冷色轮廓光。"
            "真实电影感质感，浅景深，85mm 镜头，画面稳定干净。"
        ),
        why="原生对白声 = 口型同步，这是商用数字人那一档的门槛",
    ),
    ReelShot(
        key="story_character",
        model="doubao-seedance-2-0-260128",
        label="剧情角色 · 运镜与光比",
        duration_s=8.0,
        camera="斯坦尼康跟拍，轻微横移",
        prompt=(
            "虚构成年女性深空考古学家的中景，黑色短发右侧削短，深灰色考古服，"
            "左胸有橙色任务徽标，腕部终端发出蓝光。她在废弃空间站的失重走廊里缓慢前行，"
            "手电光扫过覆满霜的金属舱壁。冷蓝色环境光配橙色实用光源，"
            "体积光穿过悬浮尘埃。镜头跟随她横移，35mm 镜头，电影质感，高动态范围。"
        ),
        why="剧情镜要的是运镜、光比、环境细节，不只是一张脸",
    ),
    ReelShot(
        key="presenter_cheap",
        model="doubao-seedance-1-0-pro-250528",
        label="同题材 · 成本下限档",
        duration_s=8.0,
        camera="固定机位",
        prompt=(
            "虚构成年男性主播的中近景，四十岁左右，浅灰色西装，"
            "站在简洁的浅色演播室背景前，面向镜头讲话，表情沉稳自然。"
            "柔和的正面主光，背景有轻微渐变。真实质感，50mm 镜头，画面稳定。"
        ),
        why="同样一条镜，便宜档和旗舰档差多少钱、差多少画质",
    ),
)


def build_request(shot: ReelShot, provider: ArkSeedanceProvider) -> GenRequest:
    caps = provider.caps
    prompt = shot.prompt
    if shot.camera:
        prompt += f" 运镜：{shot.camera}。"
    if shot.dialogue and caps.supports_native_audio:
        prompt += f' 她说："{shot.dialogue}"'
    return GenRequest(
        shot_id=shot.key,
        prompt=prompt,
        duration_s=caps.clamp_duration(shot.duration_s),
        resolution=caps.nearest_resolution((1280, 720)),
        fps=24,
        negative_prompt="字幕, 水印, 台标, 畸形手指, 多余肢体, 面部闪烁, 画面抖动",
        content_rating="g",
        idempotency_key=f"showreel-{shot.key}-{int(shot.duration_s)}",
    )


def plan(budget: float) -> list[tuple[ReelShot, ArkSeedanceProvider, GenRequest, float]]:
    rows = []
    total = 0.0
    for shot in REEL:
        p = ArkSeedanceProvider(
            model=shot.model, api_key="sk-plan-only",
            generate_audio=bool(shot.dialogue),
        )
        req = build_request(shot, p)
        cost = p.estimate_cost(req)
        total += cost
        rows.append((shot, p, req, cost))
    if total > budget:
        raise SystemExit(
            f"预估 ${total:.2f} 超出预算 ${budget:.2f}。"
            f"用 --budget 抬高上限，或用 --only 只跑其中几条。"
        )
    return rows


def render(rows, key: str, timeout_s: float) -> list[dict]:
    from longfilm.qc import QCContext, QCGate

    WORK.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    gate = QCGate.default()
    out: list[dict] = []

    for shot, _p, req, est in rows:
        provider = ArkSeedanceProvider(
            model=shot.model, api_key=key, generate_audio=bool(shot.dialogue),
        )
        print(f"\n[{shot.key}] {shot.model} · {req.duration_s:g}s · 预估 ${est:.3f}")
        t0 = time.time()
        try:
            job_id = provider.submit(req)
            print(f"  已提交 job={job_id}，等待出片…")
            res = provider.wait(job_id, timeout_s=timeout_s)
        except ProviderError as e:
            print(f"  失败（{e.kind.value}）：{e}")
            continue
        if not res.ok or not res.video_uri:
            print(f"  未出片：{res.status.value} {res.message}")
            continue

        raw = WORK / f"{shot.key}.mp4"
        src = Path(res.video_uri)
        if src.is_file():
            shutil.copy2(src, raw)
        else:  # provider 返回的是远端 url
            from longfilm.providers._http import download

            download(res.video_uri, raw)
        print(f"  出片 {raw.stat().st_size / 1e6:.1f}MB · 实付 ${res.cost_usd:.3f} · {time.time() - t0:.0f}s")

        report = gate.evaluate(str(raw), None, None, context=QCContext(
            expect_duration_s=req.duration_s, expect_fps=req.fps,
            expect_resolution=tuple(req.resolution),
        ))
        print(f"  质检 {report.verdict.value} {report.score:.3f}")

        # 合规：显式标识常驻全片，和产线交付阶段同一套
        final = MEDIA / f"{shot.key}.mp4"
        add_visible_disclosure(raw, final, text="本视频由 AI 生成 · 虚构角色")
        out.append({
            "file": final.name,
            "label": shot.label,
            "kind": "engine",
            "note": (f"{shot.model} 直接生成 · {req.duration_s:g}s · 实付 ${res.cost_usd:.3f} · "
                     f"质检 {report.verdict.value} {report.score:.2f}。{shot.why}。"),
            "pipeline": f"prompt_os → ark:{shot.model.split('-')[2]} → qc → compliance",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=5.0, help="本次花费上限（美元）")
    ap.add_argument("--only", nargs="*", help="只跑这几条（key）")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与报价，不发请求")
    a = ap.parse_args()

    rows = plan(a.budget)
    if a.only:
        rows = [r for r in rows if r[0].key in a.only]

    print("=" * 74)
    print("样片计划（报价来自 MODEL_CATALOG 的 token 计费，实付以方舟账单为准）")
    print("=" * 74)
    for shot, _p, req, cost in rows:
        audio = "原生对白声" if shot.dialogue else "无对白"
        print(f"  {shot.key:18} {shot.model:32} {req.duration_s:>4g}s  {audio:10} ≈ ${cost:.3f}")
    print(f"  {'合计':18} {'':32} {'':>5} {'':10} ≈ ${sum(r[3] for r in rows):.3f}")

    if a.dry_run:
        print("\n--dry-run：没有发出任何请求，没有产生费用。")
        return 0

    key = load_env_key()
    if not key:
        print("\n找不到 ARK_API_KEY。把它写进 .env（该文件已被 gitignore）：")
        print('  echo "ARK_API_KEY=你的密钥" >> .env')
        return 2
    print(f"\n已读到密钥（{len(key)} 字符，不回显）。开始出片…")

    items = render(rows, key, a.timeout)
    if not items:
        print("\n一条都没出成。上面的失败原因就是要查的东西。")
        return 1

    mf = MEDIA / "media.json"
    old = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else []
    keep = [m for m in old if m["file"] not in {i["file"] for i in items}]
    mf.write_text(json.dumps(items + keep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {mf}（{len(items)} 条新样片）")

    import subprocess

    subprocess.run([sys.executable, str(ROOT / "web" / "build_ui.py"), "--with-media"], check=True)
    print("页面已重建：web/index.local.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
