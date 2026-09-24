"""在自建 ComfyUI（Wan 2.2）上出数字人样片，落进 web/media/ 并重建页面。

和 make_showreel.py 的区别只有引擎：那条走火山方舟闭源 API，这条走租来的
GPU + 开源权重。**后面那半条路完全一样**：质检门禁、显式标识、媒体清单，
都用产线自己的模块，所以两边产出的片子可以直接比。

    python scripts/make_showreel_wan.py --url https://<pod>-8188.proxy.runpod.net --dry-run
    python scripts/make_showreel_wan.py --url https://<pod>-8188.proxy.runpod.net

角色一律虚构成年，提示词里写死；不指涉任何真人。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from longfilm.compliance import add_visible_disclosure  # noqa: E402

MEDIA = ROOT / "web" / "media"
WORK = ROOT / "out" / "showreel_wan"
TEMPLATE = ROOT / "deploy" / "comfy_workflows" / "wan22_ti2v_5b.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"

# Wan 的 latent 时间维是 4 的倍数，帧数必须 4k+1，否则解码掉尾帧。
NEGATIVE = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, "
    "最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, "
    "画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, "
    "杂乱的背景, 三条腿, 背景人很多, 倒着走"
)


@dataclass(frozen=True)
class ReelShot:
    key: str
    label: str
    prompt: str
    why: str
    length: int = 81          # 4k+1；81 帧 @24fps ≈ 3.4s
    width: int = 1280
    height: int = 704
    steps: int = 30
    cfg: float = 5.0
    seed: int = 0


REEL: tuple[ReelShot, ...] = (
    ReelShot(
        key="wan_presenter",
        label="虚构主播口播",
        why="口播场景的基准：正面中近景、稳定光比、可用的面部细节",
        seed=77_001,
        prompt=(
            "A medium close-up of a fictional adult female news presenter in her early thirties, "
            "wearing a navy blazer over a white shirt, standing in a modern broadcast studio. "
            "She speaks to the camera with natural mouth movement, calm expression turning into a slight smile, "
            "natural blinking and subtle head motion. Soft key light from the front left, cool rim light on the right, "
            "blurred blue screen wall in the background. Cinematic realism, shallow depth of field, 85mm lens, "
            "steady camera slowly pushing in, film grain, high detail skin texture."
        ),
    ),
    ReelShot(
        key="wan_story",
        label="剧情角色 · 失重走廊",
        why="剧情镜考的是运镜、光比和环境细节，不只是一张脸",
        seed=77_002,
        prompt=(
            "A medium shot of a fictional adult female deep-space archaeologist with short black hair shaved on the right side, "
            "wearing a dark grey excavation suit with an orange mission patch on the left chest and a glowing blue wrist terminal. "
            "She moves slowly down the corridor of an abandoned space station, her flashlight sweeping across frost-covered metal walls. "
            "Cold blue ambient light mixed with warm orange practicals, volumetric light through floating dust. "
            "Steadicam follows her laterally, 35mm lens, cinematic, high dynamic range, detailed textures."
        ),
    ),
    ReelShot(
        key="wan_closeup",
        label="特写 · 面部细节",
        why="特写最吃身份一致性：五官、皮肤质感、眼神都要经得起放大",
        seed=77_003,
        length=61,
        prompt=(
            "An extreme close-up of the face of a fictional adult male android assistant, matte pale grey synthetic skin, "
            "ring-shaped glowing cyan irises, very short silver hair with a visible seam at the hairline. "
            "He blinks slowly and turns his head a few degrees toward the camera. "
            "Cool key light from the upper left, soft cyan fill from a chest indicator ring below the frame. "
            "Photorealistic, macro detail, 100mm lens, shallow depth of field, subtle film grain."
        ),
    ),
)


class Comfy:
    """ComfyUI 的最小 HTTP 客户端。只用到提交 / 轮询 / 取文件三件事。"""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _req(self, path: str, data: bytes | None = None, timeout: float = 90):
        req = urllib.request.Request(
            self.url + path, data=data,
            headers={"User-Agent": UA, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as f:
            raw = f.read()
        return json.loads(raw) if raw[:1] in b"{[" else raw

    def submit(self, graph: dict) -> str:
        body = json.dumps({"prompt": graph, "client_id": self.client_id}).encode()
        out = self._req("/prompt", body)
        if "prompt_id" not in out:
            raise RuntimeError(f"提交被拒：{json.dumps(out, ensure_ascii=False)[:400]}")
        return out["prompt_id"]

    def wait(self, pid: str, timeout_s: float, poll_s: float = 10.0) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            hist = self._req(f"/history/{pid}")
            if pid in hist:
                rec = hist[pid]
                status = (rec.get("status") or {}).get("status_str", "")
                if status == "error":
                    raise RuntimeError(f"执行失败：{json.dumps(rec.get('status'), ensure_ascii=False)[:500]}")
                if rec.get("outputs"):
                    return rec
            time.sleep(poll_s)
        raise TimeoutError(f"等了 {timeout_s:.0f}s 还没出片")

    def fetch(self, filename: str, subfolder: str, kind: str, dest: Path) -> Path:
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": kind})
        req = urllib.request.Request(f"{self.url}/view?{q}", headers={"User-Agent": UA})
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(req, timeout=300) as f, dest.open("wb") as out:
            while chunk := f.read(1 << 20):
                out.write(chunk)
        return dest


def build_graph(shot: ReelShot) -> dict:
    tpl = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    g = {k: v for k, v in tpl.items() if not k.startswith("_")}
    g["4"]["inputs"]["text"] = shot.prompt
    g["5"]["inputs"]["text"] = NEGATIVE
    g["6"]["inputs"].update(width=shot.width, height=shot.height, length=shot.length)
    g["8"]["inputs"].update(seed=shot.seed, steps=shot.steps, cfg=shot.cfg)
    g["11"]["inputs"]["filename_prefix"] = f"longfilm/{shot.key}"
    return g


def pick_video(rec: dict) -> tuple[str, str, str]:
    """从 history 记录里找出保存的视频。SaveVideo 的输出键随版本变过，所以全扫一遍。"""
    for node_out in rec.get("outputs", {}).values():
        for key in ("images", "videos", "gifs", "video"):
            for item in node_out.get(key, []) or []:
                if isinstance(item, dict) and str(item.get("filename", "")).lower().endswith(
                        (".mp4", ".webm", ".mkv")):
                    return item["filename"], item.get("subfolder", ""), item.get("type", "output")
    raise RuntimeError(f"没找到视频输出：{json.dumps(rec.get('outputs'), ensure_ascii=False)[:400]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="ComfyUI 地址（RunPod 代理 URL）")
    ap.add_argument("--only", nargs="*", help="只跑这几条（key）")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    # 冒烟用：先用低步数小分辨率验证图能跑通，再上正式参数。租来的卡按小时计费，
    # 拿正式参数去试错是最贵的调试方式。
    ap.add_argument("--steps", type=int, help="覆盖采样步数")
    ap.add_argument("--length", type=int, help="覆盖帧数（必须 4k+1）")
    ap.add_argument("--size", help="覆盖分辨率，如 1024x576")
    a = ap.parse_args()

    shots = [s for s in REEL if not a.only or s.key in a.only]
    if a.steps or a.length or a.size:
        import dataclasses
        w, h = (int(x) for x in a.size.split("x")) if a.size else (None, None)
        shots = [dataclasses.replace(
            s,
            steps=a.steps or s.steps,
            length=a.length or s.length,
            width=w or s.width,
            height=h or s.height,
        ) for s in shots]
    print("=" * 72)
    for s in shots:
        print(f"  {s.key:16} {s.label:16} {s.width}x{s.height} {s.length}帧"
              f"({s.length / 24:.1f}s) steps={s.steps} seed={s.seed}")
    if a.dry_run:
        print("--dry-run：不提交。")
        return 0

    comfy = Comfy(a.url)
    from longfilm.qc import QCContext, QCGate

    gate = QCGate.default()
    items: list[dict] = []
    for s in shots:
        print(f"\n[{s.key}] 提交…", flush=True)
        t0 = time.time()
        try:
            pid = comfy.submit(build_graph(s))
            rec = comfy.wait(pid, a.timeout)
        except Exception as e:  # noqa: BLE001  失败原因要如实打印，不吞
            print(f"  失败：{type(e).__name__} {e}")
            continue
        fn, sub, kind = pick_video(rec)
        raw = comfy.fetch(fn, sub, kind, WORK / f"{s.key}.mp4")
        took = time.time() - t0
        print(f"  出片 {raw.stat().st_size / 1e6:.1f}MB · {took:.0f}s")

        report = gate.evaluate(str(raw), None, None, context=QCContext(
            expect_duration_s=s.length / 24, expect_fps=24,
            expect_resolution=(s.width, s.height),
        ))
        print(f"  质检 {report.verdict.value} {report.score:.3f}")

        final = MEDIA / f"{s.key}.mp4"
        add_visible_disclosure(raw, final, text="本视频由 AI 生成 · 虚构角色")
        items.append({
            "file": final.name,
            "label": s.label,
            "kind": "engine",
            "note": (f"自建 ComfyUI + Wan 2.2 TI2V-5B 直接生成 · {s.length}帧/{s.length / 24:.1f}s · "
                     f"{s.width}x{s.height} · {s.steps} 步 · 出片耗时 {took:.0f}s · "
                     f"质检 {report.verdict.value} {report.score:.2f}。{s.why}。"),
            "pipeline": "prompt → comfy:wan2.2-ti2v-5b (RTX 3090) → qc → compliance",
        })

    if not items:
        print("\n一条都没出成。")
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
