"""从一次真实产线运行生成 Web UI 的数据快照，并把它嵌进 index.html。

界面里的每个数字都来自 scripts/run_pipeline.py 的产物，不写死在页面里。
没有 out/ 时用仓库内的 web/run_snapshot.json（即最近一次运行的快照）。

用法：
    python web/build_ui.py                          # 用快照重建 index.html
    python web/build_ui.py --run out/work/full/pipeline_result.json   # 重新取数并更新快照
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def _clean(x):
    """JSON 里可能有 NaN（质检指标取不到值时）。前端拿到 null 更好处理。"""
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else round(x, 4)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def extract(run_path: Path, storyboard_path: Path) -> dict:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    sb = json.loads(storyboard_path.read_text(encoding="utf-8"))
    stages = {s["name"]: s for s in run["stages"]}
    render = stages["render"]["detail"]

    scene_title = {s["id"]: s["title"] for s in sb["scenes"]}
    dialogue_of = {
        sh["id"]: [d["text"] for d in sh.get("dialogue", [])]
        for sc in sb["scenes"] for sh in sc["shots"]
    }

    shots = []
    for r in render["rows"]:
        shots.append(_clean({
            "id": r["shot_id"],
            "scene": r["scene_id"],
            "scene_title": scene_title.get(r["scene_id"], ""),
            "duration_s": r["duration_s"],
            "shot_size": r["shot_size"],
            "camera_move": r["camera_move"],
            "lens_mm": r["lens_mm"],
            "subjects": r.get("subjects") or [],
            "engine_hint": r.get("engine_hint"),
            "provider": r.get("provider"),
            "take": r.get("take"),
            "cost": r.get("cost"),
            "qc_score": r.get("qc_score"),
            "qc_verdict": r.get("qc_verdict"),
            "qc_metrics": r.get("qc_metrics") or [],
            "qc_advice": r.get("qc_advice") or [],
            "chain": r.get("chain"),
            "chain_strategy": r.get("chain_strategy"),
            "drift_after": r.get("drift_after"),
            "reanchor": r.get("reanchor"),
            "refs": r.get("refs") or {},
            "refs_dropped": r.get("refs_dropped", 0),
            "degradations": r.get("degradations") or [],
            "dialogue": dialogue_of.get(r["shot_id"], []),
            "status": r.get("status"),
        }))

    return _clean({
        "project": run.get("project"),
        "logline": sb.get("logline"),
        "total_s": run.get("total_s"),
        "characters": [
            {
                "id": c["id"], "name": c["name"], "age_statement": c["age_statement"],
                "persona": c.get("persona", ""), "fictional": c["is_fictional"],
                "appearance": c.get("appearance") or {},
                "voice": c.get("voice") or {},
                "lora": c.get("lora"),
                "negative_prompt": c.get("negative_prompt", ""),
                # 定妆清单：uri 已登记，文件在不在本机是另一回事（exists 由 --with-portraits 时回填）
                "portraits": [
                    {"role": r["role"], "uri": r["uri"], "weight": r.get("weight"),
                     "note": r.get("note", ""), "kind": kind}
                    for kind, key in (("portrait", "portraits"), ("turnaround", "turnaround"))
                    for r in (c.get(key) or [])
                ],
                "shots": sum(1 for sc in sb["scenes"] for sh in sc["shots"]
                             if c["id"] in (sh.get("subject_ids") or [])),
            }
            for c in sb["characters"]
        ],
        "scenes": [
            {"id": s["id"], "title": s["title"], "location": s["location"],
             "time_of_day": s["time_of_day"], "shots": len(s["shots"])}
            for s in sb["scenes"]
        ],
        "stages": [
            {"name": s["name"], "ok": s["ok"], "elapsed_s": s.get("elapsed_s"),
             "detail": {k: v for k, v in (s.get("detail") or {}).items() if k != "rows"}}
            for s in run["stages"]
        ],
        "render": {k: v for k, v in render.items() if k != "rows"},
        "shots": shots,
    })


def embed_portraits(data: dict) -> int:
    """把本机存在的定妆图压小后内嵌成 data URI。

    默认**不做**这件事：assets/cast 不进仓库（每张图都要声明来源，也不放真人肖像），
    所以内嵌过的页面只写到 index.local.html，那个文件是 gitignore 的。
    """
    from io import BytesIO

    from PIL import Image

    n = 0
    for c in data["characters"]:
        for ref in c["portraits"]:
            f = ROOT / ref["uri"]
            ref["exists"] = f.is_file()
            if not ref["exists"]:
                continue
            im = Image.open(f).convert("RGB")
            im.thumbnail((420, 560))
            buf = BytesIO()
            im.save(buf, "JPEG", quality=82, optimize=True)
            ref["data"] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            n += 1
    return n


def build(data: dict, out: Path) -> Path:
    tpl = (WEB / "ui_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    assert "__RUN_DATA__" in tpl, "模板里找不到 __RUN_DATA__ 占位"
    out.write_text(tpl.replace("__RUN_DATA__", payload), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, help="pipeline_result.json；给了就重新取数并更新快照")
    ap.add_argument("--storyboard", type=Path, default=ROOT / "configs/demo_episode.json")
    ap.add_argument("--with-portraits", action="store_true",
                    help="把 assets/cast 下已有的定妆图内嵌进页面，输出到 index.local.html（不进仓库）")
    a = ap.parse_args()
    snap = WEB / "run_snapshot.json"
    if a.run:
        data = extract(a.run, a.storyboard)
        snap.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"快照已更新：{snap}（{len(data['shots'])} 镜）")
    else:
        data = json.loads(snap.read_text(encoding="utf-8"))

    out = WEB / "index.html"
    if a.with_portraits:
        n = embed_portraits(data)
        out = WEB / "index.local.html"
        print(f"已内嵌 {n} 张定妆图 —— 这份输出含图片，不要提交进仓库")
        if not n:
            print("  （assets/cast 下一张都没找到；形象区会显示示意图）")
    out = build(data, out)
    print(f"已生成：{out}（{out.stat().st_size // 1024} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
