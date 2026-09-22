"""longfilm 命令行入口。

只做转发与参数校验，不承载产线逻辑 —— 产线在各模块里，
脚本在 scripts/ 下。CLI 的价值是让人不用记住十几个脚本路径。
"""

from __future__ import annotations

import argparse
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"

MODULES = [
    "schema", "refpack", "prompt_os", "router", "chain", "audio_first",
    "timeline", "stitch", "grade", "upscale", "compliance", "qc",
    "distill", "lora", "dpo", "animate", "previz3d", "queue", "cache",
    "storyboard_gen", "charbible",
]
PROVIDERS = ["mock", "ark_seedance", "comfy_local", "dashscope_wan", "aggregator"]


def _run_script(name: str, argv: list[str]) -> int:
    path = SCRIPTS / name
    if not path.exists():
        print(f"找不到脚本 {path}", file=sys.stderr)
        return 2
    sys.argv = [str(path), *argv]
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(ROOT / "src"))
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def _selftest(pattern: str = "") -> int:
    """跑所有模块自测。这是判断「这份代码还能不能用」最快的方式。"""
    targets = [f"longfilm.{m}" for m in MODULES] + [f"longfilm.providers.{p}" for p in PROVIDERS]
    if pattern:
        targets = [t for t in targets if pattern in t]
    py = sys.executable
    env_src = str(ROOT / "src")
    ok, bad = [], []
    for t in targets:
        print(f"  {t:34} ", end="", flush=True)
        r = subprocess.run([py, "-m", t], capture_output=True, text=True,
                           env={**__import__("os").environ, "PYTHONPATH": env_src},
                           timeout=600)
        if r.returncode == 0:
            ok.append(t)
            print("✓")
        else:
            bad.append(t)
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or [""]
            print(f"✗  {tail[0][:90]}")
    print(f"\n{len(ok)}/{len(targets)} 通过")
    if bad:
        print("失败：" + ", ".join(bad))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="longfilm", description="长时间 AI 数字人拍片系统")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("storyboard", help="生成示例分镜（30 镜 / 2 个虚构成年角色）")
    p_run = sub.add_parser("run", help="端到端跑产线")
    p_run.add_argument("rest", nargs=argparse.REMAINDER)
    p_demo = sub.add_parser("demo", help="把最近一次产线运行渲成 demo 视频")
    p_demo.add_argument("rest", nargs=argparse.REMAINDER)
    p_cost = sub.add_parser("cost", help="成本对比：全 API vs 自建 GPU")
    p_cost.add_argument("rest", nargs=argparse.REMAINDER)
    p_st = sub.add_parser("selftest", help="跑全部模块自测")
    p_st.add_argument("pattern", nargs="?", default="")
    sub.add_parser("modules", help="列出所有模块")

    a = ap.parse_args(argv)
    if a.cmd == "storyboard":
        return _run_script("build_demo_storyboard.py", [])
    if a.cmd == "run":
        return _run_script("run_pipeline.py", a.rest)
    if a.cmd == "demo":
        return _run_script("make_demo.py", a.rest)
    if a.cmd == "cost":
        sys.argv = ["cost_calculator.py", *a.rest]
        sys.path.insert(0, str(ROOT / "deploy"))
        try:
            runpy.run_path(str(ROOT / "deploy" / "cost_calculator.py"), run_name="__main__")
        except SystemExit as e:
            return int(e.code or 0)
        return 0
    if a.cmd == "selftest":
        return _selftest(a.pattern)
    if a.cmd == "modules":
        print("模块   ", " ".join(MODULES))
        print("引擎   ", " ".join(PROVIDERS))
        print("\n单独跑一个模块的自测：PYTHONPATH=src python -m longfilm.<模块名>")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
