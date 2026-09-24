"""RunPod 机器控制 —— 开机、查状态、拿地址、关机。

为什么要这个脚本而不是点网页：**开着的卡一直在烧钱**。开机、出片、关机必须是
可复现的一串命令，关机那步尤其不能靠人记得去点。每条命令都打印当前时烧，
`terminate` 打印本次总花费。

密钥从 .env 的 RUNPOD_API_KEY 读，**永不回显**。

    python deploy/runpod_ctl.py gpus                    # 可开的卡与实价
    python deploy/runpod_ctl.py sshkey                  # 把本机公钥登记到账户
    python deploy/runpod_ctl.py create --gpu "RTX A6000"
    python deploy/runpod_ctl.py status                  # 所有 pod + 时烧
    python deploy/runpod_ctl.py wait <podId>            # 等到 ComfyUI 起来
    python deploy/runpod_ctl.py terminate <podId>       # 关机（必做）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"
# Cloudflare 会拦掉 urllib 的默认 UA（error 1010），所以显式给一个浏览器 UA。
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")
COMFY_IMAGE = "runpod/comfyui:1.3.2-comfyuiv0.30.0-cuda12.8"


def api_key() -> str:
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("找不到 RUNPOD_API_KEY，把它写进 .env（该文件已被 gitignore）")


def gql(query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        f"{GQL}?api_key={api_key()}", data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=90) as f:
        out = json.loads(f.read())
    if out.get("errors"):
        raise SystemExit("RunPod 报错：" + json.dumps(out["errors"], ensure_ascii=False)[:400])
    return out["data"]


def rest(path: str, method: str = "GET", body: dict | None = None):
    req = urllib.request.Request(
        REST + path, method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": "Bearer " + api_key(), "User-Agent": UA,
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as f:
            raw = f.read()
            return f.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400].decode("utf-8", "replace")


def balance() -> tuple[float, float]:
    d = gql("{ myself { clientBalance currentSpendPerHr } }")["myself"]
    return d["clientBalance"], d["currentSpendPerHr"]


def cmd_gpus(_a) -> int:
    d = gql("""{ gpuTypes { id displayName memoryInGb communityCloud secureCloud
        lowestPrice(input:{gpuCount:1}) { uninterruptablePrice minimumBidPrice } } }""")
    rows = [g for g in d["gpuTypes"]
            if g["memoryInGb"] >= 24 and (g["lowestPrice"]["uninterruptablePrice"] or 0) > 0]
    rows.sort(key=lambda g: g["lowestPrice"]["uninterruptablePrice"])
    print(f"{'GPU':28} {'显存':>5} {'按需/h':>8} {'竞价/h':>8}")
    for g in rows[:16]:
        p = g["lowestPrice"]
        print(f"{g['displayName']:28} {g['memoryInGb']:>4}G "
              f"{p['uninterruptablePrice'] or 0:>8.3f} {p['minimumBidPrice'] or 0:>8.3f}")
    return 0


def cmd_sshkey(a) -> int:
    """把本机公钥登记到账户。pod 的 SSH 登录只认账户里的公钥。"""
    priv = Path(a.key).expanduser()
    pub = priv.with_suffix(priv.suffix + ".pub") if priv.suffix else Path(str(priv) + ".pub")
    if not pub.is_file():
        priv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "longfilm",
                        "-f", str(priv)], check=True, capture_output=True)
        print(f"已生成新密钥：{priv}")
    key_line = pub.read_text(encoding="utf-8").strip()
    cur = gql("{ myself { pubKey } }")["myself"].get("pubKey") or ""
    if key_line in cur:
        print("公钥已在账户里，跳过")
        return 0
    merged = (cur.rstrip() + "\n" + key_line).strip()
    gql("mutation($input: UpdateUserSettingsInput!){ updateUserSettings(input:$input){ id } }",
        {"input": {"pubKey": merged}})
    print("公钥已登记")
    return 0


def cmd_create(a) -> int:
    bal, spend = balance()
    print(f"账户余额 ${bal:.2f}，当前时烧 ${spend:.3f}/h")
    d = gql("""{ gpuTypes { id displayName lowestPrice(input:{gpuCount:1}){ uninterruptablePrice } } }""")
    match = [g for g in d["gpuTypes"] if a.gpu.lower() in g["displayName"].lower()]
    if not match:
        raise SystemExit(f"找不到 GPU「{a.gpu}」，先跑 gpus 看有哪些")
    gpu = match[0]
    price = gpu["lowestPrice"]["uninterruptablePrice"]
    print(f"要开：{gpu['displayName']}  ${price}/h  镜像 {a.image}")
    print(f"  容器盘 {a.disk}G / 数据盘 {a.volume}G（挂在 {a.mount}）")

    st, body = rest("/pods", "POST", {
        "name": a.name,
        "imageName": a.image,
        "gpuTypeIds": [gpu["id"]],
        "gpuCount": 1,
        "cloudType": "SECURE" if a.secure else "COMMUNITY",
        "containerDiskInGb": a.disk,
        "volumeInGb": a.volume,
        "volumeMountPath": a.mount,
        "ports": ["8188/http", "8888/http", "22/tcp"],
        "env": {"JUPYTER_PASSWORD": ""},
        "interruptible": False,
    })
    if st >= 300:
        raise SystemExit(f"开机失败 HTTP {st}：{body}")
    pod_id = body["id"] if isinstance(body, dict) else None
    print(f"已下单，podId={pod_id}")
    print(f"  ComfyUI 将在 https://{pod_id}-8188.proxy.runpod.net")
    print(f"  别忘了收工时：python deploy/runpod_ctl.py terminate {pod_id}")
    return 0


def _pods() -> list[dict]:
    st, body = rest("/pods")
    if st >= 300:
        raise SystemExit(f"查询失败 HTTP {st}：{body}")
    return body or []


def cmd_status(_a) -> int:
    bal, spend = balance()
    pods = _pods()
    print(f"余额 ${bal:.2f} · 时烧 ${spend:.3f}/h · {len(pods)} 个 pod")
    for p in pods:
        gpu = (p.get("machine") or {}).get("gpuTypeId") or p.get("gpuTypeId") or "?"
        print(f"  {p['id']}  {p.get('desiredStatus','?'):10} {gpu:22} "
              f"${p.get('costPerHr', 0)}/h  {p.get('name','')}")
        if p.get("portMappings"):
            print(f"      端口映射 {p['portMappings']}")
        print(f"      ComfyUI https://{p['id']}-8188.proxy.runpod.net")
    return 0


def cmd_wait(a) -> int:
    """等 ComfyUI 起来。看的是它自己的 /system_stats，不是 pod 状态 ——
    pod 说 RUNNING 只代表容器起来了，ComfyUI 可能还在装依赖。"""
    url = f"https://{a.pod}-8188.proxy.runpod.net/system_stats"
    t0 = time.time()
    while time.time() - t0 < a.timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as f:
                if f.status == 200:
                    info = json.loads(f.read())
                    dev = (info.get("devices") or [{}])[0]
                    print(f"ComfyUI 就绪（{time.time() - t0:.0f}s）："
                          f"{dev.get('name', '?')} 显存 {dev.get('vram_total', 0) / 1e9:.0f}G")
                    return 0
        except Exception:
            pass
        print(f"  等待中 {time.time() - t0:.0f}s …", flush=True)
        time.sleep(a.interval)
    print("超时：ComfyUI 还没起来。status 看 pod 状态，或 SSH 上去看日志。")
    return 1


def cmd_terminate(a) -> int:
    for pod in (a.pods or [p["id"] for p in _pods()]):
        st, body = rest(f"/pods/{pod}", "DELETE")
        print(f"  {pod} → {'已关机' if st < 300 else f'失败 HTTP {st} {body}'}")
    bal, spend = balance()
    print(f"余额 ${bal:.2f} · 时烧 ${spend:.3f}/h")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpus").set_defaults(fn=cmd_gpus)
    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("sshkey")
    p.add_argument("--key", default="~/.ssh/id_ed25519_runpod")
    p.set_defaults(fn=cmd_sshkey)

    p = sub.add_parser("create")
    p.add_argument("--gpu", default="RTX A6000")
    p.add_argument("--image", default=COMFY_IMAGE)
    p.add_argument("--name", default="longfilm-comfy")
    p.add_argument("--disk", type=int, default=60, help="容器盘 GB")
    p.add_argument("--volume", type=int, default=80, help="数据盘 GB（放模型权重）")
    p.add_argument("--mount", default="/workspace")
    p.add_argument("--secure", action="store_true", help="用 SECURE 云（贵一点，更稳）")
    p.set_defaults(fn=cmd_create)

    p = sub.add_parser("wait")
    p.add_argument("pod")
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--interval", type=float, default=15)
    p.set_defaults(fn=cmd_wait)

    p = sub.add_parser("terminate")
    p.add_argument("pods", nargs="*", help="留空 = 关掉全部")
    p.set_defaults(fn=cmd_terminate)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
