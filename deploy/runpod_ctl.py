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


def ssh_target(pod_id: str) -> tuple[str, int] | None:
    """拿 pod 的公网 SSH 地址。没分配到就返回 None。"""
    d = gql('{ pod(input:{podId:"%s"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }' % pod_id)
    rt = (d.get("pod") or {}).get("runtime") or {}
    for port in rt.get("ports") or []:
        if port["privatePort"] == 22 and port.get("isIpPublic"):
            return port["ip"], port["publicPort"]
    return None


def cuda_ok(host: str, port: int, key: str) -> tuple[bool, str]:
    """在 pod 上实测 torch 能不能拿到卡。

    为什么不信 nvidia-smi：**它能列出卡，不代表 CUDA 能初始化**。
    实测遇到过同一台宿主机连开两个 pod，nvidia-smi 正常、torch 一律
    "CUDA unknown error"。只看 nvidia-smi 就会在坏机器上把权重下完才发现。
    """
    cmd = [
        "ssh", "-i", key, "-p", str(port), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=25", f"root@{host}",
        "python3 -c \"import torch;print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_BAD')\" 2>&1 | tail -1",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    # 在全部输出里找标记，不能只看最后一行：ssh 的提示行会盖住真正的结论
    out = (r.stdout + r.stderr).strip()
    if "CUDA_OK" in out:
        return True, "CUDA_OK"
    tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or [""]
    return False, tail[0][:160]


BAD_HOSTS_FILE = ROOT / "out" / "runpod_bad_hosts.txt"


def load_bad_hosts() -> set[str]:
    """已知坏宿主机，**落盘**保存。

    只存在内存里不够用：RunPod 会反复把新 pod 放回同一台机器上，
    而换机循环一旦重启，内存里的黑名单就没了，于是又开回那台坏的。
    实测同一个 IP 连续给了五次，其中还包括 SECURE 云。
    """
    if BAD_HOSTS_FILE.is_file():
        return {ln.split("#")[0].strip() for ln in
                BAD_HOSTS_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()}
    return set()


def mark_bad_host(host: str, why: str) -> None:
    BAD_HOSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if host in load_bad_hosts():
        return
    with BAD_HOSTS_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{host}  # {why} {time.strftime('%Y-%m-%d %H:%M')}\n")


def power_limit_w(host: str, port: int, key: str) -> float:
    """读卡的功耗上限。

    实测踩到过：同一台宿主机把 3090 和 4090 的 power limit 都锁在 150W
    （原厂分别是 350W 和 450W）。nvidia-smi 照常报 100% 利用率，
    算力却只剩三分之一，单镜从 6 分钟劣化到 25 分钟以上。
    这个值一条命令就能读到，比跑基准还快，所以放在基准前面先筛一道。
    """
    cmd = [
        "ssh", "-i", key, "-p", str(port), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=25", f"root@{host}",
        "nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return 0.0
    for line in (r.stdout + r.stderr).splitlines():
        line = line.strip()
        try:
            return float(line)
        except ValueError:
            continue
    return 0.0


BENCH_SNIPPET = (
    "import torch,time;"
    "a=torch.randn(4096,4096,device='cuda',dtype=torch.half);b=torch.randn_like(a);"
    "[a@b for _ in range(3)];torch.cuda.synchronize();t=time.time();"
    "[a@b for _ in range(50)];torch.cuda.synchronize();"
    "print('TFLOPS=%.1f'%(2*4096**3*50/(time.time()-t)/1e12))"
)


def bench_tflops(host: str, port: int, key: str) -> float:
    """实测半精度矩阵乘吞吐。

    为什么要测：社区机器会遇到**限频或被共享**的卡 —— nvidia-smi 报 100% 利用率，
    功耗却只有 149W（4090 满载 400W+），单镜从 6 分钟变成 25 分钟以上。
    这条基准 30 秒出结果，比下完 17GB 权重才发现机器不行便宜得多。
    """
    cmd = [
        "ssh", "-i", key, "-p", str(port), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=25", f"root@{host}", f'python3 -c "{BENCH_SNIPPET}"',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return 0.0
    for line in (r.stdout + r.stderr).splitlines():
        if "TFLOPS=" in line:
            return float(line.split("TFLOPS=")[1].strip())
    return 0.0


def cmd_bench(a) -> int:
    tgt = ssh_target(a.pod)
    if not tgt:
        print("这个 pod 还没分配到公网 SSH"); return 1
    tf = bench_tflops(tgt[0], tgt[1], str(Path(a.key).expanduser()))
    print(f"{a.pod}: {tf:.1f} TFLOPS（fp16 矩阵乘）")
    return 0 if tf >= a.min_tflops else 1


def cmd_provision(a) -> int:
    """开机 → 等 SSH → 实测 CUDA → 坏就换一台。返回可用 pod 的 id。"""
    key = str(Path(a.key).expanduser())
    # 同一台宿主机不重试：RunPod 会把新 pod 放回刚才那台机器上，
    # 被限功耗的卡再开一次还是被限功耗的卡。
    bad_hosts = load_bad_hosts()
    if bad_hosts:
        print(f"已知坏宿主机 {len(bad_hosts)} 台，会跳过：{', '.join(sorted(bad_hosts))}")
    for attempt in range(1, a.tries + 1):
        for gpu in a.gpus:
            print(f"\n[{attempt}/{a.tries}] 试 {gpu}")
            ns = argparse.Namespace(gpu=gpu, image=a.image, name=a.name, disk=a.disk,
                                    volume=a.volume, mount=a.mount, secure=a.secure)
            try:
                buf: list[str] = []
                import contextlib
                import io
                with contextlib.redirect_stdout(io.StringIO()) as f:
                    cmd_create(ns)
                buf = f.getvalue().splitlines()
            except SystemExit as e:
                print(f"  开机失败：{e}")
                continue
            pod = next((ln.split("podId=")[1].strip() for ln in buf if "podId=" in ln), None)
            if not pod:
                print("  没拿到 podId"); continue
            print(f"  podId={pod}，等 SSH…")
            t0 = time.time()
            tgt = None
            while time.time() - t0 < a.ssh_timeout:
                tgt = ssh_target(pod)
                if tgt:
                    break
                time.sleep(15)
            if not tgt:
                print("  等不到 SSH，换一台"); rest(f"/pods/{pod}", "DELETE"); continue
            host, port = tgt
            if host in bad_hosts:
                print(f"  又被放回已知的坏宿主机 {host}，关掉重开")
                rest(f"/pods/{pod}", "DELETE")
                continue
            print(f"  SSH {host}:{port}，验卡…")
            time.sleep(20)
            for _ in range(6):
                ok, msg = cuda_ok(host, port, key)
                if ok:
                    pw = power_limit_w(host, port, key)
                    if pw and pw < a.min_watts:
                        print(f"  ✗ 功耗上限只有 {pw:.0f}W（门槛 {a.min_watts}W）—— 卡被限功耗，换一台")
                        bad_hosts.add(host); mark_bad_host(host, f"功耗被锁 {pw:.0f}W")
                        rest(f"/pods/{pod}", "DELETE")
                        break
                    tf = bench_tflops(host, port, key)
                    if tf < a.min_tflops:
                        print(f"  ✗ 算力只有 {tf:.1f} TFLOPS（门槛 {a.min_tflops}）—— 限频或被共享，换一台")
                        bad_hosts.add(host); mark_bad_host(host, f"算力仅 {tf:.1f}T")
                        rest(f"/pods/{pod}", "DELETE")
                        break
                    print(f"  ✓ CUDA 正常 · 功耗上限 {pw:.0f}W · 实测 {tf:.1f} TFLOPS —— 用这台：{pod}")
                    print(f"    SSH  ssh -i {key} -p {port} root@{host}")
                    print(f"    HTTP https://{pod}-8188.proxy.runpod.net")
                    print(f"    收工 python deploy/runpod_ctl.py terminate {pod}")
                    return 0
                if "Connection" in msg or "closed" in msg:
                    time.sleep(20); continue
                break
            print(f"  ✗ 这台卡用不了（{msg}），关掉换下一台")
            bad_hosts.add(host); mark_bad_host(host, "CUDA 初始化失败")
            rest(f"/pods/{pod}", "DELETE")
    print("试完都没拿到可用的卡")
    return 1


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

    p = sub.add_parser("provision", help="开机 + 验卡 + 坏了自动换")
    p.add_argument("--gpus", nargs="*", default=["RTX 4090", "RTX A6000", "RTX 3090", "A40", "L40S"])
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--image", default=COMFY_IMAGE)
    p.add_argument("--name", default="longfilm-comfy")
    p.add_argument("--disk", type=int, default=60)
    p.add_argument("--volume", type=int, default=80)
    p.add_argument("--mount", default="/workspace")
    p.add_argument("--secure", action="store_true")
    p.add_argument("--key", default="~/.ssh/id_ed25519_runpod")
    p.add_argument("--ssh-timeout", type=float, default=420)
    # 门槛取 25：3090 实测 32.5，4090 正常应在 100 以上；被限频的那台
    # 功耗只有 149W、单镜从 6 分钟劣化到 25 分钟以上，实测远低于 10。
    # 定 40 会把正常的 3090 也误杀，定 25 能放行 3090 又能挡住残卡。
    p.add_argument("--min-tflops", type=float, default=25.0,
                   help="fp16 矩阵乘吞吐门槛；低于它说明卡被限频或共享")
    p.add_argument("--min-watts", type=float, default=250.0,
                   help="功耗上限门槛；3090 原厂 350W、4090 450W，150W 的是被锁了的")
    p.set_defaults(fn=cmd_provision)

    p = sub.add_parser("bench", help="实测某台 pod 的算力")
    p.add_argument("pod")
    p.add_argument("--key", default="~/.ssh/id_ed25519_runpod")
    p.add_argument("--min-tflops", type=float, default=25.0)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("terminate")
    p.add_argument("pods", nargs="*", help="留空 = 关掉全部")
    p.set_defaults(fn=cmd_terminate)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
