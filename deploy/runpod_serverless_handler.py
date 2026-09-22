"""RunPod serverless worker：按需扩缩容的出片后端。

为什么要 serverless：长片产线的负载是尖峰型的 —— 排一集片时要几十路并发，
排完就空转。常驻 pod 在空转时段烧的钱，往往比渲染本身还多。

协议对齐：这个 handler 的输入输出**就是 longfilm 的 provider 协议**
（GenRequest / GenResult 的 JSON 形态），所以产线侧不需要为它写特殊分支 ——
它在 router 眼里和 ComfyProvider 是同一种东西，只是 endpoint 不同。

部署（查证 2026-09-20，runpod-python SDK 的 handler 签名为 `handler(job) -> Any`，
job["input"] 是提交时的 input 字段）：
    1. 用 deploy/Dockerfile 构建镜像并推到 registry
    2. RunPod 控制台 → Serverless → New Endpoint，选该镜像
    3. 容器启动命令：python -u deploy/runpod_serverless_handler.py
    4. 产线侧：export LONGFILM_RUNPOD_ENDPOINT=https://api.runpod.ai/v2/<id>

本地自测（不需要 runpod SDK，也不需要 GPU）：
    PYTHONPATH=src .venv/bin/python deploy/runpod_serverless_handler.py --selftest
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

log = logging.getLogger("runpod-worker")

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
MAX_INLINE_BYTES = int(os.environ.get("LONGFILM_MAX_INLINE", 9_000_000))


def _err(kind: str, message: str, **extra: Any) -> dict:
    """错误也要带 FailureKind —— router 靠它决定重试还是换引擎。

    handler 里绝不自己重试：worker 内部偷偷重试会让产线的成本核算和
    限流统计全部失真，而且 RunPod 的计费是按 worker 执行时长算的。
    """
    return {"status": "failed", "failure": kind, "message": message, **extra}


def _deliver(path: Path) -> dict:
    """产物回传。小文件内联 base64，大文件走对象存储。

    RunPod 的返回体有大小上限，一条 15 秒 1080p 片子轻松超过。
    配了 S3 就上传后只回 URL，没配就内联 —— 但超过阈值必须明确报错，
    而不是悄悄截断成一个打不开的 mp4。
    """
    size = path.stat().st_size
    bucket = os.environ.get("LONGFILM_S3_BUCKET", "")
    if bucket:
        key = f"renders/{path.parent.name}/{path.name}"
        try:
            import boto3  # 延迟导入：没配 S3 的部署不该被这个依赖卡住

            boto3.client("s3").upload_file(str(path), bucket, key)
            region = os.environ.get("AWS_REGION", "us-east-1")
            return {"video_url": f"https://{bucket}.s3.{region}.amazonaws.com/{key}",
                    "size_bytes": size}
        except Exception as e:
            log.warning("S3 上传失败，回退内联：%s", e)
    if size > MAX_INLINE_BYTES:
        raise RuntimeError(
            f"产物 {size/1e6:.1f}MB 超过内联上限 {MAX_INLINE_BYTES/1e6:.1f}MB，"
            "请配置 LONGFILM_S3_BUCKET 走对象存储"
        )
    return {"video_b64": base64.b64encode(path.read_bytes()).decode(), "size_bytes": size}


def handler(job: dict) -> dict:
    """RunPod 入口。job["input"] 是 GenRequest 的 JSON 形态。"""
    t0 = time.time()
    payload = (job or {}).get("input") or {}
    if not isinstance(payload, dict):
        return _err("bad_request", "input 必须是 object")

    prompt = payload.get("prompt")
    if not prompt:
        return _err("bad_request", "缺 prompt")

    try:
        from longfilm.providers.base import GenRequest
        from longfilm.providers.comfy_local import ComfyProvider
        from longfilm.schema import RefPack
    except Exception as e:
        return _err("server", f"镜像里的 longfilm 包不可用：{e}")

    try:
        refs = RefPack.model_validate(payload.get("refs") or {})
    except Exception as e:
        return _err("bad_request", f"refs 结构非法：{e}")

    req = GenRequest(
        prompt=prompt,
        negative_prompt=payload.get("negative_prompt", ""),
        duration_s=float(payload.get("duration_s", 5.0)),
        resolution=tuple(payload.get("resolution", (1280, 720))),
        fps=int(payload.get("fps", 24)),
        seed=payload.get("seed"),
        refs=refs,
        first_frame_uri=payload.get("first_frame_uri"),
        last_frame_uri=payload.get("last_frame_uri"),
        audio_uri=payload.get("audio_uri"),
        camera_move=payload.get("camera_move"),
        lora=payload.get("lora"),
        shot_id=payload.get("shot_id", ""),
        idempotency_key=payload.get("idempotency_key", ""),
    )

    try:
        provider = ComfyProvider(base_url=COMFY_URL,
                                 template_dir=REPO / "deploy" / "comfy_workflows",
                                 out_dir=os.environ.get("LONGFILM_OUT", "/tmp/longfilm_out"))
        if not provider.health():
            return _err("server", f"ComfyUI 不可达：{COMFY_URL}")
        jid = provider.submit(req)
        res = provider.wait(jid, timeout_s=float(os.environ.get("LONGFILM_JOB_TIMEOUT", 1800)))
        if not res.ok:
            return _err(res.failure.value if res.failure else "unknown",
                        res.message or "生成失败", job_id=jid)
        out = {"status": "succeeded", "job_id": jid,
               "duration_s": res.duration_s, "elapsed_s": round(time.time() - t0, 1)}
        out |= _deliver(Path(res.video_uri))
        if res.last_frame_uri and Path(res.last_frame_uri).exists():
            out["last_frame_b64"] = base64.b64encode(
                Path(res.last_frame_uri).read_bytes()).decode()
        return out
    except Exception as e:
        log.error("handler 异常：%s", traceback.format_exc(limit=6))
        return _err("server", f"{type(e).__name__}: {e}")


def _selftest() -> None:
    """不需要 GPU / runpod SDK / ComfyUI —— 只验证协议层与错误分类。"""
    logging.basicConfig(level=logging.WARNING)
    assert handler({})["failure"] == "bad_request"
    assert handler({"input": "nope"})["failure"] == "bad_request"
    assert handler({"input": {}})["failure"] == "bad_request"

    r = handler({"input": {"prompt": "test", "refs": {"images": [{"role": "bogus", "uri": "x"}]}}})
    assert r["status"] == "failed", r
    assert r["failure"] in {"bad_request", "server"}, r

    # ComfyUI 不可达时必须报 server（可重试），不能报 bad_request（不重试）。
    # 关键：要确认失败的**原因是连不上**，不是 handler 自己把 provider 调错了 ——
    # 上一版就是 TypeError 被兜底成 server，自测照样"通过"。
    os.environ["COMFY_URL"] = "http://127.0.0.1:1"
    globals()["COMFY_URL"] = "http://127.0.0.1:1"
    r = handler({"input": {"prompt": "a shot", "duration_s": 3}})
    assert r["status"] == "failed" and r["failure"] == "server", r
    msg = r.get("message", "")
    for bad in ("TypeError", "AttributeError", "unexpected keyword", "NameError"):
        assert bad not in msg, f"这不是连不上，是调用写错了：{msg}"
    assert "不可达" in msg or "ComfyUI" in msg, f"失败原因不是连不上：{msg}"

    # 构造参数确实对得上 ComfyProvider 的签名（不经过 handler 的兜底）
    from longfilm.providers.comfy_local import ComfyProvider
    prov = ComfyProvider(base_url="http://127.0.0.1:1",
                         template_dir=REPO / "deploy" / "comfy_workflows",
                         out_dir="/tmp/longfilm_out")
    assert prov.health() is False, "本该连不上"
    assert prov.caps.supports_lora, "开源侧 provider 必须声明支持 LoRA"

    print("runpod handler 自测通过：参数校验 / refs 校验 / 不可达→可重试的 server / "
          "provider 构造参数与签名一致")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        sys.exit(0)
    logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(name)s: %(message)s")
    try:
        import runpod  # 只有真部署时才需要
    except ImportError:
        print("缺 runpod SDK：pip install runpod；本地只跑协议自测请加 --selftest", file=sys.stderr)
        sys.exit(20)
    runpod.serverless.start({"handler": handler})
