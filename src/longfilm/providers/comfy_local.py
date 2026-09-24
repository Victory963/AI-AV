"""ComfyUI HTTP provider —— 自建 GPU 侧的主力引擎。

为什么开源侧是「主力」而不是「备胎」：官方 API 画质高但不可微调、有审核、
无法保证跨集的角色一致性；长片产线真正的护城河是自训 LoRA + 首尾帧锁戏，
这两件事只有自建 ComfyUI 能做。官方 API 在本产线里的定位反而是「英雄镜特供」。

ComfyUI 的 API 形态（2026-09-19 查证，来源 docs.comfy.org/development/comfyui-server/comms_routes）：
  POST /prompt            提交 API 格式 workflow，body = {prompt, client_id, extra_data}，
                          返回 {prompt_id, number} 或 {error, node_errors}
  GET  /history/{id}      取执行结果，outputs 下按节点 id 归类，媒体项含 filename/subfolder/type
  GET  /view?filename=&subfolder=&type=   下载产物
  POST /upload/image      multipart 上传，字段 image/type/subfolder/overwrite，返回落盘文件名
  GET  /queue             queue_running / queue_pending，用来区分排队中与执行中
  POST /interrupt         打断当前执行
  GET  /system_stats      探活
  WS   /ws?clientId=xxx   进度推送，消息类型 status/progress/executing/executed
client_id 是「把 ws 事件路由回本客户端」的凭证，与 /prompt 的 client_id 必须一致。

设计上最关键的一条：**参数注入走 JSON 路径，不做字符串替换**。
workflow 模板里 prompt 文本、种子、分辨率都可能与模型文件名、节点标题重名，
字符串替换迟早会把 "1280" 换进某个不相干的字段里，而且出错时毫无线索。
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import _http
from .base import (
    Capabilities,
    FailureKind,
    GenRequest,
    GenResult,
    JobStatus,
    ProviderError,
    VideoProvider,
)

log = logging.getLogger(__name__)

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "deploy" / "comfy_workflows"

# ComfyUI 的 history.outputs 里，不同保存节点用不同的键放媒体项。
# VHS_VideoCombine 用 "gifs"（历史包袱，实际可能是 mp4），原生节点用 "images"/"videos"。
_MEDIA_KEYS = ("gifs", "videos", "images", "audio")
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".gif", ".webp"}


# ---------------------------------------------------------------- JSON 路径注入


def _split(path: str) -> list[str]:
    if not path:
        raise ValueError("节点路径不能为空")
    return path.split(".")


def _descend(node: Any, seg: str, path: str, upto: int) -> Any:
    where = ".".join(_split(path)[:upto])
    if isinstance(node, Mapping):
        if seg not in node:
            raise KeyError(f"workflow 路径 {path!r} 在 {where!r} 处找不到键 {seg!r}")
        return node[seg]
    if isinstance(node, list):
        if not seg.lstrip("-").isdigit():
            raise KeyError(f"workflow 路径 {path!r} 在 {where!r} 处是数组，需要数字下标而非 {seg!r}")
        idx = int(seg)
        if not -len(node) <= idx < len(node):
            raise KeyError(f"workflow 路径 {path!r} 下标 {idx} 越界（长度 {len(node)}）")
        return node[idx]
    raise KeyError(f"workflow 路径 {path!r} 在 {where!r} 处已到叶子，无法继续下钻")


def inject(template: dict[str, Any], mapping: Mapping[str, Any]) -> dict[str, Any]:
    """按节点路径把参数注入 workflow，返回新字典（不改原模板）。

    路径形如 ``"6.inputs.text"``；数组用数字下标，如 ``"11.inputs.start_image.0"``。
    路径不存在直接抛 KeyError —— 静默跳过会让「参数没生效」变成一个要肉眼比对
    两份几百行 JSON 才能发现的 bug。
    """
    out = copy.deepcopy(template)
    for path, value in mapping.items():
        segs = _split(path)
        node: Any = out
        for i, seg in enumerate(segs[:-1]):
            node = _descend(node, seg, path, i + 1)
        last = segs[-1]
        if isinstance(node, Mapping):
            if last not in node:
                raise KeyError(f"workflow 路径 {path!r} 的末级键 {last!r} 不存在")
            node[last] = value
        elif isinstance(node, list):
            if not last.lstrip("-").isdigit():
                raise KeyError(f"workflow 路径 {path!r} 末级是数组，需要数字下标")
            node[int(last)] = value
        else:
            raise KeyError(f"workflow 路径 {path!r} 的父节点不可写入")
    return out


# ---------------------------------------------------------------- 模板


@dataclass
class WorkflowTemplate:
    """ComfyUI 导出的 API 格式 workflow + 一张语义名到节点路径的绑定表。

    绑定表存在模板 JSON 自己的 ``_longfilm`` 键里而不是单独的配置文件：
    换 workflow 时只动一个文件，不会出现「模板换了、绑定表没换」的错配。
    提交前 ``graph()`` 会把下划线开头的元数据剥掉，ComfyUI 不认识它们。
    """

    name: str
    mode: str                                  # "i2v" | "t2v"
    raw: dict[str, Any]
    bind: dict[str, str] = field(default_factory=dict)
    lora_node: str | None = None
    end_image_node: str | None = None
    optional_bind: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, name: str = "") -> WorkflowTemplate:
        meta = data.get("_longfilm", {})
        return cls(
            name=meta.get("name") or name or "unnamed",
            mode=meta.get("mode", "t2v"),
            raw=data,
            bind=dict(meta.get("bind", {})),
            lora_node=meta.get("lora_node"),
            end_image_node=meta.get("end_image_node"),
            optional_bind=tuple(meta.get("optional_bind", ())),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> WorkflowTemplate:
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), name=p.stem)

    def graph(self) -> dict[str, Any]:
        """可直接 POST 给 /prompt 的纯节点图。"""
        return {k: v for k, v in self.raw.items() if not k.startswith("_")}

    def path_of(self, key: str) -> str | None:
        return self.bind.get(key)

    def render(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """按语义名填参。未在 bind 表里声明的语义名会被忽略，
        因为 t2v 模板本来就没有 start_image 这种位置。"""
        mapping = {path: values[key] for key, path in self.bind.items() if path and key in values}
        return inject(self.graph(), mapping)


def load_templates(directory: str | Path = TEMPLATE_DIR) -> dict[str, WorkflowTemplate]:
    out: dict[str, WorkflowTemplate] = {}
    for f in sorted(Path(directory).glob("*.json")):
        t = WorkflowTemplate.from_file(f)
        out[t.name] = t
    return out


# ---------------------------------------------------------------- LoRA 链


def _normalize_loras(spec: Mapping[str, Any] | Sequence[Any] | None) -> list[dict[str, Any]]:
    """把 GenRequest.lora 的几种写法收敛成 [{name, strength}, ...]。

    router 编译出来的可能是单个 LoRA、也可能是「角色 LoRA + 风格 LoRA」的串联，
    上游不该为了迁就 provider 而统一格式。
    """
    if not spec:
        return []
    items: Sequence[Any]
    if isinstance(spec, Mapping):
        items = spec.get("loras") or [spec]
    else:
        items = spec
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, Mapping):
            raise ProviderError(FailureKind.BAD_REQUEST, f"LoRA 条目必须是 dict，收到 {type(it).__name__}")
        name = it.get("name") or it.get("lora_name") or it.get("path")
        if not name:
            raise ProviderError(FailureKind.BAD_REQUEST, f"LoRA 条目缺少 name/path：{dict(it)}")
        # ComfyUI 的 LoraLoader 只认 models/loras 下的相对文件名，绝对路径必然加载失败
        out.append({
            "name": Path(str(name)).name,
            "strength": float(it.get("strength", it.get("strength_model", 0.85))),
        })
    return out


def _consumers(graph: Mapping[str, Any], node_id: str) -> list[tuple[str, str, int]]:
    """找出所有引用 [node_id, slot] 的下游输入，返回 (下游节点 id, 输入名, slot)。"""
    found: list[tuple[str, str, int]] = []
    for nid, node in graph.items():
        for key, val in (node.get("inputs") or {}).items():
            if isinstance(val, list) and len(val) == 2 and val[0] == node_id:
                found.append((nid, key, int(val[1])))
    return found


def chain_loras(
    graph: dict[str, Any], lora_node: str, loras: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """把 N 个 LoRA 串成链；N=0 时把模板里的 LoRA 节点整个旁路掉。

    旁路不是可选优化：模板里的 lora_name 是占位符，不挂 LoRA 却把节点留在图里，
    ComfyUI 会因为找不到 PLACEHOLDER_*.safetensors 直接 node_errors。
    """
    g = copy.deepcopy(graph)
    if lora_node not in g:
        raise KeyError(f"模板里没有 LoRA 节点 {lora_node!r}")
    head = g[lora_node]
    inputs = head.get("inputs") or {}
    downstream = _consumers(g, lora_node)

    if not loras:
        # slot 0 = MODEL，slot 1 = CLIP（LoraLoader 才有）。各自接回节点自己的上游。
        upstream = {0: inputs.get("model"), 1: inputs.get("clip")}
        for nid, key, slot in downstream:
            src = upstream.get(slot)
            if src is None:
                raise KeyError(f"旁路 LoRA 节点 {lora_node} 失败：slot {slot} 没有可接回的上游")
            g[nid]["inputs"][key] = copy.deepcopy(src)
        del g[lora_node]
        return g

    has_clip = "clip" in inputs
    strength_keys = [k for k in ("strength_model", "strength_clip", "strength") if k in inputs]

    def _apply(node: dict[str, Any], lora: Mapping[str, Any]) -> None:
        node["inputs"]["lora_name"] = lora["name"]
        for k in strength_keys:
            node["inputs"][k] = lora["strength"]

    _apply(head, loras[0])
    tail = lora_node
    for i, lora in enumerate(loras[1:], start=1):
        nid = f"{lora_node}_lora{i}"
        node = copy.deepcopy(head)
        node["inputs"]["model"] = [tail, 0]
        if has_clip:
            node["inputs"]["clip"] = [tail, 1]
        node.setdefault("_meta", {})["title"] = f"LoRA #{i + 1} {lora['name']}"
        _apply(node, lora)
        g[nid] = node
        tail = nid

    if tail != lora_node:
        for nid, key, slot in downstream:
            g[nid]["inputs"][key] = [tail, slot]
    return g


def drop_node(graph: dict[str, Any], node_id: str) -> dict[str, Any]:
    """摘掉一个可选节点，并把引用它的下游输入一并删掉（如未提供尾帧时的 LoadImage）。"""
    g = copy.deepcopy(graph)
    if node_id not in g:
        return g
    for nid, key, _slot in _consumers(g, node_id):
        g[nid]["inputs"].pop(key, None)
    del g[node_id]
    return g


def wan_frame_count(duration_s: float, fps: int) -> int:
    """Wan 系列的 latent 时间维是 4 的倍数，帧数必须是 4k+1，否则解码出来会掉尾帧。"""
    n = max(5, round(duration_s * fps))
    return ((n - 1) // 4) * 4 + 1


# ---------------------------------------------------------------- provider


class ComfyProvider(VideoProvider):
    """自建 ComfyUI。无审核、可挂 LoRA、可锁首尾帧，代价是画质与稳定性靠自己调。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        templates: Mapping[str, WorkflowTemplate] | None = None,
        template_dir: str | Path = TEMPLATE_DIR,
        i2v_template: str = "wan_i2v_lora",
        t2v_template: str = "wan_t2v_placeholder",
        out_dir: str | Path = "out/comfy",
        name: str = "comfy_local",
        quality_tier: int = 3,
        max_duration_s: float = 10.0,
        cost_per_second_usd: float = 0.004,   # 自建只有电费+折旧，给 router 一个非零值好比价
        timeout_s: float = 60.0,
    ) -> None:
        caps = Capabilities(
            name=name,
            kind="open",
            min_duration_s=1.0,
            max_duration_s=max_duration_s,
            resolutions=((832, 480), (1280, 720), (1920, 1080)),
            fps_options=(16, 24, 30),
            max_ref_images=2,          # 模板里两个 LoadImage：首帧 + 尾帧
            supported_image_roles=frozenset({"first_frame", "last_frame"}),
            supports_first_frame=True,
            supports_last_frame=True,
            supports_extend=False,     # 续写靠「上一镜尾帧当首帧」，不是引擎原生 extend
            supports_job_continue=False,
            supports_native_audio=False,
            supports_lora=True,
            supports_seed=True,
            supports_negative_prompt=True,
            supports_camera_control=False,
            moderated=False,           # 本机推理，无平台审核；内容边界由 router 的门禁负责
            max_concurrency=1,         # 单卡串行，写 >1 只会让显存 OOM
            cost_per_second_usd=cost_per_second_usd,
            typical_latency_s=180.0,
            quality_tier=quality_tier,
            notes="自建 ComfyUI；LoRA 与首尾帧锁戏的唯一通道",
        )
        super().__init__(caps)
        self.base_url = (base_url or os.environ.get("COMFY_URL") or DEFAULT_COMFY_URL).rstrip("/")
        self.templates = dict(templates) if templates is not None else load_templates(template_dir)
        self.i2v_template = i2v_template
        self.t2v_template = t2v_template
        self.out_dir = Path(out_dir)
        self.timeout_s = timeout_s
        # client_id 一个 provider 实例一个：ws 事件按它路由回来，换了就收不到进度。
        self.client_id = str(uuid.uuid4())
        self._jobs: dict[str, dict[str, Any]] = {}

    # ---- 基础设施

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @property
    def ws_url(self) -> str:
        """进度推送地址。调用方自己连（本产线的队列走轮询，ws 只给 CLI 看进度）。"""
        scheme = "wss" if self.base_url.startswith("https") else "ws"
        host = self.base_url.split("://", 1)[-1]
        return f"{scheme}://{host}/ws?clientId={self.client_id}"

    def health(self) -> bool:
        try:
            # 3 秒对本机够用，对跨洋的反向代理（RunPod 的 *.proxy.runpod.net）不够：
            # 实测机器明明活着却被判死，整条降级链跟着空转。跟随 provider 的超时，
            # 但压在 15 秒内 —— 探活本来就不该等太久。
            code, _ = _http.request_json(
                "GET", self._url("/system_stats"), timeout=min(15.0, max(3.0, self.timeout_s / 4)))
        except ProviderError:
            return False
        return code == 200

    def template_for(self, req: GenRequest) -> WorkflowTemplate:
        want = self.i2v_template if self._first_frame(req) else self.t2v_template
        if want not in self.templates:
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                f"没有加载到 workflow 模板 {want!r}；已加载 {sorted(self.templates)}",
            )
        return self.templates[want]

    # ---- 参考图

    @staticmethod
    def _first_frame(req: GenRequest) -> str | None:
        if req.first_frame_uri:
            return req.first_frame_uri
        hits = [i for i in req.refs.images if i.role == "first_frame"]
        return hits[0].uri if hits else None

    @staticmethod
    def _last_frame(req: GenRequest) -> str | None:
        if req.last_frame_uri:
            return req.last_frame_uri
        hits = [i for i in req.refs.images if i.role == "last_frame"]
        return hits[0].uri if hits else None

    def upload_image(self, uri: str, *, subfolder: str = "longfilm", overwrite: bool = True) -> str:
        """把参考图送进 ComfyUI 的 input 目录，返回 LoadImage 认的文件名。

        LoadImage 的 widget 值是 ``subfolder/name``（无子目录时就是 name），
        不是路径也不是 URL —— 直接塞本机绝对路径会在节点校验阶段被打回。
        """
        if uri.startswith(("http://", "https://")):
            local = self.out_dir / "_refs" / f"{uuid.uuid4().hex}{Path(uri).suffix or '.png'}"
            _http.download(uri, local, timeout=self.timeout_s)
        else:
            local = Path(uri)
        if not local.is_file():
            raise ProviderError(FailureKind.BAD_REQUEST, f"参考图不存在：{local}")

        code, body = _http.post_multipart(
            self._url("/upload/image"),
            fields={"type": "input", "subfolder": subfolder, "overwrite": str(overwrite).lower()},
            files={"image": (local.name, local.read_bytes())},
            timeout=self.timeout_s,
        )
        if code != 200 or not isinstance(body, Mapping) or "name" not in body:
            raise ProviderError(
                _http.status_to_failure(code), f"上传参考图失败 HTTP {code}: {body}", raw={"body": body}
            )
        sub = body.get("subfolder") or ""
        return f"{sub}/{body['name']}" if sub else str(body["name"])

    # ---- 提交

    def build_graph(self, req: GenRequest) -> dict[str, Any]:
        """把 GenRequest 编译成可提交的节点图。单独暴露是为了能离线做干跑校验。"""
        tpl = self.template_for(req)
        w, h = self.caps.nearest_resolution(req.resolution)
        fps = req.fps if req.fps in self.caps.fps_options else self.caps.fps_options[0]
        values: dict[str, Any] = {
            "positive": req.prompt,
            "negative": req.negative_prompt,
            "seed": req.seed if req.seed is not None else int(uuid.uuid4().int % 2**31),
            "width": w,
            "height": h,
            "length": wan_frame_count(self.caps.clamp_duration(req.duration_s), fps),
            "fps": fps,
            "filename_prefix": f"longfilm/{req.shot_id or 'shot'}",
        }
        if (ff := self._first_frame(req)) and tpl.path_of("start_image"):
            values["start_image"] = self.upload_image(ff)
        lf = self._last_frame(req)
        if lf and tpl.path_of("end_image"):
            values["end_image"] = self.upload_image(lf)

        graph = tpl.render(values)
        if not lf and tpl.end_image_node:
            # 尾帧是可选的：没有就把那条 LoadImage 支路摘掉，
            # 留着占位图会让模型往一张不相干的画面上收敛。
            graph = drop_node(graph, tpl.end_image_node)
        if tpl.lora_node:
            graph = chain_loras(graph, tpl.lora_node, _normalize_loras(req.lora))
        return graph

    def submit(self, req: GenRequest) -> str:
        if req.content_rating == "blocked":
            raise ProviderError(
                FailureKind.BAD_REQUEST, f"{req.shot_id or '该镜'} 内容分级为 blocked，产线拒单"
            )
        graph = self.build_graph(req)
        payload = {
            "prompt": graph,
            "client_id": self.client_id,
            # extra_data 只用来回溯：产物 metadata 里带上镜头号和指纹，
            # 出片后能一键定位是哪张工单生成的
            "extra_data": {
                "longfilm": {
                    "shot_id": req.shot_id,
                    "idempotency_key": req.idempotency_key,
                }
            },
        }
        code, body = _http.request_json(
            "POST", self._url("/prompt"), json_body=payload, timeout=self.timeout_s
        )
        if code != 200 or not isinstance(body, Mapping):
            raise ProviderError(_http.status_to_failure(code), f"ComfyUI /prompt HTTP {code}: {body}")
        if body.get("error") or body.get("node_errors"):
            # node_errors 是排查模板问题的唯一线索，必须原样带出来
            raise ProviderError(
                FailureKind.BAD_REQUEST,
                f"workflow 校验失败：{body.get('error')}；node_errors={body.get('node_errors')}",
                raw=dict(body),
            )
        pid = body.get("prompt_id")
        if not pid:
            raise ProviderError(FailureKind.UNKNOWN, f"/prompt 未返回 prompt_id：{body}")
        self._jobs[str(pid)] = {"submitted_at": time.time(), "shot_id": req.shot_id,
                                "duration_s": req.duration_s}
        log.info("ComfyUI 已提交 %s -> prompt_id=%s", req.shot_id or "?", pid)
        return str(pid)

    # ---- 轮询

    def _queue_state(self, job_id: str) -> JobStatus:
        code, body = _http.request_json("GET", self._url("/queue"), timeout=self.timeout_s)
        if code != 200 or not isinstance(body, Mapping):
            return JobStatus.PENDING
        for key, st in (("queue_running", JobStatus.RUNNING), ("queue_pending", JobStatus.PENDING)):
            for item in body.get(key) or []:
                if isinstance(item, list) and len(item) > 1 and str(item[1]) == job_id:
                    return st
        # 既不在队列也没进 history：多半是服务重启把队列清了
        return JobStatus.FAILED

    @staticmethod
    def _pick_media(outputs: Mapping[str, Any]) -> dict[str, Any] | None:
        """从 history.outputs 里挑出视频产物。优先 mp4/webm，其次任意媒体项。"""
        best: dict[str, Any] | None = None
        for node_out in outputs.values():
            if not isinstance(node_out, Mapping):
                continue
            for key in _MEDIA_KEYS:
                for item in node_out.get(key) or []:
                    if not isinstance(item, Mapping) or "filename" not in item:
                        continue
                    if Path(str(item["filename"])).suffix.lower() in _VIDEO_SUFFIXES:
                        return dict(item)
                    best = best or dict(item)
        return best

    def _view_url(self, item: Mapping[str, Any]) -> str:
        from urllib.parse import urlencode

        q = urlencode({
            "filename": item.get("filename", ""),
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        })
        return self._url(f"/view?{q}")

    def poll(self, job_id: str) -> GenResult:
        meta = self._jobs.get(job_id, {})
        res = GenResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            provider=self.name,
            submitted_at=meta.get("submitted_at", 0.0),
        )
        code, body = _http.request_json(
            "GET", self._url(f"/history/{job_id}"), timeout=self.timeout_s
        )
        if code != 200 or not isinstance(body, Mapping):
            res.status, res.failure = JobStatus.FAILED, _http.status_to_failure(code)
            res.message = f"/history HTTP {code}"
            return res

        entry = body.get(job_id)
        if not isinstance(entry, Mapping):
            res.status = self._queue_state(job_id)
            if res.status is JobStatus.FAILED:
                res.failure = FailureKind.SERVER
                res.message = "任务既不在队列也不在 history，ComfyUI 可能已重启"
            return res

        res.raw = {"status": entry.get("status")}
        status_obj = entry.get("status") or {}
        if isinstance(status_obj, Mapping) and status_obj.get("status_str") == "error":
            res.status, res.failure = JobStatus.FAILED, FailureKind.SERVER
            res.message = _summarize_comfy_error(status_obj)
            res.finished_at = time.time()
            return res

        item = self._pick_media(entry.get("outputs") or {})
        if item is None:
            res.status = JobStatus.RUNNING if not status_obj.get("completed") else JobStatus.FAILED
            if res.status is JobStatus.FAILED:
                res.failure = FailureKind.SERVER
                res.message = "执行完成但 outputs 里没有任何媒体产物，检查保存节点"
            return res

        dest = self.out_dir / f"{meta.get('shot_id') or job_id}_{Path(str(item['filename'])).name}"
        _http.download(self._view_url(item), dest, timeout=max(self.timeout_s, 600.0))
        res.status = JobStatus.SUCCEEDED
        res.video_uri = str(dest)
        res.duration_s = float(meta.get("duration_s", 0.0))
        res.cost_usd = round(self.caps.cost_per_second_usd * res.duration_s, 4)
        res.finished_at = time.time()
        return res

    def cancel(self, job_id: str) -> None:
        _http.request_json(
            "POST", self._url("/queue"), json_body={"delete": [job_id]}, timeout=self.timeout_s
        )
        if self._queue_state(job_id) is JobStatus.RUNNING:
            _http.request_json("POST", self._url("/interrupt"), json_body={}, timeout=self.timeout_s)


def _summarize_comfy_error(status_obj: Mapping[str, Any]) -> str:
    """把 history.status.messages 里的 execution_error 压成一行人能读的报错。"""
    for msg in status_obj.get("messages") or []:
        if isinstance(msg, list) and len(msg) == 2 and msg[0] == "execution_error":
            d = msg[1] if isinstance(msg[1], Mapping) else {}
            return (
                f"节点 {d.get('node_id')} ({d.get('node_type')}) 执行失败："
                f"{d.get('exception_type')}: {d.get('exception_message')}"
            )
    return f"执行失败：{status_obj.get('status_str')}"


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    logging.basicConfig(level=logging.WARNING)

    # 1) 模板能加载，绑定表齐全
    tpls = load_templates()
    assert {"wan_i2v_lora", "wan_t2v_placeholder"} <= set(tpls), sorted(tpls)
    i2v = tpls["wan_i2v_lora"]
    assert i2v.mode == "i2v" and i2v.lora_node == "4"
    assert "_longfilm" not in i2v.graph(), "提交前必须剥掉元数据"
    for key in ("positive", "negative", "seed", "width", "height", "length"):
        assert i2v.path_of(key), f"i2v 模板缺少绑定 {key}"

    # 2) inject 走 JSON 路径，且不改原模板
    src = {"6": {"inputs": {"text": "old", "clip": ["2", 0]}}}
    got = inject(src, {"6.inputs.text": "新提示词", "6.inputs.clip.1": 3})
    assert got["6"]["inputs"]["text"] == "新提示词"
    assert got["6"]["inputs"]["clip"] == ["2", 3]
    assert src["6"]["inputs"]["text"] == "old", "inject 不能就地改模板"
    for bad in ("6.inputs.nope", "99.inputs.text", "6.inputs.text.0"):
        try:
            inject(src, {bad: 1})
        except KeyError:
            pass
        else:
            raise AssertionError(f"路径 {bad} 应当报错")

    # 3) LoRA 串联：3 个 LoRA 要生成 3 个节点，且下游改接到链尾
    g = i2v.graph()
    loras = _normalize_loras({"loras": [
        {"name": "/abs/path/hero.safetensors", "strength": 0.9},
        {"name": "style.safetensors", "strength": 0.6},
        {"path": "motion.safetensors"},
    ]})
    assert [l["name"] for l in loras] == ["hero.safetensors", "style.safetensors", "motion.safetensors"]
    chained = chain_loras(g, "4", loras)
    lora_nodes = [n for n, v in chained.items() if v["class_type"] == "LoraLoaderModelOnly"]
    assert len(lora_nodes) == 3, lora_nodes
    assert chained["4"]["inputs"]["lora_name"] == "hero.safetensors"
    assert chained["4"]["inputs"]["strength_model"] == 0.9
    assert chained["4_lora1"]["inputs"]["model"] == ["4", 0]
    assert chained["4_lora2"]["inputs"]["model"] == ["4_lora1", 0]
    assert chained["12"]["inputs"]["model"] == ["4_lora2", 0], "KSampler 必须接链尾"
    assert chained["4_lora2"]["inputs"]["lora_name"] == "motion.safetensors"
    assert chained["4_lora2"]["inputs"]["strength_model"] == 0.85, "未给强度时用默认值"

    # 4) 无 LoRA 时整条支路旁路掉，KSampler 直接接 UNETLoader
    bypassed = chain_loras(g, "4", [])
    assert "4" not in bypassed
    assert bypassed["12"]["inputs"]["model"] == ["1", 0]

    # 5) 摘掉尾帧节点后，下游不再引用它
    no_end = drop_node(g, "8")
    assert "8" not in no_end and "end_image" not in no_end["11"]["inputs"]
    assert no_end["11"]["inputs"]["start_image"] == ["7", 0], "摘尾帧不能误伤首帧"

    # 6) 帧数对齐 4k+1
    assert wan_frame_count(3.0, 24) == 69, wan_frame_count(3.0, 24)  # 72 帧向下对齐到 4k+1
    assert wan_frame_count(0.01, 24) == 5
    for d in (1.0, 2.5, 5.0, 8.0, 10.0):
        assert (wan_frame_count(d, 16) - 1) % 4 == 0

    # 7) caps 声明（router 只读这个做决策）
    p = ComfyProvider(base_url="http://127.0.0.1:9/comfy-not-here")
    assert p.caps.kind == "open"
    assert p.caps.supports_lora and not p.caps.moderated
    assert p.caps.supports_first_frame and p.caps.supports_last_frame
    assert p.ws_url.startswith("ws://127.0.0.1:9/comfy-not-here/ws?clientId=")

    # 8) 服务不在时 health() 必须是 False 而不是抛异常
    assert p.health() is False

    # 9) 离线编图：不触网的部分（t2v 无参考图）必须能完整编出节点图
    req = GenRequest(
        prompt="a fictional adult detective walking through neon rain",
        negative_prompt="blurry, watermark",
        duration_s=5.0, resolution=(1920, 1080), fps=24, seed=42,
        shot_id="s01_sh03", lora={"name": "hero.safetensors", "strength": 0.8},
    )
    graph = p.build_graph(req)
    assert graph["4"]["inputs"]["text"] == req.prompt
    assert graph["5"]["inputs"]["text"] == req.negative_prompt
    assert graph["7"]["inputs"]["seed"] == 42
    assert (graph["6"]["inputs"]["width"], graph["6"]["inputs"]["height"]) == (1920, 1080)
    assert graph["6"]["inputs"]["length"] == wan_frame_count(5.0, 24)
    assert graph["9"]["inputs"]["filename_prefix"] == "longfilm/s01_sh03"
    assert graph["3"]["inputs"]["lora_name"] == "hero.safetensors"
    assert json.dumps(graph)  # 必须是可序列化的纯 JSON

    # 10) 提交到一个不存在的服务：抛可重试的 ProviderError，不是裸异常
    try:
        p.submit(req)
    except ProviderError as e:
        assert e.kind in {FailureKind.SERVER, FailureKind.TIMEOUT}, e.kind
    else:
        raise AssertionError("ComfyUI 不可达时 submit 应抛 ProviderError")

    # 11) 内容门禁：blocked 直接拒单，不发请求
    try:
        p.submit(GenRequest(prompt="x", duration_s=4.0, content_rating="blocked", shot_id="s99"))
    except ProviderError as e:
        assert e.kind is FailureKind.BAD_REQUEST and "拒单" in str(e)
    else:
        raise AssertionError("blocked 分级应当拒单")

    # 12) i2v 模板在缺首帧时不会被选中
    assert p.template_for(req).name == "wan_t2v_placeholder"
    req2 = GenRequest(prompt="x", duration_s=4.0, first_frame_uri="/tmp/x.png")
    assert p.template_for(req2).name == "wan_i2v_lora"

    print("comfy_local selftest OK：模板 2 个，LoRA 链/旁路/尾帧摘除/帧数对齐/离线编图全部通过")


if __name__ == "__main__":
    _selftest()
