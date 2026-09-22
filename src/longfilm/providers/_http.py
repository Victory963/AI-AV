"""provider 共用的极简 HTTP 客户端。

为什么不直接 `import requests`：本产线的硬约束是「能不加依赖就不加」，
而这三个 provider 要的全部能力不过是 JSON POST/GET、multipart 上传和文件下载，
标准库 urllib 完全够用。装了 requests 就用它（连接池更省 socket），
没装就走 urllib —— 两条路径对外暴露同一个返回约定。

返回约定：**只要服务器给了响应就返回 (status, body)，不抛异常**。
HTTP 4xx/5xx 的语义因厂商而异（同样是 400，有的是参数错、有的是审核拒），
只有 provider 自己知道怎么映射成 FailureKind，helper 不该替它决定。
真正连不上（DNS/拒连/超时）才抛 ProviderError，因为那与业务语义无关。
"""

from __future__ import annotations

import json as _json
import logging
import mimetypes
import socket
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .base import FailureKind, ProviderError

log = logging.getLogger(__name__)

# 单个响应体的内存上限。视频 provider 的 JSON 响应都是几 KB 量级，
# 超过这个数基本意味着把视频二进制当 JSON 读了 —— 宁可报错也别把 3GB 内存吃穿。
MAX_JSON_BYTES = 8 * 1024 * 1024


def _requests():
    """惰性探测 requests。缺失返回 None，不抛 —— 走 urllib 后备路径。"""
    try:
        import requests  # noqa: PLC0415  故意延迟导入：可选依赖
    except ImportError:
        return None
    return requests


def _net_error(url: str, exc: BaseException) -> ProviderError:
    """把底层网络异常翻译成可重试语义。

    连不上和超时都是 router 应当退避重试的情况，不是工单有问题。
    """
    kind = FailureKind.TIMEOUT if isinstance(exc, (socket.timeout, TimeoutError)) else FailureKind.SERVER
    host = urllib.parse.urlsplit(url).netloc or url
    return ProviderError(kind, f"连接 {host} 失败：{type(exc).__name__}: {exc}")


def _decode(raw: bytes) -> Any:
    """尽力解析 JSON；不是 JSON 就原样给回文本，便于把错误页写进日志。"""
    if not raw:
        return {}
    try:
        return _json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError):
        return {"_raw_text": raw[:2048].decode("utf-8", "replace")}


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    timeout: float = 60.0,
) -> tuple[int, Any]:
    """发一个 JSON 请求，返回 (http_status, 解析后的 body)。"""
    hdrs = dict(headers or {})
    data: bytes | None = None
    if json_body is not None:
        data = _json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    hdrs.setdefault("Accept", "application/json")

    rq = _requests()
    if rq is not None:
        try:
            resp = rq.request(method.upper(), url, headers=hdrs, data=data, timeout=timeout)
        except Exception as exc:  # requests 的异常树与 urllib 不通，统一翻译
            raise _net_error(url, exc) from exc
        return resp.status_code, _decode(resp.content[:MAX_JSON_BYTES])

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _decode(resp.read(MAX_JSON_BYTES))
    except urllib.error.HTTPError as exc:
        # HTTPError 也是一个可读的响应体 —— 厂商的错误码就在里面，不能丢。
        return exc.code, _decode(exc.read(MAX_JSON_BYTES))
    except (urllib.error.URLError, OSError) as exc:
        raise _net_error(url, exc) from exc


def post_multipart(
    url: str,
    *,
    fields: dict[str, str] | None = None,
    files: dict[str, tuple[str, bytes]] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> tuple[int, Any]:
    """multipart/form-data 上传。files 的值是 (filename, content)。"""
    boundary = f"----longfilm{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for k, v in (fields or {}).items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        )
    for k, (fname, content) in (files or {}).items():
        ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n".encode()
        )
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)

    hdrs = dict(headers or {})
    hdrs["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, _decode(resp.read(MAX_JSON_BYTES))
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read(MAX_JSON_BYTES))
    except (urllib.error.URLError, OSError) as exc:
        raise _net_error(url, exc) from exc


def download(url: str, dest: str | Path, *, headers: dict[str, str] | None = None,
             timeout: float = 600.0) -> Path:
    """流式下载到本地文件。视频动辄上百 MB，绝不整块读进内存。"""
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    req = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as fh:
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise ProviderError(
            FailureKind.SERVER if exc.code >= 500 else FailureKind.BAD_REQUEST,
            f"下载 {url} 失败：HTTP {exc.code}",
        ) from exc
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise _net_error(url, exc) from exc
    # 先落 .part 再改名：轮询器看到目标文件存在就认为这一镜拿到了，
    # 半截文件被当成成品会让 QC 环节查半天。
    tmp.replace(p)
    return p


def status_to_failure(code: int) -> FailureKind:
    """HTTP 状态码到 FailureKind 的通用映射。厂商私有错误码由各自 provider 覆盖。"""
    if code == 401 or code == 403:
        return FailureKind.AUTH
    if code == 429:
        return FailureKind.RATE_LIMIT
    if code == 408 or code == 504:
        return FailureKind.TIMEOUT
    if code >= 500:
        return FailureKind.SERVER
    if code >= 400:
        return FailureKind.BAD_REQUEST
    return FailureKind.NONE


def redact(text: str, *secrets: str | None) -> str:
    """日志脱敏。密钥进日志等于密钥泄露，所有要打日志的字符串先过这里。"""
    out = text
    for s in secrets:
        if s and len(s) >= 8:
            out = out.replace(s, f"{s[:4]}***{s[-2:]}")
    return out


def _selftest() -> None:
    logging.basicConfig(level=logging.INFO)

    assert status_to_failure(401) is FailureKind.AUTH
    assert status_to_failure(429) is FailureKind.RATE_LIMIT
    assert status_to_failure(503) is FailureKind.SERVER
    assert status_to_failure(400) is FailureKind.BAD_REQUEST
    assert status_to_failure(200) is FailureKind.NONE

    assert _decode(b'{"a":1}') == {"a": 1}
    assert _decode(b"") == {}
    assert "_raw_text" in _decode(b"<html>502</html>")

    key = "sk-abcdef0123456789"
    masked = redact(f"Authorization: Bearer {key}", key)
    assert key not in masked and "sk-a***89" in masked, masked

    # 连一个必然不存在的本地端口：应当抛 ProviderError 而不是裸 OSError
    try:
        request_json("GET", "http://127.0.0.1:9/nope", timeout=1.0)
    except ProviderError as e:
        assert e.kind in {FailureKind.SERVER, FailureKind.TIMEOUT}, e.kind
    else:
        raise AssertionError("连接被拒时应抛 ProviderError")

    print("_http selftest OK")


if __name__ == "__main__":
    _selftest()
