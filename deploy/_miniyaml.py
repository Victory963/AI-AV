"""YAML 子集读取器 —— 部署资产的公共依赖。

为什么不直接用 PyYAML：
本仓库的运行环境（以及云机器第一次开机、还没装训练栈的时刻）不保证有 PyYAML，
而 ``models.yaml`` / ``prices.yaml`` 恰恰是**在装依赖之前**就要被读的两个文件 ——
bootstrap 要先读 models.yaml 才知道下什么权重。
让部署链在第一步就依赖一个 pip 包，等于给自己埋一个「装不上就全卡住」的单点。

所以这里实现一个**刻意受限**的解析器，并在 PyYAML 可用时优先委托给它：
子集能力足够描述权重清单与价目表，超出子集的写法直接报错而不是猜，
因为静默误读一份权重清单的后果是下错 14GB 的文件。

支持的子集：
  - ``# 注释``（行首或值后，且 ``#`` 前有空白）
  - ``key: value`` 映射，按缩进嵌套（缩进必须用空格）
  - ``- item`` 列表，元素可以是标量或映射
  - 标量：单/双引号字符串、int、float、true/false、null/~、其余按裸字符串
  - 行内流式列表 ``[a, b, c]``（元素只能是标量）
不支持（遇到就抛 ValueError）：锚点/别名、多行块标量、流式映射、制表符缩进。
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["load", "loads", "MiniYamlError"]


class MiniYamlError(ValueError):
    """解析失败。一律带行号 —— 清单文件是人手维护的，不给行号等于不给线索。"""


_UNSUPPORTED = re.compile(r"^\s*(?:[&*]\w|\?\s)|[:\-]\s*[|>]\s*$")


def _scalar(raw: str, lineno: int) -> Any:
    """把一个标量文本转成 Python 值。"""
    s = raw.strip()
    if not s:
        return ""
    if s[0] in "\"'":
        if len(s) < 2 or s[-1] != s[0]:
            raise MiniYamlError(f"第 {lineno} 行：引号未闭合 -> {raw!r}")
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_scalar(p, lineno) for p in inner.split(",")]
    if s.startswith("{"):
        raise MiniYamlError(f"第 {lineno} 行：不支持流式映射，请改成缩进写法")
    if s[0] in "&*":
        raise MiniYamlError(f"第 {lineno} 行：不支持锚点/别名 -> {raw!r}")
    if s in ("|", ">") or s[:2] in ("|-", ">-", "|+", ">+"):
        raise MiniYamlError(f"第 {lineno} 行：不支持多行块标量")
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    # 数字：先整后浮，失败就当字符串。下划线分组（1_000）在权重大小里很常见。
    try:
        return int(s.replace("_", ""))
    except ValueError:
        pass
    try:
        return float(s.replace("_", ""))
    except ValueError:
        return s


def _strip_comment(line: str) -> str:
    """去掉行尾注释。引号内的 ``#`` 不算注释，否则 URL 的锚点会被腰斩。"""
    out: list[str] = []
    quote: str | None = None
    prev_space = True
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            prev_space = False
            continue
        if ch == "#" and prev_space:
            break
        out.append(ch)
        prev_space = ch.isspace()
    return "".join(out).rstrip()


def _indent_of(line: str, lineno: int) -> int:
    n = len(line) - len(line.lstrip(" "))
    if "\t" in line[:n] or line.lstrip(" ").startswith("\t"):
        raise MiniYamlError(f"第 {lineno} 行：缩进含制表符，YAML 不允许")
    return n


def _parse_block(lines: list[tuple[int, int, str]], pos: int, indent: int) -> tuple[Any, int]:
    """解析一个缩进块，返回 (值, 下一个未消费行的下标)。"""
    if pos >= len(lines):
        return None, pos
    _, first_indent, first_text = lines[pos]
    if first_text.startswith("- "):
        return _parse_list(lines, pos, first_indent)
    if first_text == "-":
        return _parse_list(lines, pos, first_indent)
    return _parse_map(lines, pos, indent if indent >= 0 else first_indent)


def _parse_list(lines: list[tuple[int, int, str]], pos: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    while pos < len(lines):
        lineno, ind, text = lines[pos]
        if ind < indent or not text.startswith("-"):
            break
        if ind > indent:
            raise MiniYamlError(f"第 {lineno} 行：列表项缩进不一致（期望 {indent}）")
        body = text[1:].lstrip()
        pos += 1
        if not body:
            # "-" 独占一行：值在下一个更深的块里
            if pos < len(lines) and lines[pos][1] > indent:
                val, pos = _parse_block(lines, pos, lines[pos][1])
                out.append(val)
            else:
                out.append(None)
            continue
        if ":" in body and not body.split(":", 1)[0].strip().startswith(("\"", "'")):
            # "- key: value"：列表元素是映射，它的首个键就落在 body 的列上
            child_indent = ind + (len(text) - len(text[1:].lstrip()) )
            inner = [(lineno, child_indent, body)]
            while pos < len(lines) and lines[pos][1] > indent:
                inner.append(lines[pos])
                pos += 1
            val, consumed = _parse_map(inner, 0, child_indent)
            if consumed != len(inner):
                raise MiniYamlError(f"第 {inner[consumed][0]} 行：列表元素内缩进异常")
            out.append(val)
        else:
            out.append(_scalar(body, lineno))
    return out, pos


def _parse_map(lines: list[tuple[int, int, str]], pos: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while pos < len(lines):
        lineno, ind, text = lines[pos]
        if ind < indent:
            break
        if ind > indent:
            raise MiniYamlError(f"第 {lineno} 行：意外的缩进（期望 {indent}，实得 {ind}）")
        if text.startswith("- "):
            break
        if ":" not in text:
            raise MiniYamlError(f"第 {lineno} 行：不是 'key: value' 形式 -> {text!r}")
        key, _, rest = text.partition(":")
        key = key.strip().strip("\"'")
        rest = rest.strip()
        pos += 1
        if rest:
            out[key] = _scalar(rest, lineno)
            continue
        # 值在下面的块里；没有更深的行就是空值
        if pos < len(lines) and lines[pos][1] > ind:
            val, pos = _parse_block(lines, pos, lines[pos][1])
            out[key] = val
        elif pos < len(lines) and lines[pos][1] == ind and lines[pos][2].startswith("-"):
            # 列表项与父键同列，是 YAML 合法写法
            val, pos = _parse_list(lines, pos, ind)
            out[key] = val
        else:
            out[key] = None
    return out, pos


def loads(text: str) -> Any:
    """解析一段 YAML 文本。PyYAML 可用时直接委托给它。"""
    try:
        import yaml  # 延迟导入：本机没有也要能跑
    except ModuleNotFoundError:
        pass
    else:
        return yaml.safe_load(text)

    lines: list[tuple[int, int, str]] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        if raw.lstrip().startswith("#"):
            continue
        body = _strip_comment(raw)
        if not body.strip():
            continue
        if body.strip() in ("---", "..."):
            continue
        if _UNSUPPORTED.search(body):
            raise MiniYamlError(f"第 {i} 行：用到了本子集不支持的 YAML 特性 -> {body.strip()!r}")
        lines.append((i, _indent_of(body, i), body.strip()))
    if not lines:
        return None
    val, pos = _parse_block(lines, 0, lines[0][1])
    if pos != len(lines):
        raise MiniYamlError(f"第 {lines[pos][0]} 行：解析在此处停住，检查缩进")
    return val


def load(path: str) -> Any:
    from pathlib import Path

    return loads(Path(path).read_text(encoding="utf-8"))


def _selftest() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sample = """
# 顶层注释
meta:
  name: "demo"     # 行内注释
  verified_at: 2026-09-20
  count: 3
  ratio: 1.5
  ok: true
  missing: null
  tags: [a, b, c]
groups:
  - id: first
    files:
      - repo: org/repo
        size_bytes: 14_294_742_832
        note: "含 # 号的值"
  - id: second
    files: []
"""
    d = loads(sample)
    assert d["meta"]["name"] == "demo", d["meta"]
    assert d["meta"]["count"] == 3
    assert abs(d["meta"]["ratio"] - 1.5) < 1e-9
    assert d["meta"]["ok"] is True
    assert d["meta"]["missing"] is None
    assert d["meta"]["tags"] == ["a", "b", "c"], d["meta"]["tags"]
    assert len(d["groups"]) == 2, d["groups"]
    g0 = d["groups"][0]
    assert g0["id"] == "first"
    assert g0["files"][0]["repo"] == "org/repo"
    assert g0["files"][0]["size_bytes"] == 14294742832
    assert g0["files"][0]["note"] == "含 # 号的值"
    assert d["groups"][1]["files"] == []
    # 不支持的特性必须报错而不是猜
    for bad in ("a: &anchor 1\n", "x:\n\ty: 1\n"):
        try:
            loads(bad)
        except (MiniYamlError, Exception) as exc:  # PyYAML 走自己的异常类型
            log.info("按预期拒绝：%s", str(exc).splitlines()[0][:60])
        else:
            raise AssertionError(f"应当拒绝：{bad!r}")
    log.info("_miniyaml 自测通过")


if __name__ == "__main__":
    _selftest()
