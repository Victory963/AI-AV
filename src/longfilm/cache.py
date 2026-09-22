"""内容寻址产物缓存 —— 让「同样的镜头只渲染一次钱只烧一次」。

三件事：
1. **ArtifactStore**：按 sha256 存放渲染产物、参考图、尾帧、音频。
   内容寻址的意义不是省磁盘，而是**去重与可复现**：两条不同的工单如果吐出
   逐字节相同的产物，它们在盘上就是同一个对象；分镜 JSON 里记的哈希也就成了
   「这一集到底用的是哪一版素材」的唯一凭据。
2. **PromptCache**：提示词+参数指纹 → 产物哈希。这是**省钱**的那一层 ——
   生成之前先查，命中就不提交任务。和 ArtifactStore 分开是因为二者生命周期不同：
   产物可以被 gc 清掉，映射也随之作废；反过来映射失效不该影响别人引用的产物。
3. **gc**：回收没有任何分镜引用的产物。长片产线跑十几轮重拍会留下大量废片，
   不回收磁盘会先于预算耗尽。

取舍：用文件系统 + SQLite，不引入对象存储/Redis。单机产线的产物量级是
GB 到百 GB，本地盘完全吃得下；跨机共享时把 root 挂到共享存储上即可，
但那时 gc 必须单点跑（见 gc 的说明）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .schema import Storyboard

log = logging.getLogger(__name__)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK = 1 << 20


# ---------------------------------------------------------------- 指纹工具


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    """流式哈希。渲染产物动辄几十 MB，一次性读进 3GB 内存的机器不合适。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(_CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def fingerprint_params(prompt: str, params: Mapping[str, Any] | None = None) -> str:
    """提示词 + 参数 → 稳定指纹。

    用 sort_keys 保证字典顺序无关；default=str 让 Enum/Path 这类值也能落地 ——
    指纹只要求「同输入同输出」，不要求可反解。
    """
    blob = json.dumps(
        {"prompt": prompt, "params": dict(params or {})},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def open_sqlite(path: str | Path, *, timeout_s: float = 30.0) -> sqlite3.Connection:
    """本包统一的 SQLite 连接方式（queue.py 也用它）。

    WAL 让「一个写者 + 多个读者」并存 —— 产线里主线程在写工单状态，
    CLI 同时在读进度，用默认的 rollback journal 会互相阻塞。
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=timeout_s, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(f"PRAGMA busy_timeout={int(timeout_s * 1000)}")
    return con


class _ThreadLocalDB:
    """每线程一个连接。sqlite3 连接默认不跨线程，线程池里必须这样拿。"""

    def __init__(self, path: str | Path, *, timeout_s: float = 30.0) -> None:
        self.path = Path(path)
        self.timeout_s = timeout_s
        self._local = threading.local()
        self._ids: set[int] = set()      # 只记「哪些线程开过连接」，不持有连接对象
        self._lock = threading.Lock()

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "con", None)
        if c is None:
            c = open_sqlite(self.path, timeout_s=self.timeout_s)
            self._local.con = c
            with self._lock:
                self._ids.add(threading.get_ident())
        return c

    def close(self) -> None:
        """只关**当前线程**的连接。

        sqlite3 禁止跨线程操作连接对象，close() 也算操作 —— 之前这里持有
        一份全局连接表并在主线程里逐个 close，线程池跑过之后必然抛
        ProgrammingError("SQLite objects created in a thread can only be
        used in that same thread")。

        其余线程的连接交给它们自己的 thread-local 随线程退出被 GC 回收：
        WAL 模式下未显式 close 的只读/已提交连接不会丢数据，而强行跨线程
        close 反而是未定义行为。要确定性释放，就在每个 worker 线程的收尾处
        各自调用一次 close()。
        """
        c = getattr(self._local, "con", None)
        if c is not None:
            c.close()
            self._local.con = None
        with self._lock:
            self._ids.discard(threading.get_ident())

    @property
    def open_threads(self) -> int:
        """还有几个线程持着连接没关 —— 排查「文件被占用」时用得上。"""
        with self._lock:
            return len(self._ids)


# ---------------------------------------------------------------- 产物库


@dataclass(frozen=True)
class ArtifactInfo:
    hash: str
    path: Path
    size: int
    mtime: float

    @property
    def ext(self) -> str:
        return self.path.suffix


@dataclass
class GCReport:
    scanned: int = 0
    kept: int = 0
    deleted: int = 0
    freed_bytes: int = 0
    deleted_hashes: list[str] = field(default_factory=list)
    pruned_cache_rows: int = 0
    dangling_links: int = 0
    dry_run: bool = False

    def report(self) -> str:
        head = "GC（试运行，未真删）" if self.dry_run else "GC"
        return (
            f"{head}：扫描 {self.scanned} 个对象，保留 {self.kept}，清理 {self.deleted}，"
            f"释放 {self.freed_bytes / 1e6:.2f} MB；"
            f"作废缓存映射 {self.pruned_cache_rows} 条，清理悬空软链 {self.dangling_links} 个"
        )


class ArtifactStore:
    """内容寻址的产物库。

    盘上结构：
        <root>/objects/ab/cdef...xyz.mp4   真身，文件名即哈希
        <root>/by-name/ep01/sc01_s003.mp4  软链，给人看的
        <root>/tmp/                        落盘中转区（保证 put 原子）
        <root>/prompt_cache.db             PromptCache

    扩展名被保留在对象名里：ffmpeg 按扩展名推断封装格式，丢掉扩展名会让
    后续 concat/probe 全线要额外指定 -f，得不偿失。
    """

    def __init__(self, root: str | Path, *, cache_db: str | Path | None = None) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.links = self.root / "by-name"
        self.tmp = self.root / "tmp"
        for d in (self.objects, self.links, self.tmp):
            d.mkdir(parents=True, exist_ok=True)
        self._cache_db = Path(cache_db) if cache_db else self.root / "prompt_cache.db"
        self._cache: PromptCache | None = None
        self._lock = threading.Lock()

    # -------------------------------------------------- 路径

    def _dir_of(self, h: str) -> Path:
        return self.objects / h[:2]

    def _find(self, h: str) -> Path | None:
        d = self._dir_of(h)
        if not d.is_dir():
            return None
        stem = h[2:]
        for p in d.glob(stem + "*"):
            # glob 的前缀匹配可能撞上更长的哈希，必须回验完整文件名
            if p.stem == stem or p.name == stem:
                return p
        return None

    def has(self, h: str) -> bool:
        return self._find(h) is not None

    def get(self, h: str) -> Path:
        p = self._find(h)
        if p is None:
            raise KeyError(f"产物库里没有哈希 {h}；可能已被 gc 回收，需要重新生成")
        return p

    def info(self, h: str) -> ArtifactInfo:
        p = self.get(h)
        st = p.stat()
        return ArtifactInfo(hash=h, path=p, size=st.st_size, mtime=st.st_mtime)

    # -------------------------------------------------- 写入

    def put(self, path: str | Path, *, move: bool = False) -> str:
        """把文件收进库，返回哈希。已存在同内容则秒退（这就是去重）。"""
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"要入库的文件不存在：{src}")
        h = hash_file(src)
        dest = self._dir_of(h) / (h[2:] + src.suffix)
        if self._find(h) is not None:
            if move:
                src.unlink()
            return h
        dest.parent.mkdir(parents=True, exist_ok=True)
        staged = self.tmp / f"{h}.{os.getpid()}.{threading.get_ident()}{src.suffix}"
        # 先落到同一文件系统的 tmp 再 rename：rename 是原子的，
        # 并发的读者要么看不到这个哈希，要么看到完整文件，不会读到半截。
        if move:
            shutil.move(str(src), str(staged))
        else:
            shutil.copyfile(src, staged)
        os.replace(staged, dest)
        return h

    def put_bytes(self, data: bytes, *, suffix: str = "") -> str:
        h = hash_bytes(data)
        if self._find(h) is not None:
            return h
        dest = self._dir_of(h) / (h[2:] + suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        staged = self.tmp / f"{h}.{os.getpid()}.{threading.get_ident()}{suffix}"
        staged.write_bytes(data)
        os.replace(staged, dest)
        return h

    def link(self, h: str, friendly_name: str) -> Path:
        """在 by-name 下建人类可读的软链（支持 "ep01/sc01_s003.mp4" 这样的层级）。

        用相对软链，整个 root 可以整体搬走而不断链。
        软链只是视图：gc 不把它当引用根，目标没了就一并清掉。
        """
        target = self.get(h)
        name = friendly_name.strip("/")
        if not name:
            raise ValueError("friendly_name 不能为空")
        if Path(name).suffix == "":
            name += target.suffix
        link_path = self.links / name
        link_path.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(target, link_path.parent)
        staged = link_path.parent / f".{link_path.name}.{os.getpid()}.tmp"
        if staged.is_symlink() or staged.exists():
            staged.unlink()
        os.symlink(rel, staged)
        os.replace(staged, link_path)
        return link_path

    # -------------------------------------------------- 遍历与统计

    def iter_objects(self) -> Iterator[ArtifactInfo]:
        for d in sorted(self.objects.iterdir()):
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if not p.is_file():
                    continue
                st = p.stat()
                yield ArtifactInfo(hash=d.name + p.stem, path=p, size=st.st_size, mtime=st.st_mtime)

    def total_size(self) -> int:
        return sum(a.size for a in self.iter_objects())

    def stats(self) -> dict[str, Any]:
        objs = list(self.iter_objects())
        return {
            "objects": len(objs),
            "bytes": sum(o.size for o in objs),
            "links": sum(1 for _ in self.links.rglob("*") if _.is_symlink()),
            "cache_entries": self.cache.stats()["entries"],
        }

    # -------------------------------------------------- 提示词缓存

    @property
    def cache(self) -> PromptCache:
        with self._lock:
            if self._cache is None:
                self._cache = PromptCache(self._cache_db)
            return self._cache

    # -------------------------------------------------- URI ↔ 哈希

    def hash_of(self, uri: str | Path | None) -> str | None:
        """把一个 uri/路径解析成库内哈希；不属于本库则返回 None。

        分镜里的 render_uri 可能是对象真身、可能是 by-name 软链、
        也可能干脆就写了个哈希串，三种都要认得出来，否则 gc 会误删在用的素材。
        """
        if not uri:
            return None
        s = str(uri)
        if s.startswith("file://"):
            s = s[7:]
        if _HEX64.match(s):
            return s if self.has(s) else None
        p = Path(s)
        if not p.exists() and not p.is_symlink():
            return None
        real = Path(os.path.realpath(p))
        try:
            rel = real.relative_to(self.objects.resolve())
        except ValueError:
            return None
        if len(rel.parts) != 2:
            return None
        h = rel.parts[0] + Path(rel.parts[1]).stem
        return h if _HEX64.match(h) else None

    # -------------------------------------------------- 回收

    def gc(
        self,
        keep_referenced_by: Storyboard | Iterable[Storyboard | str | Path],
        *,
        dry_run: bool = False,
        prune_cache: bool = True,
    ) -> GCReport:
        """回收没有被任何分镜引用的产物。

        引用根 = 传进来的分镜里出现的所有 uri（渲染产物、参考图、尾帧、
        角色定妆、音轨、LoRA 路径），外加直接给的哈希/路径。
        by-name 软链**不算**引用根 —— 它是给人看的视图，不是资产清单；
        否则一次人工浏览就能让废片永生。

        适用边界：gc 不加全局锁，必须在没有 worker 在跑的时候单点执行。
        在跑的工单其产物还没写进分镜，这时 gc 会把它当垃圾删掉。
        """
        keep: set[str] = set()
        for h in self._collect_refs(keep_referenced_by):
            keep.add(h)

        rep = GCReport(dry_run=dry_run)
        for obj in self.iter_objects():
            rep.scanned += 1
            if obj.hash in keep:
                rep.kept += 1
                continue
            rep.deleted += 1
            rep.freed_bytes += obj.size
            rep.deleted_hashes.append(obj.hash)
            if not dry_run:
                obj.path.unlink()

        if prune_cache and rep.deleted_hashes:
            rep.pruned_cache_rows = (
                0 if dry_run else self.cache.prune(set(rep.deleted_hashes))
            )
        if not dry_run:
            rep.dangling_links = self._sweep_links()
            for d in self.objects.iterdir():
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
        log.info("%s", rep.report())
        return rep

    def _collect_refs(
        self, src: Storyboard | Iterable[Storyboard | str | Path]
    ) -> Iterator[str]:
        items: Iterable[Any] = [src] if isinstance(src, (Storyboard, str, Path)) else src
        for item in items:
            if isinstance(item, Storyboard):
                for uri in iter_storyboard_uris(item):
                    h = self.hash_of(uri)
                    if h:
                        yield h
            else:
                h = self.hash_of(item)
                if h:
                    yield h

    def _sweep_links(self) -> int:
        n = 0
        for p in sorted(self.links.rglob("*"), reverse=True):
            if p.is_symlink() and not p.exists():
                p.unlink()
                n += 1
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
        return n

    def __repr__(self) -> str:
        return f"<ArtifactStore root={self.root} objects={sum(1 for _ in self.iter_objects())}>"


def iter_storyboard_uris(sb: Storyboard) -> Iterator[str]:
    """穷举一份分镜里所有指向素材的 uri。gc 的引用根就是它。

    漏一个字段就会误删在用的素材，所以这里宁可多列：新增字段时也要同步更新。
    """
    for ch in sb.characters:
        for ref in (*ch.portraits, *ch.turnaround, *ch.motion_refs):
            yield ref.uri
        if ch.voice.timbre_ref_uri:
            yield ch.voice.timbre_ref_uri
        if ch.lora:
            yield ch.lora.path
    for shot in sb.all_shots():
        if shot.render_uri:
            yield shot.render_uri
        for ref in (*shot.refs.images, *shot.refs.videos, *shot.refs.audios):
            yield ref.uri
    a = sb.audio
    for t in (a.dialogue_track, a.foley_track, a.music_track, a.ambience_track):
        if t:
            yield t


# ---------------------------------------------------------------- 提示词缓存


@dataclass(frozen=True)
class CacheEntry:
    key: str
    artifact_hash: str
    provider: str
    cost_usd: float
    created_at: float
    hits: int
    meta: dict[str, Any]


_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_cache (
    key           TEXT PRIMARY KEY,
    artifact_hash TEXT NOT NULL,
    provider      TEXT NOT NULL DEFAULT '',
    cost_usd      REAL NOT NULL DEFAULT 0.0,
    created_at    REAL NOT NULL,
    last_hit_at   REAL,
    hits          INTEGER NOT NULL DEFAULT 0,
    meta          TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_cache_hash ON prompt_cache(artifact_hash);
CREATE TABLE IF NOT EXISTS prompt_cache_stat (
    k TEXT PRIMARY KEY, v REAL NOT NULL DEFAULT 0
);
"""


class PromptCache:
    """提示词+参数指纹 → 产物哈希。

    只存映射不存产物，所以查一次的代价是一次 SQLite 点查（微秒级），
    而一次未命中的生成是分钟级 + 真金白银 —— 这笔买卖永远划算。

    命中计数与「省下的钱」是给运营看的：缓存命中率低说明分镜在无意义地抖参数。
    """

    def __init__(self, db_path: str | Path, *, timeout_s: float = 30.0) -> None:
        self.db_path = Path(db_path)
        self._db = _ThreadLocalDB(self.db_path, timeout_s=timeout_s)
        self._db.conn().executescript(_CACHE_SCHEMA)

    @staticmethod
    def key_for(prompt: str, params: Mapping[str, Any] | None = None) -> str:
        return fingerprint_params(prompt, params)

    def put(
        self,
        key: str,
        artifact_hash: str,
        *,
        provider: str = "",
        cost_usd: float = 0.0,
        meta: Mapping[str, Any] | None = None,
    ) -> CacheEntry:
        now = time.time()
        self._db.conn().execute(
            "INSERT INTO prompt_cache(key, artifact_hash, provider, cost_usd, created_at, meta) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET artifact_hash=excluded.artifact_hash, "
            "provider=excluded.provider, cost_usd=excluded.cost_usd, meta=excluded.meta",
            (key, artifact_hash, provider, float(cost_usd), now,
             json.dumps(dict(meta or {}), ensure_ascii=False)),
        )
        got = self.get(key, count_hit=False)
        assert got is not None
        return got

    def get(self, key: str, *, count_hit: bool = True) -> CacheEntry | None:
        row = self._db.conn().execute(
            "SELECT * FROM prompt_cache WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            self._bump("misses")
            return None
        if count_hit:
            self._db.conn().execute(
                "UPDATE prompt_cache SET hits=hits+1, last_hit_at=? WHERE key=?",
                (time.time(), key),
            )
            self._bump("hits")
            self._bump("saved_usd", row["cost_usd"])
        return _row_to_entry(row)

    def lookup(
        self, prompt: str, params: Mapping[str, Any] | None = None
    ) -> CacheEntry | None:
        return self.get(self.key_for(prompt, params))

    def delete(self, key: str) -> bool:
        cur = self._db.conn().execute("DELETE FROM prompt_cache WHERE key=?", (key,))
        return cur.rowcount > 0

    def prune(self, dead_hashes: Iterable[str]) -> int:
        """产物没了，映射也必须作废 —— 否则下次「命中」会拿到一个不存在的文件。"""
        dead = list(dict.fromkeys(dead_hashes))
        if not dead:
            return 0
        con = self._db.conn()
        n = 0
        for i in range(0, len(dead), 500):
            chunk = dead[i:i + 500]
            q = ",".join("?" * len(chunk))
            n += con.execute(
                f"DELETE FROM prompt_cache WHERE artifact_hash IN ({q})", chunk
            ).rowcount
        return n

    def verify(self, store: ArtifactStore) -> int:
        """对账：删掉所有指向已不存在产物的映射。返回清理条数。"""
        rows = self._db.conn().execute(
            "SELECT DISTINCT artifact_hash FROM prompt_cache"
        ).fetchall()
        dead = [r["artifact_hash"] for r in rows if not store.has(r["artifact_hash"])]
        return self.prune(dead)

    def _bump(self, k: str, by: float = 1.0) -> None:
        self._db.conn().execute(
            "INSERT INTO prompt_cache_stat(k, v) VALUES(?, ?) "
            "ON CONFLICT(k) DO UPDATE SET v = v + excluded.v",
            (k, by),
        )

    def stats(self) -> dict[str, Any]:
        con = self._db.conn()
        n = con.execute("SELECT COUNT(*) c FROM prompt_cache").fetchone()["c"]
        acc = {r["k"]: r["v"] for r in con.execute("SELECT k, v FROM prompt_cache_stat")}
        hits = int(acc.get("hits", 0))
        misses = int(acc.get("misses", 0))
        total = hits + misses
        return {
            "entries": n,
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / total if total else 0.0,
            "saved_usd": round(acc.get("saved_usd", 0.0), 4),
        }

    def close(self) -> None:
        self._db.close()

    def __repr__(self) -> str:
        return f"<PromptCache {self.db_path} entries={self.stats()['entries']}>"


def _row_to_entry(row: sqlite3.Row) -> CacheEntry:
    return CacheEntry(
        key=row["key"],
        artifact_hash=row["artifact_hash"],
        provider=row["provider"],
        cost_usd=row["cost_usd"],
        created_at=row["created_at"],
        hits=row["hits"],
        meta=json.loads(row["meta"]),
    )


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    import tempfile

    from .schema import ImageRef, Scene, Shot, Storyboard

    with tempfile.TemporaryDirectory(prefix="longfilm-cache-") as td:
        tmp = Path(td)
        store = ArtifactStore(tmp / "store")
        src = tmp / "work"
        src.mkdir()

        # --- 去重：两个不同文件名、相同内容，入库后是同一个对象
        a = src / "take1.mp4"
        b = src / "take2.mp4"
        a.write_bytes(b"\x00\x01video-bytes")
        b.write_bytes(b"\x00\x01video-bytes")
        h1 = store.put(a)
        h2 = store.put(b)
        assert h1 == h2, "相同内容必须得到相同哈希"
        assert len(list(store.iter_objects())) == 1, "相同内容不应占两份磁盘"
        assert store.get(h1).suffix == ".mp4", "扩展名要保留，ffmpeg 靠它认封装"

        # --- 幂等：重复 put 不改变库状态
        before = store.stats()
        store.put(a)
        assert store.stats() == before, "重复入库必须是无副作用的"

        # --- 内容不同则是两个对象
        c = src / "other.png"
        c.write_bytes(b"png-bytes")
        h3 = store.put(c)
        assert h3 != h1 and len(list(store.iter_objects())) == 2

        # --- 人类可读软链
        link = store.link(h1, "ep01/sc01_s001")
        assert link.is_symlink() and link.resolve() == store.get(h1).resolve()
        assert link.suffix == ".mp4", "没写扩展名时要按对象补上"
        assert not os.path.isabs(os.readlink(link)), "软链必须是相对的，整库才能搬走"

        # --- uri ↔ 哈希 三种写法都要认得
        assert store.hash_of(h1) == h1
        assert store.hash_of(store.get(h1)) == h1
        assert store.hash_of(link) == h1
        assert store.hash_of(tmp / "not-in-store.mp4") is None

        # --- PromptCache：命中即免单
        cache = store.cache
        key = PromptCache.key_for("一个女人在雨中回头", {"seed": 7, "duration_s": 8.0})
        assert cache.get(key) is None, "冷启动必须未命中"
        cache.put(key, h1, provider="mock", cost_usd=0.42, meta={"shot": "s001"})
        hit = cache.get(key)
        assert hit is not None and hit.artifact_hash == h1 and hit.meta["shot"] == "s001"
        assert PromptCache.key_for("一个女人在雨中回头", {"duration_s": 8.0, "seed": 7}) == key, \
            "参数顺序不该影响指纹"
        assert PromptCache.key_for("一个女人在雨中回头", {"seed": 8}) != key
        st = cache.stats()
        assert st["entries"] == 1 and st["hits"] == 1 and st["misses"] == 1
        assert abs(st["saved_usd"] - 0.42) < 1e-9, "省下的钱要能算出来"

        # --- gc：被分镜引用的留下，没人引用的清掉
        sb = Storyboard(
            project="demo", episode="ep01",
            scenes=[Scene(id="sc01", shots=[
                Shot(id="s001", scene_id="sc01", index=0, render_uri=str(link)),
            ])],
        )
        sb.scenes[0].shots[0].refs.images.append(
            ImageRef(role="identity", uri=str(store.get(h3)))
        )
        dry = store.gc(sb, dry_run=True)
        assert dry.scanned == 2 and dry.deleted == 0, "引用齐全时不该有垃圾"
        assert store.has(h1) and store.has(h3)

        # 再塞一个没人要的废片
        d = src / "scrap.mp4"
        d.write_bytes(b"scrap-take-bytes")
        h4 = store.put(d)
        cache.put(PromptCache.key_for("废片", {}), h4, cost_usd=0.1)
        store.link(h4, "ep01/scrap")
        rep = store.gc(sb)
        assert rep.deleted == 1 and rep.deleted_hashes == [h4], f"应只清废片，实际 {rep}"
        assert not store.has(h4) and store.has(h1) and store.has(h3)
        assert rep.pruned_cache_rows == 1, "产物没了，映射必须一起作废"
        assert rep.dangling_links == 1, "悬空软链要清掉"
        assert cache.get(PromptCache.key_for("废片", {})) is None
        assert cache.get(key) is not None, "在用的映射不能被误伤"
        assert store.hash_of(link) == h1, "幸存对象的软链仍然可用"

        # --- verify 对账
        store.get(h3).unlink()
        assert cache.verify(store) == 0, "h3 没进缓存，不该有可清的映射"
        cache.put("orphan", "0" * 64)
        assert cache.verify(store) == 1

        print("cache 自测通过：", json.dumps(store.stats(), ensure_ascii=False),
              json.dumps(cache.stats(), ensure_ascii=False))
        cache.close()


if __name__ == "__main__":
    _selftest()
