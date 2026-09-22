"""生产队列 —— 一集片从分镜到成片的可靠执行层。

要解决的现实：一集 8-12 个镜，每镜可能重拍数次，跨多个供应商，
单个任务分钟级，中途断电/Ctrl-C/OOM 随时可能发生。重跑一遍的代价是真金白银，
所以队列的第一属性不是吞吐，而是**不重复烧钱**。

为什么是 SQLite + 文件锁，而不是 Celery/Redis/RabbitMQ：
  * 工单总量是「每集几十条」，不是每秒几万条。为这个量级引入 broker，
    等于给产线加一个必须长期运维的单点故障。
  * 这条产线本来就要在一台机器上跑（渲染、ffmpeg、质检都在本地），
    队列跨机没有意义 —— 反而让「产物在哪台机器上」变成新问题。
  * SQLite 的事务是真事务：claim 用 BEGIN IMMEDIATE 就能做到原子领取，
    断电后 WAL 自动恢复，不需要额外的持久化设计。
适用边界（必须说清楚）：
  * 单机。db 文件放在 NFS/SMB 上会破坏锁语义，跨机请换成真正的 broker。
  * POSIX。文件锁用 fcntl.flock。
  * 并发规模在百级以内。再高时 claim 的串行临界区会成为瓶颈。

幂等靠三层，从便宜到贵依次拦截：
  1. 同一镜的工单行（key = episode/shot_id）：断点续跑时已完成的直接跳过；
  2. PromptCache（提示词+参数指纹 → 产物哈希）：不同镜但请求完全相同时复用产物；
  3. 真的去提交任务。
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import math
import os
import socket
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Sequence

from .cache import ArtifactStore, _ThreadLocalDB, fingerprint_params
from .providers.base import FailureKind, GenRequest, GenResult, ProviderError
from .schema import CameraMove, EngineHint, Shot, Storyboard

if TYPE_CHECKING:  # 只为类型标注，避免把 router 的依赖拖进队列的导入路径
    from .router import Router, RoutingPlan

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- 文件锁


class FileLock:
    """基于 fcntl.flock 的进程间互斥锁。

    SQLite 自己的锁足以保证单条语句的原子性，但「先回收过期租约，再挑选，
    再标记领取」是一串语句；BEGIN IMMEDIATE 能保证它们原子，
    而文件锁的作用是让并发的 worker **排队进入**这段临界区，
    而不是一起撞进去然后靠 SQLITE_BUSY 重试 —— 后者在租约回收时会退化成活锁。
    """

    def __init__(self, path: str | Path, *, timeout_s: float = 60.0) -> None:
        self.path = Path(path)
        self.timeout_s = timeout_s
        self._fd: int | None = None
        self._local = threading.Lock()

    def acquire(self, *, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local.acquire()
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fd = fd
                return True
            except OSError:
                if not blocking or time.monotonic() > deadline:
                    os.close(fd)
                    self._local.release()
                    if blocking:
                        raise TimeoutError(f"等待文件锁 {self.path} 超过 {self.timeout_s:g}s")
                    return False
                time.sleep(0.01)

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None
        self._local.release()

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# ---------------------------------------------------------------- 工单


class JobState(str, Enum):
    PENDING = "pending"      # 待领取（含退避等待）
    RUNNING = "running"      # 已被 worker 领走，持有租约
    DONE = "done"            # 出片且过质检
    FAILED = "failed"        # 重试用尽 / 不可重试 / 质检判 escalate —— 需人工
    BLOCKED = "blocked"      # 内容门禁拒单，产线不做任何规避
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {JobState.DONE, JobState.FAILED,
                        JobState.BLOCKED, JobState.CANCELLED}


@dataclass
class JobRecord:
    """一条工单。字段划分的原则：**断电后靠这一行就能决定下一步怎么走**。"""

    key: str
    episode: str
    shot_id: str
    base_fingerprint: str        # 分镜原始指纹：用来判断「分镜有没有被人改过」
    fingerprint: str             # 当前生效指纹：重拍改参数后会变
    prompt_key: str              # 提示词+参数指纹，PromptCache 的键
    state: JobState = JobState.PENDING
    provider: str = ""           # 计划主选（限流按它算）
    provider_used: str = ""      # 实际出片的那家
    fallbacks: tuple[str, ...] = ()
    attempts: int = 0            # 领取次数（含重试）
    max_attempts: int = 3
    takes: int = 0               # 重拍轮次（质检不过才加）
    priority: int = 0
    job_id: str = ""
    artifact_hash: str = ""
    render_uri: str = ""
    cost_usd: float = 0.0        # 累计花费，跨重拍累加
    est_cost_usd: float = 0.0
    qc_score: float | None = None
    qc_verdict: str = ""
    failure: FailureKind = FailureKind.NONE
    message: str = ""
    worker_id: str = ""
    lease_until: float = 0.0
    next_run_at: float = 0.0
    cache_hit: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def done(self) -> bool:
        return self.state is JobState.DONE

    def line(self) -> str:
        v = f" qc={self.qc_score:.2f}" if self.qc_score is not None else ""
        tail = f" {self.message}" if self.message and not self.done else ""
        return (
            f"{self.shot_id:<10}{self.state.value:<10}"
            f"{(self.provider_used or self.provider or '-'):<14}"
            f"take={self.takes} try={self.attempts} ${self.cost_usd:.4f}{v}"
            f"{' [缓存]' if self.cache_hit else ''}{tail}"
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    key              TEXT PRIMARY KEY,
    episode          TEXT NOT NULL DEFAULT '',
    shot_id          TEXT NOT NULL,
    base_fingerprint TEXT NOT NULL,
    fingerprint      TEXT NOT NULL,
    prompt_key       TEXT NOT NULL DEFAULT '',
    state            TEXT NOT NULL,
    provider         TEXT NOT NULL DEFAULT '',
    provider_used    TEXT NOT NULL DEFAULT '',
    fallbacks        TEXT NOT NULL DEFAULT '',
    attempts         INTEGER NOT NULL DEFAULT 0,
    max_attempts     INTEGER NOT NULL DEFAULT 3,
    takes            INTEGER NOT NULL DEFAULT 0,
    priority         INTEGER NOT NULL DEFAULT 0,
    job_id           TEXT NOT NULL DEFAULT '',
    artifact_hash    TEXT NOT NULL DEFAULT '',
    render_uri       TEXT NOT NULL DEFAULT '',
    cost_usd         REAL NOT NULL DEFAULT 0.0,
    est_cost_usd     REAL NOT NULL DEFAULT 0.0,
    qc_score         REAL,
    qc_verdict       TEXT NOT NULL DEFAULT '',
    failure          TEXT NOT NULL DEFAULT 'none',
    message          TEXT NOT NULL DEFAULT '',
    worker_id        TEXT NOT NULL DEFAULT '',
    lease_until      REAL NOT NULL DEFAULT 0,
    next_run_at      REAL NOT NULL DEFAULT 0,
    cache_hit        INTEGER NOT NULL DEFAULT 0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON jobs(state, next_run_at, priority);
CREATE INDEX IF NOT EXISTS idx_jobs_prompt ON jobs(prompt_key, state);
CREATE INDEX IF NOT EXISTS idx_jobs_ep ON jobs(episode, state);
"""


def _row_to_record(row: Any) -> JobRecord:
    return JobRecord(
        key=row["key"], episode=row["episode"], shot_id=row["shot_id"],
        base_fingerprint=row["base_fingerprint"], fingerprint=row["fingerprint"],
        prompt_key=row["prompt_key"], state=JobState(row["state"]),
        provider=row["provider"], provider_used=row["provider_used"],
        fallbacks=tuple(f for f in row["fallbacks"].split(",") if f),
        attempts=row["attempts"], max_attempts=row["max_attempts"],
        takes=row["takes"], priority=row["priority"], job_id=row["job_id"],
        artifact_hash=row["artifact_hash"], render_uri=row["render_uri"],
        cost_usd=row["cost_usd"], est_cost_usd=row["est_cost_usd"],
        qc_score=row["qc_score"], qc_verdict=row["qc_verdict"],
        failure=FailureKind(row["failure"]), message=row["message"],
        worker_id=row["worker_id"], lease_until=row["lease_until"],
        next_run_at=row["next_run_at"], cache_hit=bool(row["cache_hit"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
        started_at=row["started_at"], finished_at=row["finished_at"],
    )


@dataclass
class QueueStats:
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    cost_usd: float = 0.0
    est_cost_usd: float = 0.0
    attempts: int = 0
    takes: int = 0
    cache_hits: int = 0

    def __getitem__(self, state: str) -> int:
        return self.counts.get(state, 0)

    @property
    def done(self) -> int:
        return self["done"]

    @property
    def settled(self) -> int:
        """已成终态的数量 —— 进度条的分子。"""
        return sum(self.counts.get(s.value, 0) for s in JobState if s.terminal)

    def bar(self, width: int = 24) -> str:
        frac = self.settled / self.total if self.total else 0.0
        n = int(frac * width)
        return "#" * n + "-" * (width - n)

    def line(self) -> str:
        return (
            f"[{self.bar()}] {self.settled}/{self.total} "
            f"done={self['done']} run={self['running']} pend={self['pending']} "
            f"fail={self['failed']} block={self['blocked']} "
            f"take={self.takes} 缓存={self.cache_hits} ${self.cost_usd:.4f}"
        )


# ---------------------------------------------------------------- 提示词键


def request_cache_key(req: GenRequest, provider: str = "") -> str:
    """把一次生成请求压成缓存键。

    刻意**不含** shot_id / idempotency_key：两个不同镜号但请求完全一致的镜
    （比如反复出现的空镜），本来就该复用同一段素材。
    刻意**含** provider：同样的参数在不同引擎上画质天差地别，
    跨家复用会让「英雄镜」悄悄拿到草稿引擎的产物。要跨家复用就显式传 provider=""。
    """
    params: dict[str, Any] = {
        "provider": provider,
        "duration_s": round(req.duration_s, 4),
        "resolution": list(req.resolution),
        "fps": req.fps,
        "negative": req.negative_prompt,
        "seed": req.seed,
        "first_frame": req.first_frame_uri,
        "last_frame": req.last_frame_uri,
        "extend_from_job": req.extend_from_job,
        "audio": req.audio_uri,
        "camera": req.camera_move,
        "lora": req.lora,
        "refs": [
            [r.role, r.uri, round(r.weight, 4)]
            for r in (*req.refs.images, *req.refs.videos, *req.refs.audios)
        ],
    }
    return fingerprint_params(req.prompt, params)


# ---------------------------------------------------------------- 队列


class JobQueue:
    """单机可靠队列。所有状态都在 SQLite 里，进程随时可以死。"""

    def __init__(
        self,
        db_path: str | Path,
        *,
        lease_s: float = 600.0,
        backoff_base_s: float = 5.0,
        backoff_cap_s: float = 300.0,
        timeout_s: float = 30.0,
    ) -> None:
        self.db_path = Path(db_path)
        self.lease_s = lease_s
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self._db = _ThreadLocalDB(self.db_path, timeout_s=timeout_s)
        self.lock = FileLock(self.db_path.with_suffix(self.db_path.suffix + ".claim.lock"))
        with self.lock:
            self._db.conn().executescript(_SCHEMA)

    # -------------------------------------------------- 入队

    @staticmethod
    def job_key(episode: str, shot_id: str) -> str:
        return f"{episode}/{shot_id}" if episode else shot_id

    def enqueue(
        self,
        shot: Shot,
        plan: RoutingPlan | None = None,
        *,
        episode: str = "",
        priority: int | None = None,
        max_attempts: int = 3,
        store: ArtifactStore | None = None,
    ) -> JobRecord:
        """幂等入队。

        返回的记录可能已经是 DONE —— 那就是命中了缓存（本集旧工单 / 产物库），
        调用方看到 DONE 就不要再执行。这是「不重复烧钱」的第一道闸。
        """
        key = self.job_key(episode, shot.id)
        fp = shot.fingerprint()
        prompt_key = request_cache_key(plan.request, plan.primary or "") if (
            plan is not None and plan.request is not None
        ) else fp

        old = self.get(key)
        if old is not None:
            if old.base_fingerprint == fp:
                # 断点续跑的正常路径：这一镜的工单还在，状态就是权威，原样交回。
                return old
            log.info(
                "镜头 %s 的分镜已被修改（指纹 %s → %s），作废旧工单重新入队",
                shot.id, old.base_fingerprint, fp,
            )
            self._db.conn().execute("DELETE FROM jobs WHERE key=?", (key,))

        now = time.time()
        rec = JobRecord(
            key=key, episode=episode, shot_id=shot.id, base_fingerprint=fp,
            fingerprint=fp, prompt_key=prompt_key, max_attempts=max_attempts,
            priority=shot.index if priority is None else priority,
            created_at=now, updated_at=now,
        )
        if plan is not None:
            rec.provider = plan.primary or ""
            rec.fallbacks = tuple(plan.fallbacks)
            rec.est_cost_usd = plan.estimated_cost_usd
            if not plan.gate.allowed:
                rec.state = JobState.BLOCKED
                rec.failure = plan.gate.failure_kind
                rec.message = "内容门禁拒单：" + "；".join(plan.gate.blockers)
                rec.finished_at = now
            elif not plan.routable:
                rec.state = JobState.FAILED
                rec.failure = FailureKind.QUOTA
                rec.message = "没有可用引擎：" + "；".join(
                    f"{n}={why}" for n, why in plan.rejected
                )
                rec.finished_at = now

        if rec.state is JobState.PENDING:
            hit = self._lookup_artifact(prompt_key, store)
            if hit is not None:
                rec.state = JobState.DONE
                rec.cache_hit = True
                rec.artifact_hash, rec.render_uri, rec.provider_used = hit
                rec.cost_usd = 0.0
                rec.finished_at = now
                rec.message = "命中产物缓存，未提交任务"
                log.info("镜头 %s 命中缓存 %s，省下一次生成", shot.id, rec.artifact_hash[:12])

        self._insert(rec)
        return rec

    def mark_cache_hit(self, key: str) -> None:
        """把工单标成缓存命中。

        单列一个方法而不是给 complete() 加参数：complete 是「真的跑完了」的语义，
        缓存命中是「根本没跑」，两者的成本归因和产能统计口径不同，
        混进同一个调用里迟早会有人把缓存命中算进产能。
        """
        self._db.conn().execute("UPDATE jobs SET cache_hit=1 WHERE key=?", (key,))

    def _lookup_artifact(
        self, prompt_key: str, store: ArtifactStore | None
    ) -> tuple[str, str, str] | None:
        """(artifact_hash, render_uri, provider) 或 None。"""
        row = self._db.conn().execute(
            "SELECT artifact_hash, render_uri, provider_used FROM jobs "
            "WHERE prompt_key=? AND state='done' AND render_uri<>'' "
            "ORDER BY finished_at LIMIT 1",
            (prompt_key,),
        ).fetchone()
        if row is not None and Path(row["render_uri"]).exists():
            return row["artifact_hash"], row["render_uri"], row["provider_used"]
        if store is None:
            return None
        entry = store.cache.get(prompt_key)
        if entry is None or not store.has(entry.artifact_hash):
            return None
        return entry.artifact_hash, str(store.get(entry.artifact_hash)), entry.provider

    def _insert(self, rec: JobRecord) -> None:
        self._db.conn().execute(
            "INSERT INTO jobs(key, episode, shot_id, base_fingerprint, fingerprint, "
            "prompt_key, state, provider, provider_used, fallbacks, attempts, max_attempts, "
            "takes, priority, job_id, artifact_hash, render_uri, cost_usd, est_cost_usd, "
            "qc_score, qc_verdict, failure, message, worker_id, lease_until, next_run_at, "
            "cache_hit, created_at, updated_at, started_at, finished_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec.key, rec.episode, rec.shot_id, rec.base_fingerprint, rec.fingerprint,
             rec.prompt_key, rec.state.value, rec.provider, rec.provider_used,
             ",".join(rec.fallbacks), rec.attempts, rec.max_attempts, rec.takes,
             rec.priority, rec.job_id, rec.artifact_hash, rec.render_uri, rec.cost_usd,
             rec.est_cost_usd, rec.qc_score, rec.qc_verdict, rec.failure.value,
             rec.message, rec.worker_id, rec.lease_until, rec.next_run_at,
             int(rec.cache_hit), rec.created_at, rec.updated_at,
             rec.started_at, rec.finished_at),
        )

    # -------------------------------------------------- 领取

    def claim(
        self, worker_id: str, n: int = 1, *, lease_s: float | None = None,
        episode: str | None = None,
    ) -> list[JobRecord]:
        """原子领取至多 n 条。带租约：worker 死了，租约到期任务自动回到队列。"""
        lease = self.lease_s if lease_s is None else lease_s
        now = time.time()
        con = self._db.conn()
        with self.lock:
            con.execute("BEGIN IMMEDIATE")
            try:
                self._reclaim_expired(now)
                sql = (
                    "SELECT * FROM jobs WHERE state='pending' AND next_run_at<=? "
                    + ("AND episode=? " if episode is not None else "")
                    + "ORDER BY priority, created_at LIMIT ?"
                )
                args: list[Any] = [now] + ([episode] if episode is not None else []) + [n]
                rows = con.execute(sql, args).fetchall()
                out: list[JobRecord] = []
                for row in rows:
                    con.execute(
                        "UPDATE jobs SET state='running', worker_id=?, lease_until=?, "
                        "attempts=attempts+1, started_at=COALESCE(started_at,?), updated_at=? "
                        "WHERE key=?",
                        (worker_id, now + lease, now, now, row["key"]),
                    )
                    out.append(_row_to_record(
                        con.execute("SELECT * FROM jobs WHERE key=?", (row["key"],)).fetchone()
                    ))
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise
        for rec in out:
            log.debug("worker %s 领取 %s（第 %d 次）", worker_id, rec.shot_id, rec.attempts)
        return out

    def _reclaim_expired(self, now: float) -> int:
        """租约过期的任务回到 pending。

        不清 attempts：worker 死掉也算一次尝试，否则一个反复 OOM 的镜头会永远重试。
        """
        cur = self._db.conn().execute(
            "UPDATE jobs SET state='pending', worker_id='', lease_until=0, "
            "message='租约过期，worker 可能已死，任务回到队列', updated_at=? "
            "WHERE state='running' AND lease_until>0 AND lease_until<?",
            (now, now),
        )
        if cur.rowcount:
            log.warning("回收 %d 条租约过期的工单", cur.rowcount)
        return cur.rowcount

    def reclaim_expired(self) -> int:
        with self.lock:
            return self._reclaim_expired(time.time())

    def reclaim_running(self, reason: str = "上次运行未正常结束，恢复为待跑") -> int:
        """把所有 running 无条件拉回 pending。

        只有在**确认没有别的 worker 在跑**时才能调用（run_episode 靠独占的
        episode 文件锁来确认）；否则会把活着的任务重复派发出去。
        """
        with self.lock:
            cur = self._db.conn().execute(
                "UPDATE jobs SET state='pending', worker_id='', lease_until=0, "
                "next_run_at=0, message=?, updated_at=? WHERE state='running'",
                (reason, time.time()),
            )
        if cur.rowcount:
            log.info("恢复 %d 条上次中断的工单", cur.rowcount)
        return cur.rowcount

    def heartbeat(self, keys: Sequence[str], *, lease_s: float | None = None) -> int:
        """续租。主循环每轮调一次：进程还活着，就别让租约过期被别人抢走。"""
        if not keys:
            return 0
        lease = self.lease_s if lease_s is None else lease_s
        q = ",".join("?" * len(keys))
        cur = self._db.conn().execute(
            f"UPDATE jobs SET lease_until=?, updated_at=? "
            f"WHERE state='running' AND key IN ({q})",
            [time.time() + lease, time.time(), *keys],
        )
        return cur.rowcount

    # -------------------------------------------------- 终态

    def complete(
        self,
        key: str,
        *,
        render_uri: str,
        artifact_hash: str = "",
        cost_usd: float = 0.0,
        provider_used: str = "",
        job_id: str = "",
        qc_score: float | None = None,
        qc_verdict: str = "",
        message: str = "",
    ) -> JobRecord:
        now = time.time()
        self._db.conn().execute(
            "UPDATE jobs SET state='done', render_uri=?, artifact_hash=?, "
            "cost_usd=cost_usd+?, provider_used=?, job_id=?, qc_score=?, qc_verdict=?, "
            "failure='none', message=?, worker_id='', lease_until=0, "
            "finished_at=?, updated_at=? WHERE key=?",
            (render_uri, artifact_hash, float(cost_usd), provider_used, job_id,
             qc_score, qc_verdict, message, now, now, key),
        )
        return self.require(key)

    def fail(
        self,
        key: str,
        failure: FailureKind,
        message: str = "",
        *,
        cost_usd: float = 0.0,
        delay_s: float | None = None,
    ) -> JobRecord:
        """失败登记。可重试且次数没用尽 → 退避后回队；否则终态待人工。

        判定完全交给 FailureKind（providers/base.py 定义），队列不解释语义 ——
        队列一旦开始猜「这个错是不是能重试」，供应商换一家就得改队列。
        """
        rec = self.require(key)
        now = time.time()
        retry = failure.retryable and rec.attempts < rec.max_attempts
        if retry:
            wait_s = self.backoff(rec.attempts) if delay_s is None else delay_s
            self._db.conn().execute(
                "UPDATE jobs SET state='pending', worker_id='', lease_until=0, "
                "next_run_at=?, failure=?, message=?, cost_usd=cost_usd+?, updated_at=? "
                "WHERE key=?",
                (now + wait_s, failure.value,
                 f"{message}（第 {rec.attempts}/{rec.max_attempts} 次失败，"
                 f"{wait_s:.0f}s 后重试）", float(cost_usd), now, key),
            )
        else:
            self._db.conn().execute(
                "UPDATE jobs SET state='failed', worker_id='', lease_until=0, "
                "failure=?, message=?, cost_usd=cost_usd+?, finished_at=?, updated_at=? "
                "WHERE key=?",
                (failure.value, message, float(cost_usd), now, now, key),
            )
            log.error("工单 %s 终态失败（%s）：%s", key, failure.value, message)
        return self.require(key)

    def requeue(
        self,
        key: str,
        *,
        reason: str,
        fingerprint: str | None = None,
        prompt_key: str | None = None,
        provider: str | None = None,
        cost_usd: float = 0.0,
        bump_takes: bool = True,
        delay_s: float = 0.0,
    ) -> JobRecord:
        """重拍：换过参数的同一镜重新排队。

        fingerprint/prompt_key 必须跟着改 —— 不改的话下一轮会命中上一条废片的缓存，
        重拍就变成了「原样再拿一次」。
        """
        now = time.time()
        rec = self.require(key)
        self._db.conn().execute(
            "UPDATE jobs SET state='pending', worker_id='', lease_until=0, next_run_at=?, "
            "attempts=0, takes=takes+?, fingerprint=?, prompt_key=?, provider=?, "
            "cost_usd=cost_usd+?, qc_score=NULL, qc_verdict='', failure='none', "
            "message=?, updated_at=? WHERE key=?",
            (now + delay_s, 1 if bump_takes else 0,
             fingerprint or rec.fingerprint, prompt_key or rec.prompt_key,
             rec.provider if provider is None else provider,
             float(cost_usd), reason, now, key),
        )
        log.info("工单 %s 重拍（第 %d 次）：%s", key, rec.takes + 1, reason)
        return self.require(key)

    def block(self, key: str, reason: str, failure: FailureKind = FailureKind.MODERATION) -> JobRecord:
        now = time.time()
        self._db.conn().execute(
            "UPDATE jobs SET state='blocked', failure=?, message=?, worker_id='', "
            "lease_until=0, finished_at=?, updated_at=? WHERE key=?",
            (failure.value, reason, now, now, key),
        )
        return self.require(key)

    def cancel(self, key: str, reason: str = "人工取消") -> JobRecord:
        now = time.time()
        self._db.conn().execute(
            "UPDATE jobs SET state='cancelled', message=?, worker_id='', lease_until=0, "
            "finished_at=?, updated_at=? WHERE key=?",
            (reason, now, now, key),
        )
        return self.require(key)

    # -------------------------------------------------- 查询

    def get(self, key: str) -> JobRecord | None:
        row = self._db.conn().execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
        return _row_to_record(row) if row else None

    def require(self, key: str) -> JobRecord:
        rec = self.get(key)
        if rec is None:
            raise KeyError(f"队列里没有工单 {key}")
        return rec

    def jobs(
        self, *, episode: str | None = None, state: JobState | None = None
    ) -> list[JobRecord]:
        sql = "SELECT * FROM jobs"
        cond, args = [], []
        if episode is not None:
            cond.append("episode=?")
            args.append(episode)
        if state is not None:
            cond.append("state=?")
            args.append(state.value)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY priority, created_at"
        return [_row_to_record(r) for r in self._db.conn().execute(sql, args)]

    def pending_count(self, episode: str | None = None) -> int:
        sql = "SELECT COUNT(*) c FROM jobs WHERE state IN ('pending','running')"
        args: list[Any] = []
        if episode is not None:
            sql += " AND episode=?"
            args.append(episode)
        return self._db.conn().execute(sql, args).fetchone()["c"]

    def next_run_gap(self, episode: str | None = None) -> float | None:
        """最近一条待跑工单还要等多久（全在退避窗口里时用来决定睡多长）。"""
        sql = "SELECT MIN(next_run_at) m FROM jobs WHERE state='pending'"
        args: list[Any] = []
        if episode is not None:
            sql += " AND episode=?"
            args.append(episode)
        m = self._db.conn().execute(sql, args).fetchone()["m"]
        return None if m is None else max(0.0, m - time.time())

    def stats(self, episode: str | None = None) -> QueueStats:
        con = self._db.conn()
        where, args = ("WHERE episode=?", [episode]) if episode is not None else ("", [])
        rows = con.execute(
            f"SELECT state, COUNT(*) c FROM jobs {where} GROUP BY state", args
        ).fetchall()
        agg = con.execute(
            f"SELECT COALESCE(SUM(cost_usd),0) c, COALESCE(SUM(est_cost_usd),0) e, "
            f"COALESCE(SUM(attempts),0) a, COALESCE(SUM(takes),0) t, "
            f"COALESCE(SUM(cache_hit),0) h, COUNT(*) n FROM jobs {where}", args
        ).fetchone()
        return QueueStats(
            counts={r["state"]: r["c"] for r in rows}, total=agg["n"],
            cost_usd=agg["c"], est_cost_usd=agg["e"], attempts=agg["a"],
            takes=agg["t"], cache_hits=agg["h"],
        )

    def backoff(self, attempt: int) -> float:
        return min(self.backoff_cap_s, self.backoff_base_s * (2 ** max(0, attempt - 1)))

    def purge(self, episode: str | None = None) -> int:
        """清空（--no-resume 用）。这会丢掉花过钱的记录，CLI 必须二次确认。"""
        with self.lock:
            if episode is None:
                cur = self._db.conn().execute("DELETE FROM jobs")
            else:
                cur = self._db.conn().execute("DELETE FROM jobs WHERE episode=?", (episode,))
        return cur.rowcount

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> JobQueue:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<JobQueue {self.db_path} {self.stats().line()}>"


# ---------------------------------------------------------------- 重拍策略


def derived_seed(base: str, take: int) -> int:
    """(指纹, 轮次) → seed。必须确定性：断点续跑后第 N 次重拍要算出同一个 seed，
    否则恢复时会生成一个全新指纹，绕过缓存白烧一次钱。"""
    return int.from_bytes(hashlib.sha256(f"{base}#{take}".encode()).digest()[:4], "big")


# 激进运镜降级表。第二级重拍要削掉最容易产生形变的运动，
# 但不该一刀切成静止 —— 那等于改写分镜语言。
_CAMERA_DOWNGRADE: dict[CameraMove, CameraMove] = {
    CameraMove.WHIP: CameraMove.PAN_R,
    CameraMove.ORBIT_L: CameraMove.TRUCK_L,
    CameraMove.ORBIT_R: CameraMove.TRUCK_R,
    CameraMove.PUSH_PULL: CameraMove.DOLLY_IN,
    CameraMove.CRANE: CameraMove.PEDESTAL_U,
    CameraMove.HANDHELD: CameraMove.STEADICAM,
    CameraMove.ZOOM_IN: CameraMove.STATIC,
    CameraMove.ZOOM_OUT: CameraMove.STATIC,
}

_ARTIFACT_NEGATIVES = "morphing limbs, flickering texture, duplicated face, warped hands"


@dataclass(frozen=True)
class RetakeStrategy:
    """三级重拍：换 seed → 降级参数 → 换引擎。

    顺序是按「代价从小到大」排的：换 seed 最便宜且最常有效（大多数崩坏是
    采样运气问题）；降级参数会牺牲一点镜头语言；换引擎最贵（可能贵一个数量级，
    还要重过路由），所以放最后。
    """

    min_duration_s: float = 4.0
    shorten_ratio: float = 0.75

    def apply(self, shot: Shot, take: int, *, base: str) -> str:
        """就地改写 shot，返回这次改了什么（写进工单 message 供复盘）。"""
        seed = derived_seed(base, take)
        shot.seed = seed
        if take <= 1:
            return f"换 seed={seed}"
        if take == 2:
            notes = [f"换 seed={seed}"]
            new_move = _CAMERA_DOWNGRADE.get(shot.camera_move)
            if new_move is not None:
                shot.camera_move = new_move
                notes.append(f"运镜降级为 {new_move.name}")
            elif shot.duration_s > self.min_duration_s:
                # 已经是温和运镜还崩，多半是时间太长后半段失控：砍时长。
                old = shot.duration_s
                shot.duration_s = max(self.min_duration_s,
                                      round(shot.duration_s * self.shorten_ratio, 2))
                notes.append(f"时长 {old:g}s → {shot.duration_s:g}s")
            if _ARTIFACT_NEGATIVES not in shot.negative_prompt:
                shot.negative_prompt = ", ".join(
                    x for x in (shot.negative_prompt, _ARTIFACT_NEGATIVES) if x
                )
                notes.append("追加伪影负向词")
            return "降级参数：" + "；".join(notes)
        old_hint = shot.engine_hint
        shot.engine_hint = (
            EngineHint.OPEN if old_hint in (EngineHint.OFFICIAL, EngineHint.HERO)
            else EngineHint.HERO
        )
        return f"换引擎：engine_hint {old_hint.value} → {shot.engine_hint.value}，seed={seed}"


DEFAULT_RETAKE = RetakeStrategy()


# ---------------------------------------------------------------- 质检回调


@dataclass(frozen=True)
class QCOutcome:
    passed: bool
    score: float = 1.0
    verdict: str = "pass"          # pass / retake / escalate
    advice: tuple[str, ...] = ()

    @property
    def escalate(self) -> bool:
        return self.verdict == "escalate"


QCFn = Callable[[Shot, str], QCOutcome]


def default_qc_fn(storyboard: Storyboard | None = None, **gate_kw: Any) -> QCFn:
    """把 qc.QCGate 包成队列认识的回调。

    延迟导入：qc 依赖 numpy 与 ffmpeg 二进制，而队列本身必须在没有它们的
    环境里也能 import（比如只想看看工单状态的运维脚本）。
    """
    holder: dict[str, Any] = {}

    def _fn(shot: Shot, video_uri: str) -> QCOutcome:
        from .qc import QCGate, Verdict

        gate = holder.get("gate")
        if gate is None:
            gate = holder["gate"] = QCGate.default(**gate_kw)
        rep = gate.evaluate(video_uri, shot, storyboard)
        return QCOutcome(
            passed=rep.verdict is Verdict.PASS, score=rep.score,
            verdict=rep.verdict.value, advice=tuple(rep.advice),
        )

    return _fn


# ---------------------------------------------------------------- 进度


@dataclass(frozen=True)
class ProgressEvent:
    kind: str                # enqueued/cached/blocked/claimed/done/retake/failed/waiting
    shot_id: str
    message: str
    stats: QueueStats
    elapsed_s: float = 0.0


ProgressFn = Callable[[ProgressEvent], None]


class ConsoleProgress:
    """给 CLI 用的实时进度。tty 下就地刷新，重定向到文件时逐行追加。

    走注入的 stream 而不是 print：产线里 stdout 可能被占用（比如 ffmpeg 管道），
    进度必须能改道 stderr 或日志文件。
    """

    def __init__(self, stream: Any = None, *, quiet_kinds: Iterable[str] = ("claimed",)) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.quiet = set(quiet_kinds)
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._width = 0
        self._lock = threading.Lock()

    def __call__(self, ev: ProgressEvent) -> None:
        if ev.kind in self.quiet:
            return
        line = f"{ev.stats.line()} | {ev.shot_id} {ev.message}"
        with self._lock:
            if self._tty:
                pad = " " * max(0, self._width - len(line))
                self.stream.write("\r" + line + pad)
                self._width = len(line)
            else:
                self.stream.write(line + "\n")
            self.stream.flush()

    def finish(self, text: str = "") -> None:
        with self._lock:
            if self._tty:
                self.stream.write("\n")
            if text:
                self.stream.write(text + "\n")
            self.stream.flush()


# ---------------------------------------------------------------- 跑一集


@dataclass(frozen=True)
class ShotOutcome:
    shot_id: str
    state: JobState
    provider: str
    render_uri: str
    artifact_hash: str
    cost_usd: float
    takes: int
    attempts: int
    qc_score: float | None
    cache_hit: bool
    message: str

    @classmethod
    def of(cls, rec: JobRecord) -> ShotOutcome:
        return cls(
            shot_id=rec.shot_id, state=rec.state,
            provider=rec.provider_used or rec.provider, render_uri=rec.render_uri,
            artifact_hash=rec.artifact_hash, cost_usd=rec.cost_usd, takes=rec.takes,
            attempts=rec.attempts, qc_score=rec.qc_score, cache_hit=rec.cache_hit,
            message=rec.message,
        )


@dataclass
class EpisodeResult:
    project: str
    episode: str
    outcomes: list[ShotOutcome]
    stats: QueueStats
    elapsed_s: float
    interrupted: bool = False

    @property
    def ok(self) -> bool:
        return all(o.state is JobState.DONE for o in self.outcomes)

    @property
    def cost_usd(self) -> float:
        return self.stats.cost_usd

    def failed(self) -> list[ShotOutcome]:
        return [o for o in self.outcomes if o.state is not JobState.DONE]

    def report(self) -> str:
        lines = [
            f"=== {self.project}/{self.episode} 跑批结果 ===",
            f"{'镜头':<10}{'状态':<10}{'引擎':<14}{'次数':<16}{'产物'}",
        ]
        for o in self.outcomes:
            qc = f" qc={o.qc_score:.2f}" if o.qc_score is not None else ""
            lines.append(
                f"{o.shot_id:<10}{o.state.value:<10}{(o.provider or '-'):<14}"
                f"take={o.takes} try={o.attempts}{'':<3}"
                f"{Path(o.render_uri).name if o.render_uri else '-'}"
                f"{' [缓存]' if o.cache_hit else ''}{qc}"
            )
        lines.append(f"耗时 {self.elapsed_s:.1f}s，花费 ${self.stats.cost_usd:.4f}"
                     f"（预估 ${self.stats.est_cost_usd:.4f}），"
                     f"重拍 {self.stats.takes} 次，缓存命中 {self.stats.cache_hits} 个")
        bad = self.failed()
        if bad:
            lines.append(f"需人工处理 {len(bad)} 镜：")
            lines += [f"  - {o.shot_id} [{o.state.value}] {o.message}" for o in bad]
        if self.interrupted:
            lines.append("！本次运行被中断，已完成的镜头已落库，重跑时会自动续上")
        return "\n".join(lines)


class _ProviderLimiter:
    """按供应商分别限流。

    这点比全局并发重要得多：A 家允许 8 并发、B 家只给 2，用一个全局信号量
    要么把 A 家的额度浪费掉，要么把 B 家打到 429 —— 后者还会连累账号信誉。
    """

    def __init__(self, router: Router | None, default: int) -> None:
        self._router = router
        self._default = max(1, default)
        self._sems: dict[str, threading.Semaphore] = {}
        self._lock = threading.Lock()

    def _limit_of(self, name: str) -> int:
        if self._router is None or name not in self._router.registry:
            return self._default
        caps = self._router.registry.get(name).caps
        quota = self._router.policy.quota(name).max_concurrency
        return max(1, min(caps.max_concurrency, quota or caps.max_concurrency))

    @contextlib.contextmanager
    def slot(self, name: str) -> Iterator[None]:
        with self._lock:
            sem = self._sems.get(name)
            if sem is None:
                sem = self._sems[name] = threading.Semaphore(self._limit_of(name))
        sem.acquire()
        try:
            yield
        finally:
            sem.release()


def run_episode(
    storyboard: Storyboard,
    router: Router,
    *,
    db_path: str | Path | None = None,
    queue: JobQueue | None = None,
    store: ArtifactStore | None = None,
    concurrency: int = 4,
    resume: bool = True,
    qc_fn: QCFn | None = None,
    retake: RetakeStrategy = DEFAULT_RETAKE,
    max_takes: int = 3,
    lease_s: float = 600.0,
    poll_interval_s: float = 1.0,
    timeout_s: float | None = None,
    on_progress: ProgressFn | None = None,
    execute_fn: Callable[[Shot, Storyboard], GenResult] | None = None,
    worker_id: str | None = None,
) -> EpisodeResult:
    """端到端跑一集：规划 → 入队 → 并发执行 → 质检 → 不合格重拍 → 汇总。

    resume=True（默认）时，已完成的镜头**不会**再被执行 —— 这是本函数存在的理由。
    中断（Ctrl-C / 崩溃）不会丢失已完成的工作：每一次状态变化都同步落 SQLite。
    """
    if queue is None:
        if db_path is None:
            raise ValueError("必须给 db_path 或 queue —— 队列状态不能只活在内存里")
        queue = JobQueue(db_path, lease_s=lease_s)
    episode = f"{storyboard.project}/{storyboard.episode}"
    wid = worker_id or f"{socket.gethostname()}:{os.getpid()}"
    execute = execute_fn if execute_fn is not None else (
        lambda shot, sb: router.execute(shot, sb)
    )
    limiter = _ProviderLimiter(router, concurrency)
    t0 = time.monotonic()

    # 独占锁：拿到它就等于「本机没有第二个 run_episode 在跑这个 db」，
    # 这是下面敢无条件回收 running 工单的前提。
    ep_lock = FileLock(queue.db_path.with_suffix(queue.db_path.suffix + ".episode.lock"))
    if not ep_lock.acquire(blocking=False):
        raise RuntimeError(
            f"已有另一个进程在跑 {queue.db_path}；并行跑同一集会重复提交任务、重复扣费"
        )

    def emit(kind: str, shot_id: str, message: str) -> None:
        if on_progress is None:
            return
        on_progress(ProgressEvent(
            kind=kind, shot_id=shot_id, message=message,
            stats=queue.stats(episode), elapsed_s=time.monotonic() - t0,
        ))

    try:
        if not resume:
            n = queue.purge(episode)
            log.info("resume=False：清掉 %d 条旧工单，本集从零开始跑", n)
        else:
            queue.reclaim_running()

        shots_by_key: dict[str, Shot] = {}
        for shot in storyboard.all_shots():
            plan = router.plan(shot, storyboard)
            rec = queue.enqueue(shot, plan, episode=episode, store=store)
            shots_by_key[rec.key] = shot
            if rec.state is JobState.DONE:
                _restore_shot(shot, rec)
                emit("cached" if rec.cache_hit else "enqueued", shot.id,
                     "命中缓存，跳过" if rec.cache_hit else "已完成，跳过")
            elif rec.state is JobState.BLOCKED:
                emit("blocked", shot.id, rec.message)
            else:
                emit("enqueued", shot.id, f"入队 → {rec.provider or '待定'}")

        _drain(
            queue=queue, episode=episode, shots_by_key=shots_by_key, storyboard=storyboard,
            execute=execute, limiter=limiter, concurrency=concurrency, qc_fn=qc_fn,
            retake=retake, max_takes=max_takes, store=store, lease_s=lease_s,
            poll_interval_s=poll_interval_s, timeout_s=timeout_s, wid=wid, emit=emit, t0=t0,
        )
        interrupted = False
    except BaseException:
        # 中断/崩溃：已完成的工单早已落库，这里只负责把锁还回去。
        interrupted = True
        raise
    finally:
        ep_lock.release()
        result = EpisodeResult(
            project=storyboard.project, episode=storyboard.episode,
            outcomes=[ShotOutcome.of(r) for r in queue.jobs(episode=episode)],
            stats=queue.stats(episode), elapsed_s=time.monotonic() - t0,
            interrupted=interrupted if "interrupted" in dir() else True,
        )
    return result


def _restore_shot(shot: Shot, rec: JobRecord) -> None:
    """把工单里的产线结果写回 Shot —— 续跑时后续环节（拼接）要靠它。"""
    shot.render_uri = rec.render_uri or shot.render_uri
    shot.provider_used = rec.provider_used or shot.provider_used
    shot.job_id = rec.job_id or shot.job_id
    shot.qc_score = rec.qc_score if rec.qc_score is not None else shot.qc_score
    shot.takes = max(shot.takes, rec.takes)


def _drain(
    *, queue: JobQueue, episode: str, shots_by_key: Mapping[str, Shot],
    storyboard: Storyboard, execute: Callable[[Shot, Storyboard], GenResult],
    limiter: _ProviderLimiter, concurrency: int, qc_fn: QCFn | None,
    retake: RetakeStrategy, max_takes: int, store: ArtifactStore | None,
    lease_s: float, poll_interval_s: float, timeout_s: float | None,
    wid: str, emit: Callable[[str, str, str], None], t0: float,
) -> None:
    """主循环：领取 → 提交线程池 → 收结果。

    所有 DB 写入都在 worker 线程里完成（WAL 下并发写是安全的），主循环只做调度，
    这样一个慢镜头的质检不会卡住别的镜头的派发。
    """
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="shot")
    inflight: dict[Future[None], JobRecord] = {}
    try:
        while True:
            if timeout_s is not None and time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"本集执行超过 {timeout_s:g}s 仍未跑完")
            free = concurrency - len(inflight)
            if free > 0:
                for rec in queue.claim(wid, free, lease_s=lease_s, episode=episode):
                    shot = shots_by_key[rec.key]
                    emit("claimed", rec.shot_id, f"派给 {rec.provider or '路由决定'}")
                    fut = pool.submit(
                        _run_one, queue=queue, rec=rec, shot=shot, storyboard=storyboard,
                        execute=execute, limiter=limiter, qc_fn=qc_fn, retake=retake,
                        max_takes=max_takes, store=store, emit=emit,
                    )
                    inflight[fut] = rec
            if not inflight:
                if queue.pending_count(episode) == 0:
                    return
                gap = queue.next_run_gap(episode)
                emit("waiting", "-", f"全部在退避窗口中，等待 {gap or 0:.0f}s")
                time.sleep(min(poll_interval_s, gap if gap else poll_interval_s))
                continue
            done, _ = wait(inflight, timeout=poll_interval_s, return_when=FIRST_COMPLETED)
            queue.heartbeat([r.key for r in inflight.values()], lease_s=lease_s)
            for fut in done:
                rec = inflight.pop(fut)
                exc = fut.exception()
                if exc is None:
                    continue
                if isinstance(exc, Exception):
                    # worker 自己炸了（不是 provider 报错）：按不可重试处理，
                    # 让人去看日志，而不是拿同一个 bug 再烧三次额度。
                    log.exception("工单 %s 的执行线程异常", rec.key, exc_info=exc)
                    queue.fail(rec.key, FailureKind.UNKNOWN, f"执行线程异常：{exc}")
                    emit("failed", rec.shot_id, f"线程异常 {exc}")
                    continue
                raise exc  # KeyboardInterrupt / SystemExit：整集中断
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _run_one(
    *, queue: JobQueue, rec: JobRecord, shot: Shot, storyboard: Storyboard,
    execute: Callable[[Shot, Storyboard], GenResult], limiter: _ProviderLimiter,
    qc_fn: QCFn | None, retake: RetakeStrategy, max_takes: int,
    store: ArtifactStore | None, emit: Callable[[str, str, str], None],
) -> None:
    """跑一条工单：生成 → 入库 → 质检 → 判定。异常交给主循环统一处理。"""
    # 执行前**再查一次**缓存。入队时查过一遍，但那时同一批里参数相同的前一镜
    # 往往还没跑完，查不到；等轮到这一镜时它可能已经出片了。不复查就等于
    # 对着同一份工单重复烧一次钱 —— 一集里重复的空镜/插入镜并不罕见。
    hit = queue._lookup_artifact(rec.prompt_key, store)
    if hit is not None:
        a_hash, uri, prov = hit
        queue.complete(rec.key, render_uri=uri, artifact_hash=a_hash,
                       cost_usd=0.0, provider_used=prov,
                       message="命中产物缓存（执行前复查），未提交任务")
        queue.mark_cache_hit(rec.key)
        _restore_shot(shot, queue.require(rec.key))
        emit("done", rec.shot_id, f"命中缓存 {a_hash[:12]}，省下一次生成")
        return

    try:
        with limiter.slot(rec.provider or "-"):
            result = execute(shot, storyboard)
    except ProviderError as exc:
        queue.fail(rec.key, exc.kind, str(exc))
        emit("failed", rec.shot_id, f"{exc.kind.value}：{exc}")
        return

    if not result.ok:
        queue.fail(rec.key, result.failure, result.message, cost_usd=result.cost_usd)
        emit("failed", rec.shot_id, f"{result.failure.value}：{result.message[:60]}")
        return

    render_uri = result.video_uri or ""
    artifact_hash = ""
    if store is not None and Path(render_uri).is_file():
        artifact_hash = store.put(render_uri)
        render_uri = str(store.get(artifact_hash))
        store.link(artifact_hash, f"{storyboard.episode}/{rec.shot_id}_take{rec.takes}")

    qc = qc_fn(shot, render_uri) if qc_fn is not None else QCOutcome(passed=True)
    if qc.passed:
        if store is not None and artifact_hash:
            # 只有过了质检的产物才配进缓存：缓存里的东西是会被别的镜直接拿去用的。
            store.cache.put(rec.prompt_key, artifact_hash,
                            provider=result.provider, cost_usd=result.cost_usd,
                            meta={"shot_id": rec.shot_id, "episode": rec.episode})
        queue.complete(
            rec.key, render_uri=render_uri, artifact_hash=artifact_hash,
            cost_usd=result.cost_usd, provider_used=result.provider,
            job_id=result.job_id, qc_score=qc.score, qc_verdict=qc.verdict,
        )
        _restore_shot(shot, queue.require(rec.key))
        emit("done", rec.shot_id, f"出片 via {result.provider} ${result.cost_usd:.4f}")
        return

    advice = "；".join(qc.advice[:2])
    if rec.takes + 1 >= max_takes or qc.escalate:
        queue.fail(
            rec.key, FailureKind.UNKNOWN,
            f"质检不过（{qc.verdict} score={qc.score:.3f}），重拍 {rec.takes} 次后仍未达标：{advice}",
            cost_usd=result.cost_usd, delay_s=0,
        )
        emit("failed", rec.shot_id, f"质检 {qc.verdict}，转人工")
        return

    change = retake.apply(shot, rec.takes + 1, base=rec.base_fingerprint)
    queue.requeue(
        rec.key, reason=f"质检 {qc.score:.3f} 不达标 → {change}（{advice}）",
        fingerprint=shot.fingerprint(),
        prompt_key=fingerprint_params(shot.fingerprint(), {"take": rec.takes + 1}),
        cost_usd=result.cost_usd,
    )
    emit("retake", rec.shot_id, f"质检 {qc.score:.3f} → 重拍：{change}")


# ---------------------------------------------------------------- 自测


def _selftest() -> None:
    import tempfile

    from .providers.base import Capabilities, JobStatus, ProviderRegistry, VideoProvider
    from .router import ProviderQuota, Router, RoutingPolicy
    from .schema import Appearance, CharacterBible, Scene

    logging.basicConfig(level=logging.ERROR)

    class _FakeProvider(VideoProvider):
        """不依赖 ffmpeg 的假引擎：写几十字节的假视频就算出片。

        产物内容由 (shot_id, seed, prompt) 决定 —— 这样「换 seed 重拍」
        必然得到不同哈希，能真实检验缓存不会张冠李戴。
        """

        def __init__(self, caps: Capabilities, outdir: Path) -> None:
            super().__init__(caps)
            self.outdir = outdir
            outdir.mkdir(parents=True, exist_ok=True)
            self.calls: list[str] = []
            self.peak = 0
            self._live = 0
            self._lock = threading.Lock()
            self._jobs: dict[str, tuple[Path, GenRequest]] = {}

        def submit(self, req: GenRequest) -> str:
            with self._lock:
                self._live += 1
                self.peak = max(self.peak, self._live)
                self.calls.append(req.shot_id)
            time.sleep(0.03)   # 假装是分钟级任务，好观测并发上限
            with self._lock:
                self._live -= 1
            jid = f"{self.name}-{req.shot_id}-{req.seed}"
            p = self.outdir / f"{jid}.mp4"
            p.write_bytes(f"fake|{req.shot_id}|{req.seed}|{req.prompt}".encode())
            self._jobs[jid] = (p, req)
            return jid

        def poll(self, job_id: str) -> GenResult:
            p, req = self._jobs[job_id]
            return GenResult(
                job_id=job_id, status=JobStatus.SUCCEEDED, provider=self.name,
                video_uri=str(p), duration_s=req.duration_s,
                cost_usd=self.estimate_cost(req), submitted_at=1.0, finished_at=2.0,
            )

    def build(td: Path) -> tuple[Storyboard, Router, dict[str, _FakeProvider]]:
        reg = ProviderRegistry()
        fast = _FakeProvider(Capabilities(
            name="fast", kind="official", max_duration_s=16.0, max_ref_images=4,
            supported_image_roles=frozenset({"identity", "environment", "style"}),
            max_concurrency=4, cost_per_second_usd=0.05, quality_tier=4,
        ), td / "fast")
        slow = _FakeProvider(Capabilities(
            name="slow", kind="open", max_duration_s=16.0, max_ref_images=6,
            supported_image_roles=frozenset({"identity", "environment", "style"}),
            moderated=False, max_concurrency=1, cost_per_second_usd=0.01, quality_tier=3,
        ), td / "slow")
        reg.register(fast)
        reg.register(slow)
        policy = RoutingPolicy(quotas={"fast": ProviderQuota(max_concurrency=2)})
        router = Router(reg, policy, sleeper=lambda s: None)

        hero = CharacterBible(
            id="lin", name="林夏", age_statement="虚构角色，设定年龄 28 岁",
            appearance=Appearance(face="鹅蛋脸", hair="黑色长发", wardrobe="米色风衣"),
        )
        shots: list[Shot] = []
        for i in range(10):
            # 第 9 镜刻意做成和第 0 镜完全同参（只差镜号）：验证提示词缓存去重。
            src = 0 if i == 9 else i
            shots.append(Shot(
                id=f"s{i:03d}", scene_id=f"sc{i // 5:02d}", index=i, duration_s=8.0,
                subject_ids=["lin"], action=f"动作 {src}", environment="雨夜天台",
                lighting="霓虹侧逆光", seed=1000 + src,
            ))
        sb = Storyboard(
            project="demo", episode="ep01", characters=[hero],
            scenes=[Scene(id="sc00", shots=shots[:5]), Scene(id="sc01", shots=shots[5:])],
            style_bible="胶片颗粒，冷色调",
        )
        return sb, router, {"fast": fast, "slow": slow}

    with tempfile.TemporaryDirectory(prefix="longfilm-queue-") as td:
        tmp = Path(td)
        db = tmp / "queue.db"
        store = ArtifactStore(tmp / "store")

        # ---------- 1. 崩溃前：跑到第 4 镜时模拟进程被 Ctrl-C
        sb, router, provs = build(tmp / "gen")
        crashed: list[str] = []
        lock = threading.Lock()

        def crashing_execute(shot: Shot, s: Storyboard) -> GenResult:
            with lock:
                if len(crashed) >= 4:
                    raise KeyboardInterrupt("模拟崩溃")
                crashed.append(shot.id)
            return router.execute(shot, s)

        q1 = JobQueue(db, lease_s=30.0)
        try:
            run_episode(sb, router, queue=q1, store=store, concurrency=2,
                        execute_fn=crashing_execute, poll_interval_s=0.05)
            raise AssertionError("应该被 KeyboardInterrupt 打断")
        except KeyboardInterrupt:
            pass
        st1 = q1.stats("demo/ep01")
        done_after_crash = {r.shot_id for r in q1.jobs(episode="demo/ep01")
                            if r.state is JobState.DONE}
        assert 0 < len(done_after_crash) < 10, f"崩溃时应已完成一部分，实际 {done_after_crash}"
        assert st1["running"] + st1["pending"] > 0, "剩下的镜必须还在队列里"
        q1.close()

        # ---------- 2. 断点续跑：已完成的镜一个都不能重跑
        sb2, router2, provs2 = build(tmp / "gen2")
        executed: list[str] = []

        def counting_execute(shot: Shot, s: Storyboard) -> GenResult:
            executed.append(shot.id)
            return router2.execute(shot, s)

        q2 = JobQueue(db, lease_s=30.0)
        progress_lines: list[str] = []

        class _Cap:
            def write(self, s: str) -> int:
                progress_lines.append(s)
                return len(s)

            def flush(self) -> None:
                return None

            def isatty(self) -> bool:
                return False

        res = run_episode(
            sb2, router2, queue=q2, store=store, concurrency=3, resume=True,
            execute_fn=counting_execute, poll_interval_s=0.05,
            on_progress=ConsoleProgress(_Cap()),
        )
        assert res.ok, f"续跑后应全部完成：{res.failed()}"
        assert not (set(executed) & done_after_crash), \
            f"已完成的镜被重跑了：{set(executed) & done_after_crash}"
        assert len(res.outcomes) == 10
        assert progress_lines, "进度回调必须真的被调用"

        # 第 9 镜与第 0 镜同参：要么命中缓存，要么两者产物哈希一致（都=不重复烧钱）
        by_id = {o.shot_id: o for o in res.outcomes}
        assert by_id["s009"].artifact_hash == by_id["s000"].artifact_hash, \
            "同参镜头必须复用同一个产物"
        assert by_id["s009"].cache_hit or by_id["s009"].cost_usd == 0.0 or \
            provs2["fast"].calls.count("s009") + provs2["slow"].calls.count("s009") <= 1
        cache_stats = store.cache.stats()
        assert cache_stats["entries"] >= 1

        # 每一镜只该有一份产物被执行一次（同一次运行内不重复提交）
        all_calls = provs2["fast"].calls + provs2["slow"].calls
        assert len(all_calls) == len(set(all_calls)), f"同一镜被提交了多次：{all_calls}"
        # 按供应商分别限流：fast 被 policy 压到 2，即使全局并发是 3
        assert provs2["fast"].peak <= 2, f"fast 并发超过配额：{provs2['fast'].peak}"

        # ---------- 3. 再跑一次：全命中，零提交
        sb3, router3, provs3 = build(tmp / "gen3")
        again: list[str] = []
        res3 = run_episode(
            sb3, router3, queue=q2, store=store, concurrency=3, resume=True,
            execute_fn=lambda shot, s: again.append(shot.id) or router3.execute(shot, s),
            poll_interval_s=0.05,
        )
        assert res3.ok and not again, f"幂等失效：又跑了 {again}"
        assert res3.stats.cost_usd == res.stats.cost_usd, "续跑不该产生新花费"

        # ---------- 4. 质检不过 → 自动重拍（换 seed → 换引擎），且重拍真的换了参数
        db4 = tmp / "queue4.db"
        sb4, router4, provs4 = build(tmp / "gen4")
        seen_seeds: dict[str, list[int | None]] = {}

        def qc_fn(shot: Shot, video_uri: str) -> QCOutcome:
            seen_seeds.setdefault(shot.id, []).append(shot.seed)
            # s003 的头两次都判不过，第三次放行 —— 走满「换 seed → 降级参数」两级
            n = len(seen_seeds[shot.id])
            if shot.id == "s003" and n <= 2:
                return QCOutcome(passed=False, score=0.31, verdict="retake",
                                 advice=("画面闪烁严重，建议换 seed",))
            return QCOutcome(passed=True, score=0.88)

        q4 = JobQueue(db4, lease_s=30.0)
        res4 = run_episode(sb4, router4, queue=q4, store=ArtifactStore(tmp / "store4"),
                           concurrency=3, qc_fn=qc_fn, max_takes=3, poll_interval_s=0.05)
        rec4 = q4.require(JobQueue.job_key("demo/ep01", "s003"))
        assert rec4.state is JobState.DONE and rec4.takes == 2, f"应重拍两次：{rec4}"
        assert len(set(seen_seeds["s003"])) == 3, f"每次重拍都必须换 seed：{seen_seeds['s003']}"
        shot3 = next(s for s in sb4.all_shots() if s.id == "s003")
        assert "morphing limbs" in shot3.negative_prompt, "第二级重拍要追加伪影负向词"
        assert rec4.cost_usd > 0 and res4.stats.takes == 2

        # 确定性：同一 (指纹, 轮次) 必须算出同一个 seed，否则续跑会白烧钱
        assert derived_seed("abc", 1) == derived_seed("abc", 1)
        assert derived_seed("abc", 1) != derived_seed("abc", 2)

        # ---------- 5. 租约：worker 死掉的任务会自动回到队列
        db5 = tmp / "queue5.db"
        q5 = JobQueue(db5, lease_s=0.15)
        sb5, _, _ = build(tmp / "gen5")
        q5.enqueue(sb5.all_shots()[0], None, episode="ep-lease")
        got = q5.claim("worker-A", 5, episode="ep-lease")
        assert len(got) == 1 and not q5.claim("worker-B", 5, episode="ep-lease"), \
            "租约有效期内不能被第二个 worker 领走"
        time.sleep(0.2)
        again5 = q5.claim("worker-B", 5, episode="ep-lease")
        assert len(again5) == 1 and again5[0].attempts == 2, "租约过期后必须能被接管"

        # ---------- 6. 失败退避与终态
        key = again5[0].key
        r = q5.fail(key, FailureKind.RATE_LIMIT, "429")
        assert r.state is JobState.PENDING and r.next_run_at > time.time(), "限流要退避重试"
        assert not q5.claim("worker-C", 5, episode="ep-lease"), "退避窗口内不该被领走"
        q5.claim("worker-C", 5, episode="ep-lease", lease_s=0)  # 不影响：仍在退避
        r = q5.fail(key, FailureKind.BAD_REQUEST, "参数非法")
        assert r.state is JobState.FAILED, "不可重试的错误要直接进终态"

        print("queue 自测通过：")
        print(res.report())
        print(res4.report())
        print("队列统计：", q2.stats("demo/ep01").line())
        print("缓存统计：", json.dumps(store.cache.stats(), ensure_ascii=False))
        print("进度输出示例：", progress_lines[-1].strip())
        q2.close()
        q4.close()
        q5.close()


if __name__ == "__main__":
    _selftest()
