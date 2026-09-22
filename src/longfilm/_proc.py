"""子进程启动的统一加固。

只解决一件事：**父进程被强杀时，别把正在转码的 ffmpeg 留成孤儿。**

`subprocess.run(timeout=...)` 只在父进程还活着时有效 —— 超时了它会 kill 子进程。
但父进程自己被 OOM killer 干掉时来不及做任何清理，子进程就脱离管控继续跑。
这台 4GB 机器上实测留下过一个跑了 18 小时的转码进程，烧满一个核、
把临时目录撑到几个 G，而且没有任何日志能说明它是谁起的。

PR_SET_PDEATHSIG 由内核保证，连 SIGKILL 也绕不过去。
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any, Sequence


def _set_pdeathsig() -> None:
    """在 fork 之后、exec 之前执行：注册「父死则自杀」。"""
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL)   # 1 = PR_SET_PDEATHSIG
    except Exception:
        # 拿不到 libc 就当没有这层保护 —— 它是加固，不是功能前提
        pass


# 非 Linux 上没有 PR_SET_PDEATHSIG，preexec_fn=None 等于不做任何事。
PREEXEC = _set_pdeathsig if sys.platform.startswith("linux") else None


def run(cmd: Sequence[str], **kw: Any) -> subprocess.CompletedProcess:
    """subprocess.run 的替身，默认带父死子亡保护。

    注意 preexec_fn 在多线程程序里理论上不安全（fork 与 exec 之间只能调
    async-signal-safe 的函数）。这里只调一次 prctl，满足这个约束。
    """
    kw.setdefault("preexec_fn", PREEXEC)
    return subprocess.run(cmd, **kw)


def popen(cmd: Sequence[str], **kw: Any) -> subprocess.Popen:
    kw.setdefault("preexec_fn", PREEXEC)
    return subprocess.Popen(cmd, **kw)


def _selftest() -> None:
    import os
    import signal
    import time

    code = (
        "import subprocess,sys;sys.path.insert(0,'src');"
        "from longfilm._proc import PREEXEC;"
        "p=subprocess.Popen(['sleep','300'],preexec_fn=PREEXEC);"
        "print(p.pid,flush=True);import time;time.sleep(60)"
    )
    par = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    child = int(par.stdout.readline().strip())
    time.sleep(0.4)
    os.kill(par.pid, signal.SIGKILL)          # 模拟 OOM killer
    time.sleep(1.0)
    alive = os.path.exists(f"/proc/{child}")
    assert not alive, f"父被 SIGKILL 后子进程 {child} 仍存活，保护没生效"
    print("_proc 自测通过：父进程被 SIGKILL 时，子进程被内核连带终止")


if __name__ == "__main__":
    _selftest()
