"""本比赛专用的用户代码执行器（带内存上限 + OOM 识别与日志留存）。

背景：框架的 LocalProcessUserRunner 直接 subprocess.Popen 起用户子进程，只有超时保护、
没有任何内存限制。judge 主进程与所有用户子进程共享同一台主机（256 GiB），一个申请超大
内存的提交会吃光内存、触发内核 OOM，把 judge 主进程和其它正在跑的评测一起打挂。

本类只在比赛目录内继承并重写 _run_code，不改动 alphathonapiserver 框架代码：
    1. 用 preexec_fn + RLIMIT_AS 给用户子进程设虚拟内存上限，超限时只有该子进程自身
       malloc 失败（抛 MemoryError 非 0 退出），不会波及 judge 主进程与其它评测；
    2. 子进程非 0 退出时，扫描其 stdout 日志尾部识别是否为内存溢出（MemoryError /
       Cannot allocate memory / std::bad_alloc / CUDA out of memory 等），是则以
       reason="oom" 抛出，并把日志尾部一并带出，供上层单独记状态与留存日志。
"""
from __future__ import annotations

import os
import resource
import subprocess
import threading
import time

import structlog

from judge.runner import LocalProcessUserRunner, UserCodeRunError

logger = structlog.get_logger()

# 识别「内存溢出」的日志特征串（大小写不敏感匹配）。
# 覆盖主机内存溢出（RLIMIT_AS 触发的 MemoryError / glibc 的 Cannot allocate memory /
# C++ 的 std::bad_alloc）与 GPU 显存溢出（CUDA out of memory）。
OOM_MARKERS = (
    "memoryerror",
    "cannot allocate memory",
    "std::bad_alloc",
    "out of memory",
    "cuda out of memory",
    "outofmemoryerror",
)

# 抓取子进程 stdout 末尾多少行作为失败日志（既够定位又不至于把状态文件撑爆）。
LOG_TAIL_LINES = 80


class MemoryLimitedUserRunner(LocalProcessUserRunner):
    """在框架 LocalProcessUserRunner 之上，给用户子进程加内存上限并识别 OOM。"""

    # 单个用户子进程的虚拟内存（地址空间）上限。
    # max_workers=2：2 × 100 GiB = 200 GiB，系统与 judge 主进程留 ~56 GiB 余量。
    # 取值偏大是有意为之：RLIMIT_AS 限的是虚拟地址空间，torch/numpy 启动即 mmap
    # 一大片虚拟地址（未必真占物理内存），设紧会误杀正常大库。
    # 调整 max_workers 时需同步改此值，保证 max_workers * MEM_LIMIT 稳稳小于 256 GiB。
    MEM_LIMIT = 100 * 1024**3  # 100 GiB

    def _read_log_tail(self, log_path: str, n: int = LOG_TAIL_LINES) -> str:
        """读取子进程 stdout 日志的末尾 n 行，供失败时留存 / 排查。读不出返回空串。"""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as reader:
                lines = reader.readlines()
        except Exception:
            return ""
        return "".join(lines[-n:]).strip()

    def _is_oom(self, log_tail: str) -> bool:
        """根据日志尾部判断该次失败是否为内存溢出。"""
        low = log_tail.lower()
        return any(marker in low for marker in OOM_MARKERS)

    def _run_code(self) -> int:
        def _limit_memory() -> None:
            # 在子进程 fork 之后、exec 之前执行，只作用于用户子进程自身。
            resource.setrlimit(resource.RLIMIT_AS, (self.MEM_LIMIT, self.MEM_LIMIT))

        # 用户任务的运行日志只落盘到 {runner_dir}/stdout，绝不写到终端。
        process = subprocess.Popen(
            self.cmd,
            cwd=self.runner_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            preexec_fn=_limit_memory,
        )
        timeout = 3 * 60 * 60
        start = time.time()
        log_path = os.path.join(self.runner_dir, "stdout")

        # 在独立线程里实时把子进程输出抽干并写入日志文件，避免管道缓冲写满导致用户进程阻塞。
        def _drain() -> None:
            with open(log_path, "w", encoding="utf-8") as writer:
                for line in process.stdout:
                    writer.write(line)
                    writer.flush()

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

        timed_out = False
        while process.poll() is None:
            if time.time() - start > timeout:
                logger.warning(
                    "runner.timeout",
                    submission_id=self.submission_id,
                    elapsed=round(time.time() - start, 1),
                )
                timed_out = True
                process.kill()
                break
            time.sleep(5)

        process.wait()
        drain_thread.join(timeout=30)
        elapsed = round(time.time() - start, 1)
        logger.info(
            "runner.finished",
            submission_id=self.submission_id,
            return_code=process.returncode,
            elapsed=elapsed,
        )

        # 超时：被 judge 主动 kill，归类为 timeout。
        if timed_out:
            raise UserCodeRunError(
                "timeout",
                return_code=process.returncode,
                message=f"user code timed out after {elapsed}s",
            )
        # 非 0 退出：先看是不是内存溢出，是则单独归类为 oom 并带出日志尾部；否则按 user_error。
        if process.returncode != 0:
            log_tail = self._read_log_tail(log_path)
            if self._is_oom(log_tail):
                logger.warning(
                    "runner.oom",
                    submission_id=self.submission_id,
                    return_code=process.returncode,
                    mem_limit_gib=round(self.MEM_LIMIT / 1024**3, 1),
                    log_tail=log_tail,
                )
                raise UserCodeRunError(
                    "oom",
                    return_code=process.returncode,
                    message=f"user code out of memory (limit {round(self.MEM_LIMIT / 1024**3, 1)} GiB):\n{log_tail}",
                )
            raise UserCodeRunError(
                "user_error",
                return_code=process.returncode,
                message=f"user code exited with code {process.returncode}",
            )
        return process.returncode
