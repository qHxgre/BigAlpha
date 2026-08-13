"""私榜专用的用户代码执行器。

该模块完全位于 private 目录，不复用 public 的 runner 配置。具体内存上限由
private.py 在启动批次时通过 ``MemoryLimitedUserRunner.MEM_LIMIT`` 单独设置。
"""
from __future__ import annotations

import os
import resource
import shutil
import subprocess
import threading
import time

import structlog

from judge.runner import LocalProcessUserRunner, UserCodeRunError

logger = structlog.get_logger()

SCRATCH_DIRNAMES = (".cache", "tmp")
DATA_CACHE_DIR_PREFIX = "bigalpha_memmap_cache"
FSIZE_LIMIT = 200 * 1024**3
OOM_MARKERS = (
    "memoryerror",
    "cannot allocate memory",
    "std::bad_alloc",
    "out of memory",
    "cuda out of memory",
    "outofmemoryerror",
)
LOG_TAIL_LINES = 80


class MemoryLimitedUserRunner(LocalProcessUserRunner):
    """给私榜用户子进程设置资源限制，并保留 OOM/失败日志。"""

    # 默认值仅用于防止绕过 private.py 直接实例化；正式批次由 private.py 覆盖。
    MEM_LIMIT = 400 * 1024**3

    def _read_log_tail(self, log_path: str, n: int = LOG_TAIL_LINES) -> str:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as reader:
                lines = reader.readlines()
        except Exception:
            return ""
        return "".join(lines[-n:]).strip()

    def _is_oom(self, log_tail: str) -> bool:
        low = log_tail.lower()
        return any(marker in low for marker in OOM_MARKERS)

    def _scratch_env(self) -> dict:
        env = dict(os.environ)
        original_home = env.get("HOME") or os.path.expanduser("~")
        cache_root = os.path.join(self.runner_dir, ".cache")
        tmp_dir = os.path.join(self.runner_dir, "tmp")
        hf_home = os.path.join(cache_root, "huggingface")
        env.update({
            "HOME": self.runner_dir,
            "PYTHONUSERBASE": os.path.join(original_home, ".local"),
            "TMPDIR": tmp_dir,
            "HF_HOME": hf_home,
            "HUGGINGFACE_HUB_CACHE": hf_home,
            "HF_HUB_CACHE": hf_home,
            "TRANSFORMERS_CACHE": hf_home,
            "TORCH_HOME": os.path.join(cache_root, "torch"),
            "XDG_CACHE_HOME": cache_root,
            "PIP_CACHE_DIR": os.path.join(cache_root, "pip"),
            "OPENBLAS_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        })
        for path in (
            cache_root,
            tmp_dir,
            hf_home,
            os.path.join(cache_root, "torch"),
            os.path.join(cache_root, "pip"),
        ):
            os.makedirs(path, exist_ok=True)
        return env

    def _rmtree_logged(self, target: str) -> None:
        if not os.path.isdir(target):
            return
        try:
            shutil.rmtree(target, ignore_errors=True)
        except Exception as exc:
            logger.warning(
                "runner.cleanup_failed",
                submission_id=self.submission_id,
                path=target,
                error=str(exc),
            )

    def _cleanup_scratch(self) -> None:
        for name in SCRATCH_DIRNAMES:
            self._rmtree_logged(os.path.join(self.runner_dir, name))
        try:
            entries = os.listdir(self.runner_dir)
        except Exception as exc:
            logger.warning(
                "runner.cleanup_listdir_failed",
                submission_id=self.submission_id,
                path=self.runner_dir,
                error=str(exc),
            )
            return
        for entry in entries:
            target = os.path.join(self.runner_dir, entry)
            if entry.startswith(DATA_CACHE_DIR_PREFIX) and os.path.isdir(target):
                self._rmtree_logged(target)

    def _run_code(self) -> int:
        try:
            return self._run_code_inner()
        finally:
            self._cleanup_scratch()

    def _run_code_inner(self) -> int:
        def _limit_resources() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (self.MEM_LIMIT, self.MEM_LIMIT))
            resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_LIMIT, FSIZE_LIMIT))

        process = subprocess.Popen(
            self.cmd,
            cwd=self.runner_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            preexec_fn=_limit_resources,
            env=self._scratch_env(),
        )
        timeout = 3 * 60 * 60
        start = time.time()
        log_path = os.path.join(self.runner_dir, "stdout")

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
            mem_limit_gib=round(self.MEM_LIMIT / 1024**3, 1),
        )

        if timed_out:
            raise UserCodeRunError(
                "timeout",
                return_code=process.returncode,
                message=f"user code timed out after {elapsed}s",
            )
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
                    message=(
                        "user code out of memory "
                        f"(limit {round(self.MEM_LIMIT / 1024**3, 1)} GiB):\n{log_tail}"
                    ),
                )
            raise UserCodeRunError(
                "user_error",
                return_code=process.returncode,
                message=f"user code exited with code {process.returncode}",
            )
        return process.returncode
