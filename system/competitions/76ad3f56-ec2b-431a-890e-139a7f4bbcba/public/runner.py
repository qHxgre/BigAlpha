"""把用户子进程的缓存/临时目录重定向到持久卷（提交目录），避免写满容器 ephemeral 层。

HuggingFace / torch / pip 等库默认写 $HOME/.cache（容器 ephemeral 层），单次评测可累积
数十 GB，触发 K8s ephemeral-storage 驱逐把整个 pod 打挂。本类在起子进程前把这些缓存路径
重定向到本提交目录（持久卷 BeeGFS），跑完后统一清掉缓存只保留评分产物。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

import structlog

from judge.judgebase import LocalProcessUserRunner, UserCodeRunError

logger = structlog.get_logger()

SCRATCH_DIRNAMES = (".cache", "tmp")


class ScratchRedirectRunner(LocalProcessUserRunner):
    """在 LocalProcessUserRunner 之上，把缓存/临时目录重定向到持久卷并跑完清理。"""

    def _scratch_env(self) -> dict:
        """构造注入给用户子进程的环境变量，把缓存/临时目录指向持久卷（提交目录）。"""
        env = dict(os.environ)
        cache_root = os.path.join(self.runner_dir, ".cache")
        tmp_dir = os.path.join(self.runner_dir, "tmp")
        hf_home = os.path.join(cache_root, "huggingface")
        env.update({
            "HOME": self.runner_dir,
            "TMPDIR": tmp_dir,
            "HF_HOME": hf_home,
            "HUGGINGFACE_HUB_CACHE": hf_home,
            "HF_HUB_CACHE": hf_home,
            "TRANSFORMERS_CACHE": hf_home,
            "TORCH_HOME": os.path.join(cache_root, "torch"),
            "XDG_CACHE_HOME": cache_root,
            "PIP_CACHE_DIR": os.path.join(cache_root, "pip"),
        })
        for path in (cache_root, tmp_dir, hf_home,
                     os.path.join(cache_root, "torch"), os.path.join(cache_root, "pip")):
            os.makedirs(path, exist_ok=True)
        return env

    def _cleanup_scratch(self) -> None:
        """跑完清理缓存/临时目录，只保留评分产物（parquet / stdout / 状态文件）。"""
        for name in SCRATCH_DIRNAMES:
            target = os.path.join(self.runner_dir, name)
            if not os.path.isdir(target):
                continue
            try:
                shutil.rmtree(target, ignore_errors=True)
            except Exception as e:
                logger.warning(
                    "runner.cleanup_failed",
                    submission_id=self.submission_id,
                    path=target,
                    error=str(e),
                )

    def _run_code(self) -> int:
        """与父类相同，但给 Popen 传入重定向后的 env，跑完清理缓存目录。"""
        env = self._scratch_env()

        process = subprocess.Popen(
            self.cmd,
            cwd=self.runner_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
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
        )

        try:
            self._cleanup_scratch()
        except Exception as e:
            logger.warning("runner.cleanup_error", submission_id=self.submission_id, error=str(e))

        if timed_out:
            raise UserCodeRunError(
                "timeout",
                return_code=process.returncode,
                message=f"user code timed out after {elapsed}s",
            )
        if process.returncode != 0:
            raise UserCodeRunError(
                "user_error",
                return_code=process.returncode,
                message=f"user code exited with code {process.returncode}",
            )
        return process.returncode
