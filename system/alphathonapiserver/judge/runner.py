import os
import subprocess
import threading
import time

import structlog

from .utils import write_file
from .paths import FILE_DIR

logger = structlog.get_logger()


class UserCodeRunner:
    def __init__(self, submission_id, files: dict, cmd: list, runner_dir: str = None) -> None:
        self.submission_id = submission_id
        # 运行目录可由外部指定（默认回退到 FILE_DIR/{submission_id}）。
        # judge 会把它指向 FILE_DIR/{competition_id}/submissions/{sid}，
        # 让该提交的原始文件、注入代码、stdout 日志、产物全部收在同一个文件夹下。
        self.runner_dir = runner_dir or os.path.join(FILE_DIR, str(self.submission_id))
        self.files = files
        self.cmd = cmd

    def _pre_run(self) -> None:
        os.makedirs(self.runner_dir, exist_ok=True)
        for name, content in self.files.items():
            write_file(os.path.join(self.runner_dir, name), content)

    def run(self, _raise: bool = False) -> bool:
        try:
            self._pre_run()
            self._run_code()
            return True
        except Exception as e:
            if _raise:
                # 向上抛给调用方，由调用方决定如何记日志，避免在这里再打一遍堆栈造成重复
                raise e from e
            # 仅在吞掉异常（不向上抛）时记一行 error，不打完整 Traceback
            logger.error("[runner] 运行失败", submission_id=str(self.submission_id), error=str(e))
        return False

    def _run_code(self) -> None:
        pass


class LocalProcessUserRunner(UserCodeRunner):
    def _run_code(self) -> int:
        # 用户任务的运行日志只落盘到 {runner_dir}/stdout，绝不写到终端，
        # 终端永远只保留 judge 评估系统自身的日志。
        process = subprocess.Popen(
            self.cmd,
            cwd=self.runner_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        timeout = 3 * 60 * 60
        start = time.time()
        log_path = os.path.join(self.runner_dir, "stdout")

        # 在独立线程里实时把子进程输出抽干并写入日志文件。
        # 边跑边读，避免 stdout 管道缓冲被写满导致用户进程阻塞（之前是跑完才读，存在死锁风险）。
        def _drain() -> None:
            with open(log_path, "w", encoding="utf-8") as writer:
                for line in process.stdout:
                    writer.write(line)
                    writer.flush()

        drain_thread = threading.Thread(target=_drain, daemon=True)
        drain_thread.start()

        while process.poll() is None:
            if time.time() - start > timeout:
                logger.warning(
                    "runner.timeout",
                    submission_id=self.submission_id,
                    elapsed=round(time.time() - start, 1),
                )
                process.kill()
                break
            time.sleep(5)

        process.wait()
        drain_thread.join(timeout=30)
        logger.info(
            "runner.finished",
            submission_id=self.submission_id,
            return_code=process.returncode,
            elapsed=round(time.time() - start, 1),
        )
        return process.returncode


class K8SPodUserRunner(UserCodeRunner):
    pass
