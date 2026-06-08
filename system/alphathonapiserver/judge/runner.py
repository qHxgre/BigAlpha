import os
import subprocess
import sys
import time

import structlog

from _io import write_file
from paths import RUNNER_BASE_DIR

logger = structlog.get_logger()


class UserCodeRunner:
    def __init__(self, submission_id, files: dict, cmd: list) -> None:
        self.submission_id = submission_id
        self.runner_dir = os.path.join(RUNNER_BASE_DIR, str(self.submission_id))
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
            logger.exception(e)
            if _raise:
                raise e from e
        return False

    def _run_code(self) -> None:
        pass


class LocalProcessUserRunner(UserCodeRunner):
    def _run_code(self) -> int:
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
        with open(f"{self.runner_dir}/stdout", "w") as writer:
            while process.poll() is None:
                if time.time() - start > timeout:
                    print(f"任务超时，提交id: {self.submission_id}，运行时长：{time.time() - start}")
                    process.kill()
                    break
                time.sleep(5)
            for line in process.stdout:
                sys.stdout.write(line)
                writer.write(line)

        process.wait()
        return process.returncode


class K8SPodUserRunner(UserCodeRunner):
    pass
