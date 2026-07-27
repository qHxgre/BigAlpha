"""本比赛专用的用户代码执行器（带内存上限 + OOM 识别与日志留存）。

背景：框架的 LocalProcessUserRunner 直接 subprocess.Popen 起用户子进程，只有超时保护、
没有任何内存限制。judge 主进程与所有用户子进程共享同一台主机（256 GiB），一个申请超大
内存的提交会吃光内存、触发内核 OOM，把 judge 主进程和其它正在跑的评测一起打挂。

本类只在比赛目录内继承并重写 _run_code，不改动 alphathonapiserver 框架代码：
    1. 用 preexec_fn + RLIMIT_AS 给用户子进程设虚拟内存上限，超限时只有该子进程自身
       malloc 失败（抛 MemoryError 非 0 退出），不会波及 judge 主进程与其它评测；
       同一 preexec_fn 里再加 RLIMIT_FSIZE，限制单个文件写出大小，作为磁盘写满的第一道闸；
    2. 子进程非 0 退出时，扫描其 stdout 日志尾部识别是否为内存溢出（MemoryError /
       Cannot allocate memory / std::bad_alloc / CUDA out of memory 等），是则以
       reason="oom" 抛出，并把日志尾部一并带出，供上层单独记状态与留存日志。
    3. 起子进程前，把所有会写大文件的缓存/临时目录（HuggingFace / torch / pip / TMPDIR / $HOME）
       强制重定向到本提交目录下（持久卷 BeeGFS），而非容器可写层（ephemeral）。端到端赛道跑的是
       陌生用户提交，无法指望用户自己 export 这些变量，故由平台在起进程时注入 env 兜底。这样
       ephemeral 占用能压回几百 MB，不再触发 K8s 的 ephemeral-storage 驱逐（把整个 pod 连同
       judge 主进程一起打挂）；
    4. 子进程结束后（无论成功/失败），清掉这些缓存/临时目录，只保留产物（parquet / stdout /
       状态文件）。模型权重等缓存可重新下载，没必要长期占用持久卷（该卷已接近写满）。
    5. 用户代码里调用平台数据接口（dai.query 等）时，还会在当前工作目录（即本提交目录）下
       自行产出查询结果缓存（bigalpha_memmap_cache_ 前缀命名，如
       bigalpha_memmap_cache_P4_raw_trunk_gated_relative_aux/），不在 .cache/tmp 下，
       SCRATCH_DIRNAMES 扫不到。跑完清理时按这个前缀额外扫一遍删除；除此之外 runner_dir
       根下的其它目录不动（不确定是否为可清理的中间产物，误删风险大于占用磁盘的成本）。
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

# 起子进程前，重定向到本提交目录下的缓存/临时子目录名；跑完统一按这些名字清理。
# 相对 runner_dir，故各提交天然隔离、互不干扰。
SCRATCH_DIRNAMES = (".cache", "tmp")

# 用户代码调用平台数据接口（dai.query 等）时，在 runner_dir 根下自行产出的查询结果缓存目录
# 前缀；目录名后半段是因子/脚本名，不固定，故用前缀匹配。跑完清理时按此前缀一并删除。
DATA_CACHE_DIR_PREFIX = "bigalpha_memmap_cache"

# 单个文件写出大小上限（RLIMIT_FSIZE，字节）。限的是「单文件」不是「总量」，
# 目的是拦住失控写出的超大文件（如把整张表 dump 成一个几百 GB 的文件），
# 而不误伤正常产物（模型权重分片、parquet 一般远小于此）。总量控制靠跑完清理。
FSIZE_LIMIT = 200 * 1024**3  # 200 GiB / 单文件

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

    def _scratch_env(self) -> dict:
        """构造注入给用户子进程的环境变量：把所有会写大文件的缓存/临时目录重定向到
        本提交目录（持久卷）下，避免写满容器 ephemeral 层触发 pod 驱逐。

        继承 judge 主进程的完整环境，只覆盖缓存/临时相关的键，并提前把目录建好
        （HuggingFace/pip 等库遇到不存在的父目录时行为不一，先建好最稳妥）。
        """
        env = dict(os.environ)
        cache_root = os.path.join(self.runner_dir, ".cache")
        tmp_dir = os.path.join(self.runner_dir, "tmp")
        hf_home = os.path.join(cache_root, "huggingface")
        env.update({
            # $HOME 兜底：很多库（含未显式支持 HF_HOME 的老版本）按 ~/.cache 落缓存。
            "HOME": self.runner_dir,
            "TMPDIR": tmp_dir,
            # HuggingFace：新老版本认的 key 不同，一并设上覆盖全。
            "HF_HOME": hf_home,
            "HUGGINGFACE_HUB_CACHE": hf_home,
            "HF_HUB_CACHE": hf_home,
            "TRANSFORMERS_CACHE": hf_home,
            # torch.hub / 预训练权重
            "TORCH_HOME": os.path.join(cache_root, "torch"),
            # XDG 规范缓存根（部分库按此落盘）
            "XDG_CACHE_HOME": cache_root,
            # pip 运行时缓存
            "PIP_CACHE_DIR": os.path.join(cache_root, "pip"),
        })
        for path in (cache_root, tmp_dir, hf_home,
                     os.path.join(cache_root, "torch"), os.path.join(cache_root, "pip")):
            os.makedirs(path, exist_ok=True)
        return env

    def _cleanup_scratch(self) -> None:
        """跑完清理本提交目录下的缓存/临时目录，只保留产物（parquet/stdout/状态文件）。

        模型权重等缓存可重新下载，没必要长期占用持久卷（该卷已接近写满）。
        清理失败只记日志、不影响主流程（产物已落盘，评分不受影响）。

        除了固定名字的 SCRATCH_DIRNAMES，还按 DATA_CACHE_DIR_PREFIX 前缀扫一遍 runner_dir
        根下的目录：这是用户代码调用平台数据接口时自行产出的查询结果缓存，名字后半段
        （因子/脚本名）不固定，只有前缀固定，故用前缀匹配。除此之外根下的其它目录不动——
        不确定是否为可清理的中间产物，宁可留着占空间，不误删。
        """
        for name in SCRATCH_DIRNAMES:
            self._rmtree_logged(os.path.join(self.runner_dir, name))

        try:
            entries = os.listdir(self.runner_dir)
        except Exception as e:
            logger.warning(
                "runner.cleanup_listdir_failed",
                submission_id=self.submission_id,
                path=self.runner_dir,
                error=str(e),
            )
            return
        for entry in entries:
            if not entry.startswith(DATA_CACHE_DIR_PREFIX):
                continue
            target = os.path.join(self.runner_dir, entry)
            if os.path.isdir(target):
                self._rmtree_logged(target)

    def _rmtree_logged(self, target: str) -> None:
        """删除一个目录，失败只记日志、不影响主流程；目录不存在则跳过。"""
        if not os.path.isdir(target):
            return
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
        """跑用户子进程，无论成功/超时/OOM/报错，结束后都清理缓存/临时目录。

        清理放在 finally：_run_code_inner 在超时/非 0 退出时会抛 UserCodeRunError，
        产物此时已落盘，缓存/临时目录不再需要，清掉以免长期占用持久卷。
        """
        try:
            return self._run_code_inner()
        finally:
            self._cleanup_scratch()

    def _run_code_inner(self) -> int:
        def _limit_resources() -> None:
            # 在子进程 fork 之后、exec 之前执行，只作用于用户子进程自身。
            resource.setrlimit(resource.RLIMIT_AS, (self.MEM_LIMIT, self.MEM_LIMIT))
            # 单文件写出上限：拦住失控的超大文件写出（磁盘写满的第一道闸）。
            resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE_LIMIT, FSIZE_LIMIT))

        # 缓存/临时目录重定向到持久卷（本提交目录下），不落容器 ephemeral 层。
        env = self._scratch_env()

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
            preexec_fn=_limit_resources,
            env=env,
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
