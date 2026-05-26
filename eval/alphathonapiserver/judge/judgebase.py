"""通用比赛评测框架。

设计目标：把原本一个比赛一个 judge-{uuid}.py 脚本里的硬编码（数据集名、时间区间、评分公式、
特例补丁）抽到 CompetitionJudgeConfig，runner 一份代码跑所有比赛。

使用示例：

    from judgebase import (
        AlphathonAPI,
        CompetitionJudgeConfig,
        JudgeRunner,
        LocalProcessUserRunner,
    )

    def csi1000_score(df):
        # 入参是 raw_result 拼成的 pd.DataFrame，需要给每行算出一个 score
        return (
            df["rank_ic"].rank(pct=True) * 0.4
            + df["rank_ir"].rank(pct=True) * 0.3
            + df["sharp_ratio"].rank(pct=True) * 0.2
            + df["turnover"].rank(pct=True, ascending=False) * 0.1
        )

    config = CompetitionJudgeConfig(
        competition_id="5c3f7783-4158-4196-97ab-171b27218c7c",
        runner_code='''__USER_CODE__

def judge_runner_main():
    data = main("cpt_jyc_2025_stock_csi1000_bar1m_test", "2025-01-01", "2025-07-31 23:59:59")
    import pandas as pd
    if data["date"].dtype == "int32":
        data["date"] = pd.to_datetime(data["date"], format="%Y%m%d")
    if data["date"].max().year == 1970:
        data["date"] = pd.to_datetime(data["date"].astype("int"), format="%Y%m%d")

    from bigmodule import M
    result = M.factorlens._latest(data=data, m_cached=False)
    with open("output.data", "w") as writer:
        writer.write(result._result.id)
''',
        score_kind="public",
        score_func=csi1000_score,
    )
    JudgeRunner(config, tick_interval=60).run()
"""

import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx
import structlog

logger = structlog.get_logger()

RUNNER_BASE_DIR = os.getenv("RUNNER_BASE_DIR", "/home/aiuser/work/data/alphathon")
USER_CODE_PLACEHOLDER = "__USER_CODE__"


def _write_file(path: str, content: bytes | str) -> None:
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode=mode) as writer:
        writer.write(content)


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------


class AlphathonAPI:
    """比赛 API 客户端，认证用 cptjudge.jwt。"""

    def __init__(self) -> None:
        self.base_url = os.getenv(
            "ALPHATHON_API_BASE_URL",
            "http://alphathonapiserver.bigquant.svc.cluster.local:8000/bigapis/alphathon/v1",
        )
        self.timeout: float = float(os.getenv("ALPHATHON_API_TIMEOUT", 15.0))
        # 优先用文件中的 token；fallback 到环境变量
        token_path = os.path.join(RUNNER_BASE_DIR, "cptjudge.jwt")
        if os.path.exists(token_path):
            with open(token_path) as f:
                self.api_token = f.read().strip()
        else:
            self.api_token = os.getenv("ALPHATHON_API_TOKEN", "")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_data: Any = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        request_headers: dict[str, str] = {"accept": "application/json"}
        if self.api_token:
            request_headers["Cookie"] = f"bigjwt={self.api_token}"
        if headers:
            request_headers.update(headers)

        url = f"{self.base_url}/{path.lstrip('/')}"
        with httpx.Client(timeout=timeout or self.timeout) as client:
            response = client.request(method.upper(), url, params=params, json=json_data, headers=request_headers)
            response.raise_for_status()
            return response

    def get_competition_by_id(self, competition_id: str | uuid.UUID) -> Optional[dict[str, Any]]:
        params = {
            "constraints": json.dumps({"id": str(competition_id)}),
            "page": 1,
            "size": 1,
        }
        data = self._request("GET", "/competitions", params=params).json()
        items = (data or {}).get("data", {}).get("items", []) if isinstance(data, dict) else []
        return items[0] if items else None

    def query_submissions(
        self,
        *,
        competition_id: str | uuid.UUID | None = None,
        constraints: Optional[dict[str, Any]] = None,
        page_size: int = 5000,
        max_pages: int = 10000,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        page = 1
        while page <= max_pages:
            params: dict[str, Any] = {
                "competition_id": str(competition_id),
                "page": page,
                "size": page_size,
                "order_by": "-created_at",
                "constraints": json.dumps(constraints or {}),
            }
            data = self._request("GET", "/submissions", params=params).json()["data"]
            if not data or not data.get("items"):
                break
            items = data["items"]
            results.extend(items)
            if len(items) < page_size:
                break
            page += 1
        return results

    def get_file_content_of_submission(
        self,
        submission: dict,
        ipynb_to_py: bool = False,
        to_str: bool = False,
        save_to: Optional[str] = None,
    ) -> bytes | str:
        files = submission["data"]["files"]
        if len(files) != 1:
            raise ValueError(f"submission {submission['id']} has {len(files)} files, expected 1")
        file_id, file_info = next(iter(files.items()))
        return self.get_submission_file(submission["id"], file_id, file_info, ipynb_to_py=ipynb_to_py, to_str=to_str, save_to=save_to)

    def get_submission_file(
        self,
        submission_id: str | uuid.UUID,
        file_id: str,
        file_info: Optional[dict] = None,
        ipynb_to_py: bool = False,
        to_str: bool = False,
        save_to: Optional[str] = None,
    ) -> bytes | str:
        response = self._request("GET", f"/submissions/files/{submission_id}/{file_id}")
        raw_content: bytes | str = response.content

        if file_info and file_info.get("name", "").endswith(".ipynb") and ipynb_to_py:
            raw_content = _extract_code_from_ipynb(response.content)

        if to_str and isinstance(raw_content, bytes):
            raw_content = raw_content.decode("utf-8")

        if save_to:
            _write_file(save_to, raw_content)
        return raw_content

    def update_submission_score(self, submission_id: str | uuid.UUID, **json_data) -> dict[str, Any]:
        response = self._request("POST", f"/submissions/{submission_id}", json_data=json_data)
        return response.json()


def _extract_code_from_ipynb(raw: bytes) -> str:
    notebook_data = json.loads(raw.decode("utf-8"))
    code_cells: list[str] = []
    for cell in notebook_data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            code_cells.append("".join(source))
        elif isinstance(source, str):
            code_cells.append(source)
    return "\n\n".join(code_cells)


# ---------------------------------------------------------------------------
# 选手代码 runner
# ---------------------------------------------------------------------------


class UserCodeRunner:
    def __init__(self, submission_id: str, files: dict[str, bytes | str], cmd: list[str]) -> None:
        self.submission_id = submission_id
        self.runner_dir = os.path.join(RUNNER_BASE_DIR, str(submission_id))
        self.files = files
        self.cmd = cmd

    def _pre_run(self) -> None:
        os.makedirs(self.runner_dir, exist_ok=True)
        for name, content in self.files.items():
            _write_file(os.path.join(self.runner_dir, name), content)

    def run(self, _raise: bool = False) -> bool:
        try:
            self._pre_run()
            self._run_code()
            return True
        except Exception as e:
            logger.exception(e)
            if _raise:
                raise
        return False

    def _run_code(self) -> int:
        raise NotImplementedError


class LocalProcessUserRunner(UserCodeRunner):
    """本地子进程执行选手代码，3 小时超时。"""

    DEFAULT_TIMEOUT = 3 * 60 * 60

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
        start = time.time()
        with open(f"{self.runner_dir}/stdout", "w") as writer:
            while process.poll() is None:
                if time.time() - start > self.DEFAULT_TIMEOUT:
                    logger.warning("runner.timeout", submission_id=self.submission_id, elapsed=time.time() - start)
                    process.kill()
                    break
                time.sleep(5)
            for line in process.stdout or []:
                sys.stdout.write(line)
                writer.write(line)
        process.wait()
        return process.returncode


class K8SPodUserRunner(UserCodeRunner):
    """K8s Pod 执行（占位，未实现）。"""


# ---------------------------------------------------------------------------
# 评测器
# ---------------------------------------------------------------------------


@dataclass
class CompetitionJudgeConfig:
    """单场比赛的评测配置。

    runner_code 用 USER_CODE_PLACEHOLDER (`__USER_CODE__`) 标记选手代码插入点。
    runner_code 必须定义 judge_runner_main() 并把结果 DataSource ID 写入 ./output.data。

    score_kind: "public" 或 "private"
    score_func: 可选，把所有 raw_result 拼成的 DataFrame 转换成每行的 score
    code_patches: 选手代码字符串替换（针对个别提交的临时补丁），key=submission_id
    query_constraints: 拉取提交列表时的过滤条件
    completed_ids_file: 持久化已完成的 submission_id，断点续跑用
    """

    competition_id: str
    runner_code: str
    score_kind: str = "public"
    score_func: Optional[Callable[[Any], Any]] = None
    code_patches: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    query_constraints: dict[str, Any] = field(default_factory=dict)
    completed_ids_file: Optional[str] = None
    max_workers: int = 5
    max_pages: int = 10000

    def __post_init__(self) -> None:
        if self.score_kind not in ("public", "private"):
            raise ValueError(f"score_kind must be 'public' or 'private', got {self.score_kind!r}")
        if USER_CODE_PLACEHOLDER not in self.runner_code:
            raise ValueError(f"runner_code must contain placeholder {USER_CODE_PLACEHOLDER!r}")


class JudgeRunner:
    """通用 judge 主循环：拉取提交 → 跑代码 → 写分 → 重排榜单。"""

    def __init__(self, config: CompetitionJudgeConfig, tick_interval: int = 60) -> None:
        self.config = config
        self.tick_interval = tick_interval
        self.api = AlphathonAPI()
        self._completed_ids: set[str] = self._load_completed_ids()
        logger.info(
            "judge.init",
            competition_id=config.competition_id,
            tick_interval=tick_interval,
            completed_loaded=len(self._completed_ids),
        )

    # -- 持久化 ----------------------------------------------------------

    def _load_completed_ids(self) -> set[str]:
        path = self.config.completed_ids_file
        if not path or not os.path.exists(path):
            return set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()

    def _save_completed_ids(self) -> None:
        path = self.config.completed_ids_file
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._completed_ids), f, ensure_ascii=False, indent=2)

    # -- 单个 submission 处理 --------------------------------------------

    def _patch_user_code(self, submission_id: str, code: str) -> str:
        for old, new in self.config.code_patches.get(submission_id, []):
            code = code.replace(old, new)
        return code

    def _build_runner(self, submission: dict) -> LocalProcessUserRunner:
        user_code = self.api.get_file_content_of_submission(submission, ipynb_to_py=True, to_str=True)
        if isinstance(user_code, bytes):
            user_code = user_code.decode("utf-8")
        user_code = self._patch_user_code(submission["id"], user_code)
        return LocalProcessUserRunner(
            submission_id=submission["id"],
            files={"judge_runner.py": self.config.runner_code.replace(USER_CODE_PLACEHOLDER, user_code)},
            cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
        )

    def _read_score_data(self, runner: LocalProcessUserRunner) -> dict:
        import dai  # 延迟导入：仅 judge 环境需要

        with open(os.path.join(runner.runner_dir, "output.data")) as reader:
            datasource_id = reader.read().strip()
        return {"raw_result": dai.DataSource(datasource_id).read().iloc[0].to_dict()}

    def on_submission(self, submission: dict) -> None:
        submission_id = submission["id"]
        logger.info("submission.start", submission_id=submission_id)
        try:
            runner = self._build_runner(submission)
            runner.run(_raise=True)
            score_data = self._read_score_data(runner)
            score: float = -1
        except Exception as e:
            logger.exception("submission.run_failed", submission_id=submission_id, error=str(e))
            score = -2
            score_data = {"err_msg": "run error: check your code / get code templates in [code] tab"}

        score_field = f"{self.config.score_kind}_score"
        score_data_field = f"{self.config.score_kind}_score_data"
        self.api.update_submission_score(
            submission_id=submission_id,
            **{score_field: score, score_data_field: score_data},
        )
        logger.info("submission.scored", submission_id=submission_id, score=score)

    # -- 排名重算 --------------------------------------------------------

    def recompute_ranks(self) -> None:
        """根据 score_func 重新计算所有 submission 的最终分数。"""
        if self.config.score_func is None:
            return
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas.unavailable, skip recompute_ranks")
            return

        all_submissions = self.api.query_submissions(competition_id=self.config.competition_id)
        logger.info("rerank.fetch", count=len(all_submissions))

        score_data_field = f"{self.config.score_kind}_score_data"
        raw_results: list[dict[str, Any]] = []
        for s in all_submissions:
            data = s.get(score_data_field) or {}
            raw_result = data.get("raw_result")
            if not raw_result:
                continue
            raw_result["id"] = s["id"]
            raw_results.append(raw_result)

        if not raw_results:
            return

        df = pd.DataFrame(raw_results)
        df["score"] = self.config.score_func(df)

        score_field = f"{self.config.score_kind}_score"
        for _, row in df.iterrows():
            score = row.score
            if score != score:  # NaN
                score = -2
            self.api.update_submission_score(submission_id=row.id, **{score_field: float(score)})

    # -- 主循环 ----------------------------------------------------------

    def _fetch_pending(self, submitted_ids: set[str]) -> list[dict]:
        new_submissions = self.api.query_submissions(
            competition_id=self.config.competition_id,
            constraints=self.config.query_constraints,
            max_pages=self.config.max_pages,
        )
        pending: list[dict] = []
        for s in new_submissions:
            sid = s.get("id")
            if not sid or sid in submitted_ids or sid in self._completed_ids:
                continue
            pending.append(s)
        return pending

    def run(self) -> None:
        submitted_ids: set[str] = set()
        futures_by_id: dict[str, Future] = {}
        completed_total = 0
        executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        try:
            while True:
                logger.info("tick.start")
                added = 0
                for submission in self._fetch_pending(submitted_ids):
                    sid = submission["id"]
                    submitted_ids.add(sid)
                    futures_by_id[sid] = executor.submit(self.on_submission, submission)
                    added += 1

                done_ids = [sid for sid, f in futures_by_id.items() if f.done()]
                for sid in done_ids:
                    futures_by_id.pop(sid, None)
                    self._completed_ids.add(sid)
                completed_total += len(done_ids)

                running = len(futures_by_id)
                logger.info(
                    "tick.status",
                    added=added,
                    running=running,
                    completed_total=completed_total,
                    completed_persistent=len(self._completed_ids),
                )

                self.recompute_ranks()
                self._save_completed_ids()

                logger.info("tick.sleep", seconds=self.tick_interval)
                time.sleep(self.tick_interval)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# 兼容旧 API：保留 JudgeBase 名字让外部脚本仍可继承（推荐迁移到 JudgeRunner）
# ---------------------------------------------------------------------------


class JudgeBase:
    """Deprecated: 用 JudgeRunner + CompetitionJudgeConfig 替代。"""

    def __init__(self, competition_id: str, tick_interval: int) -> None:
        self.competition_id = competition_id
        self.tick_interval = tick_interval
        self.alphathon_api = AlphathonAPI()
        logger.warning("JudgeBase is deprecated, use JudgeRunner instead")
