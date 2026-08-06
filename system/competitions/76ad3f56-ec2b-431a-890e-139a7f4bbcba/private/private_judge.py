"""最小化私榜评测器：单次执行、全部落盘、人工确认后发布。"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from judge.judgebase import JudgeBase, LocalProcessUserRunner, setup_judge_logging
from judge.paths import FILE_DIR

from fileio import jsonable, update_manifest, write_json, write_pending_publish
from regression import run_regression
from scoring import compute_a_scores, compute_final_scores
from templates import build_sfa_runner


HERE = os.path.dirname(os.path.abspath(__file__))
COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
PRIVATE_FILES_DIR = os.path.join(FILE_DIR, COMPETITION_ID, "private")


def _notebook_code(path: str) -> str:
    with open(path, encoding="utf-8") as reader:
        notebook = json.load(reader)
    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            code_cells.append("".join(source))
        elif isinstance(source, str):
            code_cells.append(source)
    return "\n\n".join(code_cells)


def _load_submission_code(source_dir: str, record: dict, sid: str) -> tuple[str, str]:
    submission = record.get("submission") or {}
    submission_data = submission.get("data") or {}
    # API 的 submission 列表在不同版本中可能把 files 放在 data 内或顶层。
    # prepare_submissions.py 下载时已经兼容了两种结构，这里必须使用相同规则，
    # 否则顶层 files 的提交会在文件已经固化后仍被误判为 0 个 notebook。
    submission_files = submission_data.get("files")
    if submission_files is None:
        submission_files = submission.get("files")
    if not isinstance(submission_files, dict):
        raise RuntimeError(
            f"submission {sid} 的 files 字段类型错误: "
            f"{type(submission_files).__name__}"
        )
    notebooks = [
        (str(file_id), file_info)
        for file_id, file_info in submission_files.items()
        if (file_info or {}).get("name", "").endswith(".ipynb")
    ]
    if len(notebooks) != 1:
        raise RuntimeError(
            f"submission {sid} 包含 {len(notebooks)} 个 notebook，要求恰好 1 个"
        )

    file_id, file_info = notebooks[0]
    saved_by_id = {
        str(item.get("file_id")): item.get("name")
        for item in record.get("files") or []
        if item.get("file_id") is not None
    }
    saved_name = saved_by_id.get(file_id)
    if not saved_name:
        raise RuntimeError(f"submission {sid} 的 notebook {file_id} 未保存到输入包")
    path = os.path.join(source_dir, saved_name)
    if not os.path.isfile(path):
        raise RuntimeError(f"submission {sid} 缺少 notebook 文件: {path}")
    return _notebook_code(path), (file_info or {}).get("name") or saved_name


class PrivateJudge(JudgeBase):
    competition_id = COMPETITION_ID
    mode = "private"
    max_workers = 5

    DATASETS = {
        "bar1m": "bigalpha_2026_stock_bar1m_private",
        "financial": "bigalpha_2026_financial_private",
    }
    DATE_START = "2025-01-01 00:00:00"
    DATE_END = ""
    RUNS_DIR = os.path.join(PRIVATE_FILES_DIR, "runs")

    def __init__(
        self,
        input_dir: str,
        *,
        batch_id: str,
        resume: bool,
        rerun_submission_ids: list[str] | None,
        max_workers: int,
    ) -> None:
        super().__init__()
        self.input_dir = os.path.abspath(input_dir)
        self.input_metadata_path = os.path.join(self.input_dir, "metadata.json")
        if not os.path.isfile(self.input_metadata_path):
            raise RuntimeError(f"private 输入包缺少 metadata.json: {self.input_dir}")
        self.batch_id = str(batch_id).strip()
        if not self.batch_id:
            raise RuntimeError("batch_id 不能为空")
        self.run_dir = os.path.join(self.RUNS_DIR, self.batch_id)
        self.resume = bool(resume)
        rerun_ids = [str(sid).strip() for sid in (rerun_submission_ids or [])]
        if any(not sid for sid in rerun_ids):
            raise RuntimeError("rerun_submission_ids 不能包含空 ID")
        if len(rerun_ids) != len(set(rerun_ids)):
            raise RuntimeError("rerun_submission_ids 包含重复 ID")
        self.rerun_submission_ids = set(rerun_ids)
        if self.rerun_submission_ids and not self.resume:
            raise RuntimeError("RERUN_SUBMISSION_IDS 非空时必须设置 RESUME = True")
        try:
            self.max_workers = int(max_workers)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("max_workers 必须是正整数") from exc
        if self.max_workers < 1:
            raise RuntimeError("max_workers 必须是正整数")
        if os.path.exists(self.run_dir) and not self.resume:
            raise RuntimeError(
                f"批次目录已存在，拒绝覆盖: {self.run_dir}。"
                "如需断点续跑，请在 private.py 中保持相同 BATCH_ID 并设置 RESUME = True"
            )
        if self.resume and not os.path.isdir(self.run_dir):
            raise RuntimeError(f"要续跑的批次目录不存在: {self.run_dir}")

        self.submission_dir = os.path.join(self.run_dir, "submissions")
        self.artifact_dir = os.path.join(self.run_dir, "artifacts")
        self.log_dir = os.path.join(self.run_dir, "logs")
        for path in (
            self.submission_dir,
            self.artifact_dir,
            self.log_dir,
        ):
            os.makedirs(path, exist_ok=True)
        setup_judge_logging(os.path.join(self.log_dir, "judge_private.log"))

        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.pending_path = os.path.join(self.run_dir, "pending_publish.jsonl")
        if self.resume:
            if not os.path.isfile(self.manifest_path):
                raise RuntimeError(f"续跑批次缺少 manifest.json: {self.run_dir}")
            with open(self.manifest_path, encoding="utf-8") as reader:
                previous_manifest = json.load(reader)
            if previous_manifest.get("published"):
                raise RuntimeError("已发布批次不允许续跑或重跑，请创建新批次")
            # 同一批次必须沿用首次启动时确定的评估结束日，避免跨日续跑改变口径。
            self.DATE_END = previous_manifest.get("date_end") or self.DATE_END

    def _reset_submission(self, submission: dict) -> None:
        """删除指定 submission 的旧运行产物，使本次续跑必定重新执行。"""
        sid = str(submission["id"])
        path = os.path.abspath(self.submission_path(submission))
        submission_root = os.path.abspath(self.submission_dir)
        if os.path.commonpath([path, submission_root]) != submission_root or path == submission_root:
            raise RuntimeError(f"submission {sid} 的运行目录越界: {path}")
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        self.log.info(
            "submission.rerun_reset",
            submission_id=sid,
            path=path,
            msg="已清理旧产物，将强制重跑 submission",
        )

    def _prepare_submission(self, submission: dict, record: dict) -> tuple[str, str]:
        """把固化输入复制到运行目录；评测不再访问 submission API。"""
        sid = str(submission["id"])
        source_dir = os.path.join(self.input_dir, record["relative_path"])
        runner_dir = self.submission_path(submission)
        if not os.path.isdir(source_dir):
            raise RuntimeError(f"submission {sid} 的固化目录不存在: {source_dir}")
        # 与公榜先 save_submission_files 再执行代码一致：即使 notebook 不合规，
        # 也保留该 submission 的原始文件，方便审查失败原因。
        shutil.copytree(source_dir, runner_dir, dirs_exist_ok=True)
        code, code_file = _load_submission_code(source_dir, record, sid)
        code = self.preprocess_user_code(submission, code)
        return code, code_file

    def _run_submission(
        self,
        submission: dict,
        record: dict,
        *,
        position: int,
        total: int,
        progress: dict[str, int],
        progress_lock: threading.Lock,
    ) -> dict:
        sid = str(submission["id"])
        row = {"submission_id": sid, "user_id": submission.get("user_id")}
        started_at = datetime.now().astimezone()
        started = time.monotonic()
        with progress_lock:
            progress["running"] += 1
            started_progress = dict(progress)
        self.log.info(
            "submission.start",
            submission_id=sid,
            position=position,
            total=total,
            completed=started_progress["completed"],
            running=started_progress["running"],
            remaining=total - started_progress["completed"] - started_progress["running"],
            started_at=started_at.isoformat(timespec="seconds"),
            msg="开始评测 submission",
        )
        # 单条提交无论在哪个准备步骤失败，都要有独立结果目录和 result.json，
        # 不能因写失败结果时目录不存在而反过来中断整个批次。
        os.makedirs(self.submission_path(submission), exist_ok=True)
        try:
            code, row["code_file"] = self._prepare_submission(submission, record)
            runner = LocalProcessUserRunner(
                submission_id=sid,
                files={
                    "judge_runner.py": build_sfa_runner(
                        code, self.DATASETS, self.DATE_START, self.DATE_END
                    )
                },
                cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
                runner_dir=self.submission_path(submission),
            )
            runner.run(_raise=True)
            with open(os.path.join(runner.runner_dir, "factor_analyze.json"), encoding="utf-8") as reader:
                row.update(json.load(reader))
            row["status"] = "success"
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            error_type = type(exc).__name__
        elapsed_seconds = round(time.monotonic() - started, 3)
        row["elapsed_seconds"] = elapsed_seconds
        row["started_at"] = started_at.isoformat(timespec="seconds")
        row["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        # 将包含耗时和时间戳的最终结果落盘。
        write_json(os.path.join(self.submission_path(submission), "result.json"), row)
        with progress_lock:
            progress["running"] -= 1
            progress["completed"] += 1
            finished_progress = dict(progress)
        remaining = total - finished_progress["completed"] - finished_progress["running"]
        if row["status"] == "failed":
            self.log.warning(
                "submission.failed",
                submission_id=sid,
                position=position,
                total=total,
                completed=finished_progress["completed"],
                running=finished_progress["running"],
                remaining=remaining,
                elapsed_seconds=elapsed_seconds,
                error_type=error_type,
                error=row["error"],
                msg="submission 评测失败，继续处理后续提交",
            )
        self.log.info(
            "submission.finish",
            submission_id=sid,
            status=row["status"],
            position=position,
            total=total,
            completed=finished_progress["completed"],
            running=finished_progress["running"],
            remaining=remaining,
            elapsed_seconds=elapsed_seconds,
            finished_at=row["finished_at"],
            msg="submission 评测结束",
        )
        return row

    def _completed_result(self, submission: dict) -> dict | None:
        """读取已完整落盘的单条结果；缺失或损坏时返回 None 并重新执行。"""
        sid = str(submission["id"])
        path = os.path.join(self.submission_path(submission), "result.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as reader:
                row = json.load(reader)
        except (OSError, json.JSONDecodeError):
            return None
        if str(row.get("submission_id")) != sid:
            return None
        if row.get("status") not in {"success", "failed"}:
            return None
        return row

    def _save_scores(self, rows: list[dict]) -> None:
        a_scores = compute_a_scores(rows)
        b_scores = (
            run_regression(
                a_scores,
                self.submission_dir,
                self.artifact_dir,
                self.DATE_START,
                self.DATE_END,
                self.log,
            )
            if not a_scores.empty
            else {}
        )
        if not a_scores.empty:
            a_scores.to_csv(os.path.join(self.artifact_dir, "leaderboard_sfa.csv"), index=False)

        final = compute_final_scores(rows, a_scores, b_scores)
        final.to_csv(os.path.join(self.artifact_dir, "leaderboard_final.csv"), index=False)
        pd.DataFrame(rows).merge(
            final, on="submission_id", how="left", suffixes=("", "_calculated")
        ).to_csv(os.path.join(self.artifact_dir, "submissions_summary.csv"), index=False)

        by_id = {str(row["submission_id"]): row for row in rows}
        pending = []
        for record in final.to_dict("records"):
            sid = str(record["submission_id"])
            source = by_id[sid]
            score = float(record["final_score"])
            score_data = (
                {"err_msg": source.get("error", "private evaluation failed")}
                if score == -2.0
                else jsonable(record)
            )
            pending.append(
                {
                    "submission_id": sid,
                    "payload": {"private_score": score, "private_score_data": score_data},
                }
            )
        write_pending_publish(self.pending_path, pending)

    def run_once(self) -> None:
        with open(self.input_metadata_path, encoding="utf-8") as reader:
            metadata = json.load(reader)
        if metadata.get("competition_id") != self.competition_id:
            raise RuntimeError("private 输入包的 competition_id 不匹配")
        records = metadata.get("submissions") or []
        prepared_ids = [str(record["submission_id"]) for record in records]
        if len(prepared_ids) != len(set(prepared_ids)):
            raise RuntimeError("private 输入包包含重复的 submission_id")

        current_selected = self.alphathon_api.query_submissions(
            competition_id=self.competition_id,
            constraints={"selected_for_private": True},
        )
        current_ids = [str(submission["id"]) for submission in current_selected]
        prepared_id_set = set(prepared_ids)
        current_id_set = set(current_ids)
        if len(current_ids) != len(current_id_set):
            raise RuntimeError("API 返回了重复的 selected_for_private submission_id")
        if prepared_id_set != current_id_set:
            newly_selected = sorted(current_id_set - prepared_id_set)
            no_longer_selected = sorted(prepared_id_set - current_id_set)
            raise RuntimeError(
                "线上 private submission 与固化输入包不一致，拒绝评测: "
                f"线上新增={newly_selected}, 已取消选择={no_longer_selected}, "
                f"线上数量={len(current_ids)}, 输入包数量={len(prepared_ids)}。"
                "请重新运行 prepare_submissions.py。"
            )

        submissions = [dict(record["submission"]) for record in records]
        if not submissions:
            raise RuntimeError("没有 selected_for_private=True 的 submission")
        unknown_rerun_ids = sorted(self.rerun_submission_ids - prepared_id_set)
        if unknown_rerun_ids:
            raise RuntimeError(
                "RERUN_SUBMISSION_IDS 包含不在本批次中的 submission: "
                f"{unknown_rerun_ids}"
            )

        by_user: dict[str, list[str]] = {}
        for submission in submissions:
            by_user.setdefault(str(submission.get("user_id")), []).append(str(submission["id"]))
        violations = {user: ids for user, ids in by_user.items() if len(ids) > 2}
        if violations:
            raise RuntimeError(f"每个用户最多选择两个 submission: {violations}")

        write_json(os.path.join(self.run_dir, "submissions.json"), submissions)
        if self.resume:
            with open(self.manifest_path, encoding="utf-8") as reader:
                previous_manifest = json.load(reader)
            if previous_manifest.get("competition_id") != self.competition_id:
                raise RuntimeError("续跑批次的 competition_id 不匹配")
            if previous_manifest.get("input_batch_id") != metadata.get("batch_id"):
                raise RuntimeError("续跑批次使用的固化输入包已变化，拒绝混用")
            if set(previous_manifest.get("selected_submission_ids") or []) != current_id_set:
                raise RuntimeError("续跑批次的 submission 集合已变化，拒绝混用")
            update_manifest(
                self.manifest_path,
                status="resuming",
                resumed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                rerun_submission_ids=sorted(self.rerun_submission_ids),
                selected_submissions_verified_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        else:
            update_manifest(
                self.manifest_path,
                competition_id=self.competition_id,
                mode=self.mode,
                batch_id=self.batch_id,
                status="preparing",
                created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                date_start=self.DATE_START,
                date_end=self.DATE_END,
                datasets=self.DATASETS,
                submission_count=len(submissions),
                input_dir=self.input_dir,
                input_batch_id=metadata.get("batch_id"),
                input_summary=metadata.get("summary"),
                selected_submissions_verified_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                selected_submission_ids=sorted(current_id_set),
                execution_code_source="prepared/submissions/<submission_id> 中自动识别的原始代码文件",
                max_workers=self.max_workers,
                published=False,
            )
        try:
            records_by_id = {str(record["submission_id"]): record for record in records}
            batch_started = time.monotonic()
            update_manifest(
                self.manifest_path,
                status="running",
                prepared_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
            self.log.info(
                "private_batch.start",
                batch_id=self.batch_id,
                total=len(submissions),
                completed=0,
                running=0,
                remaining=len(submissions),
                resume=self.resume,
                rerun_submission_ids=sorted(self.rerun_submission_ids),
                max_workers=self.max_workers,
                msg="私榜批次开始",
            )
            progress = {"completed": 0, "running": 0}
            progress_lock = threading.Lock()
            rows_by_id: dict[str, dict] = {}
            pending: list[tuple[int, dict]] = []
            for position, submission in enumerate(submissions, 1):
                sid = str(submission["id"])
                if sid in self.rerun_submission_ids:
                    self._reset_submission(submission)
                completed = self._completed_result(submission) if self.resume else None
                if completed is not None:
                    progress["completed"] += 1
                    self.log.info(
                        "submission.resume_skip",
                        submission_id=sid,
                        status=completed["status"],
                        position=position,
                        total=len(submissions),
                        completed=progress["completed"],
                        running=0,
                        remaining=len(submissions) - progress["completed"],
                        msg="已有完整结果，断点续跑时跳过",
                    )
                    rows_by_id[sid] = completed
                    continue
                pending.append((position, submission))

            worker_count = min(self.max_workers, len(pending))
            if pending:
                self.log.info(
                    "private_batch.dispatch",
                    batch_id=self.batch_id,
                    pending=len(pending),
                    max_workers=worker_count,
                    completed=progress["completed"],
                    running=0,
                    remaining=len(pending),
                    msg="开始并行调度待评测 submission",
                )
                with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="private-judge",
                ) as executor:
                    futures = {
                        executor.submit(
                            self._run_submission,
                            submission,
                            records_by_id[str(submission["id"])],
                            position=position,
                            total=len(submissions),
                            progress=progress,
                            progress_lock=progress_lock,
                        ): str(submission["id"])
                        for position, submission in pending
                    }
                    for future in as_completed(futures):
                        sid = futures[future]
                        rows_by_id[sid] = future.result()

            # 恢复 metadata 中的稳定顺序，避免并发完成先后影响汇总文件行序。
            rows = [
                rows_by_id[str(submission["id"])]
                for submission in submissions
            ]
            self._save_scores(rows)
            batch_elapsed_seconds = round(time.monotonic() - batch_started, 3)
            update_manifest(
                self.manifest_path,
                status="review_pending",
                completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                elapsed_seconds=batch_elapsed_seconds,
            )
            self.log.info(
                "private_batch.finish",
                batch_id=self.batch_id,
                total=len(submissions),
                completed=len(submissions),
                running=0,
                remaining=0,
                elapsed_seconds=batch_elapsed_seconds,
                success=sum(row.get("status") == "success" for row in rows),
                failed=sum(row.get("status") != "success" for row in rows),
                msg="私榜批次完成，等待人工审查发布",
            )
        except Exception as exc:
            update_manifest(self.manifest_path, status="failed", error=str(exc))
            raise
