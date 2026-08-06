"""最小化私榜评测器：单次执行、全部落盘、人工确认后发布。"""
from __future__ import annotations

import json
import os
import time
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


class PrivateJudge(JudgeBase):
    competition_id = COMPETITION_ID
    mode = "private"
    max_workers = 1

    DATASETS = {
        "bar1m": "bigalpha_2026_stock_bar1m_private",
        "financial": "bigalpha_2026_financial_private",
    }
    DATE_START = "2025-01-01 00:00:00"
    DATE_END = ""
    RUNS_DIR = os.path.join(PRIVATE_FILES_DIR, "runs")
    SELECTED_SUBMISSIONS_DIR = os.path.join(PRIVATE_FILES_DIR, "selected_submissions")

    def __init__(self) -> None:
        super().__init__()
        self.batch_id = os.getenv("PRIVATE_BATCH_ID") or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.RUNS_DIR, self.batch_id)
        if os.path.exists(self.run_dir):
            raise RuntimeError(f"批次目录已存在，拒绝覆盖: {self.run_dir}")

        self.submission_dir = os.path.join(self.run_dir, "submissions")
        self.selected_submission_dir = self.SELECTED_SUBMISSIONS_DIR
        self.artifact_dir = os.path.join(self.run_dir, "artifacts")
        self.log_dir = os.path.join(self.run_dir, "logs")
        for path in (
            self.submission_dir,
            self.selected_submission_dir,
            self.artifact_dir,
            self.log_dir,
        ):
            os.makedirs(path, exist_ok=True)
        setup_judge_logging(os.path.join(self.log_dir, "judge_private.log"))

        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.pending_path = os.path.join(self.run_dir, "pending_publish.jsonl")

    def _save_selected_submission_files(self, submission: dict) -> None:
        """单独归档入围提交的原始文件，不混入任何评测运行产物。"""
        sid = str(submission["id"])
        destination = os.path.join(self.selected_submission_dir, sid)
        os.makedirs(destination, exist_ok=True)

        files = (submission.get("data") or {}).get("files") or {}
        for file_id, file_info in files.items():
            file_name = (file_info or {}).get("name") or str(file_id)
            # 与 collect_best_submissions.py 保持一致：大体积数据文件不归档。
            if os.path.splitext(file_name)[1].lower() == ".parquet":
                continue
            # API 中的文件名来自用户输入，只取 basename，避免写出归档目录。
            file_name = os.path.basename(file_name)
            save_to = os.path.join(destination, file_name)
            if os.path.isfile(save_to):
                continue
            self.alphathon_api.get_submission_file(
                sid,
                str(file_id),
                file_info,
                save_to=save_to,
            )

    def _run_submission(self, submission: dict) -> dict:
        sid = str(submission["id"])
        row = {"submission_id": sid, "user_id": submission.get("user_id")}
        try:
            self.save_submission_files(submission)
            code = self.alphathon_api.get_file_content_of_submission(
                submission, ipynb_to_py=True, to_str=True
            )
            if isinstance(code, bytes):
                code = code.decode("utf-8")
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
        write_json(os.path.join(self.submission_path(submission), "result.json"), row)
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
        submissions = self.alphathon_api.query_submissions(
            competition_id=self.competition_id,
            constraints={"selected_for_private": True},
        )
        if not submissions:
            raise RuntimeError("没有 selected_for_private=True 的 submission")

        by_user: dict[str, list[str]] = {}
        for submission in submissions:
            by_user.setdefault(str(submission.get("user_id")), []).append(str(submission["id"]))
        violations = {user: ids for user, ids in by_user.items() if len(ids) > 2}
        if violations:
            raise RuntimeError(f"每个用户最多选择两个 submission: {violations}")

        write_json(os.path.join(self.run_dir, "submissions.json"), submissions)
        for submission in submissions:
            self._save_selected_submission_files(submission)
        update_manifest(
            self.manifest_path,
            competition_id=self.competition_id,
            mode=self.mode,
            batch_id=self.batch_id,
            status="running",
            created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            date_start=self.DATE_START,
            date_end=self.DATE_END,
            datasets=self.DATASETS,
            submission_count=len(submissions),
            selected_submissions_dir=self.selected_submission_dir,
            published=False,
        )
        try:
            rows = [self._run_submission(submission) for submission in submissions]
            self._save_scores(rows)
            update_manifest(
                self.manifest_path,
                status="review_pending",
                completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            update_manifest(self.manifest_path, status="failed", error=str(exc))
            raise
