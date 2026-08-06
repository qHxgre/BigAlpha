"""最小化私榜评测器：单次执行、全部落盘、人工确认后发布。"""
from __future__ import annotations

import json
import os
import shutil
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
    submission_files = ((record.get("submission") or {}).get("data") or {}).get("files") or {}
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
    max_workers = 1

    DATASETS = {
        "bar1m": "bigalpha_2026_stock_bar1m_private",
        "financial": "bigalpha_2026_financial_private",
    }
    DATE_START = "2025-01-01 00:00:00"
    DATE_END = ""
    RUNS_DIR = os.path.join(PRIVATE_FILES_DIR, "runs")

    def __init__(self, input_dir: str) -> None:
        super().__init__()
        self.input_dir = os.path.abspath(input_dir)
        self.input_metadata_path = os.path.join(self.input_dir, "metadata.json")
        if not os.path.isfile(self.input_metadata_path):
            raise RuntimeError(f"private 输入包缺少 metadata.json: {self.input_dir}")
        self.batch_id = os.getenv("PRIVATE_BATCH_ID") or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.RUNS_DIR, self.batch_id)
        if os.path.exists(self.run_dir):
            raise RuntimeError(f"批次目录已存在，拒绝覆盖: {self.run_dir}")

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

    def _prepare_submission(self, submission: dict, record: dict) -> tuple[str, str]:
        """把固化输入复制到运行目录；评测不再访问 submission API。"""
        sid = str(submission["id"])
        source_dir = os.path.join(self.input_dir, record["relative_path"])
        runner_dir = self.submission_path(submission)
        if not os.path.isdir(source_dir):
            raise RuntimeError(f"submission {sid} 的固化目录不存在: {source_dir}")
        code, code_file = _load_submission_code(source_dir, record, sid)
        code = self.preprocess_user_code(submission, code)
        shutil.copytree(source_dir, runner_dir, dirs_exist_ok=True)
        return code, code_file

    def _run_submission(self, submission: dict) -> dict:
        sid = str(submission["id"])
        row = {"submission_id": sid, "user_id": submission.get("user_id")}
        try:
            code = submission.pop("_prepared_code")
            row["code_file"] = submission.pop("_prepared_code_file")
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

        by_user: dict[str, list[str]] = {}
        for submission in submissions:
            by_user.setdefault(str(submission.get("user_id")), []).append(str(submission["id"]))
        violations = {user: ids for user, ids in by_user.items() if len(ids) > 2}
        if violations:
            raise RuntimeError(f"每个用户最多选择两个 submission: {violations}")

        write_json(os.path.join(self.run_dir, "submissions.json"), submissions)
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
            published=False,
        )
        try:
            records_by_id = {str(record["submission_id"]): record for record in records}
            for submission in submissions:
                code, code_file = self._prepare_submission(
                    submission, records_by_id[str(submission["id"])]
                )
                submission["_prepared_code"] = code
                submission["_prepared_code_file"] = code_file
            update_manifest(
                self.manifest_path,
                status="running",
                prepared_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            )
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
