"""端到端模型赛道的可审查私榜批次评测器。"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from judge.judgebase import JudgeBase, UserCodeRunError, setup_judge_logging
from judge.paths import FILE_DIR
from runner import MemoryLimitedUserRunner
from fileio import jsonable, update_manifest, write_json, write_jsonl
from scoring import METRICS, compute_scores
from templates import build_runner

COMPETITION_ID = "523f9302-5b4b-42bd-bce1-f232e7c74316"
PRIVATE_FILES_DIR = os.path.join(FILE_DIR, COMPETITION_ID, "private")


def notebook_code(path: str) -> str:
    with open(path, encoding="utf-8") as reader:
        notebook = json.load(reader)
    return "\n\n".join(
        "".join(cell.get("source", [])) if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"
    )


def load_code(source: str, record: dict, sid: str) -> tuple[str, str]:
    submission = record.get("submission") or {}
    files = (submission.get("data") or {}).get("files")
    if files is None:
        files = submission.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("files 字段类型错误")
    notebooks = [(str(fid), info or {}) for fid, info in files.items()
                 if str((info or {}).get("name", "")).lower().endswith(".ipynb")]
    if len(notebooks) != 1:
        raise RuntimeError(f"submission {sid} 必须恰好包含一个 ipynb，实际 {len(notebooks)} 个")
    fid, info = notebooks[0]
    saved = {str(x["file_id"]): x["name"] for x in record.get("files", [])}
    name = saved.get(fid)
    if not name or not os.path.isfile(os.path.join(source, name)):
        raise RuntimeError(f"submission {sid} 的 notebook 未固化")
    return notebook_code(os.path.join(source, name)), str(info.get("name") or name)


class PrivateJudge(JudgeBase):
    competition_id = COMPETITION_ID
    mode = "private"
    DATASETS: dict[str, str] = {}
    DATE_START = ""
    DATE_END = ""
    RUNS_DIR = os.path.join(PRIVATE_FILES_DIR, "runs")
    HEARTBEAT_INTERVAL_SECONDS = 60

    def __init__(self, input_dir: str, *, batch_id: str, resume: bool,
                 rerun_submission_ids: list[str], max_workers: int) -> None:
        super().__init__()
        self.input_dir = os.path.abspath(input_dir)
        self.metadata_path = os.path.join(self.input_dir, "metadata.json")
        if not os.path.isfile(self.metadata_path):
            raise RuntimeError(f"缺少固化输入 metadata.json: {self.input_dir}")
        self.batch_id, self.resume = str(batch_id), bool(resume)
        self.rerun_ids = {str(x) for x in rerun_submission_ids}
        self.max_workers = int(max_workers)
        if self.max_workers < 1 or (self.rerun_ids and not self.resume):
            raise RuntimeError("MAX_WORKERS 必须为正整数；指定重跑 ID 时必须启用 RESUME")
        self.run_dir = os.path.join(self.RUNS_DIR, self.batch_id)
        if os.path.exists(self.run_dir) and not self.resume:
            raise RuntimeError(f"批次已存在: {self.run_dir}")
        if self.resume and not os.path.isdir(self.run_dir):
            raise RuntimeError(f"续跑批次不存在: {self.run_dir}")
        self.submission_dir = os.path.join(self.run_dir, "submissions")
        self.artifact_dir = os.path.join(self.run_dir, "artifacts")
        self.log_dir = os.path.join(self.run_dir, "logs")
        for path in (self.submission_dir, self.artifact_dir, self.log_dir):
            os.makedirs(path, exist_ok=True)
        setup_judge_logging(os.path.join(self.log_dir, "judge_private.log"))
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.pending_path = os.path.join(self.run_dir, "pending_publish.jsonl")
        if self.resume:
            manifest = self._read(self.manifest_path)
            if manifest.get("published") and not self.rerun_ids:
                raise RuntimeError(
                    "已发布批次不允许普通断点续跑；请通过 RERUN_SUBMISSION_IDS "
                    "明确指定需要重跑的 submission"
                )
            self.DATE_END = manifest.get("date_end") or self.DATE_END

    @staticmethod
    def _read(path: str) -> dict:
        with open(path, encoding="utf-8") as reader:
            return json.load(reader)

    def _result(self, sid: str) -> dict | None:
        path = os.path.join(self.submission_dir, sid, "result.json")
        try:
            row = self._read(path)
            return row if str(row.get("submission_id")) == sid and row.get("status") in {"success", "failed"} else None
        except (OSError, ValueError):
            return None

    def _run_one(self, submission: dict, record: dict, progress: dict, lock: threading.Lock) -> dict:
        sid = str(submission["id"])
        target = os.path.join(self.submission_dir, sid)
        source = os.path.abspath(os.path.join(self.input_dir, record["relative_path"]))
        input_root = os.path.abspath(self.input_dir)
        if os.path.commonpath([source, input_root]) != input_root or source == input_root:
            raise RuntimeError(f"submission {sid} 的固化输入路径越界: {source}")
        started_at, started = datetime.now().astimezone(), time.monotonic()
        with lock:
            progress["running"] += 1
        row = {"submission_id": sid, "user_id": submission.get("user_id")}
        try:
            shutil.copytree(source, target, dirs_exist_ok=True)
            code, row["code_file"] = load_code(source, record, sid)
            code = self.preprocess_user_code(submission, code)
            runner = MemoryLimitedUserRunner(
                submission_id=sid,
                files={"judge_runner.py": build_runner(code, self.DATASETS, self.DATE_START, self.DATE_END)},
                cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
                runner_dir=target,
            )
            runner.run(_raise=True)
            row.update(self._read(os.path.join(target, "score_analyze.json")))
            row["status"] = "success"
        except UserCodeRunError as exc:
            row.update(status="failed", error=str(exc), failure_type=exc.reason)
        except Exception as exc:
            row.update(status="failed", error=str(exc), failure_type=type(exc).__name__)
        row.update(started_at=started_at.isoformat(timespec="seconds"),
                   finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                   elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(os.path.join(target, "result.json"), row)
        with lock:
            progress["running"] -= 1
            progress["completed"] += 1
            progress[row["status"]] += 1
        return row

    def _heartbeat(self, total: int, progress: dict, lock: threading.Lock,
                   stop: threading.Event, started: float) -> None:
        """定时输出批次进度；Event.wait 让退出时无需等待完整间隔。"""
        sequence = 0
        while not stop.wait(self.HEARTBEAT_INTERVAL_SECONDS):
            with lock:
                snapshot = dict(progress)
            sequence += 1
            remaining = total - snapshot["completed"]
            self.log.info(
                "private_batch.heartbeat",
                seq=sequence,
                batch_id=self.batch_id,
                total=total,
                completed=snapshot["completed"],
                success=snapshot["success"],
                failed=snapshot["failed"],
                running=snapshot["running"],
                queued=max(0, remaining - snapshot["running"]),
                remaining=remaining,
                elapsed_seconds=round(time.monotonic() - started, 1),
                msg="私榜批次仍在运行",
            )

    def _save_score_pool(self, rows: list[dict]) -> None:
        process_frames, raw_frames = [], []
        for row in rows:
            if row.get("status") != "success":
                continue
            sid = str(row["submission_id"])
            for filename, frames in (("process_score.parquet", process_frames), ("raw_score.parquet", raw_frames)):
                path = os.path.join(self.submission_dir, sid, filename)
                try:
                    frame = pd.read_parquet(path)[["date", "instrument", "factor"]]
                    frames.append(frame.rename(columns={"factor": sid}).set_index(["date", "instrument"]))
                except Exception as exc:
                    self.log.warning("pool.skip", submission_id=sid, file=filename, error=str(exc))
        for name, frames in (("score_pool.parquet", process_frames), ("score_pool_raw.parquet", raw_frames)):
            if frames:
                pd.concat(frames, axis=1, join="outer").reset_index().to_parquet(os.path.join(self.artifact_dir, name))

    def _save_scores(self, rows: list[dict]) -> None:
        final = compute_scores(rows)
        successful = pd.DataFrame([r for r in rows if r.get("status") == "success"])
        if not successful.empty:
            keep = ["submission_id", *[m for m in METRICS if m in successful.columns]]
            successful[keep].merge(final, on="submission_id").to_csv(
                os.path.join(self.artifact_dir, "leaderboard_score.csv"), index=False)
        final.to_csv(os.path.join(self.artifact_dir, "leaderboard_final.csv"), index=False)
        pd.DataFrame(rows).merge(final, on="submission_id").to_csv(
            os.path.join(self.artifact_dir, "submissions_summary.csv"), index=False)
        by_id = {str(r["submission_id"]): r for r in rows}
        pending = []
        for item in final.to_dict("records"):
            sid, score = str(item["submission_id"]), float(item["score"])
            data = ({"err_msg": by_id[sid].get("error", "private evaluation failed")}
                    if score == -2 else jsonable(item))
            pending.append({"submission_id": sid, "payload": {
                "private_score": score, "private_score_data": data}})
        write_jsonl(self.pending_path, pending)
        self._save_score_pool(rows)

    def run_once(self) -> None:
        metadata = self._read(self.metadata_path)
        if metadata.get("competition_id") != self.competition_id:
            raise RuntimeError("输入包 competition_id 不匹配")
        records = metadata.get("submissions") or []
        ids = [str(r["submission_id"]) for r in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError("输入包存在重复 submission ID")
        current = self.alphathon_api.query_submissions(
            competition_id=self.competition_id, constraints={"selected_for_private": True})
        current_ids = {str(s["id"]) for s in current}
        if set(ids) != current_ids:
            raise RuntimeError(f"线上私榜选择与输入包不一致：新增={sorted(current_ids-set(ids))}，取消={sorted(set(ids)-current_ids)}")
        if not records:
            raise RuntimeError("没有私榜提交")
        if self.rerun_ids - set(ids):
            raise RuntimeError("重跑 ID 不属于本批次")
        submissions = [dict(r["submission"]) for r in records]
        for record, submission in zip(records, submissions):
            if str(submission.get("id")) != str(record["submission_id"]):
                raise RuntimeError("输入包的 submission_id 与 API 快照不一致")
        by_user: dict[str, list[str]] = {}
        for submission in submissions:
            by_user.setdefault(str(submission.get("user_id")), []).append(str(submission["id"]))
        if any(len(v) > 2 for v in by_user.values()):
            raise RuntimeError(f"每用户最多两个提交: {by_user}")
        if self.resume:
            manifest = self._read(self.manifest_path)
            if manifest.get("input_batch_id") != metadata.get("batch_id") or set(manifest.get("selected_submission_ids", [])) != current_ids:
                raise RuntimeError("续跑批次的输入包或 submission 集合已改变")
            update_manifest(
                self.manifest_path,
                status="resuming",
                resumed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                rerun_submission_ids=sorted(self.rerun_ids),
                previously_published=bool(manifest.get("published")),
            )
        else:
            update_manifest(self.manifest_path, competition_id=self.competition_id, mode="private",
                            batch_id=self.batch_id, status="preparing", date_start=self.DATE_START,
                            date_end=self.DATE_END, datasets=self.DATASETS, submission_count=len(ids),
                            input_batch_id=metadata.get("batch_id"), selected_submission_ids=sorted(ids),
                            max_workers=self.max_workers, published=False)
        update_manifest(self.manifest_path, status="running", started_at=datetime.now().astimezone().isoformat(timespec="seconds"))
        try:
            record_map = {str(r["submission_id"]): r for r in records}
            rows, pending = {}, []
            for submission in submissions:
                sid = str(submission["id"])
                if sid in self.rerun_ids:
                    shutil.rmtree(os.path.join(self.submission_dir, sid), ignore_errors=True)
                done = self._result(sid) if self.resume else None
                if done:
                    rows[sid] = done
                else:
                    pending.append(submission)
            progress = {
                "running": 0,
                "completed": len(rows),
                "success": sum(row.get("status") == "success" for row in rows.values()),
                "failed": sum(row.get("status") == "failed" for row in rows.values()),
            }
            lock = threading.Lock()
            if pending:
                heartbeat_stop = threading.Event()
                heartbeat_started = time.monotonic()
                heartbeat_thread = threading.Thread(
                    target=self._heartbeat,
                    args=(len(submissions), progress, lock, heartbeat_stop, heartbeat_started),
                    name="private-judge-heartbeat",
                    daemon=True,
                )
                heartbeat_thread.start()
                try:
                    with ThreadPoolExecutor(max_workers=min(self.max_workers, len(pending))) as pool:
                        futures = {pool.submit(self._run_one, s, record_map[str(s["id"])], progress, lock): str(s["id"]) for s in pending}
                        for future in as_completed(futures):
                            rows[futures[future]] = future.result()
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join()
            ordered = [rows[str(s["id"])] for s in submissions]
            self._save_scores(ordered)
            update_manifest(self.manifest_path, status="review_pending",
                            completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                            published=False)
        except Exception as exc:
            update_manifest(self.manifest_path, status="failed", error=str(exc))
            raise
