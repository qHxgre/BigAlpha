"""合并两个连续私榜周期的原始因子，并按完整周期重新计算全部得分。

默认采用严格口径：只有两个周期都成功、代码一致且 raw_factor.parquet 完整的
submission 才进入完整周期 SFA；其余 submission 在合并批次中记为失败。

脚本不会修改源周期，也不会生成 pending_publish.jsonl。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


HERE = Path(__file__).resolve().parent
for path in (
    Path("/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver"),
    HERE.parents[2] / "alphathonapiserver",
    HERE,
):
    value = str(path)
    if value not in sys.path:
        sys.path.append(value)

from fileio import update_manifest, write_json
from scoring import compute_a_scores, compute_final_scores
from templates import build_sfa_from_factor_runner


DATE_START = "2025-03-01 00:00:00"
DATE_END = "2026-08-10 23:59:59"
FACTOR_COLUMNS = ["date", "instrument", "factor"]
COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
DEFAULT_BATCH_ID = "20260812_180129"

# ---------------------------------------------------------------------------
# 合并运行配置：所有运行参数都在本文件中显式设置，不读取命令行参数。
# ---------------------------------------------------------------------------

# None：根据 DEFAULT_BATCH_ID 自动定位批次根目录；也可填写云端绝对路径。
RUN_ROOT: Path | None = None
FIRST_PERIOD = "20250301_20251130"
SECOND_PERIOD = "20251201_20260810"
OUTPUT_PERIOD = "20250301_20260810_merged"
MAX_WORKERS = 1

# DRY_RUN=True 只生成合并审计，不执行因子合并和评分。
DRY_RUN = False

# OVERWRITE=True 删除整个 OUTPUT_PERIOD 后全量重跑。
# RESUME=True 复用其他 submission 的已有结果，并重跑 RERUN_SUBMISSION_IDS。
# 两者不能同时为 True。本次定向重跑应保持 OVERWRITE=False、RESUME=True。
OVERWRITE = False
RESUME = True

# 仅在 RESUME = True 时生效。源周期中的这些 submission 重新评测后，需在这里
# 指定相同 ID，脚本会删除它们在 merged 周期中的旧结果并重新合并、评分；其他
# submission 继续复用 merged 周期中已有的 result.json。全部完成后会基于最新结果
# 重新生成 leaderboard_sfa.csv、回归 B 分、leaderboard_final.csv 和汇总文件。
RERUN_SUBMISSION_IDS: list[str] = [
    "65fcb0d6-42aa-4dd4-8593-8ab1437e69c9",
    "f6e44b3b-42c7-4e0c-8242-b91c89cfc509",
    "312186dc-7b00-457f-b7f3-6e885c1e5ffc",
    "7f8311ec-f7af-4867-ac95-49876c88bdc8",
    "b8f6af17-daf0-4c44-ad48-9bd6ef3d7313",
]


def _default_run_root() -> Path:
    project_root = HERE.parents[3]
    candidates = [
        project_root / "system" / "files" / COMPETITION_ID / "private" / "runs" / DEFAULT_BATCH_ID,
        project_root
        / "system"
        / "files"
        / "private"
        / COMPETITION_ID
        / "private"
        / "runs"
        / DEFAULT_BATCH_ID,
    ]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as reader:
        return json.load(reader)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _submission_map(path: Path) -> dict[str, dict]:
    values = _read_json(path)
    if not isinstance(values, list):
        raise RuntimeError(f"submissions.json 必须是列表: {path}")
    result = {}
    for item in values:
        sid = str(item.get("id") or item.get("submission_id") or "").strip()
        if not sid:
            raise RuntimeError(f"submissions.json 中存在缺少 id 的记录: {path}")
        if sid in result:
            raise RuntimeError(f"submissions.json 中存在重复 submission: {sid}")
        result[sid] = item
    return result


def _result_map(period_dir: Path) -> dict[str, dict]:
    path = period_dir / "artifacts" / "submissions_summary.csv"
    if not path.is_file():
        raise RuntimeError(f"缺少 submissions_summary.csv: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"submission_id": str})
    if frame["submission_id"].duplicated().any():
        duplicates = frame.loc[frame["submission_id"].duplicated(), "submission_id"].tolist()
        raise RuntimeError(f"汇总中存在重复 submission_id: {duplicates[:5]}")
    return {
        str(row["submission_id"]): {
            key: (None if pd.isna(value) else value)
            for key, value in row.items()
        }
        for row in frame.to_dict("records")
    }


def _code_path(period_dir: Path, sid: str, result: dict | None) -> Path | None:
    submission_dir = period_dir / "submissions" / sid
    code_file = str((result or {}).get("code_file") or "").strip()
    if code_file:
        return submission_dir / code_file
    notebooks = sorted(submission_dir.glob("*.ipynb"))
    return notebooks[0] if len(notebooks) == 1 else None


def _parquet_columns(path: Path) -> tuple[list[str] | None, str | None]:
    if not path.is_file():
        return None, "missing_raw_factor"
    try:
        import pyarrow.parquet as pq

        columns = pq.ParquetFile(path).schema_arrow.names
    except Exception as exc:
        return None, f"invalid_parquet: {exc}"
    missing = [column for column in FACTOR_COLUMNS if column not in columns]
    if missing:
        return columns, f"missing_columns: {','.join(missing)}"
    return columns, None


def build_audit(first_dir: Path, second_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    first_submissions = _submission_map(first_dir / "submissions.json")
    second_submissions = _submission_map(second_dir / "submissions.json")
    first_results = _result_map(first_dir)
    second_results = _result_map(second_dir)
    all_ids = sorted(set(first_submissions) | set(second_submissions))
    rows = []
    for sid in all_ids:
        first_result = first_results.get(sid)
        second_result = second_results.get(sid)
        first_code = _code_path(first_dir, sid, first_result)
        second_code = _code_path(second_dir, sid, second_result)
        first_hash = _sha256(first_code) if first_code else None
        second_hash = _sha256(second_code) if second_code else None
        first_raw = first_dir / "submissions" / sid / "raw_factor.parquet"
        second_raw = second_dir / "submissions" / sid / "raw_factor.parquet"
        _, first_factor_error = _parquet_columns(first_raw)
        _, second_factor_error = _parquet_columns(second_raw)
        reasons = []
        if sid not in first_submissions:
            reasons.append("missing_first_submission")
        if sid not in second_submissions:
            reasons.append("missing_second_submission")
        first_user = str((first_result or {}).get("user_id") or "")
        second_user = str((second_result or {}).get("user_id") or "")
        if first_user and second_user and first_user != second_user:
            reasons.append("user_mismatch")
        if not first_result:
            reasons.append("missing_first_result")
        elif first_result.get("status") != "success":
            reasons.append("first_not_success")
        if not second_result:
            reasons.append("missing_second_result")
        elif second_result.get("status") != "success":
            reasons.append("second_not_success")
        if first_hash is None:
            reasons.append("missing_first_code")
        if second_hash is None:
            reasons.append("missing_second_code")
        if first_hash and second_hash and first_hash != second_hash:
            reasons.append("code_mismatch")
        if first_factor_error:
            reasons.append(f"first_{first_factor_error}")
        if second_factor_error:
            reasons.append(f"second_{second_factor_error}")
        rows.append(
            {
                "submission_id": sid,
                "user_id_first": (first_result or {}).get("user_id"),
                "user_id_second": (second_result or {}).get("user_id"),
                "code_file_first": first_code.name if first_code else None,
                "code_file_second": second_code.name if second_code else None,
                "code_sha256_first": first_hash,
                "code_sha256_second": second_hash,
                "status_first": (first_result or {}).get("status"),
                "status_second": (second_result or {}).get("status"),
                "error_first": (first_result or {}).get("error"),
                "error_second": (second_result or {}).get("error"),
                "raw_factor_first": str(first_raw),
                "raw_factor_second": str(second_raw),
                "merge_status": "ready" if not reasons else "excluded",
                "merge_reason": ";".join(reasons),
            }
        )
    ordered_submissions = [first_submissions[sid] for sid in first_submissions]
    ordered_submissions.extend(
        second_submissions[sid]
        for sid in second_submissions
        if sid not in first_submissions
    )
    return pd.DataFrame(rows), ordered_submissions


def _load_and_merge_factors(first_path: Path, second_path: Path) -> pd.DataFrame:
    frames = []
    ranges = [
        (first_path, pd.Timestamp("2025-03-01"), pd.Timestamp("2025-11-30 23:59:59")),
        (second_path, pd.Timestamp("2025-12-01"), pd.Timestamp("2026-08-10 23:59:59")),
    ]
    for path, start, end in ranges:
        frame = pd.read_parquet(path, columns=FACTOR_COLUMNS)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        outside = (frame["date"] < start) | (frame["date"] > end)
        if outside.any():
            sample = frame.loc[outside, "date"].astype(str).head(5).tolist()
            raise RuntimeError(f"因子日期超出源周期 {path}: {sample}")
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if merged.duplicated(["date", "instrument"]).any():
        sample = merged.loc[
            merged.duplicated(["date", "instrument"], keep=False),
            ["date", "instrument"],
        ].head(5).to_dict("records")
        raise RuntimeError(f"合并后存在重复 date/instrument: {sample}")
    numeric = pd.to_numeric(merged["factor"], errors="coerce")
    if numeric.isna().any():
        raise RuntimeError(f"合并因子包含空值或非数值，数量={int(numeric.isna().sum())}")
    import numpy as np

    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("合并因子包含 inf/-inf")
    merged["factor"] = numeric
    return merged.sort_values(["date", "instrument"]).reset_index(drop=True)


def _run_one(
    audit_row: dict,
    output_submission_dir: Path,
    date_start: str,
    date_end: str,
) -> dict:
    sid = str(audit_row["submission_id"])
    target = output_submission_dir / sid
    target.mkdir(parents=True, exist_ok=True)
    base_row = {
        "submission_id": sid,
        "user_id": audit_row.get("user_id_first") or audit_row.get("user_id_second"),
        "code_file": audit_row.get("code_file_first") or audit_row.get("code_file_second"),
    }
    started_at = datetime.now().astimezone()
    started = time.monotonic()
    if audit_row.get("merge_status") != "ready":
        row = {
            **base_row,
            "status": "failed",
            "error": f"incomplete_period: {audit_row.get('merge_reason')}",
        }
    else:
        try:
            merged = _load_and_merge_factors(
                Path(str(audit_row["raw_factor_first"])),
                Path(str(audit_row["raw_factor_second"])),
            )
            input_path = (target / "merged_input_factor.parquet").resolve()
            merged.to_parquet(input_path, index=False)
            from judge.judgebase import LocalProcessUserRunner

            runner = LocalProcessUserRunner(
                submission_id=sid,
                files={
                    "judge_runner.py": build_sfa_from_factor_runner(
                        str(input_path), date_start, date_end
                    )
                },
                cmd=[
                    "python3",
                    "-c",
                    "from judge_runner import judge_runner_main; judge_runner_main()",
                ],
                runner_dir=str(target),
            )
            runner.run(_raise=True)
            analyze = _read_json(target / "factor_analyze.json")
            row = {**base_row, **analyze, "status": "success"}
        except Exception as exc:
            row = {**base_row, "status": "failed", "error": str(exc)}
    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    row["started_at"] = started_at.isoformat(timespec="seconds")
    row["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(str(target / "result.json"), row)
    return row


def _save_scores(rows: list[dict], submission_dir: Path, artifact_dir: Path, log: Any) -> None:
    from regression import run_regression

    a_scores = compute_a_scores(rows)
    if not a_scores.empty:
        a_scores.to_csv(artifact_dir / "leaderboard_sfa.csv", index=False)
        b_scores = run_regression(
            a_scores,
            str(submission_dir),
            str(artifact_dir),
            DATE_START,
            DATE_END,
            log,
        )
    else:
        b_scores = {}
    final = compute_final_scores(rows, a_scores, b_scores)
    final.to_csv(artifact_dir / "leaderboard_final.csv", index=False)
    pd.DataFrame(rows).merge(
        final, on="submission_id", how="left", suffixes=("", "_calculated")
    ).to_csv(artifact_dir / "submissions_summary.csv", index=False)


def main() -> None:
    if MAX_WORKERS < 1:
        raise RuntimeError("MAX_WORKERS 必须是正整数")
    rerun_submission_ids = {
        str(value).strip() for value in RERUN_SUBMISSION_IDS if str(value).strip()
    }
    if rerun_submission_ids and not RESUME:
        raise RuntimeError("RERUN_SUBMISSION_IDS 非空时必须设置 RESUME = True")
    run_root = (RUN_ROOT or _default_run_root()).resolve()
    first_dir = run_root / FIRST_PERIOD
    second_dir = run_root / SECOND_PERIOD
    output_dir = run_root / OUTPUT_PERIOD
    for path in (first_dir, second_dir):
        if not path.is_dir():
            raise RuntimeError(f"源周期目录不存在: {path}")
    if output_dir in (first_dir, second_dir):
        raise RuntimeError("输出目录不能与源周期目录相同")
    if OVERWRITE and RESUME:
        raise RuntimeError("OVERWRITE 和 RESUME 不能同时为 True")
    if output_dir.exists():
        if not OVERWRITE and not RESUME:
            raise RuntimeError(f"输出目录已存在，拒绝覆盖: {output_dir}")
        if OVERWRITE:
            shutil.rmtree(output_dir)

    artifact_dir = output_dir / "artifacts"
    submission_dir = output_dir / "submissions"
    log_dir = output_dir / "logs"
    artifact_dir.mkdir(parents=True, exist_ok=RESUME)
    audit, submissions = build_audit(first_dir, second_dir)
    audit_submission_ids = set(audit["submission_id"].astype(str))
    unknown_rerun_ids = sorted(rerun_submission_ids - audit_submission_ids)
    if unknown_rerun_ids:
        raise RuntimeError(
            "RERUN_SUBMISSION_IDS 包含不在源周期中的 submission: "
            f"{unknown_rerun_ids}"
        )
    audit.to_csv(artifact_dir / "merge_audit.csv", index=False, encoding="utf-8-sig")
    summary = audit["merge_status"].value_counts().to_dict()
    write_json(str(output_dir / "submissions.json"), submissions)
    write_json(
        str(output_dir / "manifest.json"),
        {
            "status": "audit_complete" if DRY_RUN else "running",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "date_start": DATE_START,
            "date_end": DATE_END,
            "source_periods": [str(first_dir), str(second_dir)],
            "submission_count": len(audit),
            "audit_summary": summary,
            "strict_both_success": True,
            "resume": RESUME,
            "rerun_submission_ids": sorted(rerun_submission_ids),
            "published": False,
        },
    )
    print(f"审计完成: total={len(audit)} ready={summary.get('ready', 0)} excluded={summary.get('excluded', 0)}")
    print(f"审计报告: {artifact_dir / 'merge_audit.csv'}")
    if DRY_RUN:
        return

    submission_dir.mkdir(exist_ok=RESUME)
    log_dir.mkdir(exist_ok=RESUME)
    from judge.judgebase import setup_judge_logging
    import structlog

    setup_judge_logging(str(log_dir / "merge_period_results.log"))
    log = structlog.get_logger()
    for sid in sorted(rerun_submission_ids):
        target = submission_dir / sid
        if target.exists():
            shutil.rmtree(target)
        log.info("merge.submission_reset", submission_id=sid)
    rows_by_id = {}
    records = audit.to_dict("records")
    pending_records = []
    for record in records:
        sid = str(record["submission_id"])
        result_path = submission_dir / sid / "result.json"
        if RESUME and result_path.is_file():
            try:
                completed = _read_json(result_path)
            except (OSError, json.JSONDecodeError):
                completed = None
            if (
                isinstance(completed, dict)
                and str(completed.get("submission_id")) == sid
                and completed.get("status") in {"success", "failed"}
            ):
                rows_by_id[sid] = completed
                log.info(
                    "merge.submission_resume_skip",
                    submission_id=sid,
                    status=completed["status"],
                )
                continue
        pending_records.append(record)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="period-merge") as executor:
        futures = {
            executor.submit(_run_one, record, submission_dir, DATE_START, DATE_END): str(
                record["submission_id"]
            )
            for record in pending_records
        }
        for future in as_completed(futures):
            sid = futures[future]
            rows_by_id[sid] = future.result()
            row = rows_by_id[sid]
            log.info("merge.submission_finish", submission_id=sid, status=row["status"])
    rows = [rows_by_id[str(record["submission_id"])] for record in records]
    _save_scores(rows, submission_dir, artifact_dir, log)
    update_manifest(
        str(output_dir / "manifest.json"),
        status="review_pending",
        completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        success=sum(row.get("status") == "success" for row in rows),
        failed=sum(row.get("status") != "success" for row in rows),
    )
    print(
        "完整周期评分完成: "
        f"success={sum(row.get('status') == 'success' for row in rows)} "
        f"failed={sum(row.get('status') != 'success' for row in rows)}"
    )


if __name__ == "__main__":
    main()
