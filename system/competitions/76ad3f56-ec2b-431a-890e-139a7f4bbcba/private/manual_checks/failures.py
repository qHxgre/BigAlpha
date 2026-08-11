"""失败提交及参赛者联系信息检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import read_summary, show
from .config import CheckPaths


def _read_stdout(path: Path) -> str:
    """读取提交日志；日志缺失或不可读时返回可直接写入报告的说明。"""
    if not path.is_file():
        return f"[stdout 不存在: {path}]"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[stdout 读取失败: {path}; {type(exc).__name__}: {exc}]"


def _display_value(value: object) -> str:
    """将 DataFrame 中的缺失值转换成适合文本报告的占位符。"""
    return "-" if pd.isna(value) or str(value) == "" else str(value)


def _write_failure_log(
    by_submission: pd.DataFrame,
    output_path: Path,
    *,
    total_count: int,
) -> Path:
    """将全部失败提交及其完整 stdout 写入一个便于搜索的文本文件。"""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_counts = (
        by_submission["error"].fillna("未记录错误").astype(str).value_counts(dropna=False)
    )
    header_lines = [
        "失败 Submission 日志汇总",
        "=" * 100,
        f"总提交数: {total_count}",
        f"失败提交数: {len(by_submission)}",
        f"涉及参赛方: {by_submission['participant_name'].nunique(dropna=True)}",
        "",
        "错误类型统计:",
    ]
    header_lines.extend(f"  {count:>4}  {error}" for error, count in error_counts.items())

    # 流式写入，避免失败日志较多或单份 stdout 很大时把所有内容同时留在内存中。
    with output_path.open("w", encoding="utf-8") as output:
        output.write("\n".join(header_lines).rstrip() + "\n")
        for index, row in enumerate(by_submission.itertuples(index=False), start=1):
            stdout_path = Path(row.stdout_path)
            submission_lines = [
                "",
                "#" * 100,
                f"[{index}/{len(by_submission)}] submission_id: {row.submission_id}",
                "#" * 100,
                f"participant: {_display_value(row.participant_name)} "
                f"({_display_value(row.participant_type)})",
                f"contacts: {_display_value(row.contact_names)}",
                f"contact_user_ids: {_display_value(row.contact_user_ids)}",
                f"code_file: {_display_value(row.code_file)}",
                f"status: {_display_value(row.status)}",
                f"error: {_display_value(row.error)}",
                f"elapsed_seconds: {_display_value(row.elapsed_seconds)}",
                f"stdout_path: {stdout_path}",
                "-" * 100,
                "STDOUT (完整日志)",
                "-" * 100,
                _read_stdout(stdout_path).rstrip(),
            ]
            output.write("\n".join(submission_lines).rstrip() + "\n")

    # 即使没有失败提交，也产出带统计信息的文件，避免误读上一次运行的旧结果。
    return output_path


def analyze_failed_submissions(
    paths: CheckPaths,
    *,
    stdout_tail_lines: int = 20,
    log_output_path: str | Path | None = None,
    display: bool = True,
) -> dict[str, Any]:
    """返回失败明细，并将所有失败提交的完整 stdout 汇总到单个日志文件。"""
    if stdout_tail_lines < 0:
        raise ValueError("stdout_tail_lines 不能小于 0")
    summary = read_summary(paths)
    failed = summary.loc[summary["status"].fillna("").ne("success")].copy()
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    contacts_by_submission, contacts_by_user = {}, {}
    for participant in metadata["participants"]:
        if participant["type"] == "team":
            participant_name = participant.get("team_name", "")
            contacts = participant.get("members", [])
        else:
            contact = participant.get("user", {})
            participant_name, contacts = contact.get("name", ""), [contact]
        contact_info = {
            "participant_type": participant["type"], "participant_name": participant_name,
            "contact_names": "、".join(contact.get("name", "") for contact in contacts),
            "contact_user_ids": "、".join(str(contact.get("user_id", "")) for contact in contacts),
        }
        for submission_id in participant.get("private_submission_ids", []):
            contacts_by_submission[str(submission_id)] = contact_info
        for contact in contacts:
            contacts_by_user[str(contact.get("user_id", ""))] = contact_info

    contact_table = pd.DataFrame.from_dict(contacts_by_submission, orient="index")
    contact_table.index.name = "submission_id"
    by_submission = failed.merge(contact_table.reset_index(), on="submission_id", how="left")
    for column in ("participant_type", "participant_name", "contact_names", "contact_user_ids"):
        fallback = by_submission["user_id"].map(
            lambda user_id: contacts_by_user.get(str(user_id), {}).get(column)
        )
        by_submission[column] = by_submission[column].fillna(fallback)

    def stdout_tail(submission_id: str) -> str:
        path = paths.run_dir / "submissions" / str(submission_id) / "stdout"
        if not path.is_file():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-stdout_tail_lines:]) if stdout_tail_lines else ""

    by_submission["failure_detail"] = by_submission["submission_id"].map(stdout_tail)
    by_submission["stdout_path"] = by_submission["submission_id"].map(
        lambda submission_id: str(paths.run_dir / "submissions" / submission_id / "stdout")
    )
    columns = [
        "submission_id", "code_file", "status", "error", "failure_detail", "participant_type",
        "participant_name", "contact_names", "contact_user_ids", "elapsed_seconds", "stdout_path",
    ]
    by_submission = by_submission[columns].sort_values(["participant_name", "submission_id"])
    by_contact = (
        by_submission.groupby(
            ["participant_type", "participant_name", "contact_names", "contact_user_ids"], dropna=False
        )
        .agg(
            failed_count=("submission_id", "size"),
            submission_ids=("submission_id", lambda values: "、".join(values)),
            code_files=("code_file", lambda values: "、".join(values.fillna("").astype(str))),
            errors=("error", lambda values: " | ".join(dict.fromkeys(values.fillna("").astype(str)))),
        )
        .reset_index()
        .sort_values(["failed_count", "participant_name"], ascending=[False, True])
    )
    failure_log_path = _write_failure_log(
        by_submission,
        Path(log_output_path) if log_output_path else paths.artifacts_dir / "failed_submissions.log",
        total_count=len(summary),
    )
    print(f"总提交数: {len(summary)}，未跑成功提交: {len(by_submission)}，需联系参赛方: {len(by_contact)}")
    print(f"失败日志汇总: {failure_log_path}")
    if display:
        show(by_contact, by_submission)
    return {
        "by_contact": by_contact,
        "by_submission": by_submission,
        "log_path": failure_log_path,
    }
