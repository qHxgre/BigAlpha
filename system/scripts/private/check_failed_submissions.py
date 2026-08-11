"""检查私榜批次中的失败 submission，并输出内部日志和公开通知。

日常使用时只需修改下方 ``competition_id`` 和 ``batch_id``，然后运行：

    python3 system/scripts/private/check_failed_submissions.py

输出文件：
    system/files/scripts/private/<competition_id>__<batch_id>__failed_submissions.log
    system/files/scripts/private/<competition_id>__<batch_id>__failed_submissions.md

``.log`` 仅供主办方内部排查，包含联系方式和运行日志；``.md`` 可直接用于
通知参赛者，仅列出 submission ID 和参赛方名称，不包含联系方式、用户 ID、
报错详情或文件路径等敏感信息。
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# 运行配置：每次只需修改这两个值。
# ---------------------------------------------------------------------------
competition_id = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
batch_id = "20260811_102014"


SYSTEM_DIR = Path(__file__).resolve().parents[2]
FILES_DIR = SYSTEM_DIR / "files"
OUTPUT_ROOT = FILES_DIR / "scripts" / "private"
USERS_CSV = OUTPUT_ROOT / "alphathon__user.csv"

# 报名信息中可能使用的电话字段名。
PHONE_FIELDS = (
    "phone",
    "mobile",
    "phone_number",
    "phoneNumber",
    "mobile_phone",
    "mobilePhone",
    "telephone",
    "tel",
)


def _resolve_private_root(cid: str) -> Path:
    """兼容云端和本地下载数据的两种目录布局。"""
    candidates = (
        FILES_DIR / cid / "private",
        FILES_DIR / "private" / cid / "private",
    )
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        matches = [path for path in existing if (path / "runs" / batch_id).is_dir()]
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(
            "检测到多个私榜数据目录，无法唯一确定批次位置：\n  "
            + "\n  ".join(str(path) for path in existing)
        )
    raise FileNotFoundError(
        "找不到比赛私榜数据目录，已检查：\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def _required_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"缺少{description}: {path}")
    return path


def _read_stdout(path: Path) -> str:
    if not path.is_file():
        return f"[stdout 不存在: {path}]"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"[stdout 读取失败: {path}; {type(exc).__name__}: {exc}]"


def _display_value(value: object) -> str:
    return "-" if pd.isna(value) or str(value) == "" else str(value)


def _markdown_cell(value: object) -> str:
    """将动态内容安全地放入 Markdown 表格单元格。"""
    normalized = " ".join(_display_value(value).splitlines())
    return escape(normalized, quote=False).replace("|", "\\|")


def _extract_phone(data: object) -> str:
    """从报名信息中提取电话，兼容常见字段名及一层 contact 嵌套。"""
    if not isinstance(data, dict):
        return ""
    for field in PHONE_FIELDS:
        value = data.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    contact = data.get("contact")
    if isinstance(contact, dict):
        return _extract_phone(contact)
    return ""


def _load_phones(cid: str) -> dict[str, str]:
    """从 alphathon__user.csv 的报名 data 中读取 user_id -> 电话。"""
    _required_file(USERS_CSV, "参赛者报名数据文件")
    users = pd.read_csv(
        USERS_CSV,
        usecols=["competition_id", "user_id", "data"],
        dtype={"competition_id": str, "user_id": str, "data": str},
    )
    users = users.loc[users["competition_id"].eq(cid)]

    phones: dict[str, str] = {}
    invalid_data_count = 0
    for row in users.itertuples(index=False):
        try:
            registration_data = json.loads(row.data) if row.data else {}
        except (TypeError, json.JSONDecodeError):
            invalid_data_count += 1
            continue
        phone = _extract_phone(registration_data)
        if phone:
            phones[str(row.user_id)] = phone

    if invalid_data_count:
        print(f"警告: {invalid_data_count} 条报名 data 不是有效 JSON，已跳过")
    return phones


def _contact_maps(
    metadata: dict[str, Any], phones_by_user: dict[str, str]
) -> tuple[dict[str, dict], dict[str, dict]]:
    by_submission: dict[str, dict] = {}
    by_user: dict[str, dict] = {}

    for participant in metadata.get("participants", []):
        participant_type = str(participant.get("type", ""))
        if participant_type == "team":
            participant_name = participant.get("team_name", "")
            contacts = participant.get("members", []) or []
        else:
            contact = participant.get("user", {}) or {}
            participant_name = contact.get("name", "")
            contacts = [contact]

        contact_info = {
            "participant_type": participant_type,
            "participant_name": participant_name,
            "contact_names": "、".join(str(item.get("name", "")) for item in contacts),
            "contact_user_ids": "、".join(
                str(item.get("user_id", "")) for item in contacts
            ),
            "contact_phones": "、".join(
                phones_by_user.get(str(item.get("user_id", "")), "") or "-"
                for item in contacts
            ),
        }
        for submission_id in participant.get("private_submission_ids", []) or []:
            by_submission[str(submission_id)] = contact_info
        for contact in contacts:
            user_id = str(contact.get("user_id", ""))
            if user_id:
                by_user[user_id] = contact_info

    return by_submission, by_user


def _build_failure_tables(
    summary: pd.DataFrame,
    metadata: dict[str, Any],
    run_dir: Path,
    phones_by_user: dict[str, str],
) -> pd.DataFrame:
    required_columns = {"submission_id", "user_id", "status"}
    missing = sorted(required_columns.difference(summary.columns))
    if missing:
        raise ValueError(f"submissions_summary.csv 缺少列: {', '.join(missing)}")

    failed = summary.loc[summary["status"].fillna("").ne("success")].copy()
    contacts_by_submission, contacts_by_user = _contact_maps(metadata, phones_by_user)
    contact_columns = [
        "participant_type",
        "participant_name",
        "contact_names",
        "contact_user_ids",
        "contact_phones",
    ]

    for column in contact_columns:
        failed[column] = failed["submission_id"].map(
            lambda sid: contacts_by_submission.get(str(sid), {}).get(column)
        )
        fallback = failed["user_id"].map(
            lambda uid: contacts_by_user.get(str(uid), {}).get(column)
        )
        failed[column] = failed[column].fillna(fallback)

    failed["stdout_path"] = failed["submission_id"].map(
        lambda sid: str(run_dir / "submissions" / str(sid) / "stdout")
    )
    output_columns = [
        "submission_id",
        "code_file",
        "status",
        "error",
        "participant_type",
        "participant_name",
        "contact_names",
        "contact_user_ids",
        "contact_phones",
        "elapsed_seconds",
        "stdout_path",
    ]
    for column in output_columns:
        if column not in failed.columns:
            failed[column] = pd.NA
    by_submission = failed[output_columns].sort_values(
        ["participant_name", "submission_id"], na_position="last"
    )

    return by_submission


def _write_full_log(
    by_submission: pd.DataFrame,
    output_path: Path,
    total_count: int,
) -> None:
    error_counts = (
        by_submission["error"]
        .fillna("未记录错误")
        .astype(str)
        .value_counts(dropna=False)
    )
    lines = [
        "失败 Submission 日志汇总",
        "=" * 100,
        f"competition_id: {competition_id}",
        f"batch_id: {batch_id}",
        f"总提交数: {total_count}",
        f"失败提交数: {len(by_submission)}",
        f"涉及参赛方: {by_submission['participant_name'].nunique(dropna=True)}",
        "",
        "错误类型统计:",
    ]
    lines.extend(f"  {count:>4}  {error}" for error, count in error_counts.items())

    with output_path.open("w", encoding="utf-8") as output:
        output.write("\n".join(lines).rstrip() + "\n")
        for index, row in enumerate(by_submission.itertuples(index=False), start=1):
            stdout_path = Path(row.stdout_path)
            block = [
                "",
                "#" * 100,
                f"[{index}/{len(by_submission)}] submission_id: {row.submission_id}",
                "#" * 100,
                f"participant: {_display_value(row.participant_name)} "
                f"({_display_value(row.participant_type)})",
                f"contacts: {_display_value(row.contact_names)}",
                f"phones: {_display_value(row.contact_phones)}",
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
            output.write("\n".join(block).rstrip() + "\n")


def _write_public_markdown(
    by_submission: pd.DataFrame,
    output_path: Path,
) -> None:
    """生成可公开转发的失败通知，仅公开提交与参赛方的对应关系。"""
    failed_submission_count = len(by_submission)
    submissions_by_participant: dict[str, list[object]] = {}
    for row in by_submission.itertuples(index=False):
        participant_name = _display_value(row.participant_name)
        submission_ids = submissions_by_participant.setdefault(participant_name, [])
        if row.submission_id not in submission_ids:
            submission_ids.append(row.submission_id)
    failed_participant_count = len(submissions_by_participant)

    if failed_submission_count == 0:
        lines = [
            "# 提交运行结果通知",
            "",
            "本次检查未发现运行失败的提交，无需参赛者进行额外处理。",
        ]
    else:
        lines = [
            "# 提交运行异常通知",
            "",
            "本次检查发现部分参赛者的提交未能成功完成运行。",
            "",
            f"- 受影响提交数：{failed_submission_count}",
            f"- 受影响参赛方数：{failed_participant_count}",
            "",
            "## 运行失败的提交",
            "",
            "| 参赛方 | Submission IDs |",
            "| --- | --- |",
        ]
        for participant_name, submission_ids in submissions_by_participant.items():
            escaped_name = _markdown_cell(participant_name)
            escaped_ids = "<br>".join(
                _markdown_cell(submission_id) for submission_id in submission_ids
            )
            lines.append(f"| {escaped_name} | {escaped_ids} |")

        lines.extend([
            "",
            "## 请受影响的参赛者及时处理",
            "",
            "请主动联系主办方，确认提交失败的具体原因及后续处理方式。联系时，"
            "请私聊主办方并提供必要信息，不要在公开群聊或公开文档中"
            "发送账号、手机号、代码、日志等敏感内容。",
            "",
            "> 本通知为公开版汇总，除上述 Submission ID 和参赛方名称外，不包含"
            "联系方式、用户 ID、错误详情或内部文件路径。具体情况请以主办方核查"
            "结果为准。",
        ])

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    cid = competition_id.strip()
    bid = batch_id.strip()
    if not cid or not bid:
        raise ValueError("competition_id 和 batch_id 不能为空")

    private_root = _resolve_private_root(cid)
    run_dir = private_root / "runs" / bid
    summary_path = _required_file(
        run_dir / "artifacts" / "submissions_summary.csv",
        "提交汇总文件",
    )
    metadata_path = _required_file(
        private_root / "prepared" / "metadata.json",
        "参赛者 metadata 文件",
    )

    summary = pd.read_csv(
        summary_path,
        dtype={"submission_id": str, "user_id": str},
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    phones_by_user = _load_phones(cid)
    by_submission = _build_failure_tables(summary, metadata, run_dir, phones_by_user)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT / f"{cid}__{bid}__failed_submissions.log"
    markdown_path = OUTPUT_ROOT / f"{cid}__{bid}__failed_submissions.md"
    _write_full_log(by_submission, log_path, len(summary))
    _write_public_markdown(by_submission, markdown_path)

    print(f"总提交数: {len(summary)}")
    print(f"未跑成功提交: {len(by_submission)}")
    print(f"需联系参赛方: {by_submission['participant_name'].nunique(dropna=True)}")
    print(f"完整日志: {log_path}")
    print(f"公开通知: {markdown_path}")


if __name__ == "__main__":
    main()
