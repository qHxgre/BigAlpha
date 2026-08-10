"""失败提交及参赛者联系信息检查。"""

from __future__ import annotations

import json

import pandas as pd

from .common import read_summary, show
from .config import CheckPaths


def analyze_failed_submissions(
    paths: CheckPaths, *, stdout_tail_lines: int = 20, display: bool = True
) -> dict[str, pd.DataFrame]:
    """返回逐提交失败明细和按联系人聚合的私聊清单。"""
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
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-stdout_tail_lines:])

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
    print(f"总提交数: {len(summary)}，未跑成功提交: {len(by_submission)}，需联系参赛方: {len(by_contact)}")
    if display:
        show(by_contact, by_submission)
    return {"by_contact": by_contact, "by_submission": by_submission}
