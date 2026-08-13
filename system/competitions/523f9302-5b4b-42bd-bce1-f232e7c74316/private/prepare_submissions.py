"""固化私榜提交；未主动选择者自动取公榜得分最高的一份。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
API_SERVER = os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver"))
for path in (API_SERVER, HERE):
    if path not in sys.path:
        sys.path.append(path)

from judge.api import AlphathonAPI
from judge.paths import FILE_DIR
from fileio import jsonable, write_json

COMPETITION_ID = "523f9302-5b4b-42bd-bce1-f232e7c74316"
DEFAULT_OUTPUT = os.path.join(FILE_DIR, COMPETITION_ID, "private", "prepared")


def paginate(api: AlphathonAPI, path: str, **params) -> list[dict]:
    """读取分页 API 的全部记录。"""
    results = []
    page = 1
    while True:
        response = api._request(
            "GET", path, params={**params, "page": page, "size": 1000}
        ).json()
        items = ((response or {}).get("data") or {}).get("items") or []
        results.extend(items)
        if len(items) < 1000:
            return results
        page += 1


def profile(user: dict | None, user_id: str) -> dict:
    """提取检查失败 submission 时所需的参赛者公开资料。"""
    data = (user or {}).get("data") or {}
    return {
        "user_id": user_id,
        "name": data.get("name") or user_id,
        "school": data.get("school") or "（未填写）",
    }


def submission_files(submission: dict) -> dict:
    files = (submission.get("data") or {}).get("files")
    if files is None:
        files = submission.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"files 字段类型错误: {type(files).__name__}")
    return files


def safe_name(file_id: str, info: dict | None, used: set[str]) -> str:
    raw = (info or {}).get("name") or file_id
    name = os.path.basename(str(raw).replace("\\", "/")) or file_id
    stem, suffix = os.path.splitext(name)
    candidate, index = name, 2
    while candidate.casefold() in used:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def select_submissions(
    all_submissions: list[dict],
    explicitly_selected: list[dict],
    team_by_user: dict[str, str],
) -> tuple[list[dict], dict[str, str]]:
    """按参赛者归组，未主动选择者自动取公榜得分最高的一份。"""
    explicit_ids = {str(item["id"]) for item in explicitly_selected}

    def owner(submission: dict) -> tuple[str, str]:
        user_id = str(submission.get("user_id"))
        team_id = team_by_user.get(user_id)
        return ("team", team_id) if team_id else ("individual", user_id)

    explicitly_selected_owners = {owner(item) for item in explicitly_selected}
    fallback_by_owner: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for submission in all_submissions:
        submission_owner = owner(submission)
        if submission_owner in explicitly_selected_owners:
            continue
        if submission.get("public_score") is None:
            continue
        fallback_by_owner[submission_owner].append(submission)

    fallback = []
    for submissions in fallback_by_owner.values():
        submissions.sort(
            key=lambda item: (
                float(item["public_score"]),
                str(item.get("created_at") or ""),
                str(item["id"]),
            ),
            reverse=True,
        )
        fallback.extend(submissions[:1])

    selected = [*explicitly_selected, *fallback]
    sources = {
        str(item["id"]): (
            "selected_for_private" if str(item["id"]) in explicit_ids else "public_top_1"
        )
        for item in selected
    }
    return selected, sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-id", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    output = os.path.abspath(args.output_root)
    os.makedirs(output, exist_ok=True)
    staging = os.path.join(output, f".preparing-{os.getpid()}-{time.time_ns()}")
    os.makedirs(os.path.join(staging, "submissions"))

    api = AlphathonAPI()
    explicitly_selected = api.query_submissions(
        competition_id=COMPETITION_ID,
        constraints={"selected_for_private": True},
    )
    all_submissions = api.query_submissions(competition_id=COMPETITION_ID)

    users = paginate(api, "/users", competition_id=COMPETITION_ID)
    teams = paginate(api, "/teams", competition_id=COMPETITION_ID)
    users_by_id = {str(user.get("user_id")): user for user in users}

    team_by_user: dict[str, str] = {}
    teams_by_id: dict[str, tuple[dict, list[str]]] = {}
    for team in teams:
        team_id = str(team["id"])
        roster = list(
            dict.fromkeys(
                [str(team.get("creator")), *(str(uid) for uid in team.get("members") or [])]
            )
        )
        teams_by_id[team_id] = (team, roster)
        for user_id in roster:
            team_by_user[user_id] = team_id

    selected, selection_sources = select_submissions(
        all_submissions, explicitly_selected, team_by_user
    )
    if not selected:
        shutil.rmtree(staging, ignore_errors=True)
        print("没有主动选择或可按公榜得分自动选择的 submission", file=sys.stderr)
        return 1

    selected_by_owner: dict[tuple[str, str], list[dict]] = defaultdict(list)
    records, errors = [], []
    for submission in selected:
        sid = str(submission["id"])
        user_id = str(submission.get("user_id"))
        team_id = team_by_user.get(user_id)
        destination = os.path.join(staging, "submissions", sid)
        os.makedirs(destination)
        downloaded, used = [], set()
        try:
            for file_id, info in submission_files(submission).items():
                if info is not None and not isinstance(info, dict):
                    raise RuntimeError(f"文件 {file_id} 元数据类型错误")
                name = safe_name(str(file_id), info, used)
                api.get_submission_file(sid, str(file_id), info, save_to=os.path.join(destination, name))
                downloaded.append({"file_id": str(file_id), "name": name})
            record = {
                "submission_id": sid,
                "user_id": user_id,
                "team_id": team_id,
                "public_score": submission.get("public_score"),
                "selection_source": selection_sources[sid],
                "relative_path": f"submissions/{sid}",
                "files": downloaded,
                "submission": jsonable(submission),
            }
            records.append(record)
            owner = ("team", team_id) if team_id else ("individual", user_id)
            selected_by_owner[owner].append(record)
        except Exception as exc:
            errors.append({"submission_id": sid, "error": f"{type(exc).__name__}: {exc}"})
            shutil.rmtree(destination, ignore_errors=True)

    if errors:
        write_json(os.path.join(output, "preparation_errors.json"), {"errors": errors})
        shutil.rmtree(staging, ignore_errors=True)
        print(f"{len(errors)} 个 submission 下载失败", file=sys.stderr)
        return 1

    participants = []
    for (participant_type, participant_id), owner_records in selected_by_owner.items():
        if participant_type == "team":
            team, roster = teams_by_id[participant_id]
            participant = {
                "type": "team",
                "team_id": participant_id,
                "team_name": team.get("name") or participant_id,
                "members": [profile(users_by_id.get(uid), uid) for uid in roster],
            }
        else:
            participant = {
                "type": "individual",
                "user": profile(users_by_id.get(participant_id), participant_id),
            }
        participant.update(
            private_submission_count=len(owner_records),
            private_submission_ids=[item["submission_id"] for item in owner_records],
        )
        participants.append(participant)
    participants.sort(
        key=lambda item: (
            item["type"],
            str(item.get("team_name") or (item.get("user") or {}).get("name") or ""),
        )
    )

    metadata = {
        "competition_id": COMPETITION_ID,
        "batch_id": args.batch_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {
            "private_submission_count": len(records),
            "explicit_submission_count": len(explicitly_selected),
            "fallback_submission_count": len(records) - len(explicitly_selected),
            "private_participant_count": len(participants),
            "private_team_count": sum(item["type"] == "team" for item in participants),
            "private_individual_count": sum(
                item["type"] == "individual" for item in participants
            ),
        },
        "participants": participants,
        "submissions": records,
    }
    write_json(os.path.join(staging, "metadata.json"), metadata)
    target = os.path.join(output, "submissions")
    if os.path.exists(target):
        shutil.rmtree(target)
    os.replace(os.path.join(staging, "submissions"), target)
    os.replace(os.path.join(staging, "metadata.json"), os.path.join(output, "metadata.json"))
    shutil.rmtree(staging, ignore_errors=True)
    error_path = os.path.join(output, "preparation_errors.json")
    if os.path.exists(error_path):
        os.remove(error_path)
    print(f"prepared submissions: {len(records)}; output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
