"""固化私榜入围 submission 及其公榜背景信息，供 private.py 离线评测。"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
API_SERVER = os.getenv(
    "ALPHATHON_API_SERVER_DIR",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver")),
)
if API_SERVER not in sys.path:
    sys.path.append(API_SERVER)

from judge.api import AlphathonAPI
from judge.paths import FILE_DIR

from fileio import jsonable, write_json


COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
DEFAULT_OUTPUT_ROOT = os.path.join(FILE_DIR, COMPETITION_ID, "private", "prepared")


def _paginate(api: AlphathonAPI, path: str, **params) -> list[dict]:
    results = []
    page = 1
    while True:
        response = api._request(
            "GET", path, params={**params, "page": page, "size": 1000}
        ).json()
        data = (response or {}).get("data") or {}
        items = data.get("items") or []
        results.extend(items)
        if not items or len(items) < 1000:
            return results
        page += 1


def _profile(user: dict | None, user_id: str) -> dict:
    data = (user or {}).get("data") or {}
    return {
        "user_id": user_id,
        "name": data.get("name") or user_id,
        "school": data.get("school") or "（未填写）",
    }


def _public_ranks(entries: list[dict], rank_order: str) -> dict[tuple[str, str], int]:
    scored = [entry for entry in entries if entry.get("public_score") is not None]
    scored.sort(
        key=lambda entry: float(entry["public_score"]),
        reverse=rank_order != "asc",
    )
    return {
        (entry["type"], str(entry.get("team_id") or entry.get("user_id"))): rank
        for rank, entry in enumerate(scored, 1)
    }


def _submission_files(submission: dict) -> dict:
    """兼容 files 位于 data 内或提交顶层的 API 返回。"""
    data = submission.get("data") or {}
    files = data.get("files")
    if files is None:
        files = submission.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"files 字段类型错误: {type(files).__name__}")
    return files


def _load_notebook(path: str) -> dict | None:
    """按内容识别 notebook，避免只依赖不可靠的文件名后缀。"""
    try:
        with open(path, encoding="utf-8-sig") as reader:
            value = json.load(reader)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("cells"), list):
        return None
    if "nbformat" not in value:
        return None
    return value


def _safe_file_name(file_id: str, file_info: dict | None, used: set[str]) -> str:
    info = file_info if isinstance(file_info, dict) else {}
    raw_name = info.get("name") or str(file_id)
    name = os.path.basename(str(raw_name).replace("\\", "/")) or str(file_id)
    stem, suffix = os.path.splitext(name)
    candidate = name
    index = 2
    while candidate.casefold() in used:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def _download_submission(api: AlphathonAPI, submission: dict, destination: str) -> list[dict]:
    sid = str(submission["id"])
    os.makedirs(destination)
    downloaded = []
    notebooks: list[tuple[str, dict]] = []
    files = _submission_files(submission)
    used_names: set[str] = set()
    for file_id, file_info in files.items():
        if file_info is not None and not isinstance(file_info, dict):
            raise RuntimeError(
                f"文件 {file_id} 的元数据类型错误: {type(file_info).__name__}"
            )
        name = _safe_file_name(str(file_id), file_info, used_names)
        if os.path.splitext(name)[1].lower() == ".parquet":
            continue
        path = os.path.join(destination, name)
        api.get_submission_file(sid, str(file_id), file_info, save_to=path)
        downloaded.append({"file_id": str(file_id), "name": name})
        notebook = _load_notebook(path)
        if notebook is not None:
            notebooks.append((path, notebook))
    if len(notebooks) != 1:
        file_summary = [
            {
                "file_id": str(file_id),
                "name": file_info.get("name") if isinstance(file_info, dict) else None,
            }
            for file_id, file_info in files.items()
        ]
        raise RuntimeError(
            f"submission {sid} 包含 {len(notebooks)} 个有效 notebook，要求恰好 1 个；"
            f"API 文件清单={json.dumps(file_summary, ensure_ascii=False)}"
        )

    notebook_path, notebook = notebooks[0]
    notebook_name = os.path.basename(notebook_path)
    notebook_file = next(item for item in downloaded if item["name"] == notebook_name)
    code_cells = []
    for cell in notebook.get("cells", []):
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            source = cell.get("source", [])
            code_cells.append("".join(source) if isinstance(source, list) else str(source))
    code_path = os.path.join(destination, "submission_code.py")
    with open(code_path, "w", encoding="utf-8") as writer:
        writer.write("\n\n".join(code_cells))
    os.remove(notebook_path)
    downloaded.remove(notebook_file)
    downloaded.append(
        {
            "file_id": None,
            "name": "submission_code.py",
            "generated": True,
            "source_file_id": notebook_file["file_id"],
            "source_name": notebook_file["name"],
        }
    )
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="固定输入包目录")
    parser.add_argument(
        "--batch-id",
        default=time.strftime("%Y%m%d_%H%M%S"),
        help="仅写入 metadata，不再用于创建分批目录",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_root)
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, f".preparing-{os.getpid()}-{time.time_ns()}")
    os.makedirs(os.path.join(work_dir, "submissions"))
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)

    api = AlphathonAPI()
    competition = api.get_competition_by_id(COMPETITION_ID) or {}
    selected = api.query_submissions(
        competition_id=COMPETITION_ID,
        constraints={"selected_for_private": True},
    )
    if not selected:
        print("没有 selected_for_private=True 的 submission", file=sys.stderr)
        return 1

    users = _paginate(api, "/users", competition_id=COMPETITION_ID)
    teams = _paginate(api, "/teams", competition_id=COMPETITION_ID)
    leaderboard = _paginate(api, f"/leaderboard/{COMPETITION_ID}")
    users_by_id = {str(user.get("user_id")): user for user in users}
    rank_order = ((competition.get("data") or {}).get("competition") or {}).get(
        "rank_order", "desc"
    )
    public_ranks = _public_ranks(leaderboard, rank_order)
    leaderboard_by_key = {
        (entry["type"], str(entry.get("team_id") or entry.get("user_id"))): entry
        for entry in leaderboard
    }

    team_by_user = {}
    teams_by_id = {}
    for team in teams:
        tid = str(team["id"])
        roster = list(dict.fromkeys([str(team.get("creator")), *(str(x) for x in team.get("members") or [])]))
        teams_by_id[tid] = (team, roster)
        for uid in roster:
            team_by_user[uid] = tid

    selected_by_owner = defaultdict(list)
    submission_records = []
    preparation_errors = []
    for submission in selected:
        sid = str(submission["id"])
        uid = str(submission.get("user_id"))
        tid = team_by_user.get(uid)
        destination = os.path.join(work_dir, "submissions", sid)
        try:
            files = _download_submission(api, submission, destination)
        except Exception as exc:
            shutil.rmtree(destination, ignore_errors=True)
            preparation_errors.append(
                {
                    "submission_id": sid,
                    "user_id": uid,
                    "error": f"{type(exc).__name__}: {exc}",
                    "submission": jsonable(submission),
                }
            )
            print(f"准备 submission {sid} 失败: {exc}", file=sys.stderr)
            continue
        record = {
            "submission_id": sid,
            "user_id": uid,
            "team_id": tid,
            "public_score": submission.get("public_score"),
            "created_at": submission.get("created_at"),
            "relative_path": f"submissions/{sid}",
            "code_file": f"submissions/{sid}/submission_code.py",
            "files": files,
            "submission": submission,
        }
        submission_records.append(record)
        selected_by_owner[("team", tid) if tid else ("individual", uid)].append(record)

    if preparation_errors:
        error_path = os.path.join(output_dir, "preparation_errors.json")
        write_json(
            error_path,
            {
                "competition_id": COMPETITION_ID,
                "batch_id": args.batch_id,
                "selected_submission_count": len(selected),
                "prepared_submission_count": len(submission_records),
                "failed_submission_count": len(preparation_errors),
                "errors": preparation_errors,
            },
        )
        shutil.rmtree(work_dir, ignore_errors=True)
        print(
            f"准备失败: {len(preparation_errors)}/{len(selected)} 个 submission；"
            f"详情: {error_path}",
            file=sys.stderr,
        )
        return 1

    participant_records = []
    for (participant_type, participant_id), records in selected_by_owner.items():
        key = (participant_type, participant_id)
        board = leaderboard_by_key.get(key, {})
        if participant_type == "team":
            team, roster = teams_by_id[participant_id]
            participant = {
                "type": "team",
                "team_id": participant_id,
                "team_name": team.get("name"),
                "members": [_profile(users_by_id.get(uid), uid) for uid in roster],
            }
        else:
            participant = {
                "type": "individual",
                "user": _profile(users_by_id.get(participant_id), participant_id),
            }
        participant.update(
            public_score=board.get("public_score"),
            public_rank=public_ranks.get(key),
            private_submission_count=len(records),
            private_submission_ids=[item["submission_id"] for item in records],
        )
        participant_records.append(participant)
    participant_records.sort(key=lambda item: item.get("public_rank") or float("inf"))

    user_counts = Counter(str(item.get("user_id")) for item in selected)
    metadata = {
        "competition_id": COMPETITION_ID,
        "competition_name": competition.get("name"),
        "batch_id": args.batch_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "rank_order": rank_order,
        "summary": {
            "private_submission_count": len(selected),
            "private_user_count": len(user_counts),
            "private_team_count": sum(x["type"] == "team" for x in participant_records),
            "private_individual_count": sum(x["type"] == "individual" for x in participant_records),
        },
        "participants": participant_records,
        "submissions": submission_records,
    }
    staged_metadata = os.path.join(work_dir, "metadata.json")
    write_json(staged_metadata, jsonable(metadata))

    submissions_dir = os.path.join(output_dir, "submissions")
    if os.path.exists(submissions_dir):
        shutil.rmtree(submissions_dir)
    os.replace(os.path.join(work_dir, "submissions"), submissions_dir)
    os.replace(staged_metadata, os.path.join(output_dir, "metadata.json"))
    shutil.rmtree(work_dir, ignore_errors=True)
    error_path = os.path.join(output_dir, "preparation_errors.json")
    if os.path.exists(error_path):
        os.remove(error_path)
    print(f"private submissions: {len(selected)}")
    print(f"participants: {len(participant_records)}")
    print(f"prepared input: {output_dir}")
    print(f"run: python private.py --input {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
