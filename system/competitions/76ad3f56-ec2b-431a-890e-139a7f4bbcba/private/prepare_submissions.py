"""固化私榜入围 submission 及其公榜背景信息，供 private.py 离线评测。"""
from __future__ import annotations

import argparse
import json
import os
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


def _download_submission(api: AlphathonAPI, submission: dict, destination: str) -> list[dict]:
    os.makedirs(destination)
    downloaded = []
    notebooks = []
    files = (submission.get("data") or {}).get("files") or {}
    for file_id, file_info in files.items():
        name = os.path.basename((file_info or {}).get("name") or str(file_id))
        if os.path.splitext(name)[1].lower() == ".parquet":
            continue
        path = os.path.join(destination, name)
        api.get_submission_file(str(submission["id"]), str(file_id), file_info, save_to=path)
        downloaded.append({"file_id": str(file_id), "name": name})
        if name.lower().endswith(".ipynb"):
            notebooks.append(path)
    if len(notebooks) != 1:
        raise RuntimeError(
            f"submission {submission['id']} 包含 {len(notebooks)} 个 notebook，要求恰好 1 个"
        )

    with open(notebooks[0], encoding="utf-8") as reader:
        notebook = json.load(reader)
    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            code_cells.append("".join(source) if isinstance(source, list) else str(source))
    code_path = os.path.join(destination, "submission_code.py")
    with open(code_path, "w", encoding="utf-8") as writer:
        writer.write("\n\n".join(code_cells))
    downloaded.append({"file_id": None, "name": "submission_code.py", "generated": True})
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="输入包父目录")
    parser.add_argument("--batch-id", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    output_dir = os.path.abspath(os.path.join(args.output_root, args.batch_id))
    if os.path.exists(output_dir):
        print(f"目标目录已存在，拒绝覆盖: {output_dir}", file=sys.stderr)
        return 1
    os.makedirs(os.path.join(output_dir, "submissions"))

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
    for submission in selected:
        sid = str(submission["id"])
        uid = str(submission.get("user_id"))
        tid = team_by_user.get(uid)
        files = _download_submission(
            api, submission, os.path.join(output_dir, "submissions", sid)
        )
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
    write_json(os.path.join(output_dir, "metadata.json"), jsonable(metadata))
    print(f"private submissions: {len(selected)}")
    print(f"participants: {len(participant_records)}")
    print(f"prepared input: {output_dir}")
    print(f"run: python private.py --input {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
