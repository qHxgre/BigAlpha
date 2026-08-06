"""按团队取最高分 submission，把对应文件夹复制到 downloads 目录，方便下载。

口径（与 competition_stats.py 的团队排行一致）：
    团队成员 = 队长（team.creator）+ team.members，去重保序；
    团队最高分 submission = 全体成员名下所有 submission 中 public_score 最大的一条
    （未评分/None 不参与比较）；若并列最高分，取先遇到的那条（不保证时间顺序）。

复制规则：
    源目录 system/files/{competition_id}/submissions/{submission_id}/ 复制到
    目标目录 system/files/{competition_id}/downloads/
        {rank}_{team_name}_{team_id}/{user_id}/{submission_id}/，
    即先按团队、再按成员分组；入选团队中每位成员只复制得分最高的 3 个
    submission（未评分的不参与排名），
    只复制、不移动、不删除原目录；目标目录已存在则跳过（不重复复制）。
    复制时跳过：submission 目录下的子文件夹（如 __pycache__、数据缓存目录等）、
    以及 .parquet 文件，只保留根一级的其它文件（代码、stdout、json 等）。

要查哪场比赛，改下面的 COMPETITION_ID，然后跑：
    python system/scripts/submissions/collect_best_submissions.py       # 先看会复制哪些
    python system/scripts/submissions/collect_best_submissions.py -y    # 确认后直接复制

输出：downloads 目录下每队一个文件夹、队内按成员分组；另在该目录写一份
best_submissions_<date>.json 汇总。每条记录包含团队得分，以及每个成员的姓名、
学校、全部 submission 和对应得分（按得分从高到低排序）。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.client import AlphathonClient
from common.ids import dedup_keep_order
from common.paths import resolve_downloads_dir, resolve_submissions_dir

# ===== 配置：要查哪场比赛 =====================================================
COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
# 只转移团队得分排名前 N 的 submission（按 team_score 从高到低）
TOP_N = 50
# 每位成员最多转移得分最高的 N 个 submission（未评分的不参与排名）
MEMBER_TOP_N = 3
# ===========================================================================


def _to_float(x: object) -> float | None:
    """把 API 返回的分数（数字/字符串/None）安全转成 float，转不了则 None。"""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def copy_submission_files(src: Path, dest: Path) -> None:
    """把 src 下根一级的文件复制到 dest，跳过子文件夹和 .parquet 文件。"""
    dest.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_dir():
            continue
        if entry.suffix.lower() == ".parquet":
            continue
        shutil.copy2(entry, dest / entry.name)


def _safe_path_part(value: object, fallback: str) -> str:
    """生成适合目录名的短文本；保留中文，只替换路径/控制字符。"""
    text = str(value or "").strip()
    text = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text[:80] or fallback


def _member_submissions(uid: str, subs_by_user: dict[str, list[dict]]) -> list[dict]:
    """某成员名下所有 submission 及得分，按得分从高到低排序（未评分排最后）。"""
    items = [
        {
            "submission_id": str(s.get("id")),
            "score": _to_float(s.get("public_score")),
        }
        for s in subs_by_user.get(uid, [])
    ]
    items.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0.0)))
    return items


def _build_user_index(users: list[dict]) -> dict[str, dict[str, str]]:
    """报名记录 -> 用户姓名/学校；缺失字段使用稳定的兜底值。"""
    result: dict[str, dict[str, str]] = {}
    for user in users:
        uid = str(user.get("user_id"))
        data = user.get("data") or {}
        result[uid] = {
            "name": str(data.get("name") or uid),
            "school": str(data.get("school") or "").strip() or "（未填写）",
        }
    return result


def best_submission_per_team(client: AlphathonClient, competition_id: str) -> list[dict]:
    """按团队取最高分 submission，返回每队一条记录（team_id/team_name/best_score/
    best_submission_id/best_user_id/members），无有效评分的团队 best_submission_id
    为 None。members 中记录姓名、学校及每个成员名下的所有 submission 和得分。
    """
    users = client.list_users(competition_id)
    teams = client.list_teams(competition_id)
    subs = client.list_submissions(competition_id)
    user_index = _build_user_index(users)

    subs_by_user: dict[str, list[dict]] = defaultdict(list)
    for s in subs:
        subs_by_user[str(s.get("user_id"))].append(s)

    rows: list[dict] = []
    for t in teams:
        roster = dedup_keep_order([t.get("creator"), *(t.get("members") or [])])
        best_score: float | None = None
        best_sub_id: str | None = None
        best_user_id: str | None = None
        for uid in roster:
            for s in subs_by_user.get(uid, []):
                score = _to_float(s.get("public_score"))
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = score
                    best_sub_id = str(s.get("id"))
                    best_user_id = uid
        members = []
        for uid in roster:
            profile = user_index.get(uid, {"name": uid, "school": "（未填写）"})
            members.append({
                "user_id": uid,
                "name": profile["name"],
                "school": profile["school"],
                "submissions": _member_submissions(uid, subs_by_user),
            })
        rows.append({
            "team_id": str(t.get("id")),
            "team_name": t.get("name"),
            "team_score": best_score,
            "best_submission_id": best_sub_id,
            "best_user_id": best_user_id,
            "members": members,
        })

    rows.sort(key=lambda r: (r["team_score"] is None, -(r["team_score"] or 0.0)))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _find_member(row: dict, user_id: str) -> dict:
    """从团队记录找到 submission 所属成员。正常情况下必定存在。"""
    return next(
        (member for member in row["members"] if member["user_id"] == user_id),
        {"user_id": user_id, "name": user_id, "school": "（未填写）", "submissions": []},
    )


def _submission_dest(downloads_dir: Path, row: dict, member: dict, submission_id: str) -> Path:
    """团队/成员/submission 三级目标目录。ID 保证同名团队/成员不会冲突。"""
    team_dir = (
        f"{row['rank']:02d}_"
        f"{_safe_path_part(row['team_name'], 'unnamed_team')}_"
        f"{_safe_path_part(row['team_id'], 'unknown_team')}"
    )
    member_dir = _safe_path_part(member["user_id"], "unknown_user")
    return downloads_dir / team_dir / member_dir / submission_id


def _copy_jobs(downloads_dir: Path, submissions_dir: Path, rows: list[dict]) -> list[dict]:
    """把入选团队展开为逐成员的高分 submission 复制任务。"""
    jobs: list[dict] = []
    for row in rows:
        for member in row["members"]:
            scored_submissions = [
                submission
                for submission in member["submissions"]
                if submission["score"] is not None
            ]
            for submission in scored_submissions[:MEMBER_TOP_N]:
                submission_id = submission["submission_id"]
                jobs.append({
                    "team": row,
                    "member": member,
                    "submission": submission,
                    "src": submissions_dir / submission_id,
                    "dest": _submission_dest(downloads_dir, row, member, submission_id),
                })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-y", "--yes", action="store_true", help="跳过确认，直接复制")
    args = parser.parse_args()

    submissions_dir = resolve_submissions_dir(COMPETITION_ID)
    if not submissions_dir.is_dir():
        print(f"提交目录不存在: {submissions_dir}", file=sys.stderr)
        sys.exit(1)

    client = AlphathonClient()
    rows = best_submission_per_team(client, COMPETITION_ID)

    print(f"=== 比赛 {COMPETITION_ID}：共 {len(rows)} 支团队 ===")
    no_score = [r for r in rows if r["best_submission_id"] is None]
    with_score = [r for r in rows if r["best_submission_id"] is not None]
    for r in with_score:
        member = _find_member(r, r["best_user_id"])
        print(f"  #{r['rank']} {r['team_name']}  团队得分: {r['team_score']:.5f}  "
              f"submission: {r['best_submission_id']}  提交人: {member['name']}")
    if no_score:
        print(f"\n以下 {len(no_score)} 支团队没有已评分的 submission，跳过：")
        for r in no_score:
            print(f"  {r['team_name']} ({r['team_id']})")

    if not with_score:
        print("\n没有任何团队有可复制的 submission，结束。")
        return

    with_score = with_score[:TOP_N]
    print(f"\n按得分取前 {TOP_N} 支团队，实际 {len(with_score)} 支参与复制。")

    downloads_dir = resolve_downloads_dir(COMPETITION_ID)
    print(f"目标目录: {downloads_dir}")

    jobs = _copy_jobs(downloads_dir, submissions_dir, with_score)
    print(f"每位成员最多取得分前 {MEMBER_TOP_N} 名，共包含 {len(jobs)} 个 submission。")
    missing_jobs = [job for job in jobs if not job["src"].is_dir()]
    for job in missing_jobs:
        print(
            f"  [警告] 源目录不存在，跳过: {job['submission']['submission_id']} "
            f"({job['team']['team_name']} / {job['member']['name']})",
            file=sys.stderr,
        )
    todo = [job for job in jobs if job["src"].is_dir()]

    if not args.yes:
        answer = input(f"\n确认复制以上 {len(todo)} 个 submission 目录到 downloads 吗？[y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消，未复制任何文件。")
            return

    downloads_dir.mkdir(parents=True, exist_ok=True)
    copied = skipped = failed = 0
    copy_results: dict[tuple[str, str, str], dict] = {}
    for row in with_score:
        for member in row["members"]:
            for submission in member["submissions"]:
                key = (row["team_id"], member["user_id"], submission["submission_id"])
                copy_results[key] = {"copy_status": "not_selected"}

    for job in missing_jobs:
        key = (
            job["team"]["team_id"],
            job["member"]["user_id"],
            job["submission"]["submission_id"],
        )
        copy_results[key] = {
            "dest": str(job["dest"]),
            "copy_status": "source_missing",
        }

    for job in todo:
        member = job["member"]
        submission = job["submission"]
        sub_id = submission["submission_id"]
        src = job["src"]
        dest = job["dest"]
        key = (job["team"]["team_id"], member["user_id"], sub_id)
        if dest.exists():
            print(f"  [已存在，跳过] {dest}")
            skipped += 1
            copy_results[key] = {"dest": str(dest), "copy_status": "already_exists"}
            continue
        try:
            copy_submission_files(src, dest)
            print(f"  [已复制] {src} -> {dest}")
            copied += 1
            copy_results[key] = {"dest": str(dest), "copy_status": "copied"}
        except OSError as e:
            print(f"  [失败] {sub_id} -> {e}", file=sys.stderr)
            failed += 1
            copy_results[key] = {
                "dest": str(dest),
                "copy_status": "failed",
                "error": str(e),
            }

    print(
        f"\n汇总：已复制 {copied}，已存在跳过 {skipped}，"
        f"源目录缺失 {len(missing_jobs)}，失败 {failed}"
    )

    manifest = []
    for row in with_score:
        manifest_row = {**row, "members": []}
        for member in row["members"]:
            manifest_member = {**member, "submissions": []}
            for submission in member["submissions"]:
                key = (row["team_id"], member["user_id"], submission["submission_id"])
                manifest_member["submissions"].append({
                    **submission,
                    **copy_results[key],
                })
            manifest_row["members"].append(manifest_member)
        manifest.append(manifest_row)

    date_str = datetime.now().strftime("%Y%m%d")
    manifest_path = downloads_dir / f"best_submissions_{date_str}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出汇总: {manifest_path}")


if __name__ == "__main__":
    main()
