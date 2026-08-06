"""查询两个比赛的参赛人数、提交次数、学校分布，以及按团队汇总的排行。

口径：
    参赛人数 = 该比赛全部报名记录数（AlphathonClient.list_users，不筛状态，
               含 pending/approved/approved_join_space/rejected，与
               participants.py 拉取全量报名的口径一致）。
    提交次数 = 该比赛全部 submission 记录数（AlphathonClient.list_submissions）。
    学校分布 = 按报名记录 data.school 分组计数（同一学校下的报名记录数，
               不去重 user_id；school 为空/缺失的归入"（未填写）"）。
    团队排行 = 团队成员（队长 + members，去重）中所有 submission 的
               public_score 最大值作为团队得分，按团队得分从高到低排序；
               每行列出成员名单、成员学校、总提交次数，以及每个成员的
               submission 得分均值与最高分（未评分的 submission 不计入
               均值/最高分，但计入提交次数）。
    学校排行榜 = 按报名记录 data.school 分组（同一 school 下的全部 user_id，
               不去重团队），每所学校内部按成员最高 public_score 排出校内
               排行；学校得分取校内最高分，学校之间按该分数从高到低排序。

要查哪两场比赛，改下面的 COMPETITION_IDS 即可，然后跑：
    python system/scripts/submissions/competition_stats.py

输出（DATA_ROOT/submissions，见 common.paths.DATA_ROOT）：
    competition_stats_<date>.json   结构化统计结果
    competition_stats_<date>.md     基于上述 json 生成的对外可读文档
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.client import AlphathonClient
from common.ids import dedup_keep_order
from common.paths import DATA_ROOT

# ===== 配置：要查哪两场比赛 ===================================================
COMPETITION_IDS: dict[str, str] = {
    "76ad3f56-ec2b-431a-890e-139a7f4bbcba": "赛道一·AI因子挖掘",
    "523f9302-5b4b-42bd-bce1-f232e7c74316": "赛道二·端到端AI量化模型",
}
# ===========================================================================

OUT_DIR = DATA_ROOT / "submissions"


def school_breakdown(users: list[dict]) -> Counter:
    """按 data.school 分组计数，school 缺失/空归入"（未填写）"。"""
    counter: Counter = Counter()
    for u in users:
        school = ((u.get("data") or {}).get("school") or "").strip() or "（未填写）"
        counter[school] += 1
    return counter


def _to_float(x: object) -> float | None:
    """把 API 返回的分数（数字/字符串/None）安全转成 float，转不了则 None。"""
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def build_user_index(users: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """报名记录 -> {user_id: 姓名} / {user_id: 学校}，姓名缺失回退 user_id。"""
    name_map: dict[str, str] = {}
    school_map: dict[str, str] = {}
    for u in users:
        uid = str(u.get("user_id"))
        data = u.get("data") or {}
        name_map[uid] = data.get("name") or uid
        school_map[uid] = (data.get("school") or "").strip() or "（未填写）"
    return name_map, school_map


def _make_member_stats_fn(users: list[dict], subs: list[dict]):
    """返回 (stats_for, name_map, school_map)：stats_for(uid) 计算该用户的
    name/school/submission_count/avg_score/max_score（未评分/None 不计入均值与
    最高分，但计入提交次数）。team/school 两种汇总口径共用这份逐用户统计。
    """
    name_map, school_map = build_user_index(users)

    subs_by_user: dict[str, list[dict]] = defaultdict(list)
    for s in subs:
        subs_by_user[str(s.get("user_id"))].append(s)

    def stats_for(uid: str) -> dict:
        user_subs = subs_by_user.get(uid, [])
        scores = [sc for sc in (_to_float(s.get("public_score")) for s in user_subs) if sc is not None]
        return {
            "user_id": uid,
            "name": name_map.get(uid, uid),
            "school": school_map.get(uid, "（未填写）"),
            "submission_count": len(user_subs),
            "avg_score": (sum(scores) / len(scores)) if scores else None,
            "max_score": max(scores) if scores else None,
        }

    return stats_for, name_map, school_map


def build_team_stats(client: AlphathonClient, competition_id: str) -> list[dict]:
    """按团队汇总：团队得分取全体成员所有 submission 中的最高 public_score，
    按该分数从高到低排序；每行含成员名单/学校/提交次数，以及每个成员的
    submission 得分均值与最高分。

    口径说明：
        - 团队成员 = 队长（team.creator）+ team.members，去重保序；
        - 每个成员的"得分"取其名下全部 submission 的 public_score（未评分/None 不计入
          均值/最高分，但计入提交次数）；
        - 团队总提交次数 = 全体成员提交次数之和。
    """
    users = client.list_users(competition_id)
    teams = client.list_teams(competition_id)
    subs = client.list_submissions(competition_id)
    stats_for, _, _ = _make_member_stats_fn(users, subs)

    rows: list[dict] = []
    for t in teams:
        roster = dedup_keep_order([t.get("creator"), *(t.get("members") or [])])
        members = [stats_for(uid) for uid in roster]
        max_scores = [m["max_score"] for m in members if m["max_score"] is not None]
        rows.append({
            "team_id": str(t.get("id")),
            "team_name": t.get("name"),
            "team_score": max(max_scores) if max_scores else None,
            "total_submissions": sum(m["submission_count"] for m in members),
            "schools": dedup_keep_order(m["school"] for m in members),
            "members": members,
        })

    rows.sort(key=lambda r: (r["team_score"] is None, -(r["team_score"] or 0.0)))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def build_school_stats(client: AlphathonClient, competition_id: str) -> list[dict]:
    """按学校汇总：每所学校内部按成员最高 public_score 排出校内排行榜；
    学校得分取校内最高分，学校之间按该分数从高到低排序。

    口径说明：
        - 学校成员 = 该报名记录 data.school 等于此学校的全部 user_id（不去重团队）；
        - 每个成员的"得分"口径与 build_team_stats 一致：其名下全部 submission 的
          public_score（未评分/None 不计入均值/最高分，但计入提交次数）；
        - 学校总提交次数 = 该校全体成员提交次数之和。
    """
    users = client.list_users(competition_id)
    subs = client.list_submissions(competition_id)
    stats_for, name_map, school_map = _make_member_stats_fn(users, subs)

    uids_by_school: dict[str, list[str]] = defaultdict(list)
    for uid in name_map:
        uids_by_school[school_map.get(uid, "（未填写）")].append(uid)

    rows: list[dict] = []
    for school, uids in uids_by_school.items():
        members = [stats_for(uid) for uid in uids]
        members.sort(key=lambda m: (m["max_score"] is None, -(m["max_score"] or 0.0)))
        max_scores = [m["max_score"] for m in members if m["max_score"] is not None]
        rows.append({
            "school": school,
            "school_score": max(max_scores) if max_scores else None,
            "n_members": len(members),
            "total_submissions": sum(m["submission_count"] for m in members),
            "members": members,
        })

    rows.sort(key=lambda r: (r["school_score"] is None, -(r["school_score"] or 0.0)))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def collect_stats(client: AlphathonClient) -> list[dict]:
    """按 COMPETITION_IDS 逐场查询，返回结构化统计结果（供落 json / 生成 md 复用）。"""
    stats: list[dict] = []
    for cid, name in COMPETITION_IDS.items():
        users = client.list_users(cid)
        n_subs = len(client.list_submissions(cid))
        schools = school_breakdown(users)
        stats.append({
            "competition_id": cid,
            "name": name,
            "n_users": len(users),
            "n_submissions": n_subs,
            "schools": [
                {"school": school, "count": n} for school, n in schools.most_common()
            ],
            "teams": build_team_stats(client, cid),
            "school_ranking": build_school_stats(client, cid),
        })
    return stats


def _fmt_score(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "-"


def print_stats(stats: list[dict]) -> None:
    print("=== 参赛人数 / 提交次数 ===")
    name_width = max(len(s["name"]) for s in stats)
    for s in stats:
        print(f"  {s['name']:<{name_width}}  ({s['competition_id']})  "
              f"参赛人数: {s['n_users']}  提交次数: {s['n_submissions']}")

    print("\n=== 学校分布（按报名记录计数，从多到少）===")
    for s in stats:
        print(f"\n  --- {s['name']} ({s['competition_id']})  共 {len(s['schools'])} 所学校 ---")
        for row in s["schools"]:
            print(f"    {row['school']}: {row['count']}")

    print("\n=== 团队排行（按团队最高 public_score 从高到低）===")
    for s in stats:
        print(f"\n  --- {s['name']} ({s['competition_id']})  共 {len(s['teams'])} 支团队 ---")
        for t in s["teams"]:
            print(f"    #{t['rank']} {t['team_name']}  团队得分: {_fmt_score(t['team_score'])}  "
                  f"总提交: {t['total_submissions']}  学校: {', '.join(t['schools'])}")
            for m in t["members"]:
                print(f"        {m['name']} ({m['school']})  提交次数: {m['submission_count']}  "
                      f"均分: {_fmt_score(m['avg_score'])}  最高分: {_fmt_score(m['max_score'])}")

    print("\n=== 学校排行榜（学校得分 = 校内成员最高 public_score，从高到低）===")
    for s in stats:
        print(f"\n  --- {s['name']} ({s['competition_id']})  共 {len(s['school_ranking'])} 所学校 ---")
        for sc in s["school_ranking"]:
            print(f"    #{sc['rank']} {sc['school']}  学校得分: {_fmt_score(sc['school_score'])}  "
                  f"成员数: {sc['n_members']}  总提交: {sc['total_submissions']}")
            for m in sc["members"]:
                print(f"        {m['name']}  提交次数: {m['submission_count']}  "
                      f"均分: {_fmt_score(m['avg_score'])}  最高分: {_fmt_score(m['max_score'])}")


def build_markdown(stats: list[dict], date_str: str) -> str:
    """把结构化统计结果渲染成对外可读的 Markdown 文档。"""
    lines: list[str] = [
        f"# {date_str} 比赛参赛统计",
        "",
        "## 参赛人数 / 提交次数",
        "",
        "| 比赛 | competition_id | 参赛人数 | 提交次数 |",
        "|---|---|---|---|",
    ]
    for s in stats:
        lines.append(f"| {s['name']} | `{s['competition_id']}` | {s['n_users']} | {s['n_submissions']} |")

    lines += ["", "## 学校分布", ""]
    for s in stats:
        lines.append(f"### {s['name']}（共 {len(s['schools'])} 所学校）")
        lines.append("")
        lines.append("| 学校 | 报名人数 |")
        lines.append("|---|---|")
        for row in s["schools"]:
            lines.append(f"| {row['school']} | {row['count']} |")
        lines.append("")

    lines += ["", "## 团队排行（按团队最高 public_score 从高到低）", ""]
    for s in stats:
        lines.append(f"### {s['name']}（共 {len(s['teams'])} 支团队）")
        lines.append("")
        lines.append("| 排名 | 团队 | 团队得分 | 总提交次数 | 涉及学校 | 成员明细（姓名/学校/提交次数/均分/最高分） |")
        lines.append("|---|---|---|---|---|---|")
        for t in s["teams"]:
            member_cells = "<br>".join(
                f"{m['name']}（{m['school']}，提交{m['submission_count']}次，"
                f"均分{_fmt_score(m['avg_score'])}，最高{_fmt_score(m['max_score'])}）"
                for m in t["members"]
            )
            lines.append(
                f"| {t['rank']} | {t['team_name']} | {_fmt_score(t['team_score'])} | "
                f"{t['total_submissions']} | {', '.join(t['schools'])} | {member_cells} |"
            )
        lines.append("")

    lines += ["", "## 学校排行榜（学校得分 = 校内成员最高 public_score，从高到低）", ""]
    for s in stats:
        lines.append(f"### {s['name']}（共 {len(s['school_ranking'])} 所学校）")
        lines.append("")
        lines.append("| 排名 | 学校 | 学校得分 | 成员数 | 总提交次数 | 校内成员明细（姓名/提交次数/均分/最高分） |")
        lines.append("|---|---|---|---|---|---|")
        for sc in s["school_ranking"]:
            member_cells = "<br>".join(
                f"{m['name']}（提交{m['submission_count']}次，"
                f"均分{_fmt_score(m['avg_score'])}，最高{_fmt_score(m['max_score'])}）"
                for m in sc["members"]
            )
            lines.append(
                f"| {sc['rank']} | {sc['school']} | {_fmt_score(sc['school_score'])} | "
                f"{sc['n_members']} | {sc['total_submissions']} | {member_cells} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    client = AlphathonClient()
    stats = collect_stats(client)
    print_stats(stats)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    json_path = OUT_DIR / f"competition_stats_{date_str}.json"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = OUT_DIR / f"competition_stats_{date_str}.md"
    md_path.write_text(build_markdown(stats, date_str), encoding="utf-8")

    print(f"\n已写出: {json_path}")
    print(f"已写出: {md_path}")


if __name__ == "__main__":
    main()
