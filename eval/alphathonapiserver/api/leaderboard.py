"""排行榜与比赛统计接口（带 2 分钟内存缓存）"""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Path, Request

from bigshared2.auth import Credential, anonymous_authenticator, authenticator
from bigshared2.schemas.http import PagingQueryMixin, ResponseModel

from .. import models
from ..constants import UserStatus
from ._helpers import get_competition_or_404, slice_for_paging

router = APIRouter()

CACHE_DURATION_SECONDS = 2 * 60

_leaderboard_cache: dict[str, dict[str, Any]] = {}
_summary_cache: dict[str, dict[str, Any]] = {}


def _read_cache(cache: dict[str, dict[str, Any]], key: str) -> Any | None:
    entry = cache.get(key)
    if not entry:
        return None
    if (datetime.now() - entry["timestamp"]).total_seconds() >= CACHE_DURATION_SECONDS:
        return None
    return entry["data"]


def _write_cache(cache: dict[str, dict[str, Any]], key: str, data: Any) -> None:
    cache[key] = {"data": data, "timestamp": datetime.now()}


@router.get("/{competition_id}/summary")
async def get_competition_summary(
    _: Credential = Depends(authenticator | anonymous_authenticator),
    competition_id: uuid.UUID = Path(description="比赛ID"),
) -> ResponseModel:
    """比赛统计摘要"""
    await get_competition_or_404(competition_id)

    cache_key = str(competition_id)
    if cached := _read_cache(_summary_cache, cache_key):
        return ResponseModel(data=cached)

    users = await models.User.filter(competition_id=competition_id).all()
    approved_users_count = sum(1 for u in users if u.status == UserStatus.APPROVED)
    submissions_count = await models.Submission.filter(competition_id=competition_id).count()
    teams_count = await models.Team.filter(competition_id=competition_id).count()

    summary_data = {
        "approved_users_count": approved_users_count,
        "submissions_count": submissions_count,
        "teams_count": teams_count,
        "total_users_count": len(users),
    }
    _write_cache(_summary_cache, cache_key, summary_data)
    return ResponseModel(data=summary_data)


@router.get("/{competition_id}")
async def get_leaderboard(
    request: Request,
    _: Credential = Depends(authenticator | anonymous_authenticator),
    competition_id: uuid.UUID = Path(description="比赛ID"),
    paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
) -> ResponseModel:
    """比赛排行榜"""
    competition = await get_competition_or_404(competition_id)
    cache_key = str(competition_id)

    leaderboard_data = _read_cache(_leaderboard_cache, cache_key)
    if leaderboard_data is None:
        leaderboard_data = await _calculate_leaderboard(competition)
        _write_cache(_leaderboard_cache, cache_key, leaderboard_data)

    paginated, page, _ = slice_for_paging(leaderboard_data, paging)
    request.state.log_data["items"] = len(paginated)
    return ResponseModel(data={"items": paginated, "total": len(leaderboard_data), "page": page, "size": len(paginated)})


async def _calculate_leaderboard(competition: models.Competition) -> list[dict[str, Any]]:
    """聚合用户/团队最高分，返回带 rank 的有序列表。"""
    rank_order = competition.data.get("competition", {}).get("rank_order", "desc")
    user_scores = await _get_user_max_scores(competition.id, rank_order)
    teams = await models.Team.filter(competition_id=competition.id).all()

    # user_id -> team；含队长本人
    team_members_map: dict[str, models.Team] = {}
    for team in teams:
        team_members_map[str(team.creator)] = team
        for user_id in team.members:
            team_members_map[str(user_id)] = team

    team_scores: dict[str, dict[str, Any]] = {
        str(team.id): {
            "type": "team",
            "team_id": str(team.id),
            "team_name": team.name,
            "public_score": None,
            "private_score": None,
            "submission_count": 0,
            "last_submission_time": None,
            "members": [],
        }
        for team in teams
    }

    individual_scores: list[dict[str, Any]] = []
    for entry in user_scores:
        user_id = str(entry["user_id"])
        if user_id in team_members_map:
            team = team_members_map[user_id]
            team_data = team_scores[str(team.id)]

            if _is_score_better(entry["max_public_score"], team_data["public_score"], rank_order):
                team_data["public_score"] = entry["max_public_score"]
            if _is_score_better(entry["max_private_score"], team_data["private_score"], rank_order):
                team_data["private_score"] = entry["max_private_score"]

            team_data["submission_count"] += entry["submission_count"]

            user_last = entry["last_submission_time"]
            if user_last is not None and (team_data["last_submission_time"] is None or user_last > team_data["last_submission_time"]):
                team_data["last_submission_time"] = user_last

            team_data["members"].append(
                {
                    "user_id": user_id,
                    "public_score": entry["max_public_score"],
                    "private_score": entry["max_private_score"],
                    "submission_count": entry["submission_count"],
                    "last_submission_time": entry["last_submission_time"],
                }
            )
        else:
            individual_scores.append(
                {
                    "type": "individual",
                    "user_id": user_id,
                    "public_score": entry["max_public_score"],
                    "private_score": entry["max_private_score"],
                    "submission_count": entry["submission_count"],
                    "last_submission_time": entry["last_submission_time"],
                }
            )

    all_entries = list(team_scores.values()) + individual_scores
    scored = [e for e in all_entries if e["public_score"] is not None or e["private_score"] is not None]
    unscored = [e for e in all_entries if e["public_score"] is None and e["private_score"] is None]

    sort_key = _sort_key_desc if rank_order == "desc" else _sort_key_asc
    sorted_entries = sorted(scored, key=sort_key) + unscored
    for i, entry in enumerate(sorted_entries):
        entry["rank"] = i + 1
    return sorted_entries


async def _get_user_max_scores(competition_id: uuid.UUID, rank_order: str = "desc") -> list[dict[str, Any]]:
    """获取每个 approved 用户的最高（或最低）分数。"""
    from tortoise import connections

    func = "MAX" if rank_order == "desc" else "MIN"
    query = f"""
    SELECT
        u.user_id,
        s.max_public_score,
        s.max_private_score,
        s.submission_count,
        s.last_submission_time
    FROM alphathon__user u
    LEFT JOIN (
        SELECT
            user_id,
            {func}(public_score) as max_public_score,
            {func}(private_score) as max_private_score,
            COUNT(*) as submission_count,
            MAX(created_at) as last_submission_time
        FROM alphathon__submission
        WHERE competition_id = %s
        GROUP BY user_id
    ) s ON u.user_id = s.user_id
    WHERE u.competition_id = %s and u.status = 'approved'
    """
    conn = connections.get("default")
    results = await conn.execute_query(query, [str(competition_id), str(competition_id)])
    return [
        {
            "user_id": row["user_id"],
            "max_public_score": row["max_public_score"],
            "max_private_score": row["max_private_score"],
            "submission_count": row["submission_count"] or 0,
            "last_submission_time": row["last_submission_time"],
        }
        for row in results[1]
    ]


def _is_score_better(new_score: float | None, current_score: float | None, rank_order: str = "desc") -> bool:
    if new_score is None:
        return False
    if current_score is None:
        return True
    return new_score > current_score if rank_order == "desc" else new_score < current_score


def _sort_key_desc(entry: dict[str, Any]) -> tuple:
    private_score = entry.get("private_score")
    public_score = entry.get("public_score")
    p = float(private_score) if private_score is not None else float("-inf")
    s = float(public_score) if public_score is not None else float("-inf")
    return (-p, -s)


def _sort_key_asc(entry: dict[str, Any]) -> tuple:
    private_score = entry.get("private_score")
    public_score = entry.get("public_score")
    p = float(private_score) if private_score is not None else float("inf")
    s = float(public_score) if public_score is not None else float("inf")
    return (p, s)
