import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Path, Request

from bigshared2.auth import Credential, anonymous_authenticator, authenticator
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, ResponseModel

import models
from constants import UserStatus

router = APIRouter()

# Memory cache for leaderboard results
_leaderboard_cache: dict[str, dict[str, Any]] = {}
# Memory cache for competition summary results
_summary_cache: dict[str, dict[str, Any]] = {}
CACHE_DURATION_MINUTES = 2


@router.get("/{competition_id}/summary")
async def get_competition_summary(
    _: Credential = Depends(authenticator | anonymous_authenticator),
    competition_id: uuid.UUID = Path(description="比赛ID"),
) -> ResponseModel:
    """获取比赛统计摘要"""
    # 检查比赛是否存在
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 检查缓存
    cache_key = f"summary_{str(competition_id)}"
    now = datetime.now()

    if cache_key in _summary_cache:
        cache_data = _summary_cache[cache_key]
        cache_time = cache_data.get("timestamp")
        if cache_time and (now - cache_time).total_seconds() < CACHE_DURATION_MINUTES * 60:
            # 返回缓存结果
            return ResponseModel(data=cache_data["data"])

    # 重新计算统计数据
    # 统计参赛用户数（状态为approved）
    users = await models.User.filter(competition_id=competition_id).all()
    approved_users_count = sum(1 for user in users if user.status == UserStatus.APPROVED)

    # 统计提交数
    submissions_count = await models.Submission.filter(competition_id=competition_id).count()

    # 统计团队数
    teams_count = await models.Team.filter(competition_id=competition_id).count()

    summary_data = {
        "approved_users_count": approved_users_count,
        "submissions_count": submissions_count,
        "teams_count": teams_count,
        "total_users_count": len(users),
    }

    # 更新缓存
    _summary_cache[cache_key] = {"data": summary_data, "timestamp": now}

    return ResponseModel(data=summary_data)


@router.get("/{competition_id}")
async def get_leaderboard(
    request: Request,
    _: Credential = Depends(authenticator | anonymous_authenticator),
    competition_id: uuid.UUID = Path(description="比赛ID"),
    paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
) -> ResponseModel:
    """获取比赛排行榜"""
    # 检查比赛是否存在
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 检查缓存
    cache_key = str(competition_id)
    now = datetime.now()

    if cache_key in _leaderboard_cache:
        cache_data = _leaderboard_cache[cache_key]
        cache_time = cache_data.get("timestamp")
        if cache_time and (now - cache_time).total_seconds() < CACHE_DURATION_MINUTES * 60:
            # 返回缓存结果（应用分页）
            cached_results = cache_data["results"]
            return _apply_pagination(cached_results, paging, request)

    # 重新计算排行榜
    leaderboard_data = await _calculate_leaderboard(competition_id)

    # 更新缓存
    _leaderboard_cache[cache_key] = {"results": leaderboard_data, "timestamp": now}

    # 应用分页并返回
    return _apply_pagination(leaderboard_data, paging, request)


async def _calculate_leaderboard(competition_id: uuid.UUID) -> list[dict[str, Any]]:
    """计算排行榜数据"""
    competition = await models.Competition.filter(id=competition_id).first()
    if competition is None:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    rank_order = competition.data.get("competition", {}).get("rank_order", "desc")
    # 1. 获取所有比赛用户及其最高分数
    user_scores = await _get_user_max_scores(competition_id, rank_order)

    # 2. 获取团队信息
    teams = await models.Team.filter(competition_id=competition_id).all()

    # 创建团队成员映射 (user_id -> team)，包括团队创建者
    team_members_map = {}
    for team in teams:
        # 团队创建者
        team_members_map[str(team.creator)] = team
        # 团队成员
        for user_id in team.members:
            team_members_map[str(user_id)] = team

    # 初始化所有团队
    team_scores = {}  # team_id -> team_data
    for team in teams:
        team_id = str(team.id)
        team_scores[team_id] = {
            "type": "team",
            "team_id": team_id,
            "team_name": team.name,
            "public_score": None,
            "private_score": None,
            "submission_count": 0,
            "last_submission_time": None,
            "members": [],
        }

    # 4. 处理所有比赛用户（user_scores 已包含所有用户）
    individual_scores = []  # 非团队成员的个人分数

    for user_score_data in user_scores:
        user_id = str(user_score_data["user_id"])

        if user_id in team_members_map:
            # 团队成员
            team = team_members_map[user_id]
            team_id = str(team.id)
            team_data = team_scores[team_id]

            # 更新团队最高分数
            if _is_score_better(user_score_data["max_public_score"], team_data["public_score"], rank_order):
                team_data["public_score"] = user_score_data["max_public_score"]

            if _is_score_better(user_score_data["max_private_score"], team_data["private_score"], rank_order):
                team_data["private_score"] = user_score_data["max_private_score"]

            # 累加团队提交次数
            team_data["submission_count"] += user_score_data["submission_count"]

            # 更新团队最后提交时间（取最晚的时间）
            user_last_time = user_score_data["last_submission_time"]
            if user_last_time is not None:
                if team_data["last_submission_time"] is None or user_last_time > team_data["last_submission_time"]:
                    team_data["last_submission_time"] = user_last_time

            # 添加成员信息
            team_data["members"].append(
                {
                    "user_id": user_id,
                    "public_score": user_score_data["max_public_score"],
                    "private_score": user_score_data["max_private_score"],
                    "submission_count": user_score_data["submission_count"],
                    "last_submission_time": user_score_data["last_submission_time"],
                }
            )
        else:
            # 个人用户（不在团队中）
            individual_scores.append(
                {
                    "type": "individual",
                    "user_id": user_id,
                    "public_score": user_score_data["max_public_score"],
                    "private_score": user_score_data["max_private_score"],
                    "submission_count": user_score_data["submission_count"],
                    "last_submission_time": user_score_data["last_submission_time"],
                }
            )

    # 5. 合并所有条目并排序
    all_entries = list(team_scores.values()) + individual_scores

    # 6. 筛选已评分
    unscored_entries = []
    scored_entries = []
    for entry in all_entries:
        if entry["public_score"] is None and entry["private_score"] is None:
            unscored_entries.append(entry)
        else:
            scored_entries.append(entry)

    # 7. 按排名规则排序
    sort_key = _leaderboard_sort_key_desc if rank_order == "desc" else _leaderboard_sort_key_asc
    sorted_entries = sorted(scored_entries, key=sort_key)
    sorted_entries.extend(unscored_entries)

    # 8. 添加排名
    for i, entry in enumerate(sorted_entries):
        entry["rank"] = i + 1

    return sorted_entries


async def _get_user_max_scores(competition_id: uuid.UUID, rank_order: str = "desc") -> list[dict[str, Any]]:
    """获取每个用户的最高分数"""
    # 使用原生SQL查询获取每个用户的最高分数
    from tortoise import connections

    conn = connections.get("default")

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

    results = await conn.execute_query(query, [str(competition_id), str(competition_id)])

    return [
        {
            "user_id": row["user_id"],
            "max_public_score": row["max_public_score"],
            "max_private_score": row["max_private_score"],
            "submission_count": row["submission_count"] or 0,
            "last_submission_time": row["last_submission_time"],
        }
        for row in results[1]  # results[1] contains the actual data rows
    ]


def _is_score_better(new_score: float | None, current_score: float | None, rank_order: str = "desc") -> bool:
    """判断新分数是否比当前分数更好"""
    if new_score is None:
        return False
    if current_score is None:
        return True
    return new_score > current_score if rank_order == "desc" else new_score < current_score


def _leaderboard_sort_key_desc(entry: dict[str, Any]) -> tuple:
    """排行榜排序键：先按私榜分数降序，再按公榜分数降序"""
    private_score = entry.get("private_score")
    public_score = entry.get("public_score")

    # 将None转换为最小值用于排序
    private_score_sort = float(private_score) if private_score is not None else float("-inf")
    public_score_sort = float(public_score) if public_score is not None else float("-inf")

    return (-private_score_sort, -public_score_sort)


def _leaderboard_sort_key_asc(entry: dict[str, Any]) -> tuple:
    """排行榜排序键：先按私榜分数升序，再按公榜分数升序"""
    private_score = entry.get("private_score")
    public_score = entry.get("public_score")

    # 将None转换为最大值用于排序
    private_score_sort = float(private_score) if private_score is not None else float("inf")
    public_score_sort = float(public_score) if public_score is not None else float("inf")

    return (private_score_sort, public_score_sort)


def _apply_pagination(data: list[dict[str, Any]], paging: PagingQueryMixin | None, request: Request) -> ResponseModel:
    """应用分页逻辑"""
    total = len(data)

    if paging:
        page = paging.page or 0
        size = paging.size or 50
        paginated_data = data[(page - 1) * size : page * size]
    else:
        page = 0
        size = 50
        paginated_data = data[:size]

    request.state.log_data["items"] = len(paginated_data)

    return ResponseModel(data={"items": paginated_data, "total": total, "page": page, "size": len(paginated_data)})
