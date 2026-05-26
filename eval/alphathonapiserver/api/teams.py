"""团队相关接口"""

import uuid

from fastapi import APIRouter, Body, Depends, Path, Query, Request
from tortoise.expressions import Q

from bigshared2.auth import Credential, authenticator
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import ResponseModel

from .. import models, schemas
from ._helpers import check_deadline, get_competition_or_404, get_user_registration_or_403, safe_notify

router = APIRouter()


def _check_team_merger_deadline(competition: models.Competition) -> None:
    check_deadline(competition, "team_merger_deadline", "团队组建已截止")


async def get_user_team(competition_id: uuid.UUID, user_id: uuid.UUID) -> models.Team | None:
    """查找用户在该比赛中所在的团队（创建者 / 成员 / 待审批）。

    依赖 MySQL 8 的 JSON_CONTAINS / 多值索引。
    """
    user_id_str = str(user_id)
    return await models.Team.filter(
        Q(competition_id=competition_id)
        & (
            Q(creator=user_id_str)
            | Q(members__contains=[user_id_str])
            | Q(pending_users__contains=[user_id_str])
        )
    ).first()


async def _get_team_or_404(team_id: uuid.UUID) -> models.Team:
    team = await models.Team.filter(id=team_id).first()
    if not team:
        raise HTTPException(Errors.NOT_FOUND.with_message("团队不存在"))
    return team


async def _load_team_and_competition(team_id: uuid.UUID) -> tuple[models.Team, models.Competition]:
    """读取团队 + 所属比赛，并校验团队组建截止时间。"""
    team = await _get_team_or_404(team_id)
    competition = await get_competition_or_404(team.competition_id)
    _check_team_merger_deadline(competition)
    return team, competition


def _ensure_captain(team: models.Team, credential: Credential, action: str) -> None:
    if team.creator != credential.user_id:
        raise HTTPException(Errors.FORBIDDEN.with_message(f"只有团队「{team.name}」的队长可以{action}"))


@router.post("")
async def create(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_in: schemas.TeamIn = Body(),
) -> ResponseModel:
    """创建团队"""
    data = team_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data
    competition_id = data["competition_id"]

    competition = await get_competition_or_404(competition_id)
    _check_team_merger_deadline(competition)
    await get_user_registration_or_403(competition_id, credential.user_id)

    existing_team = await get_user_team(competition_id, credential.user_id)
    if existing_team:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户已在团队「{existing_team.name}」中，无法创建新团队"))

    if await models.Team.filter(competition_id=competition_id, name=data["name"]).exists():
        raise HTTPException(Errors.BAD_REQUEST.with_message("团队名称在该比赛中已存在"))

    team = await models.Team.create(creator=credential.user_id, members=[], pending_users=[], **data)
    request.state.log_data["team.id"] = team.id
    return ResponseModel(data=schemas.Team.model_validate(team))


@router.post("/{team_id}")
async def update(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(),
    team_update: schemas.TeamUpdate = Body(),
) -> ResponseModel:
    """更新团队信息（仅队长）"""
    team, _ = await _load_team_and_competition(team_id)
    _ensure_captain(team, credential, "更新团队信息")

    data = team_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    if "name" in data and data["name"] != team.name:
        if await models.Team.filter(competition_id=team.competition_id, name=data["name"]).exists():
            raise HTTPException(Errors.BAD_REQUEST.with_message("团队名称在该比赛中已存在"))

    await team.update_from_dict(data).save()
    return ResponseModel(data=schemas.Team.model_validate(team))


@router.post("/{team_id}/apply")
async def apply_to_join(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(),
) -> ResponseModel:
    """申请加入团队"""
    team, _ = await _load_team_and_competition(team_id)
    await get_user_registration_or_403(team.competition_id, credential.user_id)

    existing_team = await get_user_team(team.competition_id, credential.user_id)
    if existing_team:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户已在团队「{existing_team.name}」中，无法申请加入新团队"))

    pending_users = team.pending_users or []
    pending_users.append(str(credential.user_id))
    team.pending_users = pending_users
    await team.save()

    return ResponseModel(data=schemas.Team.model_validate(team))


@router.post("/{team_id}/approve/{user_id}")
async def approve_application(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(),
    user_id: uuid.UUID = Path(),
    approve: bool = Body(embed=True, description="true为通过，false为拒绝"),
) -> ResponseModel:
    """队长审批加入申请"""
    request.state.log_data["approved"] = approve

    team, competition = await _load_team_and_competition(team_id)
    _ensure_captain(team, credential, "审批加入申请")

    pending_users = team.pending_users or []
    if str(user_id) not in pending_users:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户不在团队「{team.name}」的待审批列表中"))

    pending_users.remove(str(user_id))
    team.pending_users = pending_users

    if approve:
        # 防并发：再查一次确认用户没有进别的队
        existing_team = await get_user_team(team.competition_id, user_id)
        if existing_team and existing_team.id != team.id:
            raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户已在团队「{existing_team.name}」中，无法加入团队「{team.name}」"))

        members = team.members or []
        if str(user_id) not in members:
            members.append(str(user_id))
            team.members = members

        content = (
            f"【{competition.name}】恭喜！您申请加入【{team.name}】的审核已通过。"
            f"[快去看看吧>>](https://bigquant.com/square/competition/{competition.id})"
        )
    else:
        content = (
            f"【{competition.name}】很抱歉！您申请加入【{team.name}】的审核未通过。"
            f"[去看看别的团队吧>>](https://bigquant.com/square/competition/{competition.id})"
        )

    await safe_notify(request, user_id=user_id, title="【比赛团队审核通知】", content=content)
    await team.save()

    return ResponseModel(data=schemas.Team.model_validate(team))


@router.delete("/{team_id}/members/{user_id}")
async def remove_member(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(),
    user_id: uuid.UUID = Path(),
) -> ResponseModel:
    """队长移除队员"""
    team, competition = await _load_team_and_competition(team_id)
    _ensure_captain(team, credential, "移除队员")

    members = team.members or []
    if str(user_id) not in members:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户不在团队「{team.name}」的成员中"))

    members.remove(str(user_id))
    team.members = members
    await team.save()

    content = (
        f"【{competition.name}】很抱歉！您已被移出{team.name}，"
        f"[去看看别的团队吧>>](https://bigquant.com/square/competition/{competition.id})"
    )
    await safe_notify(request, user_id=user_id, title="【比赛团队移出通知】", content=content)
    return ResponseModel()


@router.delete("/{team_id}")
async def delete_team(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(),
) -> ResponseModel:
    """解散团队（必须先清空成员/待审批）"""
    team, _ = await _load_team_and_competition(team_id)
    _ensure_captain(team, credential, "解散团队")

    members = team.members or []
    pending_users = team.pending_users or []
    if members or pending_users:
        msg_parts = []
        if members:
            msg_parts.append(f"{len(members)}名成员")
        if pending_users:
            msg_parts.append(f"{len(pending_users)}名待审批用户")
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"团队「{team.name}」还有{' 和 '.join(msg_parts)}，需要先清空才能解散团队"))

    await team.delete()
    request.state.log_data["team.id"] = team.id
    return ResponseModel()


@router.get("/my")
async def get_my_team(
    request: Request,
    credential: Credential = Depends(authenticator),
    competition_id: uuid.UUID = Query(description="比赛ID"),
) -> ResponseModel:
    """获取我的团队"""
    await get_competition_or_404(competition_id)

    user_team = await get_user_team(competition_id, credential.user_id)
    if not user_team:
        return ResponseModel(data=None)

    user_id_str = str(credential.user_id)
    if user_team.creator == credential.user_id or user_id_str in (user_team.members or []):
        return ResponseModel(data={"team": schemas.Team.model_validate(user_team), "status": "member"})

    if user_id_str in (user_team.pending_users or []):
        team_data = schemas.Team.model_validate(user_team).model_dump()
        team_data["pending_users"] = len(user_team.pending_users or [])
        return ResponseModel(data={"team": team_data, "status": "pending"})

    return ResponseModel(data=None)


@router.delete("/{team_id}/leave")
async def leave_team(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(description="团队ID"),
) -> ResponseModel:
    """退出团队 / 取消加入申请"""
    team, _ = await _load_team_and_competition(team_id)
    user_id_str = str(credential.user_id)

    members = team.members or []
    pending_users = team.pending_users or []

    if user_id_str in members:
        members.remove(user_id_str)
        team.members = members
        await team.save()
        request.state.log_data["action"] = "left_team"
        return ResponseModel(data={"message": f"已退出团队「{team.name}」"})

    if user_id_str in pending_users:
        pending_users.remove(user_id_str)
        team.pending_users = pending_users
        await team.save()
        request.state.log_data["action"] = "cancelled_application"
        return ResponseModel(data={"message": f"已取消加入团队「{team.name}」的申请"})

    raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户不在团队「{team.name}」中"))
