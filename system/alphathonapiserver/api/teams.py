import uuid
from datetime import datetime

from fastapi import APIRouter, Body, Depends, Path, Query, Request

from bigshared2.auth import Credential, authenticator
from bigshared2.auth.schemas import BIGQUANT_SPACE_ID
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import ResponseModel

import constants
import models
import schemas
from utils import create_notice, send_wechat_message

router = APIRouter()


async def check_team_merger_deadline(competition: models.Competition) -> None:
    """检查团队组建截止时间"""
    if competition.summary and "team_merger_deadline" in competition.summary:
        team_merger_deadline_str = competition.summary["team_merger_deadline"]
        try:
            team_merger_deadline = datetime.strptime(team_merger_deadline_str[:10], "%Y-%m-%d").date()
            current_date = datetime.now().date()
            if current_date > team_merger_deadline:
                raise HTTPException(Errors.BAD_REQUEST.with_message("团队组建已截止"))
        except (ValueError, TypeError):
            # 如果时间格式不正确，记录但不阻止操作
            pass


async def get_user_team(competition_id: uuid.UUID, user_id: uuid.UUID) -> models.Team | None:
    """获取用户所在的团队（包括队长、成员、待审批）

    使用 MySQL 8 的多值索引来查询:
    competition_id=competition_id and (creator=user_id or user_id in members or user_id in pending_users)
    """
    from tortoise.expressions import Q

    user_id_str = str(user_id)

    # 构建查询条件：competition_id 匹配 且 (创建者是用户 或 用户在成员列表中 或 用户在待审批列表中)
    team = await models.Team.filter(
        Q(competition_id=competition_id) & (Q(creator=user_id_str) | Q(members__contains=[user_id_str]) | Q(pending_users__contains=[user_id_str]))
    ).first()

    return team


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

    # 检查比赛是否存在
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 检查团队组建截止时间
    await check_team_merger_deadline(competition)

    # 检查用户是否已报名该比赛且审批通过
    user_registration = await models.User.filter(competition_id=competition_id, user_id=credential.user_id).first()
    if not user_registration:
        raise HTTPException(Errors.FORBIDDEN.with_message("请先报名参加比赛"))

    if user_registration.status != constants.UserStatus.APPROVED:
        raise HTTPException(Errors.FORBIDDEN.with_message("用户报名尚未审批通过，无法创建团队"))

    # 检查用户是否已在其他团队中
    existing_team = await get_user_team(competition_id, credential.user_id)
    if existing_team:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户已在团队「{existing_team.name}」中，无法创建新团队"))

    # 检查团队名称在同一比赛中是否唯一
    existing_team_name = await models.Team.filter(competition_id=competition_id, name=data["name"]).first()
    if existing_team_name:
        raise HTTPException(Errors.BAD_REQUEST.with_message("团队名称在该比赛中已存在"))

    # 创建团队，创建者自动成为队长
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
    """更新团队信息"""
    team = await models.Team.filter(id=team_id).first()
    if not team:
        raise HTTPException(Errors.NOT_FOUND.with_message("团队不存在"))

    # 检查团队组建截止时间
    competition = await models.Competition.filter(id=team.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
    await check_team_merger_deadline(competition)

    # 检查权限：只有队长可以更新团队信息
    if team.creator != credential.user_id:
        raise HTTPException(Errors.FORBIDDEN.with_message(f"只有团队「{team.name}」的队长可以更新团队信息"))

    data = team_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    # 如果要更新团队名称，检查新名称在同一比赛中是否唯一
    if "name" in data and data["name"] != team.name:
        existing_team_name = await models.Team.filter(competition_id=team.competition_id, name=data["name"]).first()
        if existing_team_name:
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
    team = await models.Team.filter(id=team_id).first()
    if not team:
        raise HTTPException(Errors.NOT_FOUND.with_message("团队不存在"))

    # 检查团队组建截止时间
    competition = await models.Competition.filter(id=team.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
    await check_team_merger_deadline(competition)

    # 检查用户是否已报名该比赛且审批通过
    user_registration = await models.User.filter(competition_id=team.competition_id, user_id=credential.user_id).first()
    if not user_registration:
        raise HTTPException(Errors.FORBIDDEN.with_message("请先报名参加比赛"))

    if user_registration.status != constants.UserStatus.APPROVED:
        raise HTTPException(Errors.FORBIDDEN.with_message("用户报名尚未审批通过，无法申请加入团队"))

    # 检查用户是否已在其他团队中
    existing_team = await get_user_team(team.competition_id, credential.user_id)
    if existing_team:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户已在团队「{existing_team.name}」中，无法申请加入新团队"))

    pending_users = team.pending_users or []

    # 添加到待审批列表
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

    team = await models.Team.filter(id=team_id).first()
    if not team:
        raise HTTPException(Errors.NOT_FOUND.with_message("团队不存在"))

    # 检查团队组建截止时间
    competition = await models.Competition.filter(id=team.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
    await check_team_merger_deadline(competition)

    # 检查权限：只有队长可以审批
    if team.creator != credential.user_id:
        raise HTTPException(Errors.FORBIDDEN.with_message(f"只有团队「{team.name}」的队长可以审批加入申请"))

    # 检查用户是否在待审批列表中
    pending_users = team.pending_users or []
    if str(user_id) not in pending_users:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户不在团队「{team.name}」的待审批列表中"))

    # 从待审批列表中移除
    pending_users.remove(str(user_id))
    team.pending_users = pending_users

    if approve:
        # 再次检查用户是否在其他团队中（防止并发问题）
        existing_team = await get_user_team(team.competition_id, user_id)
        if existing_team and existing_team.id != team.id:
            raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户已在团队「{existing_team.name}」中，无法加入团队「{team.name}」"))

        # 通过：添加到成员列表
        members = team.members or []
        if str(user_id) not in members:
            members.append(str(user_id))
            team.members = members

        content = f"【{competition.name}】恭喜！您申请加入【{team.name}】的审核已通过。[快去看看吧>>](https://bigquant.com/square/competition/{competition.id})"
    else:
        content = f"【{competition.name}】很抱歉！您申请加入【{team.name}】的审核未通过。[去看看别的团队吧>>](https://bigquant.com/square/competition/{competition.id})"
    try:
        await create_notice(user_id=str(user_id), space_id=BIGQUANT_SPACE_ID, title="【比赛团队审核通知】", content=content, channel="system")
        await send_wechat_message(title="比赛团队审核通知", user_id=user_id)
    except Exception as e:
        request.state.log_data["notice_exception"] = str(e)

    await team.save()

    return ResponseModel(data=schemas.Team.model_validate(team))


@router.delete("/{team_id}/members/{user_id}")
async def remove_member(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(),
    user_id: uuid.UUID = Path(),
) -> ResponseModel:
    """移除队员"""
    team = await models.Team.filter(id=team_id).first()
    if not team:
        raise HTTPException(Errors.NOT_FOUND.with_message("团队不存在"))

    # 检查团队组建截止时间
    competition = await models.Competition.filter(id=team.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
    await check_team_merger_deadline(competition)

    # 检查权限：只有队长可以移除队员
    if team.creator != credential.user_id:
        raise HTTPException(Errors.FORBIDDEN.with_message(f"只有团队「{team.name}」的队长可以移除队员"))

    # 从成员列表中移除
    members = team.members or []
    if str(user_id) in members:
        members.remove(str(user_id))
        team.members = members
        await team.save()

        content = f"【{competition.name}】很抱歉！您已被移出{team.name}，[去看看别的团队吧>>](https://bigquant.com/square/competition/{competition.id})"
        try:
            await create_notice(user_id=str(user_id), space_id=BIGQUANT_SPACE_ID, title="【比赛团队移出通知】", content=content, channel="system")
            await send_wechat_message(title="比赛团队移出通知", user_id=user_id)
        except Exception as e:
            request.state.log_data["notice_exception"] = str(e)
    else:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户不在团队「{team.name}」的成员中"))

    return ResponseModel()


@router.delete("/{team_id}")
async def delete_team(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(),
) -> ResponseModel:
    """解散团队"""
    team = await models.Team.filter(id=team_id).first()
    if not team:
        raise HTTPException(Errors.NOT_FOUND.with_message("团队不存在"))

    # 检查团队组建截止时间
    competition = await models.Competition.filter(id=team.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
    await check_team_merger_deadline(competition)

    # 检查权限：只有队长可以解散团队
    if team.creator != credential.user_id:
        raise HTTPException(Errors.FORBIDDEN.with_message(f"只有团队「{team.name}」的队长可以解散团队"))

    # 检查是否还有成员或待审批用户
    members = team.members or []
    pending_users = team.pending_users or []
    if members or pending_users:
        member_count = len(members)
        pending_count = len(pending_users)
        msg_parts = []
        if member_count > 0:
            msg_parts.append(f"{member_count}名成员")
        if pending_count > 0:
            msg_parts.append(f"{pending_count}名待审批用户")
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
    # 检查比赛是否存在
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 查找用户所在的团队
    user_team = await get_user_team(competition_id, credential.user_id)
    if not user_team:
        # 没有找到任何团队
        return ResponseModel(data=None)

    # 判断用户在团队中的身份
    user_id_str = str(credential.user_id)

    # 如果是队长或成员，返回完整信息
    if user_team.creator == credential.user_id or user_id_str in (user_team.members or []):
        return ResponseModel(data={"team": schemas.Team.model_validate(user_team), "status": "member"})

    # 如果在待审批列表中，返回简化信息
    if user_id_str in (user_team.pending_users or []):
        team_data = schemas.Team.model_validate(user_team).model_dump()
        # 隐藏待审批用户详细信息，只返回数量
        team_data["pending_users"] = len(user_team.pending_users or [])
        return ResponseModel(data={"team": team_data, "status": "pending"})

    # 理论上不会到达这里
    return ResponseModel(data=None)


@router.delete("/{team_id}/leave")
async def leave_team(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(description="团队ID"),
) -> ResponseModel:
    """用户退出团队或取消申请"""
    # 检查团队是否存在
    team = await models.Team.filter(id=team_id).first()
    if not team:
        raise HTTPException(Errors.NOT_FOUND.with_message("团队不存在"))

    # 检查团队组建截止时间
    competition = await models.Competition.filter(id=team.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
    await check_team_merger_deadline(competition)

    user_id_str = str(credential.user_id)

    # 检查用户是否在该团队中（成员或待审批）
    members = team.members or []
    pending_users = team.pending_users or []

    if user_id_str not in members and user_id_str not in pending_users:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户不在团队「{team.name}」中"))

    # 如果在成员列表中，从成员中移除
    if user_id_str in members:
        members.remove(user_id_str)
        team.members = members
        await team.save()
        request.state.log_data["action"] = "left_team"
        return ResponseModel(data={"message": f"已退出团队「{team.name}」"})

    # 如果在待审批列表中，从待审批中移除
    if user_id_str in pending_users:
        pending_users.remove(user_id_str)
        team.pending_users = pending_users
        await team.save()
        request.state.log_data["action"] = "cancelled_application"
        return ResponseModel(data={"message": f"已取消加入团队「{team.name}」的申请"})

    # 理论上不会到达这里
    raise HTTPException(Errors.BAD_REQUEST.with_message("用户状态异常"))


# @router.get("")
# async def reads(
#     request: Request,
#     credential: Credential = Depends(authenticator),
#     constraints: QueryConstraintsMixin | None = Depends(QueryConstraintsMixin.q),
#     order_by: list[str] | None = Query([], description="排序字段"),
#     include_fields: list[str] | None = Query([], description="只返回指定包含的字段"),
#     exclude_fields: list[str] | None = Query([], description="排除的字段"),
#     paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
#     competition_id: uuid.UUID | None = Query(None, description="比赛ID，用于获取指定比赛的团队列表"),
# ) -> ResponseModel:
#     """获取团队列表"""
#     # 构建基础查询条件
#     base_constraints = {}
#     if competition_id:
#         # 检查比赛是否存在
#         competition = await models.Competition.filter(id=competition_id).first()
#         if not competition:
#             raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

#         base_constraints["competition_id"] = competition_id

#     # 合并约束条件
#     all_constraints = base_constraints
#     if constraints and constraints.data:
#         all_constraints.update(constraints.data)

#     items = sql_utils.to_schema(
#         await sql_utils.paginate(
#             sql_utils.selects(
#                 model=models.Team,
#                 include_fields=include_fields,
#                 exclude_fields=exclude_fields,
#                 constraints=all_constraints,
#                 order_by=order_by,
#             ),
#             paging=paging,
#         ),
#         schemas.Team,
#     )
#     request.state.log_data["items"] = len(items.items)
#     return ResponseModel(data=items)
