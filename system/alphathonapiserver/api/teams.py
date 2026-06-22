import uuid
from datetime import datetime

from fastapi import APIRouter, Body, Depends, Path, Query, Request
from tortoise.transactions import in_transaction

from bigshared2.auth import Credential, authenticator
from bigshared2.auth.schemas import BIGQUANT_SPACE_ID
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from .. import constants, models, schemas
from ..constants import TeamApplyStatus
from ..utils import create_notice, send_wechat_message

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

    if user_registration.status not in (constants.UserStatus.APPROVED, constants.UserStatus.APPROVED_JOIN_SPACE):
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

    if user_registration.status not in (constants.UserStatus.APPROVED, constants.UserStatus.APPROVED_JOIN_SPACE):
        raise HTTPException(Errors.FORBIDDEN.with_message("用户报名尚未审批通过，无法申请加入团队"))

    # 检查用户是否已在其他团队中
    existing_team = await get_user_team(team.competition_id, credential.user_id)
    if existing_team:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户已在团队「{existing_team.name}」中，无法申请加入新团队"))

    # 检查是否已有待审批的申请
    existing_application = await models.TeamApply.filter(team_id=team_id, user_id=credential.user_id, status=TeamApplyStatus.PENDING).first()
    if existing_application:
        raise HTTPException(Errors.BAD_REQUEST.with_message("您已申请加入该团队，请等待审批"))

    # 创建申请记录，并更新团队的待审批用户列表
    user_id_str = str(credential.user_id)

    # 使用事务处理，保证 team 和 team_apply 的数据一致性
    async with in_transaction(connection_name="primary"):
        pending_users = team.pending_users or []
        if user_id_str not in pending_users:
            pending_users.append(user_id_str)
            team.pending_users = pending_users
            await team.save()

        await models.TeamApply.create(team_id=team_id, user_id=credential.user_id, competition_id=team.competition_id, status=TeamApplyStatus.PENDING)

    return ResponseModel(data=schemas.Team.model_validate(team))


@router.post("/{team_id}/approve/{user_id}")
async def approve_application(
    request: Request,
    credential: Credential = Depends(authenticator),
    team_id: uuid.UUID = Path(),
    user_id: uuid.UUID = Path(),
    approve: bool = Body(embed=True, description="true为通过，false为拒绝"),
    rejection_reason: str | None = Body(None, description="拒绝理由"),
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

    # 使用事务处理，保证 team 和 team_apply 的数据一致性
    async with in_transaction(connection_name="primary"):
        if approve:
            # 再次检查用户是否在其他团队中（防止并发问题）
            existing_team = await get_user_team(team.competition_id, user_id)
            if existing_team and existing_team.id != team.id:
                raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户已在团队「{existing_team.name}」中，无法加入团队「{team.name}」"))

            # 通过：添加到成员列表，从待审批列表中移除
            user_id_str = str(user_id)
            members = team.members or []
            if user_id_str not in members:
                members.append(user_id_str)
                team.members = members

            # 从待审批列表中移除
            pending_users = team.pending_users or []
            if user_id_str in pending_users:
                pending_users.remove(user_id_str)
                team.pending_users = pending_users

            content = (
                f"【{competition.name}】恭喜！您申请加入【{team.name}】的审核已通过。[快去看看吧>>](https://bigquant.com/square/competition/{competition.id})"
            )
        else:
            # 拒绝：从待审批列表中移除
            user_id_str = str(user_id)
            pending_users = team.pending_users or []
            if user_id_str in pending_users:
                pending_users.remove(user_id_str)
                team.pending_users = pending_users

            content = f"【{competition.name}】很抱歉！您申请加入【{team.name}】的审核未通过。[去看看别的团队吧>>](https://bigquant.com/square/competition/{competition.id})"

        # 检查申请是否存在
        team_apply = await models.TeamApply.filter(team_id=team_id, user_id=user_id, status=TeamApplyStatus.PENDING).order_by("-created_at").first()
        if team_apply:
            team_apply.status = TeamApplyStatus.APPROVED if approve else TeamApplyStatus.REJECTED
            if team_apply.status == TeamApplyStatus.REJECTED:
                team_apply.rejection_reason = rejection_reason
            await team_apply.save()
        await team.save()
    try:
        await create_notice(user_id=str(user_id), space_id=BIGQUANT_SPACE_ID, title="【比赛团队审核通知】", content=content, channel="system")
        await send_wechat_message(title="比赛团队审核通知", user_id=user_id)
    except Exception as e:
        request.state.log_data["notice_exception"] = str(e)

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
        # 使用事务处理，保证 team 和 team_apply 的数据一致性
        async with in_transaction(connection_name="primary"):
            members.remove(str(user_id))
            team.members = members

            # Create or update TeamApply record with REMOVED status
            team_apply = await models.TeamApply.filter(team_id=team_id, user_id=user_id, status=TeamApplyStatus.APPROVED).order_by("-created_at").first()
            if team_apply:
                team_apply.status = TeamApplyStatus.REMOVED
                await team_apply.save()
            else:
                await models.TeamApply.create(team_id=team_id, user_id=user_id, status=TeamApplyStatus.REMOVED)

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

    if team.members:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"团队「{team.name}」有成员，请先移除所有成员"))

    # 检查是否还有成员或待审批申请
    team_applies = await models.TeamApply.filter(team_id=team_id, status__in=(TeamApplyStatus.PENDING, TeamApplyStatus.APPROVED)).all()

    # Get all members to notify
    pending_users = team.pending_users or []

    # 使用事务处理，保证 team 和 team_apply 的数据一致性
    async with in_transaction(connection_name="primary"):
        # Update all TeamApply records to TEAM_DELETED status instead of deleting them
        for team_apply in team_applies:
            team_apply.status = TeamApplyStatus.TEAM_DELETED
            await team_apply.save()
        # 删除团队
        await team.delete()

    request.state.log_data["team.id"] = team.id

    # Send notifications to all members
    content = f"【{competition.name}】您所在的团队【{team.name}】已被解散，[去看看别的团队吧>>](https://bigquant.com/square/competition/{competition.id})"
    for pending_user_id in pending_users:
        try:
            await create_notice(user_id=pending_user_id, space_id=BIGQUANT_SPACE_ID, title="【比赛团队解散通知】", content=content, channel="system")
            await send_wechat_message(title="比赛团队解散通知", user_id=pending_user_id)
        except Exception as e:
            request.state.log_data["notice_exception"] = str(e)

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

    user_id = str(credential.user_id)
    # 查找用户所在的团队
    user_team = await get_user_team(competition_id, user_id)
    if not user_team:
        # 没有找到任何团队，检查是否有待审批的申请
        # 直接查询当前比赛的待审批申请
        last_team_apply = await models.TeamApply.filter(user_id=user_id, competition_id=competition_id).order_by("-created_at").first()

        if last_team_apply:
            team = await models.Team.filter(id=last_team_apply.team_id, competition_id=last_team_apply.competition_id).get_or_none()
            if team:
                # 找到属于当前比赛的待审批申请
                team_data = schemas.Team.model_validate(team).model_dump()
                team_data["pending_users"] = len(team.pending_users or [])
                return ResponseModel(data={"team": team_data, "status": last_team_apply.status, "role": None})

        # 没有找到任何团队或申请
        return ResponseModel(data=None)
    else:
        status = "approved"
        if user_id in user_team.pending_users:
            status = "pending"

        # 如果是团队创建者，则返回团队创建者角色
        return ResponseModel(
            data={
                "team": schemas.Team.model_validate(user_team),
                "status": status,
                "role": "creator" if user_team.creator == credential.user_id else "member" if status == "approved" else None,
            }
        )


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

    # 检查用户是否在该团队中（成员）
    members = team.members or []
    pending_users = team.pending_users or []

    # 检查是否有该团队的申请记录
    team_apply = await models.TeamApply.filter(team_id=team_id, user_id=credential.user_id).order_by("-created_at").first()

    if user_id_str not in members and user_id_str not in pending_users:
        raise HTTPException(Errors.BAD_REQUEST.with_message(f"用户不在团队「{team.name}」中"))

    # 使用事务处理，保证 team 和 team_apply 的数据一致性
    async with in_transaction(connection_name="primary"):
        team.members = list(filter(lambda uid: uid != user_id_str, members))
        team.pending_users = list(filter(lambda uid: uid != user_id_str, pending_users))
        await team.save()

        # 如果存在对应的 TeamApply 记录，更新为 LEAVE 状态
        if team_apply:
            team_apply.status = TeamApplyStatus.LEAVE
            await team_apply.save()
        else:
            # 如果没有 TeamApply 记录，创建一条 LEAVE 状态的记录
            await models.TeamApply.create(team_id=team_id, user_id=credential.user_id, status=TeamApplyStatus.LEAVE)

    request.state.log_data["action"] = "leave_team"

    return ResponseModel()


@router.get("")
async def reads(
    request: Request,
    credential: Credential = Depends(authenticator),
    constraints: QueryConstraintsMixin | None = Depends(QueryConstraintsMixin.q),
    order_by: list[str] | None = Query([], description="排序字段"),
    include_fields: list[str] | None = Query([], description="只返回指定包含的字段"),
    exclude_fields: list[str] | None = Query([], description="排除的字段"),
    paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
    competition_id: uuid.UUID | None = Query(None, description="比赛ID，用于获取指定比赛的团队列表"),
) -> ResponseModel:
    """获取团队列表"""
    base_constraints = {}
    if competition_id:
        competition = await models.Competition.filter(id=competition_id).first()
        if not competition:
            raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

        base_constraints["competition_id"] = competition_id

    all_constraints = base_constraints
    if constraints and constraints.data:
        all_constraints.update(constraints.data)

    items = sql_utils.to_schema(
        await sql_utils.paginate(
            sql_utils.selects(
                model=models.Team,
                constraints=all_constraints,
                order_by=order_by,
            ),
            paging=paging,
        ),
        schemas.Team,
    )
    new_items = schemas.schema_to_dict(schemas.Team, items.items, include_fields, exclude_fields)
    request.state.log_data["items"] = len(new_items)
    return ResponseModel(
        data={
            "page": items.page,
            "size": items.size,
            "total": items.total,
            "count": len(new_items),
            "items": new_items,
        }
    )
