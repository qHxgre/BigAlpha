import uuid
from datetime import datetime

from fastapi import APIRouter, Body, Depends, Path, Query, Request

from bigshared2.auth import Credential, authenticator, authorizer
from bigshared2.auth.schemas import ANY_SPACE_ID, BIGQUANT_SPACE_ID
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

import constants
import models
import schemas
from utils import create_notice, send_wechat_message

router = APIRouter()


@router.post("")
async def create(
    request: Request,
    credential: Credential = Depends(authenticator),
    user_in: schemas.UserIn = Body(),
) -> ResponseModel:
    """用户报名比赛"""
    data = user_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    # 检查比赛是否存在
    competition = await models.Competition.filter(id=data["competition_id"]).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 检查报名截止时间
    if competition.summary and "entry_deadline" in competition.summary:
        entry_deadline_str = competition.summary["entry_deadline"]
        try:
            entry_deadline = datetime.strptime(entry_deadline_str[:10], "%Y-%m-%d").date()
            current_date = datetime.now().date()
            if current_date > entry_deadline:
                raise HTTPException(Errors.BAD_REQUEST.with_message("报名已截止"))
        except (ValueError, TypeError):
            # 如果时间格式不正确，记录但不阻止报名
            pass

    # 检查用户是否已经报名，如果已报名则返回现有状态
    existing_user = await models.User.filter(competition_id=data["competition_id"], user_id=credential.user_id).first()
    if existing_user:
        if existing_user.status == constants.UserStatus.REJECTED:
            existing_user.status = constants.UserStatus.PENDING
            existing_user.data = user_in.data
            await existing_user.save()
        return ResponseModel(data=schemas.User.model_validate(existing_user))
    status = constants.UserStatus.PENDING
    if competition.summary.get("approve_disabled", False):
        status = constants.UserStatus.APPROVED

    user = await models.User.create(user_id=credential.user_id, status=status, **data)
    request.state.log_data["user.id"] = user.id

    return ResponseModel(data=schemas.User.model_validate(user))


@router.get("")
async def reads(
    request: Request,
    credential: Credential = Depends(authenticator),
    constraints: QueryConstraintsMixin | None = Depends(QueryConstraintsMixin.q),
    order_by: list[str] | None = Query([], description="排序字段"),
    include_fields: list[str] | None = Query([], description="只返回指定包含的字段"),
    exclude_fields: list[str] | None = Query([], description="排除的字段"),
    paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
    competition_id: uuid.UUID | None = Query(None, description="比赛ID，用于获取指定比赛的用户列表"),
) -> ResponseModel:
    """获取比赛用户列表"""
    # 构建基础查询条件
    all_constraints = constraints.data if constraints and constraints.data else {}
    base_constraints = {}
    should_apply_privacy_protection = True
    if competition_id:
        competition = await models.Competition.filter(id=competition_id).first()
        if not competition:
            raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
        # 判断是否需要应用隐私保护
        should_apply_privacy_protection = await _should_apply_privacy_protection(request, competition, credential, all_constraints)

        base_constraints["competition_id"] = competition_id
    else:
        should_apply_privacy_protection = False
        # 如果没有指定比赛ID，先判定是否是比赛管理员，是则获取所有用户列表
        try:
            await authorizer.requires(
                request,
                ANY_SPACE_ID,
                [constants.Privileges.competition_manage],
            )
        except HTTPException:
            # 非管理员只能查看自己的报名记录
            base_constraints["user_id"] = credential.user_id

    # 合并约束条件
    all_constraints.update(base_constraints)
    items = sql_utils.to_schema(
        await sql_utils.paginate(
            sql_utils.selects(
                model=models.User,
                constraints=all_constraints,
                order_by=order_by,
            ),
            paging=paging,
        ),
        schemas.User,
    )
    if should_apply_privacy_protection:
        _apply_privacy_protection(items)

    new_items = schemas.schema_to_dict(schemas.User, items.items, include_fields, exclude_fields)
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


@router.post("/{user_id}")
async def update(
    request: Request,
    credential: Credential = Depends(authenticator),
    user_id: uuid.UUID = Path(),
    user_update: schemas.UserUpdate = Body(),
) -> ResponseModel:
    """更新用户报名状态（审批）"""
    user = await models.User.filter(id=user_id).first()
    if not user:
        raise HTTPException(Errors.NOT_FOUND.with_message("用户报名记录不存在"))

    # 检查权限：比赛创建者可以审批，用户本人可以更新自己的报名信息
    competition = await models.Competition.filter(id=user.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    data = user_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    # 如果没有传data字段，则使用原数据
    data["data"] = data["data"] if user_update.data else user.data
    if "status" in data:
        # 更新状态，先删除拒绝理由
        data["data"].pop("reject_reason", None)
        # 检查权限：只有比赛创建者或管理员可以审批
        if competition.creator != credential.user_id:
            await authorizer.requires(
                request,
                ANY_SPACE_ID,
                [constants.Privileges.competition_manage],
            )
        # 如果要修改状态，检查报名截止时间
        if competition.summary and "entry_deadline" in competition.summary:
            entry_deadline_str = competition.summary["entry_deadline"]
            try:
                entry_deadline = datetime.strptime(entry_deadline_str[:10], "%Y-%m-%d").date()
                current_date = datetime.now().date()
                if current_date > entry_deadline:
                    raise HTTPException(Errors.BAD_REQUEST.with_message("报名已截止，无法审批"))
            except (ValueError, TypeError):
                # 如果时间格式不正确，记录但不阻止审批
                pass
        try:
            title = "【比赛报名审核通知】"
            content = None
            if data["status"] == constants.UserStatus.APPROVED.value:
                content = f"【{competition.name}】恭喜！您申请的比赛报名审核已通过。[快去看看吧>>](https://bigquant.com/square/competition/{competition.id})"
            elif data["status"] == constants.UserStatus.REJECTED.value:
                reject_reason = data.get("reject_reason")
                if reject_reason:
                    data["data"]["reject_reason"] = reject_reason
                    reject_reason = f"，拒绝原因：{reject_reason}"
                content = (
                    f"【{competition.name}】很抱歉！您申请的比赛报名审核未通过{reject_reason}。[去看看别的比赛吧>>](https://bigquant.com/square/competition)"
                )
            if content:
                await create_notice(user_id=str(user.user_id), space_id=BIGQUANT_SPACE_ID, title=title, content=content, channel="system")
                await send_wechat_message(title=title, user_id=user.user_id)
        except Exception as e:
            request.state.log_data["notice_exception"] = str(e)

    # 如果要修改其他信息，只有用户本人或比赛创建者才能操作
    if user.user_id != credential.user_id and competition.creator != credential.user_id:
        await authorizer.requires(
            request,
            ANY_SPACE_ID,
            [constants.Privileges.competition_manage],
        )

    await user.update_from_dict(data).save()

    return ResponseModel(data=schemas.User.model_validate(user))


@router.delete("/{user_id}")
async def delete(
    request: Request,
    credential: Credential = Depends(authenticator),
    user_id: uuid.UUID = Path(),
) -> ResponseModel:
    """取消报名"""
    user = await models.User.filter(id=user_id).first()
    if not user:
        raise HTTPException(Errors.NOT_FOUND.with_message("用户报名记录不存在"))

    # 检查权限：只有用户本人或比赛创建者才能删除
    competition = await models.Competition.filter(id=user.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    if user.user_id != credential.user_id and competition.creator != credential.user_id:
        raise HTTPException(Errors.FORBIDDEN.with_message("只能取消自己的报名或删除自己创建比赛的用户"))

    await user.delete()

    return ResponseModel()


async def _should_apply_privacy_protection(request, competition, credential, constraints: dict) -> bool:
    """判断是否应该应用隐私保护模式

    隐私保护规则：
    1. 如果用户是比赛创建者，无需隐私保护
    2. 如果用户不是创建者，且查询条件包含其他用户ID，则需要隐私保护

    Args:
        request: 请求对象
        competition: 比赛对象
        credential: 当前用户凭据
        constraints: 查询约束条件

    Returns:
        bool: True表示需要应用隐私保护，False表示不需要
    """
    # 如果是比赛创建者，可以查看所有信息，不需要隐私保护

    if competition.creator == credential.user_id:
        return False

    try:
        await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.competition_manage])
        return False
    except Exception:
        pass
    # 检查是否查询其他用户的信息
    is_querying_specific_user = "user_id" in constraints

    if is_querying_specific_user:
        # 如果查询特定用户且不是自己，则需要隐私保护
        return str(constraints["user_id"]) != str(credential.user_id)

    # 默认情况下需要隐私保护
    return True


def _apply_privacy_protection(items):
    """对用户列表应用隐私保护，只保留姓名信息

    Args:
        items: 用户列表对象
    """
    for item in items.items:
        if hasattr(item, "data") and isinstance(item.data, dict):
            item.data = {"name": item.data.get("name")}
