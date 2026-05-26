"""用户报名相关接口"""

import uuid

from fastapi import APIRouter, Body, Depends, Path, Query, Request

from bigshared2.auth import Credential, authenticator
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from .. import constants, models, schemas
from ._helpers import (
    check_deadline,
    get_competition_or_404,
    has_manage_privilege,
    paginated_response,
    require_manager_or_creator,
    safe_notify,
)

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

    competition = await get_competition_or_404(data["competition_id"])
    check_deadline(competition, "entry_deadline", "报名已截止")

    # 已报名：被拒后允许重新提交
    existing_user = await models.User.filter(competition_id=data["competition_id"], user_id=credential.user_id).first()
    if existing_user:
        if existing_user.status == constants.UserStatus.REJECTED:
            existing_user.status = constants.UserStatus.PENDING
            existing_user.data = user_in.data
            await existing_user.save()
        return ResponseModel(data=schemas.User.model_validate(existing_user))

    status = constants.UserStatus.APPROVED if competition.summary.get("approve_disabled", False) else constants.UserStatus.PENDING
    user = await models.User.create(user_id=credential.user_id, status=status, **data)
    request.state.log_data["user.id"] = user.id

    return ResponseModel(data=schemas.User.model_validate(user))


@router.get("")
async def reads(
    request: Request,
    credential: Credential = Depends(authenticator),
    constraints: QueryConstraintsMixin | None = Depends(QueryConstraintsMixin.q),
    order_by: list[str] | None = Query([], description="排序字段"),
    include_fields: list[str] | None = Query([], description="只返回指定字段"),
    exclude_fields: list[str] | None = Query([], description="排除字段"),
    paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
    competition_id: uuid.UUID | None = Query(None, description="比赛ID，用于获取指定比赛的用户列表"),
) -> ResponseModel:
    """获取比赛用户列表"""
    all_constraints = dict(constraints.data) if constraints and constraints.data else {}
    should_apply_privacy = True

    if competition_id:
        competition = await get_competition_or_404(competition_id)
        should_apply_privacy = await _should_apply_privacy(request, competition, credential, all_constraints)
        all_constraints["competition_id"] = competition_id
    else:
        should_apply_privacy = False
        if not await has_manage_privilege(request):
            # 非管理员只能看到自己的报名记录
            all_constraints["user_id"] = credential.user_id

    items = sql_utils.to_schema(
        await sql_utils.paginate(
            sql_utils.selects(model=models.User, constraints=all_constraints, order_by=order_by),
            paging=paging,
        ),
        schemas.User,
    )
    if should_apply_privacy:
        _apply_privacy_protection(items)

    response = paginated_response(items, schemas.User, include_fields=include_fields, exclude_fields=exclude_fields)
    request.state.log_data["items"] = response.data["count"]
    return response


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

    competition = await get_competition_or_404(user.competition_id)

    data = user_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data
    data["data"] = data["data"] if user_update.data else user.data

    if "status" in data:
        # 改状态前清掉旧的拒绝理由
        data["data"].pop("reject_reason", None)
        # 只有比赛创建者或管理员可以审批
        await require_manager_or_creator(request, credential, competition)
        check_deadline(competition, "entry_deadline", "报名已截止，无法审批")
        await _send_review_notice(request, user, competition, data)

    if user.user_id != credential.user_id:
        await require_manager_or_creator(request, credential, competition)

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

    competition = await get_competition_or_404(user.competition_id)
    if user.user_id != credential.user_id and competition.creator != credential.user_id:
        raise HTTPException(Errors.FORBIDDEN.with_message("只能取消自己的报名或删除自己创建比赛的用户"))

    await user.delete()
    return ResponseModel()


async def _send_review_notice(request: Request, user: models.User, competition: models.Competition, data: dict) -> None:
    """根据审批状态构造通知文案并发送。"""
    title = "【比赛报名审核通知】"
    content: str | None = None

    if data.get("status") == constants.UserStatus.APPROVED.value:
        content = (
            f"【{competition.name}】恭喜！您申请的比赛报名审核已通过。"
            f"[快去看看吧>>](https://bigquant.com/square/competition/{competition.id})"
        )
    elif data.get("status") == constants.UserStatus.REJECTED.value:
        reject_reason = data.get("reject_reason")
        suffix = ""
        if reject_reason:
            data["data"]["reject_reason"] = reject_reason
            suffix = f"，拒绝原因：{reject_reason}"
        content = (
            f"【{competition.name}】很抱歉！您申请的比赛报名审核未通过{suffix}。"
            f"[去看看别的比赛吧>>](https://bigquant.com/square/competition)"
        )

    if content:
        await safe_notify(request, user_id=user.user_id, title=title, content=content)


async def _should_apply_privacy(request, competition: models.Competition, credential: Credential, constraints: dict) -> bool:
    """判断是否对返回的 user 列表应用隐私保护。

    规则：
    - 比赛创建者 / 管理员 → 不脱敏
    - 普通用户查询其他人 → 脱敏
    - 普通用户查询自己 → 不脱敏
    """
    if competition.creator == credential.user_id:
        return False
    if await has_manage_privilege(request):
        return False
    if "user_id" in constraints:
        return str(constraints["user_id"]) != str(credential.user_id)
    return True


def _apply_privacy_protection(items) -> None:
    """非授权方查询时只保留 data.name，其他字段抹除。"""
    for item in items.items:
        if hasattr(item, "data") and isinstance(item.data, dict):
            item.data = {"name": item.data.get("name")}
