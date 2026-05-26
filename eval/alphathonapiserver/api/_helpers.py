"""API 通用 helper：404 校验、截止时间校验、通知发送、分页响应、权限判断"""

from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from fastapi import Request

from bigshared2.auth import Credential, authorizer
from bigshared2.auth.schemas import ANY_SPACE_ID, BIGQUANT_SPACE_ID
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, ResponseModel

from .. import constants, models, schemas
from ..utils import create_notice, send_wechat_message

log = structlog.get_logger(__name__)


async def get_competition_or_404(competition_id: UUID | str) -> models.Competition:
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
    return competition


async def get_user_registration_or_403(competition_id: UUID | str, user_id: UUID | str) -> models.User:
    """获取用户报名记录，未报名/未通过审批则抛 403。"""
    user = await models.User.filter(competition_id=competition_id, user_id=user_id).first()
    if not user:
        raise HTTPException(Errors.FORBIDDEN.with_message("请先报名参加比赛"))
    if user.status != constants.UserStatus.APPROVED:
        raise HTTPException(Errors.FORBIDDEN.with_message("用户报名尚未审批通过"))
    return user


def check_deadline(competition: models.Competition, key: str, message: str) -> None:
    """检查 competition.summary 中的截止时间字段，过期则抛 BAD_REQUEST。

    时间格式异常时静默放行，与原行为保持一致。
    """
    if not competition.summary:
        return
    raw = competition.summary.get(key)
    if not raw:
        return
    try:
        deadline = datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return
    if datetime.now().date() > deadline:
        raise HTTPException(Errors.BAD_REQUEST.with_message(message))


async def has_manage_privilege(request: Request) -> bool:
    """判断当前请求是否具备 competition_manage 权限。"""
    try:
        await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.competition_manage])
        return True
    except HTTPException:
        return False


async def require_manager_or_creator(request: Request, credential: Credential, competition: models.Competition) -> None:
    """要求当前用户是比赛创建者或比赛管理员。"""
    if competition.creator == credential.user_id:
        return
    await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.competition_manage])


async def safe_notify(request: Request, *, user_id: UUID | str, title: str, content: str, channel: str = "system") -> None:
    """发送站内信 + 微信通知，失败仅记录到 log_data，不阻塞主流程。"""
    try:
        await create_notice(user_id=str(user_id), space_id=BIGQUANT_SPACE_ID, title=title, content=content, channel=channel)
        await send_wechat_message(title=title, user_id=user_id)
    except Exception as e:
        request.state.log_data["notice_exception"] = str(e)
        log.warning("notify.failed", user_id=str(user_id), error=str(e))


def paginated_response(items, schema: Any, *, include_fields=None, exclude_fields=None) -> ResponseModel:
    """统一构造分页响应。items 为 sql_utils.paginate 返回值。"""
    new_items = schemas.schema_to_dict(schema, items.items, include_fields, exclude_fields)
    return ResponseModel(
        data={
            "page": items.page,
            "size": items.size,
            "total": items.total,
            "count": len(new_items),
            "items": new_items,
        }
    )


def slice_for_paging(data: list[dict[str, Any]], paging: PagingQueryMixin | None) -> tuple[list[dict[str, Any]], int, int]:
    """对内存数据应用分页，返回 (paginated, page, size)。page 从 1 开始。"""
    page = (paging.page if paging and paging.page else 1) or 1
    size = (paging.size if paging and paging.size else 50) or 50
    if page < 1:
        page = 1
    return data[(page - 1) * size : page * size], page, size
