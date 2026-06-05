"""API 通用 helper：404 校验、权限判断、分页响应"""

from uuid import UUID

from fastapi import Request

from bigshared2.auth import authorizer
from bigshared2.auth.schemas import ANY_SPACE_ID
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import ResponseModel

from . import constants, models


async def get_competition_or_404(competition_id: UUID | str) -> models.Competition:
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))
    return competition


async def has_manage_privilege(request: Request) -> bool:
    """判断当前请求是否具备 competition_manage 权限。"""
    try:
        await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.competition_manage])
        return True
    except HTTPException:
        return False


async def require_manage_privilege(request: Request) -> None:
    """要求当前用户具备 competition_manage 权限。"""
    await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.competition_manage])


def paginated_response(items, *, include_fields=None, exclude_fields=None) -> ResponseModel:
    """统一构造分页响应。items 为 sql_utils.paginate 返回值。"""
    if include_fields:
        new_items = [item.dict(include=include_fields) for item in items.items]
    elif exclude_fields:
        new_items = [item.dict(exclude=exclude_fields) for item in items.items]
    else:
        new_items = [item.dict() for item in items.items]
    return ResponseModel(
        data={
            "page": items.page,
            "size": items.size,
            "total": items.total,
            "count": len(new_items),
            "items": new_items,
        }
    )
