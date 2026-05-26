"""代码广场相关接口"""

import uuid

from fastapi import APIRouter, Body, Depends, Path, Query, Request
from tortoise.functions import Max

from bigshared2.auth import Credential, anonymous_authenticator, authenticator
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from .. import constants, models, schemas
from ._helpers import get_competition_or_404, paginated_response, require_manager_or_creator

router = APIRouter()


@router.post("")
async def create(
    request: Request,
    credential: Credential = Depends(authenticator),
    code_in: schemas.CodeIn = Body(),
) -> ResponseModel:
    data = code_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    await get_competition_or_404(code_in.competition_id)
    user = await models.User.filter(
        competition_id=code_in.competition_id,
        user_id=credential.user_id,
        status=constants.UserStatus.APPROVED,
    ).first()
    if not user:
        raise HTTPException(Errors.FORBIDDEN.with_message("用户未加入比赛"))

    code = await models.Code.create(creator=credential.user_id, **data)
    request.state.log_data["code.id"] = code.id
    return ResponseModel(data=schemas.Code.model_validate(code))


@router.get("")
async def reads(
    request: Request,
    credential: Credential = Depends(authenticator | anonymous_authenticator),
    constraints: QueryConstraintsMixin | None = Depends(QueryConstraintsMixin.q),
    order_by: list[str] | None = Query([], description="排序字段"),
    include_fields: list[str] | None = Query([], description="只返回指定字段"),
    exclude_fields: list[str] | None = Query([], description="排除字段"),
    paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
) -> ResponseModel:
    items = sql_utils.to_schema(
        await sql_utils.paginate(
            sql_utils.selects(
                model=models.Code,
                constraints=constraints and constraints.data,
                order_by=order_by,
            ),
            paging=paging,
        ),
        schemas.Code,
    )
    response = paginated_response(items, schemas.Code, include_fields=include_fields, exclude_fields=exclude_fields)
    request.state.log_data["items"] = response.data["count"]
    return response


@router.post("/{code_id}")
async def update(
    request: Request,
    credential: Credential = Depends(authenticator),
    code_id: uuid.UUID = Path(),
    code_update: schemas.CodeUpdate = Body(),
) -> ResponseModel:
    code = await models.Code.filter(id=code_id).first()
    if not code:
        raise HTTPException(Errors.NOT_FOUND.with_message("代码不存在"))

    code.data = code.data or {}
    data = code_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    bq_user_id = str(credential.user_id)
    likers = code.data.get("likers", [])
    if code_update.like == 1 and bq_user_id not in likers:
        likers.append(bq_user_id)
    elif code_update.like == -1 and bq_user_id in likers:
        likers.remove(bq_user_id)
    code.like_count = len(likers)
    code.data["likers"] = likers

    if code_update.top == 1 and not code.data.get("is_top", False):
        max_rank = await models.Code.annotate(max_rank=Max("rank")).values("max_rank")
        code.rank = max_rank[0]["max_rank"] + 1
        code.data["is_top"] = True
    elif code_update.top == -1 and code.data.get("is_top", False):
        code.rank = 1
        code.data["is_top"] = False

    if code_update.data:
        # 不允许从外部覆盖系统维护的字段
        extra = {k: v for k, v in code_update.data.items() if k not in ("likers", "is_top")}
        if extra:
            code.data.update(extra)

    await code.save()
    return ResponseModel(data=schemas.Code.model_validate(code))


@router.delete("/{code_id}")
async def delete(
    request: Request,
    credential: Credential = Depends(authenticator),
    code_id: uuid.UUID = Path(),
) -> ResponseModel:
    code = await models.Code.filter(id=code_id).first()
    if not code:
        raise HTTPException(Errors.NOT_FOUND.with_message("代码不存在"))

    if credential.user_id != code.creator:
        competition = await get_competition_or_404(code.competition_id)
        await require_manager_or_creator(request, credential, competition)

    await code.delete()
    return ResponseModel()
