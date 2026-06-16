import uuid

from fastapi import APIRouter, Body, Depends, Path, Query, Request
from tortoise.functions import Max

from bigshared2.auth import Credential, anonymous_authenticator, authenticator, authorizer
from bigshared2.auth.schemas import ANY_SPACE_ID
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

import constants
import models
import schemas

router = APIRouter()


@router.post("")
async def create(
    request: Request,
    credential: Credential = Depends(authenticator),
    code_in: schemas.CodeIn = Body(),
) -> ResponseModel:
    data = code_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    # Verify competition exists
    competition = await models.Competition.filter(id=code_in.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    user = await models.User.filter(competition_id=code_in.competition_id, user_id=credential.user_id, status=constants.UserStatus.APPROVED).first()
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
    include_fields: list[str] | None = Query([], description="只返回指定包含的字段，用于减少不必要的请求，比如列表页只需要特定字段"),
    exclude_fields: list[str] | None = Query([], description="排除的字段，用于减少不必要的请求"),
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

    new_items = schemas.schema_to_dict(schemas.Code, items.items, include_fields, exclude_fields)
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


# @router.get("/{code_id}")
async def read(
    request: Request,
    credential: Credential = Depends(authenticator | anonymous_authenticator),
    code_id: uuid.UUID = Path(),
) -> ResponseModel:
    code = await models.Code.filter(id=code_id).first()
    if not code:
        raise HTTPException(Errors.NOT_FOUND.with_message("代码不存在"))
    if credential.user_id != code.creator and credential.user_id != code.competition.creator:
        await authorizer.requires(
            request,
            ANY_SPACE_ID,
            [constants.Privileges.competition_manage],
        )

    return ResponseModel(data=schemas.Code.model_validate(code))


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
    code.data = code.data if code.data else {}
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
        # 去除传入数据中的 likers, is_top
        code_update.data.pop("likers", None)
        code_update.data.pop("is_top", None)
        if code_update.data:
            code.data = code.data.update(code_update.data)
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
        await authorizer.requires(
            request,
            ANY_SPACE_ID,
            [constants.Privileges.competition_manage],
        )

    await code.delete()

    return ResponseModel()
