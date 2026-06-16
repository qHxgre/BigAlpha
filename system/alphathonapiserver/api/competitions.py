import uuid

from fastapi import APIRouter, Body, Depends, Path, Query, Request

from bigshared2.auth import Credential, anonymous_authenticator, authenticator, authorizer
from bigshared2.auth.schemas import ANY_SPACE_ID
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from .. import constants, models, schemas

router = APIRouter()


@router.post("")
async def create(
    request: Request,
    credential: Credential = Depends(authenticator),
    competition_in: schemas.CompetitionIn = Body(),
) -> ResponseModel:
    # TODO: 是否任何人都可以创建比赛

    data = competition_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    competition = await models.Competition.create(creator=credential.user_id, **data)
    request.state.log_data["competition.id"] = competition.id

    return ResponseModel(data=schemas.Competition.model_validate(competition))


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
                model=models.Competition,
                constraints=constraints and constraints.data,
                order_by=order_by,
            ),
            paging=paging,
        ),
        schemas.Competition,
    )

    new_items = schemas.schema_to_dict(schemas.Competition, items.items, include_fields, exclude_fields)
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


@router.post("/{competition_id}")
async def update(
    request: Request,
    credential: Credential = Depends(authenticator),
    competition_id: uuid.UUID = Path(),
    competition_update: schemas.CompetitionUpdate = Body(),
) -> ResponseModel:
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    if competition.creator != credential.user_id:
        await authorizer.requires(
            request,
            ANY_SPACE_ID,
            [constants.Privileges.competition_manage],
        )

    data = competition_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    await competition.update_from_dict(data).save()

    return ResponseModel(data=schemas.Competition.model_validate(competition))


@router.delete("/{competition_id}")
async def delete(
    request: Request,
    credential: Credential = Depends(authenticator),
    competition_id: uuid.UUID = Path(),
) -> ResponseModel:
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    if competition.creator != credential.user_id:
        await authorizer.requires(
            request,
            ANY_SPACE_ID,
            [constants.Privileges.competition_manage],
        )

    await competition.delete()

    return ResponseModel()
