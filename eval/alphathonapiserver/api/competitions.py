"""比赛 CRUD"""

import uuid

from fastapi import APIRouter, Body, Depends, Path, Query, Request

from bigshared2.auth import Credential, anonymous_authenticator, authenticator
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from .. import models, schemas
from ._helpers import get_competition_or_404, paginated_response, require_manager_or_creator

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
    include_fields: list[str] | None = Query([], description="只返回指定字段"),
    exclude_fields: list[str] | None = Query([], description="排除字段"),
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
    response = paginated_response(items, schemas.Competition, include_fields=include_fields, exclude_fields=exclude_fields)
    request.state.log_data["items"] = response.data["count"]
    return response


@router.post("/{competition_id}")
async def update(
    request: Request,
    credential: Credential = Depends(authenticator),
    competition_id: uuid.UUID = Path(),
    competition_update: schemas.CompetitionUpdate = Body(),
) -> ResponseModel:
    competition = await get_competition_or_404(competition_id)
    await require_manager_or_creator(request, credential, competition)

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
    competition = await get_competition_or_404(competition_id)
    await require_manager_or_creator(request, credential, competition)
    await competition.delete()
    return ResponseModel()
