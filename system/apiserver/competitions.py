"""比赛只读接口（评测系统不再创建/修改/删除比赛）"""

import uuid

from fastapi import APIRouter, Depends, Path, Query, Request

from bigshared2.auth import Credential, anonymous_authenticator, authenticator
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from . import models, schemas
from .helpers import get_competition_or_404, paginated_response

router = APIRouter()


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
    """获取比赛列表"""
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
    response = paginated_response(items, include_fields=include_fields, exclude_fields=exclude_fields)
    request.state.log_data["items"] = response.data["count"]
    return response


@router.get("/{competition_id}")
async def read(
    _: Credential = Depends(authenticator | anonymous_authenticator),
    competition_id: uuid.UUID = Path(),
) -> ResponseModel:
    """获取单个比赛详情"""
    competition = await get_competition_or_404(competition_id)
    return ResponseModel(data=schemas.Competition.model_validate(competition))
