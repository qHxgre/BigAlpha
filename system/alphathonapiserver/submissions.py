"""作品提交相关接口（评测系统：仅查询 + 回写分数 + 文件下载）"""

import uuid
from pathlib import Path as PathLib

from fastapi import APIRouter, Body, Depends, Path, Query, Request
from fastapi.responses import FileResponse

from bigshared2.auth import Credential, authenticator
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from . import models, schemas, settings
from .helpers import (
    get_competition_or_404,
    has_manage_privilege,
    paginated_response,
    require_manage_privilege,
)

router = APIRouter()


@router.get("")
async def reads(
    request: Request,
    credential: Credential = Depends(authenticator),
    constraints: QueryConstraintsMixin | None = Depends(QueryConstraintsMixin.q),
    order_by: list[str] | None = Query([], description="排序字段"),
    include_fields: list[str] | None = Query([], description="只返回指定字段"),
    exclude_fields: list[str] | None = Query([], description="排除字段"),
    paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
    competition_id: uuid.UUID | None = Query(None, description="比赛ID"),
) -> ResponseModel:
    """获取提交列表"""
    base_constraints: dict = {}
    if competition_id:
        await get_competition_or_404(competition_id)
        base_constraints["competition_id"] = competition_id

    # 非管理员只能看自己的提交
    if not await has_manage_privilege(request):
        base_constraints["user_id"] = credential.user_id

    if constraints and constraints.data:
        base_constraints.update(constraints.data)

    items = sql_utils.to_schema(
        await sql_utils.paginate(
            sql_utils.selects(model=models.Submission, constraints=base_constraints, order_by=order_by),
            paging=paging,
        ),
        schemas.Submission,
    )
    response = paginated_response(items, include_fields=include_fields, exclude_fields=exclude_fields)
    request.state.log_data["items"] = response.data["count"]
    return response


@router.post("/{submission_id}")
async def update_score(
    request: Request,
    credential: Credential = Depends(authenticator),
    submission_id: uuid.UUID = Path(),
    score_update: schemas.SubmissionScoreUpdate = Body(),
) -> ResponseModel:
    """评测系统回写分数。仅 competition_manage 权限可调用。"""
    await require_manage_privilege(request)

    submission = await models.Submission.filter(id=submission_id).first()
    if not submission:
        raise HTTPException(Errors.NOT_FOUND.with_message("提交不存在"))

    data = score_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    await submission.update_from_dict(data).save()
    return ResponseModel(data=schemas.Submission.model_validate(submission))


@router.get("/files/{submission_id}/{file_id}")
async def get_file(
    request: Request,
    credential: Credential = Depends(authenticator),
    submission_id: uuid.UUID = Path(description="提交ID"),
    file_id: str = Path(description="文件ID"),
) -> FileResponse:
    """文件下载（评测系统拉选手代码用）"""
    submission = await models.Submission.filter(id=submission_id).first()
    if not submission:
        raise HTTPException(Errors.NOT_FOUND.with_message("提交不存在"))

    competition = await get_competition_or_404(submission.competition_id)

    is_owner = submission.user_id == credential.user_id
    is_creator = competition.creator == credential.user_id
    if not is_owner and not is_creator:
        await require_manage_privilege(request)

    submission_files = submission.data.get("files", {})
    if file_id not in submission_files:
        raise HTTPException(Errors.NOT_FOUND.with_message("文件不存在于该提交中"))

    upload_dir = PathLib(settings.FILE_UPLOAD_PATH) / str(submission.competition_id)
    file_path = upload_dir / f"{submission.user_id}-{file_id}"
    if not file_path.exists():
        raise HTTPException(Errors.NOT_FOUND.with_message("文件不存在"))

    original_filename = submission_files[file_id].get("name", file_path.name)
    return FileResponse(path=file_path, filename=original_filename, media_type="application/octet-stream")
