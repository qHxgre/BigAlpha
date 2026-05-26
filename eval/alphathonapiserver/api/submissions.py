"""作品提交相关接口"""

import uuid
from pathlib import Path as PathLib

from fastapi import APIRouter, Body, Depends, File, Path, Query, Request, UploadFile
from fastapi.responses import FileResponse

from bigshared2.auth import Credential, authenticator
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from .. import models, schemas, settings
from ._helpers import (
    get_competition_or_404,
    get_user_registration_or_403,
    has_manage_privilege,
    paginated_response,
    require_manager_or_creator,
)

router = APIRouter()

SCORE_FIELDS = frozenset({"public_score", "public_score_data", "private_score", "private_score_data"})


@router.post("")
async def create(
    request: Request,
    credential: Credential = Depends(authenticator),
    submission_in: schemas.SubmissionIn = Body(),
) -> ResponseModel:
    """用户提交比赛作品"""
    data = submission_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    await get_competition_or_404(data["competition_id"])
    await get_user_registration_or_403(data["competition_id"], credential.user_id)

    submission = await models.Submission.create(user_id=credential.user_id, **data)
    request.state.log_data["submission.id"] = submission.id
    return ResponseModel(data=schemas.Submission.model_validate(submission))


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
    response = paginated_response(items, schemas.Submission, include_fields=include_fields, exclude_fields=exclude_fields)
    request.state.log_data["items"] = response.data["count"]
    return response


@router.post("/{submission_id}")
async def update(
    request: Request,
    credential: Credential = Depends(authenticator),
    submission_id: uuid.UUID = Path(),
    submission_update: schemas.SubmissionUpdate = Body(),
) -> ResponseModel:
    """更新提交（用户更新提交数据 / 比赛创建者更新分数）"""
    data = submission_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    submission = await models.Submission.filter(id=submission_id).first()
    if not submission:
        raise HTTPException(Errors.NOT_FOUND.with_message("提交不存在"))

    competition = await get_competition_or_404(submission.competition_id)

    is_creator = competition.creator == credential.user_id
    is_owner = submission.user_id == credential.user_id

    if not is_creator and not is_owner:
        await require_manager_or_creator(request, credential, competition)

    # 分数字段只允许比赛创建者 / 管理员写
    if any(f in data for f in SCORE_FIELDS) and not is_creator:
        await require_manager_or_creator(request, credential, competition)

    await submission.update_from_dict(data).save()
    return ResponseModel(data=schemas.Submission.model_validate(submission))


@router.delete("/{submission_id}")
async def delete(
    request: Request,
    credential: Credential = Depends(authenticator),
    submission_id: uuid.UUID = Path(),
) -> ResponseModel:
    """删除提交"""
    submission = await models.Submission.filter(id=submission_id).first()
    if not submission:
        raise HTTPException(Errors.NOT_FOUND.with_message("提交不存在"))

    competition = await get_competition_or_404(submission.competition_id)
    if submission.user_id != credential.user_id and competition.creator != credential.user_id:
        raise HTTPException(Errors.FORBIDDEN.with_message("只能删除自己的提交或自己创建比赛的提交"))

    await submission.delete()
    return ResponseModel()


@router.post("/files/upload")
async def upload_file(
    request: Request,
    credential: Credential = Depends(authenticator),
    competition_id: uuid.UUID = Query(description="比赛ID"),
    file: UploadFile = File(description="上传的文件"),
) -> ResponseModel:
    """文件上传"""
    await get_competition_or_404(competition_id)
    await get_user_registration_or_403(competition_id, credential.user_id)

    upload_dir = PathLib(settings.FILE_UPLOAD_PATH) / str(competition_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex
    secure_filename = f"{credential.user_id}-{file_id}"
    file_path = upload_dir / secure_filename

    content = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(content)

    request.state.log_data["file_upload"] = {
        "file_id": file_id,
        "original_filename": file.filename,
        "size": len(content),
        "competition_id": str(competition_id),
    }
    return ResponseModel(data={"file_id": file_id, "original_filename": file.filename, "size": len(content)})


@router.get("/files/{submission_id}/{file_id}")
async def get_file(
    request: Request,
    credential: Credential = Depends(authenticator),
    submission_id: uuid.UUID = Path(description="提交ID"),
    file_id: str = Path(description="文件ID"),
) -> FileResponse:
    """文件下载"""
    submission = await models.Submission.filter(id=submission_id).first()
    if not submission:
        raise HTTPException(Errors.NOT_FOUND.with_message("提交不存在"))

    competition = await get_competition_or_404(submission.competition_id)

    if submission.user_id != credential.user_id and competition.creator != credential.user_id:
        await require_manager_or_creator(request, credential, competition)

    submission_files = submission.data.get("files", {})
    if file_id not in submission_files:
        raise HTTPException(Errors.NOT_FOUND.with_message("文件不存在于该提交中"))

    upload_dir = PathLib(settings.FILE_UPLOAD_PATH) / str(submission.competition_id)
    file_path = upload_dir / f"{submission.user_id}-{file_id}"
    if not file_path.exists():
        raise HTTPException(Errors.NOT_FOUND.with_message("文件不存在"))

    original_filename = submission_files[file_id].get("name", file_path.name)
    return FileResponse(path=file_path, filename=original_filename, media_type="application/octet-stream")
