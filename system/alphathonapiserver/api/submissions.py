import uuid
from datetime import datetime
from pathlib import Path as PathLib

from fastapi import APIRouter, Body, Depends, File, Path, Query, Request, UploadFile
from fastapi.responses import FileResponse

from bigshared2.auth import Credential, authenticator, authorizer
from bigshared2.auth.schemas import ANY_SPACE_ID
from bigshared2.db.sql import utils as sql_utils
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import PagingQueryMixin, QueryConstraintsMixin, ResponseModel

from .. import constants, models, schemas, settings

router = APIRouter()


@router.post("")
async def create(
    request: Request,
    credential: Credential = Depends(authenticator),
    submission_in: schemas.SubmissionIn = Body(),
) -> ResponseModel:
    """用户提交比赛作品"""
    data = submission_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    # 检查比赛是否存在
    competition = await models.Competition.filter(id=data["competition_id"]).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 检查用户是否已报名该比赛且审批通过
    user_registration = await models.User.filter(competition_id=data["competition_id"], user_id=credential.user_id).first()
    if not user_registration:
        raise HTTPException(Errors.FORBIDDEN.with_message("请先报名参加比赛"))

    if user_registration.status not in (constants.UserStatus.APPROVED, constants.UserStatus.APPROVED_JOIN_SPACE):
        raise HTTPException(Errors.FORBIDDEN.with_message("用户报名尚未审批通过，无法提交作品"))

    # TODO: 支持直接从 AIStudio 上传文件(mount & copy)

    submission = await models.Submission.create(user_id=credential.user_id, **data)
    request.state.log_data["submission.id"] = submission.id

    return ResponseModel(data=schemas.Submission.model_validate(submission))


@router.get("")
async def reads(
    request: Request,
    credential: Credential = Depends(authenticator),
    constraints: QueryConstraintsMixin | None = Depends(QueryConstraintsMixin.q),
    order_by: list[str] | None = Query([], description="排序字段"),
    include_fields: list[str] | None = Query([], description="只返回指定包含的字段"),
    exclude_fields: list[str] | None = Query([], description="排除的字段"),
    paging: PagingQueryMixin | None = Depends(PagingQueryMixin.q),
    competition_id: uuid.UUID | None = Query(None, description="比赛ID，用于获取指定比赛的提交列表"),
) -> ResponseModel:
    """获取提交列表"""
    # 构建基础查询条件
    base_constraints = {}
    if competition_id:
        # 检查是否有权限查看该比赛的提交列表（比赛创建者可以查看所有提交，普通用户只能查看自己的）
        competition = await models.Competition.filter(id=competition_id).first()
        if not competition:
            raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

        base_constraints["competition_id"] = competition_id

        try:
            # 创建者和管理员才能查看所有提交
            if competition.creator != credential.user_id:
                await authorizer.requires(
                    request,
                    ANY_SPACE_ID,
                    [constants.Privileges.competition_manage],
                )
        except Exception:
            # 非管理员，只能查看自己的提交
            base_constraints["user_id"] = credential.user_id

    else:
        try:
            # 管理员才能查看所有提交
            await authorizer.requires(
                request,
                ANY_SPACE_ID,
                [constants.Privileges.competition_manage],
            )
        except Exception:
            # 非管理员，只能查看自己的提交
            base_constraints["user_id"] = credential.user_id

    # 合并约束条件
    all_constraints = base_constraints
    if constraints and constraints.data:
        all_constraints.update(constraints.data)

    items = sql_utils.to_schema(
        await sql_utils.paginate(
            sql_utils.selects(
                model=models.Submission,
                constraints=all_constraints,
                order_by=order_by,
            ),
            paging=paging,
        ),
        schemas.Submission,
    )

    new_items = schemas.schema_to_dict(schemas.Submission, items.items, include_fields, exclude_fields)
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


@router.post("/{submission_id}")
async def update(
    request: Request,
    credential: Credential = Depends(authenticator),
    submission_id: uuid.UUID = Path(),
    submission_update: schemas.SubmissionUpdate = Body(),
) -> ResponseModel:
    """更新提交（用户更新提交数据或比赛创建者更新分数）"""
    data = submission_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    submission = await models.Submission.filter(id=submission_id).first()
    if not submission:
        raise HTTPException(Errors.NOT_FOUND.with_message("提交不存在"))

    # 检查比赛是否存在
    competition = await models.Competition.filter(id=submission.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 权限检查
    is_competition_creator = competition.creator == credential.user_id
    is_submission_owner = submission.user_id == credential.user_id

    if not is_competition_creator and not is_submission_owner:
        await authorizer.requires(
            request,
            ANY_SPACE_ID,
            [constants.Privileges.competition_manage],
        )

    # 分数相关字段只有比赛创建者可以更新
    score_fields = {"public_score", "public_score_data", "private_score", "private_score_data"}
    if any(field in data for field in score_fields) and not is_competition_creator:
        await authorizer.requires(
            request,
            ANY_SPACE_ID,
            [constants.Privileges.competition_manage],
        )
    else:
        pull_deadline = competition.summary.get("pull_deadline")
        pull_deadline = datetime.fromisoformat(pull_deadline.replace("Z", "+00:00")).date() if pull_deadline else datetime.now().date()
        if datetime.now().date() > pull_deadline:
            raise HTTPException(Errors.FORBIDDEN.with_message("更新已截止"))

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

    # 检查比赛是否存在
    competition = await models.Competition.filter(id=submission.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 权限检查：只有提交者本人或比赛创建者才能删除
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
    """文件上传接口"""
    # 检查比赛是否存在
    competition = await models.Competition.filter(id=competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 检查用户是否已报名该比赛且审批通过
    user_registration = await models.User.filter(competition_id=competition_id, user_id=credential.user_id).first()
    if not user_registration:
        raise HTTPException(Errors.FORBIDDEN.with_message("请先报名参加比赛"))

    if user_registration.status not in (constants.UserStatus.APPROVED, constants.UserStatus.APPROVED_JOIN_SPACE):
        raise HTTPException(Errors.FORBIDDEN.with_message("用户报名尚未审批通过，无法上传文件"))

    # 创建文件保存目录
    upload_dir = PathLib(settings.FILE_UPLOAD_PATH) / str(competition_id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 生成唯一文件名，包含用户ID以防止文件访问权限问题
    file_id = str(uuid.uuid4().hex)
    secure_filename = f"{credential.user_id}-{file_id}"
    file_path = upload_dir / secure_filename

    # 保存文件
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # 记录文件信息到日志
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
    """文件获取接口"""
    # 获取提交记录
    submission = await models.Submission.filter(id=submission_id).first()
    if not submission:
        raise HTTPException(Errors.NOT_FOUND.with_message("提交不存在"))

    # 检查比赛是否存在
    competition = await models.Competition.filter(id=submission.competition_id).first()
    if not competition:
        raise HTTPException(Errors.NOT_FOUND.with_message("比赛不存在"))

    # 权限检查：只有提交者本人或比赛创建者才能获取文件
    if submission.user_id != credential.user_id and competition.creator != credential.user_id:
        await authorizer.requires(
            request,
            ANY_SPACE_ID,
            [constants.Privileges.competition_manage],
        )

    # 检查文件是否在提交数据中
    submission_files = submission.data.get("files", {})
    if file_id not in submission_files:
        raise HTTPException(Errors.NOT_FOUND.with_message("文件不存在于该提交中"))

    # 构建安全文件路径，包含用户ID验证
    upload_dir = PathLib(settings.FILE_UPLOAD_PATH) / str(submission.competition_id)
    secure_filename = f"{submission.user_id}-{file_id}"
    file_path = upload_dir / secure_filename
    if not file_path.exists():
        raise HTTPException(Errors.NOT_FOUND.with_message("文件不存在"))

    # 获取原始文件名
    original_filename = submission_files[file_id].get("name", file_path.name)

    return FileResponse(path=file_path, filename=original_filename, media_type="application/octet-stream")
