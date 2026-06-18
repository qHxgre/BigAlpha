"""导师相关接口"""

import uuid

from fastapi import APIRouter, Body, Depends, Path, Query, Request
from tortoise.exceptions import IntegrityError

from bigshared2.auth import Credential, anonymous_authenticator, authenticator, authorizer
from bigshared2.auth.schemas import ANY_SPACE_ID
from bigshared2.schemas.exceptions import Errors, HTTPException
from bigshared2.schemas.http import ResponseModel

from .. import constants, models, schemas

router = APIRouter()


# ---------------------------------------------------------------------------
# 导师管理（需要管理员权限）
# ---------------------------------------------------------------------------


@router.post("")
async def create_mentor(
    request: Request,
    credential: Credential = Depends(authenticator),
    mentor_in: schemas.MentorIn = Body(),  # codespell:ignore
) -> ResponseModel:
    """创建导师"""
    await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.mentor_manage])

    data = mentor_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    mentor = await models.Mentor.create(**data)
    request.state.log_data["mentor.id"] = str(mentor.id)

    return ResponseModel(data=schemas.Mentor.model_validate(mentor))


# ---------------------------------------------------------------------------
# 导师愿望单（路由必须在 /{mentor_id} 之前注册，否则会被参数路由捕获）
# ---------------------------------------------------------------------------


@router.post("/wishlists")
async def create_wishlist(
    request: Request,
    credential: Credential = Depends(authenticator),
    wishlist_in: schemas.MentorWishlistIn = Body(),
) -> ResponseModel:
    """提交导师愿望单"""
    data = wishlist_in.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    wishlist = await models.MentorWishlist.create(user_id=credential.user_id, **data)
    request.state.log_data["wishlist.id"] = str(wishlist.id)

    return ResponseModel(data=schemas.MentorWishlist.model_validate(wishlist))


@router.get("/wishlists/my")
async def get_my_wishlists(
    request: Request,
    credential: Credential = Depends(authenticator),
) -> ResponseModel:
    """获取当前用户提交的所有愿望单"""
    wishlists = await models.MentorWishlist.filter(user_id=credential.user_id).order_by("-created_at").all()
    return ResponseModel(data=[schemas.MentorWishlist.model_validate(w) for w in wishlists])


@router.get("/wishlists")
async def list_wishlists(
    request: Request,
    credential: Credential = Depends(authenticator),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
) -> ResponseModel:
    """获取所有愿望单（管理员）"""
    await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.mentor_manage])

    total = await models.MentorWishlist.all().count()
    wishlists = await models.MentorWishlist.all().order_by("-created_at").offset((page - 1) * page_size).limit(page_size)

    return ResponseModel(data={"total": total, "page": page, "page_size": page_size, "items": [schemas.MentorWishlist.model_validate(w) for w in wishlists]})


@router.delete("/wishlists/{wishlist_id}")
async def delete_wishlist(
    request: Request,
    credential: Credential = Depends(authenticator),
    wishlist_id: uuid.UUID = Path(),
) -> ResponseModel:
    """删除愿望单（本人或管理员）"""
    wishlist = await models.MentorWishlist.filter(id=wishlist_id).first()
    if not wishlist:
        raise HTTPException(Errors.NOT_FOUND.with_message("愿望单不存在"))

    if wishlist.user_id != credential.user_id:
        await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.mentor_manage])

    await wishlist.delete()
    request.state.log_data["wishlist.id"] = str(wishlist_id)

    return ResponseModel()


# ---------------------------------------------------------------------------
# 导师列表、详情、更新、删除（参数路由放在固定路由之后）
# ---------------------------------------------------------------------------


@router.get("/actions/my")
async def get_my_mentor_actions(
    request: Request,
    credential: Credential = Depends(authenticator),
    mentor_ids: list[uuid.UUID] | None = Query(None, description="导师ID列表，不传则返回当前用户对所有导师的互动状态"),
) -> ResponseModel:
    """获取当前用户对导师的互动状态，返回以 mentor_id 为 key 的字典：{liked: bool, joined: bool}"""
    filters: dict = {"user_id": credential.user_id}
    if mentor_ids is not None:
        filters["mentor_id__in"] = [str(mid) for mid in mentor_ids]

    records = await models.MentorUserAction.filter(**filters).values("mentor_id", "action")

    result: dict[str, dict] = {}
    for row in records:
        mid = row["mentor_id"]
        if mid not in result:
            result[mid] = {"liked": False, "joined": False}
        if row["action"] == constants.MentorAction.LIKE:
            result[mid]["liked"] = True
        elif row["action"] == constants.MentorAction.JOIN:
            result[mid]["joined"] = True

    return ResponseModel(data=result)


@router.get("")
async def list_mentors(
    request: Request,
    credential: Credential = Depends(authenticator | anonymous_authenticator),
    status: int | None = Query(None, description="状态过滤：1=展示, 0=隐藏"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
) -> ResponseModel:
    """获取导师列表"""
    filters: dict = {}
    if status is not None:
        filters["status"] = status

    total = await models.Mentor.filter(**filters).count()
    mentors = await models.Mentor.filter(**filters).order_by("-created_at").offset((page - 1) * page_size).limit(page_size)

    mentor_ids = [str(m.id) for m in mentors]

    # 批量查关注数
    like_counts: dict[str, int] = {}
    join_counts: dict[str, int] = {}
    if mentor_ids:
        for row in await models.MentorUserAction.filter(mentor_id__in=mentor_ids, action=constants.MentorAction.LIKE).values("mentor_id"):
            like_counts[row["mentor_id"]] = like_counts.get(row["mentor_id"], 0) + 1
        for row in await models.MentorUserAction.filter(mentor_id__in=mentor_ids, action=constants.MentorAction.JOIN).values("mentor_id"):
            join_counts[row["mentor_id"]] = join_counts.get(row["mentor_id"], 0) + 1

    # 批量查当前用户的互动状态
    user_liked: set[str] = set()
    user_joined: set[str] = set()
    if mentor_ids and credential and credential.user:
        for row in await models.MentorUserAction.filter(user_id=credential.user_id, mentor_id__in=mentor_ids).values("mentor_id", "action"):
            if row["action"] == constants.MentorAction.LIKE:
                user_liked.add(row["mentor_id"])
            elif row["action"] == constants.MentorAction.JOIN:
                user_joined.add(row["mentor_id"])

    items = []
    for m in mentors:
        mid = str(m.id)
        item = schemas.Mentor.model_validate(m).model_dump()
        item["like_count"] = like_counts.get(mid, 0)
        item["join_count"] = join_counts.get(mid, 0)
        item["liked"] = mid in user_liked
        item["joined"] = mid in user_joined
        items.append(item)

    return ResponseModel(data={"total": total, "page": page, "page_size": page_size, "items": items})


@router.get("/{mentor_id}")
async def get_mentor(
    request: Request,
    credential: Credential = Depends(authenticator | anonymous_authenticator),
    mentor_id: uuid.UUID = Path(),
) -> ResponseModel:
    """获取导师详情"""
    mentor = await models.Mentor.filter(id=mentor_id).first()
    if not mentor:
        raise HTTPException(Errors.NOT_FOUND.with_message("导师不存在"))

    mentor_id_str = str(mentor_id)

    like_count = await models.MentorUserAction.filter(mentor_id=mentor_id_str, action=constants.MentorAction.LIKE).count()
    join_count = await models.MentorUserAction.filter(mentor_id=mentor_id_str, action=constants.MentorAction.JOIN).count()

    liked = False
    joined = False
    if credential and credential.user:
        user_actions = await models.MentorUserAction.filter(user_id=credential.user_id, mentor_id=mentor_id_str).values_list("action", flat=True)
        liked = constants.MentorAction.LIKE in user_actions
        joined = constants.MentorAction.JOIN in user_actions

    data = schemas.Mentor.model_validate(mentor).model_dump()
    data["like_count"] = like_count
    data["join_count"] = join_count
    data["liked"] = liked
    data["joined"] = joined

    return ResponseModel(data=data)


@router.post("/{mentor_id}")
async def update_mentor(
    request: Request,
    credential: Credential = Depends(authenticator),
    mentor_id: uuid.UUID = Path(),
    mentor_update: schemas.MentorUpdate = Body(),
) -> ResponseModel:
    """更新导师信息"""
    await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.mentor_manage])

    mentor = await models.Mentor.filter(id=mentor_id).first()
    if not mentor:
        raise HTTPException(Errors.NOT_FOUND.with_message("导师不存在"))

    data = mentor_update.model_dump(exclude_unset=True, exclude_none=True)
    request.state.log_data["data"] = data

    await mentor.update_from_dict(data).save()

    return ResponseModel(data=schemas.Mentor.model_validate(mentor))


@router.delete("/{mentor_id}")
async def delete_mentor(
    request: Request,
    credential: Credential = Depends(authenticator),
    mentor_id: uuid.UUID = Path(),
) -> ResponseModel:
    """删除导师"""
    await authorizer.requires(request, ANY_SPACE_ID, [constants.Privileges.mentor_manage])

    mentor = await models.Mentor.filter(id=mentor_id).first()
    if not mentor:
        raise HTTPException(Errors.NOT_FOUND.with_message("导师不存在"))

    await mentor.delete()
    request.state.log_data["mentor.id"] = str(mentor_id)

    return ResponseModel()


# ---------------------------------------------------------------------------
# 用户与导师互动（点赞 / 申请加入小组）
# ---------------------------------------------------------------------------


@router.post("/{mentor_id}/action")
async def mentor_action(
    request: Request,
    credential: Credential = Depends(authenticator),
    mentor_id: uuid.UUID = Path(),
    action: int = Body(embed=True, description="动作类型：1=关注/点赞, 2=申请加入小组"),
) -> ResponseModel:
    """对导师执行互动动作（幂等：重复操作返回已存在的记录）"""
    if action not in (constants.MentorAction.LIKE.value, constants.MentorAction.JOIN.value):
        raise HTTPException(Errors.BAD_REQUEST.with_message("无效的动作类型，支持：1=关注/点赞, 2=申请加入小组"))

    mentor = await models.Mentor.filter(id=mentor_id).first()
    if not mentor:
        raise HTTPException(Errors.NOT_FOUND.with_message("导师不存在"))

    mentor_id_str = str(mentor_id)

    try:
        record, created = await models.MentorUserAction.get_or_create(
            user_id=credential.user_id,
            mentor_id=mentor_id_str,
            action=action,
        )
    except IntegrityError:
        record = await models.MentorUserAction.filter(user_id=credential.user_id, mentor_id=mentor_id_str, action=action).first()
        if not record:
            raise HTTPException(Errors.INTERNAL_SERVER_ERROR.with_message("互动记录创建失败")) from None
        created = False

    request.state.log_data["created"] = created
    request.state.log_data["action_id"] = record.id

    return ResponseModel(data={"record": schemas.MentorUserAction.model_validate(record), "created": created})


@router.delete("/{mentor_id}/action")
async def cancel_mentor_action(
    request: Request,
    credential: Credential = Depends(authenticator),
    mentor_id: uuid.UUID = Path(),
    action: int = Query(description="要取消的动作类型：1=关注/点赞, 2=申请加入小组"),
) -> ResponseModel:
    """取消对导师的互动动作"""
    if action not in (constants.MentorAction.LIKE.value, constants.MentorAction.JOIN.value):
        raise HTTPException(Errors.BAD_REQUEST.with_message("无效的动作类型，支持：1=关注/点赞, 2=申请加入小组"))

    mentor_id_str = str(mentor_id)
    deleted_count = await models.MentorUserAction.filter(user_id=credential.user_id, mentor_id=mentor_id_str, action=action).delete()
    request.state.log_data["deleted_count"] = deleted_count

    return ResponseModel()
