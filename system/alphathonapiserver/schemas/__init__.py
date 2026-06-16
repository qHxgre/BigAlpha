"""schemas"""

from typing import Any, TypeVar

from pydantic import BaseModel, Field, validator
from tortoise.contrib.pydantic.creator import pydantic_model_creator

import models

# Competition schemas
Competition = pydantic_model_creator(models.Competition, name="Competition")
CompetitionIn = pydantic_model_creator(models.Competition, name="CompetitionIn", exclude=("id", "created_at", "updated_at", "creator"))
CompetitionUpdate = pydantic_model_creator(
    models.Competition,
    name="CompetitionUpdate",
    include=("name", "description", "start_date", "end_date", "organizer", "prize", "rank", "summary", "data"),
    optional=("name", "description", "start_date", "end_date", "organizer", "prize", "rank", "summary", "data"),
)

# User schemas
User = pydantic_model_creator(models.User, name="User")
UserIn = pydantic_model_creator(models.User, name="UserIn", exclude=("id", "user_id", "created_at", "updated_at", "status"))


# UserUpdate = pydantic_model_creator(models.User, name="UserUpdate", include=("status", "data"), optional=("status", "data"))
class UserUpdate(BaseModel):
    """UserUpdate schema"""

    status: str | None = Field(default=None, description="用户状态")
    data: dict[str, Any] | None = Field(default=None, description="用户数据")
    reject_reason: str | None = Field(default=None, description="拒绝理由")


# Submission schemas
Submission = pydantic_model_creator(models.Submission, name="Submission")
SubmissionIn = pydantic_model_creator(models.Submission, name="SubmissionIn", exclude=("id", "user_id", "created_at", "updated_at"))
SubmissionUpdate = pydantic_model_creator(
    models.Submission,
    name="SubmissionUpdate",
    include=("data", "public_score", "public_score_data", "private_score", "private_score_data", "selected_for_private"),
    optional=("data", "public_score", "public_score_data", "private_score", "private_score_data", "selected_for_private"),
)

# Team schemas
Team = pydantic_model_creator(models.Team, name="Team")
TeamIn = pydantic_model_creator(models.Team, name="TeamIn", exclude=("id", "created_at", "updated_at", "creator", "members", "pending_users"))
TeamUpdate = pydantic_model_creator(models.Team, name="TeamUpdate", include=("name",))

# Code schemas
Code = pydantic_model_creator(models.Code, name="Code")
CodeIn = pydantic_model_creator(models.Code, name="CodeIn", exclude=("id", "created_at", "updated_at", "creator", "like_count", "rank"))


class CodeUpdate(BaseModel):
    """CodeUpdate schema"""

    data: dict[str, Any] | None = Field(default=None, description="代码内容")
    like: int | None = Field(default=0, description="点赞，-1是取消点赞，0表示不做任何操作，1表示点赞")
    top: int | None = Field(default=0, description="是否置顶，-1是取消置顶，0表示做任何操作，1表示置顶")

    @validator("like")
    @classmethod
    def validate_like(cls, v):
        """验证like"""
        if v not in [-1, 0, 1]:
            raise ValueError("like must be -1, 0, or 1")
        return v

    @validator("top")
    @classmethod
    def validate_top(cls, v):
        """验证top"""
        if v not in [-1, 0, 1]:
            raise ValueError("top must be -1, 0, or 1")
        return v


T = TypeVar("T")


def schema_to_dict(schema: T, items: list[T], include_fields=None, exclude_fields=None) -> list[dict[str, Any]]:
    """Convert pydantic model to dict"""
    if not issubclass(schema, BaseModel):
        raise ValueError("items must be a list of pydantic models")
    if include_fields:
        return [item.dict(include=include_fields) for item in items]
    if exclude_fields:
        return [item.dict(exclude=exclude_fields) for item in items]
    return [item.dict() for item in items]
