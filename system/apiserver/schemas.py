"""schemas"""

from typing import Any

from pydantic import BaseModel, Field
from tortoise.contrib.pydantic.creator import pydantic_model_creator

from . import models

# Competition schemas（评测系统只读）
Competition = pydantic_model_creator(models.Competition, name="Competition")

# Submission schemas
Submission = pydantic_model_creator(models.Submission, name="Submission")


class SubmissionScoreUpdate(BaseModel):
    """评测系统回写分数的输入。仅允许 4 个分数字段。"""

    public_score: float | None = Field(default=None, description="公榜分数")
    public_score_data: dict[str, Any] | None = Field(default=None, description="公榜分数详细数据")
    private_score: float | None = Field(default=None, description="私榜分数")
    private_score_data: dict[str, Any] | None = Field(default=None, description="私榜分数详细数据")
