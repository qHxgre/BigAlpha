"""数据模型"""

from tortoise import fields
from tortoise.models import Model

from bigshared2.db.sql.models import (
    CreatedAtMixin,
    CreatorAndIndexMixin,
    SpaceIDAndIndexMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)


class Competition(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, SpaceIDAndIndexMixin, CreatorAndIndexMixin, Model):
    name = fields.CharField(max_length=128, description="比赛名称")
    description = fields.TextField(null=True, description="比赛简介")
    start_date = fields.DatetimeField(null=True, description="开始日期")
    end_date = fields.DatetimeField(null=True, description="结束日期")
    organizer = fields.CharField(max_length=255, null=True, description="主办方名称")
    prize = fields.IntField(null=True, description="奖金")
    rank = fields.IntField(default=1, description="列表页排序，0表示不在列表页出现")
    summary = fields.JSONField(default=dict, description="比赛列表摘要字段")
    data = fields.JSONField(default=dict, description="比赛详情扩展字段")

    class Meta:
        table = "alphathon__competition"


class Submission(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Model):
    competition_id = fields.UUIDField(description="比赛ID", index=True)
    user_id = fields.UUIDField(description="用户ID", index=True)
    data = fields.JSONField(default=dict, description="提交的数据内容")
    public_score = fields.DecimalField(max_digits=10, decimal_places=5, null=True, description="公榜分数")
    public_score_data = fields.JSONField(default=dict, description="公榜分数详细数据")
    private_score = fields.DecimalField(max_digits=10, decimal_places=5, null=True, description="私榜分数")
    private_score_data = fields.JSONField(default=dict, description="私榜分数详细数据")
    selected_for_private = fields.BooleanField(default=False, description="是否被选中用于私榜评分")

    class Meta:
        table = "alphathon__submission"
