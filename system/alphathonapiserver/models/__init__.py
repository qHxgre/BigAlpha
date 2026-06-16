"""数据模型"""

from tortoise import fields, models
from tortoise.models import Model

from bigshared2.db.sql.models import (
    CreatedAtMixin,
    CreatorAndIndexMixin,
    SpaceIDAndIndexMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)

from constants import UserStatus


class Competition(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, SpaceIDAndIndexMixin, CreatorAndIndexMixin, Model):
    name = fields.CharField(max_length=128, description="比赛名称")
    description = fields.TextField(null=True, description="比赛简介")
    start_date = fields.DatetimeField(null=True, description="开始日期")
    end_date = fields.DatetimeField(null=True, description="结束日期")
    organizer = fields.CharField(max_length=255, null=True, description="主办方名称")
    prize = fields.IntField(null=True, description="奖金")
    rank = fields.IntField(default=1, description="列表页排序，0表示不在列表页出现")
    summary = fields.JSONField(
        default=dict, description="Extended summary content for list display (e.g. registration_deadline, team_building_deadline, submission_deadline, rewards)"
    )
    data = fields.JSONField(default=dict, description="Extended data content for detail display (e.g. detailed_description)")

    class Meta:
        table = "alphathon__competition"


class User(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Model):
    competition_id = fields.UUIDField(description="比赛ID", index=True)
    user_id = fields.UUIDField(description="用户ID", index=True)
    status = fields.CharEnumField(UserStatus, default=UserStatus.PENDING, description="用户状态")
    data = fields.JSONField(default=dict, description="Extended data content for user details")

    class Meta:
        table = "alphathon__user"
        unique_together = [("competition_id", "user_id")]


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


# TODO: 在 migration 中为 members 和 pending_users 手动创建多值索引
#   ALTER TABLE team ADD INDEX idx_users_mv ((CAST(members AS UNSIGNED ARRAY)));
#   teams = await Team.filter(members__contains=user_id).all()
class Team(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, CreatorAndIndexMixin, models.Model):
    competition_id = fields.UUIDField(description="比赛ID", index=True)
    name = fields.CharField(max_length=255, description="团队名称")
    members = fields.JSONField(default=list, description="团队成员用户ID列表")
    pending_users = fields.JSONField(default=list, description="请求加入的用户ID列表")

    class Meta:
        table = "alphathon__team"


class Code(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, CreatorAndIndexMixin, Model):
    """上传代码"""

    competition_id = fields.UUIDField(description="比赛ID", index=True)
    like_count = fields.IntField(default=0, description="点赞数")
    rank = fields.IntField(default=1, description="列表页排序，置顶的逻辑当前(最大rank+1)达到置顶效果")
    data = fields.JSONField(default={}, description="代码内容")

    class Meta:
        table = "alphathon__code"
