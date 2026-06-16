"""所有常量"""

from enum import Enum

from bigshared2.auth import Privilege, PrivilegeSet, Roles


class UserStatus(str, Enum):
    """用户状态枚举"""

    PENDING = "pending"  # 待定/待审核
    APPROVED = "approved"  # 通过审核
    REJECTED = "rejected"  # 拒绝


# BigAuth 定义的 Privileges
class Privileges(PrivilegeSet):
    # 优惠券模版权限
    competition_manage = Privilege("/alphathon/competition/manage", [Roles.super_admin, Roles.operation_manager], "管理比赛")
