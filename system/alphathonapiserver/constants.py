"""所有常量"""

from bigshared2.auth import Privilege, PrivilegeSet, Roles


class Privileges(PrivilegeSet):
    competition_manage = Privilege(
        "/alphathon/competition/manage",
        [Roles.competition_admin, Roles.super_admin, Roles.operation_manager],
        "管理比赛",
    )
