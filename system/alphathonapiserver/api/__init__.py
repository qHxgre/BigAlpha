"""接口路由"""

from fastapi import APIRouter

from api import code, competitions, leaderboard, submissions, teams, users

router = APIRouter()

router.include_router(competitions.router, prefix="/competitions", tags=["比赛"])
router.include_router(users.router, prefix="/users", tags=["用户"])
router.include_router(submissions.router, prefix="/submissions", tags=["提交"])
router.include_router(teams.router, prefix="/teams", tags=["团队"])
router.include_router(leaderboard.router, prefix="/leaderboard", tags=["排行榜"])
router.include_router(code.router, prefix="/code", tags=["代码"])
