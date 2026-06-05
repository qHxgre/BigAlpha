"""项目入口"""

from fastapi import APIRouter

from bigshared2.bigapi import BigAPIApp

from . import competitions, constants, settings, submissions

router = APIRouter()
router.include_router(competitions.router, prefix="/competitions", tags=["比赛"])
router.include_router(submissions.router, prefix="/submissions", tags=["提交"])

app = BigAPIApp(
    name="alphathon",
    api_router=router,
    tortoise_orm=settings.TORTOISE_ORM,
    privileges=constants.Privileges,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("alphathon.main:app", host="0.0.0.0", port=8000, reload=True)
