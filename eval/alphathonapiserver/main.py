"""项目入口"""

from bigshared2.bigapi import BigAPIApp

from . import api, constants, settings

app = BigAPIApp(
    name="alphathon",
    api_router=api.router,
    tortoise_orm=settings.TORTOISE_ORM,
    privileges=constants.Privileges,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("alphathon.main:app", host="0.0.0.0", port=8000, reload=True)
