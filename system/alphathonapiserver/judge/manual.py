import structlog

from .judgebase import AlphathonAPI


logger = structlog.get_logger()

alphathon_api = AlphathonAPI()

alphathon_api.update_submission_score(
    submission_id="48be5669-7b2d-41a5-9de0-fd1592851d9c",
    public_score=-2.,
    public_score_data={"err_msg": "使用LEAD函数, 存在未来信息"},
)


