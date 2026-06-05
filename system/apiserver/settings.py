"""项目配置"""

from bigshared2.db.sql import settings as db_settings
from bigshared2.utils.env import Env

TORTOISE_ORM: dict = {
    **db_settings.BASE_TORTOISE_ORM,
    "apps": {
        "alphathonapiserver": {
            "models": ["alphathonapiserver.models", "aerich.models"],
        },
    },
}

FILE_UPLOAD_PATH = Env.string("FILE_UPLOAD_PATH", "/var/app/data/uploads")

Env.auto_load()
