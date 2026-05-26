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

# 消息通知服务
MESSAGE_HOST = Env.string("MESSAGE_HOST", "http://messageapiserver:8000")

WECHAT_TRADE_TEMPLATE_ID = Env.string("WECHAT_TRADE_TEMPLATE_ID", "8TLWGGMsDfdxVEJPqRjbSkN-oqH3PR13zXBIoIxnNl0")
WECHAT_MESSAGE_JUMP_LINK = Env.string("WECHAT_MESSAGE_JUMP_LINK", "https://bigquant.com/aiuser/account/notifications")
# 系统消息通知服务
NOTIFICATION_HOST = Env.string("NOTIFICATION_HOST", "http://notificationapiserver:8000")

Env.auto_load()
