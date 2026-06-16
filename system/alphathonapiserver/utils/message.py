"""message api operations"""

import uuid
from datetime import datetime
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, Field

import settings

log = structlog.get_logger(__name__)


class WechatTemplateItem(BaseModel):
    """微信模板数据项"""

    value: str


class SignalWechatTemplateData(BaseModel):
    """微信模板数据"""

    keyword1: WechatTemplateItem = Field(alias="strategy_name")
    keyword2: WechatTemplateItem = Field(alias="execution_time")


class SignalWechatTemplateSchema(BaseModel):
    """微信模板推送"""

    template_id: str
    template_data: SignalWechatTemplateData
    jump_link: str


async def send_wechat_message(title: str, user_id: uuid.UUID):
    template_data = SignalWechatTemplateData(
        strategy_name=WechatTemplateItem(value=title),
        execution_time=WechatTemplateItem(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    signal_wechat = SignalWechatTemplateSchema(
        template_id=settings.WECHAT_TRADE_TEMPLATE_ID,
        template_data=template_data,
        jump_link=settings.WECHAT_MESSAGE_JUMP_LINK,
    )
    message = signal_wechat.model_dump()
    mid = str(uuid.uuid4())
    message_data = {
        "id": mid,
        "space_id": "00000000-0000-0000-0000-000000000000",
        "type": "wechat",
        "user_id": str(user_id),
        "priority": 20,
        "data": message,
        "meta": {"source": f"alphathonapiserver:{str(user_id)}"},
    }
    await _send_wechat_message(data=message_data)


async def _send_wechat_message(data: dict[str, Any]) -> Any:
    """调用 messageapiserver 接口发送微信消息

    Args:
        data (Dict[str, Any]): 待发送消息数据.

    Returns:
        Dict[str, Any]: 发送结果.
    """
    async with httpx.AsyncClient() as client:
        url = f"{settings.MESSAGE_HOST}/bigapis/message/v1/wechat"
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()
