"""notification api operations"""

import httpx

from ..settings import NOTIFICATION_HOST


async def create_notice(user_id: str, space_id: str, title: str, content: str, channel: str):
    """创建消息通知"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{NOTIFICATION_HOST}/bigapis/notify/v1/notice",
            json={"notice": {"title": title, "content": content, "channel": channel, "recipient_id": user_id, "notice_type": "signal"}, "space_id": space_id},
            timeout=5,
        )
        response.raise_for_status()
        return response.json()["data"]
