"""Alphathon API 客户端 — 给 scripts/ 下的统计脚本复用。

通过 HTTP 调用运行中的 alphathonapiserver 查询数据，不直连数据库。
配置全部走环境变量，方便本地连远程：

    ALPHATHON_API_BASE_URL   API 根地址（默认指向集群内地址）
    ALPHATHON_API_TOKEN      bigjwt token，请求会带成 Cookie: bigjwt=<token>
    ALPHATHON_JWT_FILE       token 文件路径（当 ALPHATHON_API_TOKEN 未设置时读它）
    ALPHATHON_API_TIMEOUT    单次请求超时秒数（默认 30）

用法见同目录下的 participants.py。
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

DEFAULT_BASE_URL = "http://alphathonapiserver.bigquant.svc.cluster.local:8000/bigapis/alphathon/v1"
DEFAULT_JWT_FILE = "/home/aiuser/work/workspace/BigAlpha/system/files/cptjudge.jwt"


class AlphathonClient:
    """对 alphathonapiserver REST API 的薄封装，只做查询。"""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("ALPHATHON_API_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout if timeout is not None else float(os.getenv("ALPHATHON_API_TIMEOUT", "30"))
        self.token = token or self._load_token()
        self._session = requests.Session()

    @staticmethod
    def _load_token() -> str:
        token = os.getenv("ALPHATHON_API_TOKEN")
        if token:
            return token.strip()
        jwt_file = os.getenv("ALPHATHON_JWT_FILE", DEFAULT_JWT_FILE)
        try:
            with open(jwt_file, encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    def _request(self, method: str, path: str, *, params: dict | None = None, json_data: Any = None) -> Any:
        headers = {"accept": "application/json"}
        if self.token:
            headers["Cookie"] = f"bigjwt={self.token}"
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self._session.request(method.upper(), url, params=params, json=json_data, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---- 分页拉全量 ----------------------------------------------------

    def _paginate(self, path: str, *, params: dict, page_size: int = 1000, max_pages: int = 100000) -> list[dict[str, Any]]:
        """把一个返回 {data: {items, total, ...}} 的列表接口翻完，返回全部 items。"""
        results: list[dict[str, Any]] = []
        page = 1
        params = dict(params)
        while page <= max_pages:
            params.update({"page": page, "size": page_size})
            data = self._request("GET", path, params=params).get("data") or {}
            items = data.get("items") or []
            if not items:
                break
            results.extend(items)
            if len(items) < page_size:
                break
            page += 1
        return results

    # ---- 业务查询 ------------------------------------------------------

    def get_competition(self, competition_id: str) -> dict[str, Any] | None:
        params = {"constraints": json.dumps({"id": str(competition_id)})}
        items = self._paginate("/competitions", params=params, page_size=1)
        return items[0] if items else None

    def list_users(
        self,
        competition_id: str,
        *,
        constraints: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """获取某场比赛的参赛者（报名记录）列表。

        需要有 competition_manage 权限或为该比赛创建者，否则返回的字段会被隐私保护裁剪。
        """
        params: dict[str, Any] = {"competition_id": str(competition_id)}
        if constraints:
            params["constraints"] = json.dumps(constraints)
        if order_by:
            params["order_by"] = order_by
        return self._paginate("/users", params=params)

    def list_teams(self, competition_id: str, *, order_by: list[str] | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"competition_id": str(competition_id)}
        if order_by:
            params["order_by"] = order_by
        return self._paginate("/teams", params=params)

    def list_submissions(
        self,
        competition_id: str,
        *,
        constraints: dict[str, Any] | None = None,
        order_by: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"competition_id": str(competition_id)}
        if constraints:
            params["constraints"] = json.dumps(constraints)
        if order_by:
            params["order_by"] = order_by
        return self._paginate("/submissions", params=params)

    # ---- 业务写入 ------------------------------------------------------

    def update_user_status(self, user_id: str, status: str) -> dict[str, Any]:
        """更新某条报名记录的状态（审批）。

        对应 POST /users/{user_id}。需要比赛创建者或 competition_manage 权限。
        status 取值见 alphathonapiserver.constants.UserStatus：
        pending / approved / approved_join_space / rejected。
        """
        return self._request("POST", f"/users/{user_id}", json_data={"status": status})
