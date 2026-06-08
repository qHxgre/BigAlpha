import json
import os
import uuid
from typing import Any, Dict, List, Optional

import httpx

from utils import write_file
from paths import ALPHATHON_API_BASE_URL, JWT_FILE


class AlphathonAPI:
    def __init__(self) -> None:
        self.base_url = os.getenv("ALPHATHON_API_BASE_URL", ALPHATHON_API_BASE_URL)
        self.timeout: float = float(os.getenv("ALPHATHON_API_TIMEOUT", 15.0))
        self.api_token: str = os.getenv("ALPHATHON_API_TOKEN") or open(JWT_FILE).read().strip()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_data: Any = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        request_headers: Dict[str, str] = {"accept": "application/json"}
        if self.api_token:
            request_headers["Cookie"] = f"bigjwt={self.api_token}"
        if headers:
            request_headers.update(headers)

        url = f"{self.base_url}/{path.lstrip('/')}"
        with httpx.Client(timeout=timeout or self.timeout) as client:
            response = client.request(method.upper(), url, params=params, json=json_data, headers=request_headers)
            response.raise_for_status()
            return response

    def get_competition_by_id(self, competition_id: str | uuid.UUID) -> Optional[Dict[str, Any]]:
        params = {
            "constraints": json.dumps({"id": str(competition_id)}),
            "page": 1,
            "size": 1,
        }
        data = self._request("GET", "/competitions", params=params).json()
        items = (data or {}).get("data", {}).get("items", []) if isinstance(data, dict) else []
        return items[0] if items else None

    def query_submissions(
        self,
        *,
        competition_id: str | uuid.UUID | None = None,
        constraints: Optional[Dict[str, Any]] = None,
        page_size: int = 5000,
        max_pages: int = 10000,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        page = 1
        constraints = constraints or {}

        while page <= max_pages:
            params: Dict[str, Any] = {
                "competition_id": str(competition_id),
                "page": page,
                "size": page_size,
                "order_by": "-created_at",
                "constraints": json.dumps(constraints),
            }
            data = self._request("GET", "/submissions", params=params).json()["data"]
            if not data or not data.get("items"):
                break
            items = data["items"]
            results.extend(items)
            if len(items) < page_size:
                break
            page += 1

        return results

    def get_file_content_of_submission(self, submission: dict, ipynb_to_py: bool = False, to_str: bool = False, save_to: Optional[str] = None) -> bytes | str:
        if len(submission["data"]["files"]) != 1:
            raise Exception(f"submission {submission['id']} has {len(submission['data']['files'])} files, while only 1 is expected")

        file_id, file_info = list(submission["data"]["files"].items())[0]
        return self.get_submission_file(submission["id"], file_id, file_info, ipynb_to_py=ipynb_to_py, to_str=to_str, save_to=save_to)

    def get_submission_file(
        self,
        submission_id: str | uuid.UUID,
        file_id: str,
        file_info: Optional[dict] = None,
        ipynb_to_py: bool = False,
        to_str: bool = False,
        save_to: Optional[str] = None,
    ) -> bytes | str:
        path = f"/submissions/files/{submission_id}/{file_id}"
        response = self._request("GET", path)
        raw_content: bytes | str = response.content

        if file_info is not None and file_info["name"].endswith(".ipynb") and ipynb_to_py:
            notebook_data = json.loads(raw_content.decode("utf-8"))
            code_cells: List[str] = []
            for cell in notebook_data.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", [])
                if isinstance(source, list):
                    code_cells.append("".join(source))
                elif isinstance(source, str):
                    code_cells.append(source)
            raw_content = "\n\n".join(code_cells)

        if to_str and isinstance(raw_content, bytes):
            raw_content = raw_content.decode("utf8")

        if save_to:
            write_file(save_to, raw_content)

        return raw_content

    def update_submission_score(self, submission_id: str | uuid.UUID, **json_data) -> Dict[str, Any]:
        path = f"/submissions/{submission_id}"
        response = self._request("POST", path, json_data=json_data)
        return response.json()
