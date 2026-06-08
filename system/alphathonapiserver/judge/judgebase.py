import json
import os
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, Optional

import httpx
import structlog

logger = structlog.get_logger()

RUNNER_BASE_DIR = os.getenv("RUNNER_BASE_DIR", "/home/aiuser/work/data/alphathon")


def _write_file(path, content):
    mode = "w"
    if isinstance(content, bytes):
        mode = "wb"
    with open(path, mode=mode) as writer:
        writer.write(content)


class AlphathonAPI:
    def __init__(self) -> None:
        self.base_url = os.getenv("ALPHATHON_API_BASE_URL", "http://alphathonapiserver.bigquant.svc.cluster.local:8000/bigapis/alphathon/v1")
        self.timeout: float = float(os.getenv("ALPHATHON_API_TIMEOUT", 15.0))
        self.api_token: str = os.getenv("ALPHATHON_API_TOKEN")
        self.api_token = open(os.path.join(RUNNER_BASE_DIR, "cptjudge.jwt")).read().strip()

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
        request_headers: Dict[str, str] = {
            "accept": "application/json",
        }
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
        """获取单个比赛信息。未找到则返回 None。"""
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
        constraints: Dict[str, Any] = {},
        page_size: int = 5000,
        max_pages: int = 10000,
    ) -> list[Dict[str, Any]]:
        results: list[Dict[str, Any]] = []
        page = 1

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

    def get_file_content_of_submission(self, submission: dict, ipynb_to_py=False, to_str=False, save_to: Optional[str]=None) -> bytes:
        if len(submission["data"]["files"]) != 1:
            raise Exception(f"submission {submission['id']} has {len(submission['files'])} files, while only 1 is expected")

        file_id, file_info = list(submission["data"]["files"].items())[0]
        return self.get_submission_file(submission['id'], file_id, file_info, ipynb_to_py=True)

    def get_submission_file(
        self,
        submission_id: str | uuid.UUID,
        file_id: str,
        file_info: Optional[dict]=None,
        ipynb_to_py=False,
        to_str=False,
        save_to: Optional[str]=None
    ) -> bytes | str:
        path = f"/submissions/files/{submission_id}/{file_id}"
        response = self._request("GET", path)
        raw_content = response.content

        if file_info is not None and file_info["name"].endswith(".ipynb") and ipynb_to_py:
            notebook_data = json.loads(raw_content.decode("utf-8"))
            code_cells = []
            for cell in notebook_data.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", [])
                    if isinstance(source, list):
                        code_cells.append("".join(source))
                    elif isinstance(source, str):
                        code_cells.append(source)
            raw_content = "\n\n".join(code_cells)

        if to_str and isinstance(raw_content, bytes):
            raw_content = raw_content.decode("utf8")

        if save_to:
            _write_file(save_to, raw_content)

        return raw_content

    def update_submission_score(self, submission_id: str | uuid.UUID, **json_data) -> Dict[str, Any]:
        """更新提交的 public_score 和 public_score_data"""
        path = f"/submissions/{submission_id}"

        # json_data = {}

        # if public_score is not None:
        #     json_data["public_score"] = public_score
        
        # if public_score_data is not None:
        #     json_data["public_score_data"] = public_score_data

        response = self._request("POST", path, json_data=json_data)
        return response.json()


class UserCodeRunner:
    def __init__(self, submission_id, files: dict, cmd: list) -> None:
        self.submission_id = submission_id
        self.runner_dir = os.path.join(RUNNER_BASE_DIR, str(self.submission_id))
        self.files = files
        self.cmd = cmd

    def _pre_run(self):
        runner_dir = os.path.join(RUNNER_BASE_DIR, str(self.submission_id))
        os.makedirs(runner_dir, exist_ok=True)

        for name, content in self.files.items():
            mode = "w"
            if isinstance(content, bytes):
                mode = "wb"
            with open(os.path.join(runner_dir, name), mode=mode) as writer:
                writer.write(content)

    def run(self, _raise=False):
        try:
            self._pre_run()
            self._run_code()
            return True
        except Exception as e:
            logger.exception(e)
            if _raise:
                raise e from e
        return False

    def _run_code(self):
        pass


class LocalProcessUserRunner(UserCodeRunner):
    def _run_code(self):
        process = subprocess.Popen(
            # ["python3", "judge_runner.py"],
            # ["python3", "-c", "print(123)"],
            self.cmd,
            cwd=self.runner_dir,
            # TODO
            # timeout=3*60*60,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        # assert process is not None
        timeout = 3 * 60 * 60
        start = time.time()
        # 打开日志文件用于写入
        with open(f"{self.runner_dir}/stdout", 'w') as writer:
            while process.poll() is None:
                if time.time() - start > timeout:
                    print(f"任务超时，提交id: {self.submission_id}，运行时长：{time.time() - start}")
                    process.kill()
                    break
                # print(f"任务状态检查，提交id：{self.submission_id}")
                time.sleep(5)

            # 读取剩余输出
            for line in process.stdout:
                sys.stdout.write(line)
                writer.write(line)

        # 等待子进程结束，并获取返回码
        process.wait()
        return_code = process.returncode
        # print(f"\n命令执行完毕，返回码: {return_code}")
        return return_code


class K8SPodUserRunner(UserCodeRunner):
    pass


class JudgeBase:
    def __init__(self, competition_id, tick_interval) -> None:
        self.competition_id = competition_id
        self.tick_interval = tick_interval
        self.alphathon_api = AlphathonAPI()

        logger.info(f"Judge: {self.competition_id}, tick_interval: {self.tick_interval}")

    def on_tick(self) -> None:
        try:
            assert self.competition_id
            self.log.info("fetch_competition.start")
            competition = self.client.get_competition_by_id(self.competition_id)
            if competition is None:
                self.log.warning("competition.not_found")
                return
            comp_id = competition.get("id")
            comp_name = competition.get("name")
            self.log.info("competition.fetched", id=comp_id, name=comp_name)

            # 获取该比赛的所有提交
            self.log.info("submissions.fetch_start", competition_id=str(comp_id))
            submissions = self.client.get_all_submissions(competition_id=str(comp_id))
            self.log.info("submissions.fetched", count=len(submissions))

            # 如果有提交，获取第一个提交的文件
            if submissions:
                for submission in submissions:
                    if submission.get("public_score_data", None):
                        print("------ already scored, skip", submission["id"])
                        continue

                    submission_id = submission.get("id")
                    if not submission_id:
                        self.log.warning("submission.missing_id")
                        return
                        
                    submission_files = submission.get("data", {}).get("files", {})

                    self.log.info("submission.files", submission_id=submission_id, file_count=len(submission_files))
                    
                    # 获取每个文件
                    for file_id in submission_files:
                        try:
                            file_content = self.client.get_submission_file(submission_id, file_id)
                            file_info = submission_files[file_id]
                            self.log.info("submission.file_retrieved", 
                                        file_id=file_id, 
                                        filename=file_info.get("name"),
                                        size=len(file_content))
                            
                            # 将文件内容写入 user_code.py 并获取评分结果
                            score, score_data = self._write_user_code(file_content, file_info)

                            # 如果评分成功，更新 submission 的 public_score 和 score_data
                            if score is not None and score_data is not None:
                                try:
                                    self.client.update_submission_score(
                                        submission_id=submission_id,
                                        public_score=score,
                                        public_score_data=score_data
                                    )
                                    self.log.info("submission.score_updated", 
                                                submission_id=submission_id, 
                                                score=score)
                                except Exception as update_exc:
                                    self.log.exception("submission.score_update_error", 
                                                     submission_id=submission_id, 
                                                     error=str(update_exc))
                            
                        except Exception as exc:
                            self.log.exception("submission.file_error", file_id=file_id, error=str(exc))
        except httpx.HTTPError as exc:
            self.log.exception("competition.fetch_http_error", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            self.log.exception("competition.fetch_unexpected_error", error=str(exc))

    def private_score(self) -> None:
        # TODO
        pass

    def run(self) -> None:
        import concurrent.futures
        submitted_ids: set[str] = set()
        futures_by_id: dict[str, concurrent.futures.Future] = {}
        completed_total = 0
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        complete_ids_file = '/home/aiuser/work/data/alphathon/bf1b4468-6b4d-43dc-98e1-8c2358c61793-final-ids-20260324.json'
        try:
            with open(complete_ids_file, 'r', encoding='utf-8') as f:
                complete_ids = json.load(f)
        except:
            complete_ids = []
        logger.info(f"find complete_ids, total= {len(complete_ids)} ..")

        try:
            while True:
                logger.info("tick ..")
                added = 0
                if hasattr(self, "on_submission"):
                    new_submissions = self.alphathon_api.query_submissions(
                        competition_id=self.competition_id,
                        # constraints={"public_score": None},
                        constraints={"id__in": [
                            '5672095b-7c4a-4802-80ac-340821649d47',
                            '4c734b98-4f62-4c12-b26b-214c1bb94358',
                        ]},
                    )
                    pending = [s for s in new_submissions if s.get("id") not in submitted_ids]
                    for submission in pending:
                        sid = submission.get("id")
                        if sid is None:
                            continue
                        if sid in complete_ids:
                            continue
                        submitted_ids.add(sid)
                        fut = executor.submit(self.on_submission, submission)
                        futures_by_id[str(sid)] = fut
                        added += 1
                    logger.info("check complete, total={total}, complete={complete}, left={left}".format(
                        total=len(pending),
                        complete=len(complete_ids),
                        left=len(list(futures_by_id.keys()))
                    ))
                if futures_by_id:
                    done_ids = [sid for sid, f in futures_by_id.items() if f.done()]
                    if done_ids:
                        completed_total += len(done_ids)
                        for sid in done_ids:
                            futures_by_id.pop(sid, None)
                    complete_ids = list(set(complete_ids + done_ids))
                running = sum(1 for f in futures_by_id.values() if not f.done())
                logger.info("judge.status", added=added, running=running, completed_total=completed_total, total_tracked=len(futures_by_id))
                if hasattr(self, "on_tick"):
                    self.on_tick()
                    
                with open(complete_ids_file, 'w', encoding='utf-8') as f:
                    json.dump(complete_ids, f, ensure_ascii=False, indent=4)
                logger.info(f"save complete_ids, total= {len(complete_ids)} ..")


                logger.info(f"sleep {self.tick_interval}s ..")
                time.sleep(self.tick_interval)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
