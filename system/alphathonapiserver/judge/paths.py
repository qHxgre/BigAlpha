import os
from pathlib import Path

# judge 目录（存放 judgebase.py），用于让 competitions/ 下的入口脚本能 import judgebase
JUDGE_DIR: str = str(Path(__file__).resolve().parent)

# 文件根目录
RUNNER_BASE_DIR: str = str(Path(__file__).resolve().parents[2] / "files")

# JWT 文件路径，AlphathonAPI 用它给请求带上 cookie。
JWT_FILE: str = os.path.join(RUNNER_BASE_DIR, "cptjudge.jwt")

# 排行榜 csv / 已完成 id 持久化目录
LEADERBOARD_DIR: str = os.path.join(RUNNER_BASE_DIR, "leaderboard")
COMPLETE_IDS_DIR: str = os.path.join(RUNNER_BASE_DIR, "complete_ids")

# alphathon api
ALPHATHON_API_BASE_URL: str = "http://alphathonapiserver.bigquant.svc.cluster.local:8000/bigapis/alphathon/v1"