import os
from pathlib import Path

# 文件根目录
FILE_DIR: str = '/Users/xiehao/Desktop/workspace/BigQuant/BigAlpha/system/files'


# JWT 文件路径，AlphathonAPI 用它给请求带上 cookie。
JWT_FILE: str = os.path.join(FILE_DIR, "cptjudge.jwt")

# alphathon api
ALPHATHON_API_BASE_URL: str = "http://alphathonapiserver.bigquant.svc.cluster.local:8000/bigapis/alphathon/v1"