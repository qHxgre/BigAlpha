"""本比赛评测用到的常量与自定义异常。"""

# ---- 产物文件名（落在每个提交目录 / 榜单目录下）----------------------------
RAW_FACTOR_FILE = "raw_factor.parquet"
PROCESS_FACTOR_FILE = "process_factor.parquet"
FACTOR_ANALYZE_FILE = "factor_analyze.json"
# 每个提交单因子分析的运行状态记录文件：无论成功/失败都会落盘一份，
# is_done() 据此判断该提交是否已跑过，避免进程重启后重复执行（尤其是注定失败的提交）。
SFA_STATUS_FILE = "sfa_status.json"

# ---- 运行状态 --------------------------------------------------------------
# 失败再细分四类，便于排查与决定是否重试：
STATUS_SUCCESS = "success"          # 跑通
STATUS_USER_ERROR = "user_error"    # 用户代码本身报错（子进程非 0 退出）
STATUS_TIMEOUT = "timeout"          # 运行超时（被 judge kill）
STATUS_FILE_ERROR = "file_error"    # 用户提交的文件本身有问题（缺失/数量不对/无法解析）
STATUS_ENV_ERROR = "env_error"      # 评测环境自身问题（拉取/落盘/注入失败等）

# 这些终态视为「已完成、不再重跑」：成功是真完成，用户报错/超时/文件错误再跑也是同样结果。
# 唯独 env_error 不在此列——多半是临时性问题，重启/下个 tick 应当重试。
TERMINAL_STATUSES = {STATUS_SUCCESS, STATUS_USER_ERROR, STATUS_TIMEOUT, STATUS_FILE_ERROR}

# 各失败状态回写给前端的提示语（统一记 -2 分）。
STATUS_ERR_MSG = {
    STATUS_TIMEOUT: "timeout: your code exceeded the time limit",
    STATUS_USER_ERROR: "run error: check your code / get code templates in [code] tab",
    STATUS_FILE_ERROR: "file error: check your submission file (exactly 1 valid .ipynb expected)",
    STATUS_ENV_ERROR: "evaluation system error, will retry automatically",
}


class SubmissionFileError(Exception):
    """用户提交的文件本身有问题：缺失、notebook 数量不对、ipynb 无法解析等。

    属于用户侧错误（重试也是同样结果），与「评测环境异常」区分开，单独记为终态。
    """
