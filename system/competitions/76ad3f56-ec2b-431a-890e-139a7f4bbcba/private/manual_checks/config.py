"""人工复核的唯一配置入口。

修改运行目录、日期、阈值或输出文件名时，只需要编辑本文件。其他模块不应再
声明业务配置或硬编码路径。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# 用户配置区
# ---------------------------------------------------------------------------
USE_LOCAL = True
COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
RUN_ID = "20260812_180129"
PERIOD_RUN_DIRS = {
    "period_1": "20250301_20251130",
    "period_2": "20251201_20260810",
    "merged": "20250301_20260810_merged",
}
DEFAULT_PERIOD = "merged"

LOCAL_WORKSPACE_DIR = Path("/Users/xiehao/Desktop/workspace/BigAlpha")
CLOUD_WORKSPACE_DIR = Path("/home/aiuser/work/workspace/BigAlpha")

# 数据字段
METRICS = ("ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir")
REGRESSION_METRICS = (
    "model_score", "abs_weight_mean", "abs_weight_std", "selection_rate",
)
# 产物文件名
FACTOR_POOL_FILENAME = "factor_pool.parquet"
FACTOR_POOL_RAW_FILENAME = "factor_pool_raw.parquet"
SUMMARY_FILENAME = "submissions_summary.csv"
FINAL_FILENAME = "leaderboard_final.csv"
REGRESSION_FILENAME = "leaderboard_reg.csv"
METADATA_FILENAME = "metadata.json"
REPORT_FILENAME = "manual_check_report.md"
TEAM_PRIVATE_LEADERBOARD_FILENAME = "team_private_leaderboard.csv"
TEAM_LEADERBOARD_SUMMARY_FILENAME = "team_leaderboard_summary.csv"
PERIOD_COMPARISON_FILENAME = "period_score_comparison.csv"
TEAM_PERIOD_COMPARISON_FILENAME = "team_period_score_comparison.csv"
# 兼容旧导出代码读取可选的主体内相关性文件；精简流程不会主动计算它。
HIGH_CORRELATION = 0.95
FACTOR_SIMILARITY_SUMMARY_FILENAME = "factor_similarity_summary.csv"


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class CheckPaths:
    """一次私榜运行的所有输入、输出路径。"""

    run_dir: Path
    prepared_dir: Path
    private_code_dir: Path
    public_leaderboard_dir: Path | None = None
    bigalpha_eval_src: Path | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_dir", "prepared_dir", "private_code_dir", "public_leaderboard_dir",
            "bigalpha_eval_src",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _resolved(value))

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def factor_pool_path(self) -> Path:
        return self.artifacts_dir / FACTOR_POOL_FILENAME

    @property
    def factor_pool_raw_path(self) -> Path:
        """由各 submission 的 raw_factor.parquet 合并得到的原始因子池。"""
        return self.artifacts_dir / FACTOR_POOL_RAW_FILENAME

    @property
    def summary_path(self) -> Path:
        return self.artifacts_dir / SUMMARY_FILENAME

    @property
    def final_path(self) -> Path:
        return self.artifacts_dir / FINAL_FILENAME

    @property
    def regression_path(self) -> Path:
        return self.artifacts_dir / REGRESSION_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.prepared_dir / METADATA_FILENAME

    @property
    def report_path(self) -> Path:
        return self.artifacts_dir / REPORT_FILENAME

    @property
    def team_private_leaderboard_path(self) -> Path:
        return self.artifacts_dir / TEAM_PRIVATE_LEADERBOARD_FILENAME

    @property
    def team_leaderboard_summary_path(self) -> Path:
        return self.artifacts_dir / TEAM_LEADERBOARD_SUMMARY_FILENAME

    @property
    def public_summary_path(self) -> Path:
        if self.public_leaderboard_dir is None:
            raise ValueError("未配置公榜 leaderboard 目录")
        return self.public_leaderboard_dir / SUMMARY_FILENAME

    @property
    def public_regression_path(self) -> Path:
        if self.public_leaderboard_dir is None:
            raise ValueError("未配置公榜 leaderboard 目录")
        return self.public_leaderboard_dir / REGRESSION_FILENAME

    @property
    def submission_source_path(self) -> Path:
        return self.run_dir.parents[4] / "alphathon__submission.csv"

    @property
    def team_source_path(self) -> Path:
        return self.run_dir.parents[4] / "alphathon__team.csv"

    @property
    def user_source_path(self) -> Path:
        return self.run_dir.parents[5] / "scripts" / "private" / "alphathon__user.csv"

    @property
    def incremental_dir(self) -> Path:
        return self.artifacts_dir / "incremental_analysis"

@dataclass(frozen=True)
class ManualCheckConfig:
    """人工复核导出配置。"""

    paths: CheckPaths
    period_paths: dict[str, CheckPaths]
    high_correlation: float = HIGH_CORRELATION

    def activate(self) -> None:
        """注册项目源码目录，确保 Notebook 和脚本使用相同导入环境。"""
        for path in (self.paths.private_code_dir, self.paths.bigalpha_eval_src):
            if path is not None and str(path) not in sys.path:
                sys.path.insert(0, str(path))


def build_config() -> ManualCheckConfig:
    """根据用户配置区构造当前环境配置。"""
    workspace = _resolved(LOCAL_WORKSPACE_DIR if USE_LOCAL else CLOUD_WORKSPACE_DIR)
    files_dir = workspace / "system" / "files"
    competition_files = (
        files_dir / "private" / COMPETITION_ID
        if USE_LOCAL
        else files_dir / COMPETITION_ID
    )
    private_code_dir = workspace / "system" / "competitions" / COMPETITION_ID / "private"
    run_root = competition_files / "private" / "runs" / RUN_ID
    common = {
        "prepared_dir": competition_files / "private" / "prepared",
        "private_code_dir": private_code_dir,
        "public_leaderboard_dir": files_dir / "public" / COMPETITION_ID / "leaderboard",
        "bigalpha_eval_src": workspace / "eval" / "bigalpha_eval" / "src",
    }
    period_paths = {
        name: CheckPaths(run_dir=run_root / dirname, **common)
        for name, dirname in PERIOD_RUN_DIRS.items()
    }
    return ManualCheckConfig(paths=period_paths[DEFAULT_PERIOD], period_paths=period_paths)


CONFIG = build_config()
PATHS = CONFIG.paths
