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
CHECK_START_DATE = "2025-12-01"
CHECK_END_DATE = "2026-08-10"

LOCAL_WORKSPACE_DIR = Path("/Users/xiehao/Desktop/workspace/BigAlpha")
CLOUD_WORKSPACE_DIR = Path("/home/aiuser/work/workspace/BigAlpha")

# 分析参数
HIGH_CORRELATION = 0.95
MAX_SIMILARITY_SAMPLES = 50_000
REPORT_TOP_N = 20
REGRESSION_OVERVIEW_TOP_N = 30
AB_WEIGHT_STEPS = 10
REGRESSION_TOLERANCE = 1e-8
MAX_AB_RANK_GAP = 20
MAX_FINAL_RANK_CHANGE = 10
MAX_ABS_STYLE_CORRELATION = 0.10
MAX_STYLE_REGRESSION_R2 = 0.10
STYLE_QUANTILE = 0.95
INCREMENTAL_TRAIN_WINDOW = 60
INCREMENTAL_TEST_WINDOW = 20
INCREMENTAL_STEP = 20
INCREMENTAL_ALPHA = 1e-3
INCREMENTAL_L1_RATIO = 0.5
SIMILARITY_CLUSTER_THRESHOLD = 0.80
SIMILARITY_HIGH_THRESHOLD = 0.90
SIMILARITY_NEAR_DUPLICATE_THRESHOLD = 0.95
SIMILARITY_TOP_PEERS = 3
MIN_WEIGHT_FOR_SIGN = 1e-4

# 数据字段
METRICS = ("ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir")
REGRESSION_METRICS = (
    "model_score", "abs_weight_mean", "abs_weight_std", "selection_rate",
)
EXPOSURE_DATASOURCE = "bigalpha_2026_exposure"
STYLE_COLUMNS = (
    "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
    "EARNYILD", "GROWTH", "LEVERAGE",
)
INDUSTRY_COLUMNS = (
    "AGRIFOREST", "MINING", "CHEM", "IRONSTEEL", "NONFERMETAL", "ELECTRONICS",
    "AUTO", "HOUSEAPP", "FOODBEVER", "TEXTILE", "LIGHTINDUS", "HEALTH",
    "UTILITIES", "TRANSPORTATION", "REALESTATE", "COMMETRADE", "LEISERVICE",
    "BANK", "NONBANKFINAN", "CONGLOMERATES", "CONMAT", "BUILDDECO", "ELECEQP",
    "AERODEF", "COMPUTER", "MEDIA", "TELECOM", "COAL", "PETRO", "ENVP", "BEAUTY",
)
EXPOSURE_COLUMNS = (*STYLE_COLUMNS, *INDUSTRY_COLUMNS)

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
REGRESSION_RERUN_DIRNAME = "regression_rerun"
RERUN_SUMMARY_FILENAME = "regression_rerun_summary.json"
RERUN_COMPARISON_FILENAME = "regression_rerun_comparison.csv"
RERUN_SCORES_FILENAME = "regression_rerun_scores.csv"
RERUN_WEIGHTS_FILENAME = "regression_rerun_weights_history.parquet"
STYLE_EXPOSURE_DIRNAME = "style_exposure"
STYLE_EXPOSURE_SUMMARY_FILENAME = "style_exposure_summary.csv"
STYLE_EXPOSURE_DETAIL_FILENAME = "style_exposure_by_style.csv"
STYLE_EXPOSURE_DAILY_FILENAME = "style_exposure_daily.parquet"
STYLE_EXPOSURE_METADATA_FILENAME = "style_exposure_metadata.json"
STYLE_EXPOSURE_FIGURE_FILENAME = "style_exposure_overview.png"
INCREMENTAL_DIRNAME = "incremental_analysis"
INCREMENTAL_BY_WINDOW_FILENAME = "incremental_by_window.parquet"
INCREMENTAL_SUMMARY_FILENAME = "incremental_summary.csv"
GROUP_INCREMENTAL_BY_WINDOW_FILENAME = "group_incremental_by_window.parquet"
GROUP_INCREMENTAL_SUMMARY_FILENAME = "group_incremental_summary.csv"
FACTOR_SIMILARITY_DAILY_FILENAME = "factor_similarity_daily.parquet"
FACTOR_SIMILARITY_SUMMARY_FILENAME = "factor_similarity_summary.csv"
FACTOR_CLUSTERS_FILENAME = "factor_clusters.csv"
INCREMENTAL_METADATA_FILENAME = "analysis_metadata.json"


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
    def regression_rerun_dir(self) -> Path:
        return self.artifacts_dir / REGRESSION_RERUN_DIRNAME

    @property
    def style_exposure_dir(self) -> Path:
        return self.artifacts_dir / STYLE_EXPOSURE_DIRNAME

    @property
    def incremental_dir(self) -> Path:
        return self.artifacts_dir / INCREMENTAL_DIRNAME

@dataclass(frozen=True)
class ManualCheckConfig:
    """供全部检查脚本使用的统一配置。"""

    paths: CheckPaths
    start_date: str = CHECK_START_DATE
    end_date: str = CHECK_END_DATE
    high_correlation: float = HIGH_CORRELATION
    max_similarity_samples: int = MAX_SIMILARITY_SAMPLES
    report_top_n: int = REPORT_TOP_N
    regression_overview_top_n: int = REGRESSION_OVERVIEW_TOP_N
    ab_weight_steps: int = AB_WEIGHT_STEPS
    regression_tolerance: float = REGRESSION_TOLERANCE
    max_abs_style_correlation: float = MAX_ABS_STYLE_CORRELATION
    max_style_regression_r2: float = MAX_STYLE_REGRESSION_R2
    style_quantile: float = STYLE_QUANTILE
    incremental_train_window: int = INCREMENTAL_TRAIN_WINDOW
    incremental_test_window: int = INCREMENTAL_TEST_WINDOW
    incremental_step: int = INCREMENTAL_STEP
    incremental_alpha: float = INCREMENTAL_ALPHA
    incremental_l1_ratio: float = INCREMENTAL_L1_RATIO
    similarity_cluster_threshold: float = SIMILARITY_CLUSTER_THRESHOLD
    similarity_high_threshold: float = SIMILARITY_HIGH_THRESHOLD
    similarity_near_duplicate_threshold: float = SIMILARITY_NEAR_DUPLICATE_THRESHOLD
    similarity_top_peers: int = SIMILARITY_TOP_PEERS
    min_weight_for_sign: float = MIN_WEIGHT_FOR_SIGN

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
    paths = CheckPaths(
        run_dir=competition_files / "private" / "runs" / RUN_ID,
        prepared_dir=competition_files / "private" / "prepared",
        private_code_dir=private_code_dir,
        public_leaderboard_dir=files_dir / "public" / COMPETITION_ID / "leaderboard",
        bigalpha_eval_src=workspace / "eval" / "bigalpha_eval" / "src",
    )
    return ManualCheckConfig(paths=paths)


CONFIG = build_config()
PATHS = CONFIG.paths
