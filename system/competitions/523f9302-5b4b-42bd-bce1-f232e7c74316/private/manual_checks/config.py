"""人工复核的唯一配置入口。

本比赛只按四项指标的百分位排名等权计算 ``score``，没有因子池回归模型评价。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


USE_LOCAL = True
COMPETITION_ID = "523f9302-5b4b-42bd-bce1-f232e7c74316"
RUN_ID = "20260812_180115"

LOCAL_WORKSPACE_DIR = Path("/Users/xiehao/Desktop/workspace/BigAlpha")
CLOUD_WORKSPACE_DIR = Path("/home/aiuser/work/workspace/BigAlpha")

METRICS = ("ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir")
SCORE_TOLERANCE = 1e-12
REPORT_TOP_N = 30
MAX_FINAL_RANK_CHANGE = 10
HIGH_CORRELATION = 0.95
MAX_SIMILARITY_SAMPLES = 50_000

SUMMARY_FILENAME = "submissions_summary.csv"
SCORE_FILENAME = "leaderboard_score.csv"
FINAL_FILENAME = "leaderboard_final.csv"
PROCESS_POOL_FILENAME = "score_pool.parquet"
RAW_POOL_FILENAME = "score_pool_raw.parquet"
METADATA_FILENAME = "metadata.json"
REPORT_FILENAME = "manual_check_report.md"
TEAM_PRIVATE_LEADERBOARD_FILENAME = "team_private_leaderboard.csv"
SIMILARITY_SUMMARY_FILENAME = "prediction_similarity_summary.csv"


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class CheckPaths:
    run_dir: Path
    prepared_dir: Path
    private_code_dir: Path
    public_leaderboard_dir: Path | None = None

    def __post_init__(self) -> None:
        for name in ("run_dir", "prepared_dir", "private_code_dir", "public_leaderboard_dir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _resolved(value))

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def summary_path(self) -> Path:
        return self.artifacts_dir / SUMMARY_FILENAME

    @property
    def score_path(self) -> Path:
        return self.artifacts_dir / SCORE_FILENAME

    @property
    def final_path(self) -> Path:
        return self.artifacts_dir / FINAL_FILENAME

    @property
    def process_pool_path(self) -> Path:
        return self.artifacts_dir / PROCESS_POOL_FILENAME

    @property
    def raw_pool_path(self) -> Path:
        return self.artifacts_dir / RAW_POOL_FILENAME

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
    def public_summary_path(self) -> Path:
        if self.public_leaderboard_dir is None:
            raise ValueError("未配置公榜 leaderboard 目录")
        return self.public_leaderboard_dir / SUMMARY_FILENAME

    @property
    def similarity_path(self) -> Path:
        return self.artifacts_dir / SIMILARITY_SUMMARY_FILENAME


@dataclass(frozen=True)
class ManualCheckConfig:
    paths: CheckPaths
    metrics: tuple[str, ...] = METRICS
    score_tolerance: float = SCORE_TOLERANCE
    report_top_n: int = REPORT_TOP_N
    max_final_rank_change: int = MAX_FINAL_RANK_CHANGE
    high_correlation: float = HIGH_CORRELATION
    max_similarity_samples: int = MAX_SIMILARITY_SAMPLES

    def activate(self) -> None:
        for path in (self.paths.private_code_dir, self.paths.private_code_dir.parent):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))


def build_config() -> ManualCheckConfig:
    workspace = _resolved(LOCAL_WORKSPACE_DIR if USE_LOCAL else CLOUD_WORKSPACE_DIR)
    files_dir = workspace / "system" / "files"
    competition_files = (
        files_dir / "private" / COMPETITION_ID
        if USE_LOCAL
        else files_dir / COMPETITION_ID
    )
    private_dir = workspace / "system" / "competitions" / COMPETITION_ID / "private"
    return ManualCheckConfig(CheckPaths(
        run_dir=competition_files / "private" / "runs" / RUN_ID,
        prepared_dir=competition_files / "private" / "prepared",
        private_code_dir=private_dir,
        public_leaderboard_dir=files_dir / "public" / COMPETITION_ID / "leaderboard",
    ))


CONFIG = build_config()
PATHS = CONFIG.paths
