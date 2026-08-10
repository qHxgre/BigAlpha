"""人工复核所需的路径配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckPaths:
    """一次私榜运行的复核路径集合。"""

    run_dir: Path
    prepared_dir: Path
    private_code_dir: Path
    bigalpha_eval_src: Path | None = None

    def __post_init__(self) -> None:
        for field_name in ("run_dir", "prepared_dir", "private_code_dir", "bigalpha_eval_src"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value).expanduser().resolve())

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def factor_pool_path(self) -> Path:
        return self.artifacts_dir / "factor_pool.parquet"

    @property
    def summary_path(self) -> Path:
        return self.artifacts_dir / "submissions_summary.csv"

    @property
    def final_path(self) -> Path:
        return self.artifacts_dir / "leaderboard_final.csv"

    @property
    def regression_path(self) -> Path:
        return self.artifacts_dir / "leaderboard_reg.csv"

    @property
    def metadata_path(self) -> Path:
        return self.prepared_dir / "metadata.json"
