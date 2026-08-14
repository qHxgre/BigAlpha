"""私榜人工复核工具，面向 Notebook 的统一导入入口。"""

from .config import CONFIG, PATHS, ManualCheckConfig, CheckPaths
from .factors import analyze_cross_sectional_similarity, analyze_factor_similarity
from .incremental import analyze_incremental_contribution
from .ranking import (
    analyze_a_metric_sensitivity,
    analyze_ab_weight_sensitivity,
    analyze_rank_conflicts,
    check_score_consistency,
    compare_public_private_ranking,
)
from .regression import (
    analyze_b_score_robustness,
    analyze_regression_explainability,
    analyze_regression_stability,
    check_regression_integrity,
)
from .style_exposure import (
    StyleExposureCheckResult,
    analyze_style_exposure,
    load_style_exposure_results,
    plot_style_exposure_results,
    save_style_exposure_results,
)
from .team_private_leaderboard import (
    build_team_leaderboard_summary,
    build_team_private_leaderboard,
    build_team_private_report,
    export_team_private_leaderboard,
)
from .visualization import plot_regression_overview, rerun_regression_explanation


def generate_markdown_report(*args, **kwargs):
    """延迟导入报告模块，兼容 Notebook 导入和 ``python -m`` 执行。"""
    from .report import generate_markdown_report as generate

    return generate(*args, **kwargs)

__all__ = [
    "CONFIG", "PATHS", "ManualCheckConfig", "CheckPaths",
    "check_score_consistency", "analyze_rank_conflicts",
    "compare_public_private_ranking",
    "analyze_ab_weight_sensitivity", "analyze_a_metric_sensitivity",
    "check_regression_integrity", "analyze_regression_stability",
    "analyze_regression_explainability",
    "analyze_factor_similarity",
    "analyze_cross_sectional_similarity", "analyze_incremental_contribution",
    "analyze_style_exposure", "StyleExposureCheckResult",
    "load_style_exposure_results", "plot_style_exposure_results", "save_style_exposure_results",
    "analyze_b_score_robustness", "plot_regression_overview",
    "rerun_regression_explanation", "generate_markdown_report",
    "build_team_leaderboard_summary", "build_team_private_leaderboard",
    "build_team_private_report", "export_team_private_leaderboard",
]
