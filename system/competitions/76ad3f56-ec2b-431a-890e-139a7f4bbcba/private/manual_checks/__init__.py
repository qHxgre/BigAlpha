"""私榜人工复核工具，面向 Notebook 的统一导入入口。"""

from .config import CheckPaths
from .factors import analyze_factor_similarity
from .failures import analyze_failed_submissions
from .ranking import (
    analyze_a_metric_sensitivity,
    analyze_ab_weight_sensitivity,
    analyze_rank_conflicts,
    check_score_consistency,
)
from .regression import (
    analyze_b_score_robustness,
    analyze_regression_stability,
    check_regression_integrity,
)
from .visualization import plot_regression_overview, rerun_regression_explanation


def generate_markdown_report(*args, **kwargs):
    """延迟导入报告模块，兼容 Notebook 导入和 ``python -m`` 执行。"""
    from .report import generate_markdown_report as generate

    return generate(*args, **kwargs)

__all__ = [
    "CheckPaths", "check_score_consistency", "analyze_rank_conflicts",
    "analyze_ab_weight_sensitivity", "analyze_a_metric_sensitivity",
    "check_regression_integrity", "analyze_regression_stability",
    "analyze_factor_similarity", "analyze_failed_submissions",
    "analyze_b_score_robustness", "plot_regression_overview",
    "rerun_regression_explanation", "generate_markdown_report",
]
