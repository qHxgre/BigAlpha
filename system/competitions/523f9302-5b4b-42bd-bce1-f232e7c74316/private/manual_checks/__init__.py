"""端到端模型私榜人工复核工具。"""

from .config import CONFIG, PATHS, CheckPaths, ManualCheckConfig
from .ranking import (
    analyze_metric_rank_conflicts,
    analyze_metric_sensitivity,
    check_score_consistency,
)
from .similarity import analyze_prediction_similarity


def generate_markdown_report(*args, **kwargs):
    """延迟导入报告模块，兼容 Notebook 和命令行执行。"""
    from .report import generate_markdown_report as generate

    return generate(*args, **kwargs)


__all__ = [
    "CONFIG", "PATHS", "CheckPaths", "ManualCheckConfig",
    "check_score_consistency", "analyze_metric_rank_conflicts",
    "analyze_metric_sensitivity", "analyze_prediction_similarity",
    "generate_markdown_report",
]
