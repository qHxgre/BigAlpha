"""按参赛主体计算 merge 周期内 submission 因子的两两截面相关性。"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG, CheckPaths
from .team_private_leaderboard import build_team_private_report


PAIR_FILENAME = "team_submission_similarity_pairs.csv"
SUMMARY_FILENAME = "team_submission_similarity_summary.csv"


def _load_factor(paths: CheckPaths, submission_id: str) -> pd.DataFrame | None:
    path = paths.run_dir / "submissions" / submission_id / "process_factor.parquet"
    if not path.is_file():
        return None
    frame = pd.read_parquet(path, columns=["date", "instrument", "factor"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["instrument"] = frame["instrument"].astype(str)
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    return frame.dropna(subset=["date", "instrument", "factor"])


def _pair_similarity(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    merged = left.merge(right, on=["date", "instrument"], suffixes=("_1", "_2"))
    if merged.empty:
        return {
            "overlap_rows": 0, "valid_days": 0, "mean_correlation": None,
            "mean_abs_correlation": None, "median_correlation": None,
            "p95_abs_correlation": None, "min_correlation": None,
            "max_correlation": None,
        }
    daily = (
        merged.groupby("date", sort=True)
        .apply(lambda group: group["factor_1"].corr(group["factor_2"]), include_groups=False)
        .dropna()
    )
    if daily.empty:
        return {
            "overlap_rows": len(merged), "valid_days": 0, "mean_correlation": None,
            "mean_abs_correlation": None, "median_correlation": None,
            "p95_abs_correlation": None, "min_correlation": None,
            "max_correlation": None,
        }
    return {
        "overlap_rows": len(merged),
        "valid_days": len(daily),
        "mean_correlation": daily.mean(),
        "mean_abs_correlation": daily.abs().mean(),
        "median_correlation": daily.median(),
        "p95_abs_correlation": daily.abs().quantile(0.95),
        "min_correlation": daily.min(),
        "max_correlation": daily.max(),
    }


def analyze_team_submission_similarity(
    paths: CheckPaths | None = None,
    *,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """读取或计算 merge 周期团队内部 submission 两两相关性。"""
    paths = paths or CONFIG.period_paths["merged"]
    output = Path(output_dir).expanduser().resolve() if output_dir else paths.artifacts_dir
    output.mkdir(parents=True, exist_ok=True)
    pair_path = output / PAIR_FILENAME
    summary_path = output / SUMMARY_FILENAME
    if not force and pair_path.is_file() and summary_path.is_file():
        return {
            "summary": pd.read_csv(summary_path, dtype={"participant_id": str}),
            "pairs": pd.read_csv(
                pair_path,
                dtype={
                    "participant_id": str,
                    "submission_id_1": str,
                    "submission_id_2": str,
                },
            ),
            "summary_csv": summary_path,
            "pairs_csv": pair_path,
            "factor_type": "process_factor（去极值、标准化、风格及行业剔除后）",
            "period": "2025-03-01 至 2026-08-10",
        }

    report = build_team_private_report(paths)
    factor_cache: dict[str, pd.DataFrame | None] = {}
    pair_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for participant in report["participants"]:
        submission_ids = [str(item["submission_id"]) for item in participant["submissions"]]
        for submission_id in submission_ids:
            if submission_id not in factor_cache:
                factor_cache[submission_id] = _load_factor(paths, submission_id)
        available = [submission_id for submission_id in submission_ids if factor_cache[submission_id] is not None]
        team_pairs = []
        for left_id, right_id in combinations(available, 2):
            metrics = _pair_similarity(factor_cache[left_id], factor_cache[right_id])
            row = {
                "participant_type": participant["participant_type"],
                "participant_id": participant["participant_id"],
                "team_name": participant["participant_name"],
                "submission_id_1": left_id,
                "submission_id_2": right_id,
                **metrics,
            }
            pair_rows.append(row)
            team_pairs.append(row)

        pair_frame = pd.DataFrame(team_pairs)
        abs_values = (
            pd.to_numeric(pair_frame.get("mean_abs_correlation"), errors="coerce")
            if not pair_frame.empty else pd.Series(dtype=float)
        )
        summary_rows.append({
            "participant_type": participant["participant_type"],
            "participant_id": participant["participant_id"],
            "team_name": participant["participant_name"],
            "submission_count": len(submission_ids),
            "available_submission_count": len(available),
            "missing_submission_count": len(submission_ids) - len(available),
            "expected_pair_count": len(submission_ids) * (len(submission_ids) - 1) // 2,
            "computed_pair_count": len(team_pairs),
            "mean_pair_abs_correlation": abs_values.mean(),
            "max_pair_abs_correlation": abs_values.max(),
            "high_correlation_pair_count_0_8": int(abs_values.ge(0.8).sum()),
            "near_duplicate_pair_count_0_95": int(abs_values.ge(0.95).sum()),
            "missing_submission_ids": ";".join(
                submission_id for submission_id in submission_ids
                if factor_cache[submission_id] is None
            ),
        })

    pairs = pd.DataFrame(pair_rows)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["max_pair_abs_correlation", "team_name"],
        ascending=[False, True], na_position="last", kind="stable",
    )
    pairs.to_csv(pair_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return {
        "summary": summary,
        "pairs": pairs,
        "summary_csv": summary_path,
        "pairs_csv": pair_path,
        "factor_type": "process_factor（去极值、标准化、风格及行业剔除后）",
        "period": "2025-03-01 至 2026-08-10",
    }
