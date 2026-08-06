"""私榜因子池构建和回归执行。"""
from __future__ import annotations

import os

import pandas as pd

from judge.judgebase import LocalProcessUserRunner

from scoring import compute_b_scores
from templates import build_regression_runner


def run_regression(
    successful: pd.DataFrame,
    submission_dir: str,
    artifact_dir: str,
    date_start: str,
    date_end: str,
    logger,
) -> dict[str, float]:
    frames = []
    for sid in successful["submission_id"].astype(str):
        path = os.path.join(submission_dir, sid, "process_factor.parquet")
        try:
            factor = pd.read_parquet(path)[["date", "instrument", "factor"]]
            frames.append(factor.rename(columns={"factor": sid}).set_index(["date", "instrument"]))
        except Exception as exc:
            logger.warning("pool.skip", submission_id=sid, error=str(exc))
    if len(frames) < 2:
        return {}

    pool = pd.concat(frames, axis=1, join="outer").reset_index()
    pool_path = os.path.abspath(os.path.join(artifact_dir, "factor_pool.parquet"))
    regression_csv = os.path.abspath(os.path.join(artifact_dir, "leaderboard_reg.csv"))
    pool.to_parquet(pool_path)

    runner = LocalProcessUserRunner(
        submission_id="_regression",
        files={
            "judge_runner.py": build_regression_runner(
                pool_path, regression_csv, date_start, date_end
            )
        },
        cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
        runner_dir=os.path.join(artifact_dir, "regression"),
    )
    runner.run(_raise=True)
    return compute_b_scores(pd.read_csv(regression_csv, encoding="utf-8-sig"))
