"""最终打分与运行结果汇总阶段。

职责：
    - score_final：合成最终得分 Score_i = 0.3 * A_i + 0.7 * B_i 并回写；
    - summarize_submissions：把所有提交的运行状态、单因子指标、各项得分、产物情况汇总成一张表。

作为 mixin 混入 BigAlphaJudge，依赖 BigAlphaJudgeBase 提供的 mode 感知路径与 load_b_scores。
"""
from __future__ import annotations

import os

import pandas as pd

import scoring
from fileio import csv_to_map, read_csv, read_json


class ScoringMixin:
    """最终得分合成 + 提交运行结果汇总。"""

    # ---- 最终得分 ---------------------------------------------------------

    def score_final(self) -> None:
        """合成并回写每个提交的最终得分：Score_i = 0.3 * A_i + 0.7 * B_i。

        三类结果分别落盘，互不覆盖：
            - 单因子分析（A 项）：leaderboard_sfa.csv，由 score_sfa() 产出；
            - 因子池回归（B 项来源）：leaderboard_reg.csv，由 run_regression() 产出；
            - 最终得分：leaderboard_final.csv，由本方法产出（id / a_score / b_score / final_score）。

        - A_i：单因子截面排名得分，取自 leaderboard_sfa.csv 的 score 列；
        - B_i：因子池回归的 ModelScore 百分位归一化（load_b_scores），
                回归尚未产出或该因子未入池时记 0，此时 Score_i = 0.3 * A_i，后续 tick 会自动补齐。
        """
        if not os.path.exists(self.leaderboard_sfa_csv):
            self.log.warning("final.no_sfa", msg="缺少单因子得分，跳过最终合成")
            return
        sfa_df = read_csv(self.leaderboard_sfa_csv, logger=self.log)
        if sfa_df is None:
            return
        if "id" not in sfa_df.columns:
            self.log.warning("final.no_id", msg="单因子榜单缺少 id 列")
            return

        b_scores = self.load_b_scores()

        # 最终得分单独成表，只保留评分相关列，不污染单因子分析榜单
        final = pd.DataFrame({"id": sfa_df["id"]})
        final["a_score"] = pd.to_numeric(sfa_df.get("score"), errors="coerce")
        final["b_score"] = final["id"].astype(str).map(b_scores).astype(float).fillna(0.0)
        final["final_score"] = 0.3 * final["a_score"] + 0.7 * final["b_score"]
        final = final.sort_values("final_score", ascending=False, na_position="last")

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        final.to_csv(self.leaderboard_final_csv, index=False)

        for _, row in final.iterrows():
            score = row["final_score"]
            # NaN != NaN：A 缺失（如指标全空）时最终分也为 NaN，统一记为 -2 失败
            if score != score:
                score = -2
            self.alphathon_api.update_submission_score(
                submission_id=row["id"],
                **{self.score_field: float(score)},
            )
        self.log.info(
            "final.scored",
            count=len(final),
            with_b=int((final["b_score"] > 0).sum()),
            msg="合成最终得分 0.3*A + 0.7*B 完成",
        )

    # ---- 运行结果汇总 -----------------------------------------------------

    def summarize_submissions(self) -> None:
        """把所有提交的运行结果汇总到一个统计文件 submissions_summary.csv。

        逐个提交收集：
            - 运行状态（sfa_status.json：status / finished_at / elapsed_ms / error）；
            - 单因子分析指标（factor_analyze.json：ic_mean / ic_ir / sharpe_ratio / stress_ic_ir 等）；
            - A 项截面排名得分（leaderboard_sfa.csv 的 score）与最终得分（leaderboard_final.csv 的 a/b/final）；
            - 各产物是否落盘（raw/process factor、回归得分）。
        汇总后按 score 倒序落盘，方便整体观察各提交的成败与表现。
        """
        # A 项（单因子截面排名得分）取自 leaderboard_sfa.csv 的 score 列
        a_scores = csv_to_map(read_csv(self.leaderboard_sfa_csv, logger=self.log), "id", "score")

        # 最终得分（a_score / b_score / final_score）整行取自 leaderboard_final.csv
        final_df = read_csv(self.leaderboard_final_csv, logger=self.log)
        final_scores = (
            {str(r["id"]): r.to_dict() for _, r in final_df.iterrows()}
            if final_df is not None and "id" in final_df.columns
            else {}
        )

        rows = []
        for sid, submission, sub_dir in self._iter_submission_dirs():
            row: dict = {
                "submission_id": sid,
                "user_id": submission.get("user_id"),
                "group": scoring.group_key(submission),
            }

            # 运行状态
            status = self.read_sfa_status(submission) or {}
            row["status"] = status.get("status")
            row["finished_at"] = status.get("finished_at")
            row["elapsed_ms"] = status.get("elapsed_ms")
            row["error"] = status.get("error")

            # 单因子分析指标
            fa = read_json(os.path.join(sub_dir, self.factor_analyze_file), logger=self.log)
            if fa is not None:
                for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir"]:
                    row[col] = fa.get(col)

            # 截面排名得分：A 项、B 项与最终得分
            final_row = final_scores.get(sid, {})
            row["a_score"] = final_row.get("a_score", a_scores.get(sid))
            row["b_score"] = final_row.get("b_score")
            row["score"] = final_row.get("final_score", a_scores.get(sid))

            # 产物是否落盘
            row["has_raw_factor"] = os.path.exists(os.path.join(sub_dir, self.raw_factor_file))
            row["has_process_factor"] = os.path.exists(os.path.join(sub_dir, self.process_factor_file))
            row["has_regression_score"] = os.path.exists(os.path.join(sub_dir, self.factor_regression_score_file))

            rows.append(row)

        if not rows:
            self.log.warning("summary.empty", msg="没有任何提交运行结果可汇总")
            return

        df = pd.DataFrame(rows)
        # 指标统一转数值，便于排序与后续分析
        for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir", "a_score", "b_score", "score", "elapsed_ms"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("score", ascending=False, na_position="last")

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        df.to_csv(self.submissions_summary_csv, index=False)
        self.log.info("summary.saved", count=len(df), path=self.submissions_summary_csv, msg="汇总提交运行结果完成")
