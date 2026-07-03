"""最终打分与运行结果汇总阶段。

职责：
    - score_final：读取截面排名得分快照，回写每个提交的最终得分（本赛道最终得分即四指标排名加权）；
    - summarize_submissions：把所有提交的运行状态、单因子指标、最终得分、产物情况汇总成一张表。

作为 mixin 混入 EndToEndJudge，依赖 EndToEndJudgeBase 提供的 mode 感知路径。
"""
from __future__ import annotations

import os

import pandas as pd

import scoring
from fileio import csv_to_map, read_csv, read_json
from constants import (
    STATUS_ENV_ERROR,
    STATUS_ERR_MSG
)


class ScoringMixin:
    """最终得分回写 + 提交运行结果汇总。"""

    # ---- 最终得分 ---------------------------------------------------------

    def score_final(self) -> None:
        """回写每个提交的最终得分（本赛道最终得分即截面排名加权得分，无因子池回归 B 项）。

        - 得分快照取自 leaderboard_score.csv 的 score 列（由 score_models() 产出）；
        - 单独落盘 leaderboard_final.csv（id / score），与得分快照表分开，便于观察。
        """
        if not os.path.exists(self.leaderboard_score_csv):
            self.log.warning("final.no_score", msg="缺少截面排名得分，跳过最终回写")
            return
        score_df = read_csv(self.leaderboard_score_csv, logger=self.log)
        if score_df is None:
            return
        if "id" not in score_df.columns:
            self.log.warning("final.no_id", msg="得分榜单缺少 id 列")
            return

        final = pd.DataFrame({"id": score_df["id"]})
        final["score"] = pd.to_numeric(score_df.get("score"), errors="coerce")
        final = final.sort_values("score", ascending=False, na_position="last")

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        final.to_csv(self.leaderboard_final_csv, index=False)

        for _, row in final.iterrows():
            score = row["score"]
            # NaN != NaN：指标全空导致得分缺失时，统一记为 -2 失败
            if score != score:
                self.alphathon_api.update_submission_score(
                    submission_id=row["id"],
                    **{
                        self.score_field: -2,
                        self.score_data_field: {"err_msg": STATUS_ERR_MSG[STATUS_ENV_ERROR]},
                    },
                )
            else:
                self.alphathon_api.update_submission_score(
                    submission_id=row["id"],
                    **{
                        self.score_field: float(score),
                        self.score_data_field: self._row_to_jsonable(row),
                    },
                )
        self.log.info("final.scored", count=len(final), msg="回写最终得分完成")

    @staticmethod
    def _row_to_jsonable(row: pd.Series) -> dict:
        """把 DataFrame 的一行（pandas Series，含 numpy 标量）转成可 JSON 序列化的纯 dict。

        iterrows() 产出的 Series 及其 numpy 标量（np.float64 等）不能直接 json.dumps，
        update_submission_score 走 httpx json= 序列化时会触发
        'Object of type Series is not JSON serializable'。这里逐项转成原生 Python
        类型，NaN/Inf 一律转 None，保证回写时可序列化。
        """
        import math

        out: dict = {}
        for key, val in row.items():
            # numpy 标量 -> python 原生
            if hasattr(val, "item"):
                val = val.item()
            # NaN / Inf -> None（JSON 不支持）
            if isinstance(val, float) and not math.isfinite(val):
                val = None
            out[str(key)] = val
        return out

    # ---- 运行结果汇总 -----------------------------------------------------

    def summarize_submissions(self) -> None:
        """把所有提交的运行结果汇总到一个统计文件 submissions_summary.csv。

        逐个提交收集：
            - 运行状态（score_status.json：status / finished_at / elapsed_ms / error）；
            - 单因子分析指标（score_analyze.json：ic_mean / ic_ir / sharpe_ratio / stress_ic_ir 等）；
            - 最终得分（leaderboard_score.csv 的 score）；
            - 各产物是否落盘（raw/process score）。
        汇总后按 score 倒序落盘，方便整体观察各提交的成败与表现。
        """
        # 最终得分取自 leaderboard_score.csv 的 score 列
        scores = csv_to_map(read_csv(self.leaderboard_score_csv, logger=self.log), "id", "score")

        rows = []
        for sid, submission, sub_dir in self._iter_submission_dirs():
            row: dict = {
                "submission_id": sid,
                "user_id": submission.get("user_id"),
                "group": scoring.group_key(submission),
            }

            # 运行状态
            status = self.read_score_status(submission) or {}
            row["status"] = status.get("status")
            row["finished_at"] = status.get("finished_at")
            row["elapsed_ms"] = status.get("elapsed_ms")
            row["error"] = status.get("error")

            # 单因子分析指标
            analyze = read_json(os.path.join(sub_dir, self.score_analyze_file), logger=self.log)
            if analyze is not None:
                for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir"]:
                    row[col] = analyze.get(col)

            # 最终得分
            row["score"] = scores.get(sid)

            # 产物是否落盘
            row["has_raw_score"] = os.path.exists(os.path.join(sub_dir, self.raw_score_file))
            row["has_process_score"] = os.path.exists(os.path.join(sub_dir, self.process_score_file))

            rows.append(row)

        if not rows:
            self.log.warning("summary.empty", msg="没有任何提交运行结果可汇总")
            return

        df = pd.DataFrame(rows)

        # 设了 SUBMISSION_IDS 只跑子集时，本轮 rows 只含该子集，若直接整表覆盖会抹掉
        # 其他提交的历史行。此处做 upsert：读入旧表，剔除本轮涉及的 id，再拼回本轮结果，
        # 保证该文件只新增或替换指定 submission 的记录，不删除其余提交的记录。
        # 全量模式（SUBMISSION_IDS 为空）本轮即全体，旧表整体被替换，行为不变。
        if self.SUBMISSION_IDS:
            old_df = read_csv(self.submissions_summary_csv, logger=self.log)
            if old_df is not None and "submission_id" in old_df.columns:
                current_ids = set(df["submission_id"].astype(str))
                kept = old_df[~old_df["submission_id"].astype(str).isin(current_ids)]
                df = pd.concat([kept, df], ignore_index=True)

        # 指标统一转数值，便于排序与后续分析
        for col in ["ic_mean", "ic_ir", "sharpe_ratio", "stress_ic_ir", "score", "elapsed_ms"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("score", ascending=False, na_position="last")

        os.makedirs(self.leaderboard_dir, exist_ok=True)
        df.to_csv(self.submissions_summary_csv, index=False)
        self.log.info("summary.saved", count=len(df), path=self.submissions_summary_csv, msg="汇总提交运行结果完成")
