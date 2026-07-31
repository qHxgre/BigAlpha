"""因子池回归阶段（B 项来源）。

职责：把各队伍排名靠前的因子拼成因子池落盘，跑一次 Elastic Net 回归得到 per_factor_scores，
并据此算出每个因子的 B 项得分。

作为 mixin 混入 BigAlphaJudge，依赖 BigAlphaJudgeBase 提供的 mode 感知路径与 JUDGE_REG 模板。
"""
from __future__ import annotations

import os
import time

import pandas as pd
import pyarrow.parquet as pq

from judge.judgebase import LocalProcessUserRunner, log_timer

import scoring
from fileio import csv_to_map, read_csv


class RegressionMixin:
    """因子池构建 + 回归 + B 项得分计算。"""

    # ---- 因子池构建 -------------------------------------------------------

    @staticmethod
    def _pool_existing_columns(path: str) -> set[str]:
        """只读已有因子池 parquet 的 schema（不读数据），返回已落盘的因子（列）名集合。

        用于判断哪些 kept 因子已经在旧因子池文件里，可以直接按列复用，不必重新读取
        各提交目录下的 process_factor/raw_factor 并重新做 outer join。
        """
        if not os.path.exists(path):
            return set()
        try:
            names = pq.ParquetFile(path).schema.names
        except Exception:
            return set()
        return {n for n in names if n not in ("date", "instrument") and not n.startswith("__index_level")}

    def _load_reused_pool_columns(self, path: str, sids: list[str]) -> pd.DataFrame | None:
        """从旧因子池 parquet 按列只读出 date/instrument + 指定因子列（列裁剪，不读整张表）。"""
        if not sids:
            return None
        try:
            return pd.read_parquet(path, columns=["date", "instrument", *sids])
        except Exception as e:
            self.log.error("pool.reuse_read_failed", path=path, count=len(sids), error=str(e), msg="复用旧因子池列失败")
            return None

    def save_factor_pool(self) -> int:
        """汇总优质因子，构建因子池 parquet。

        1. 遍历 submissions 目录读取处理后因子文件，因子名取该提交的 id；
        2. 按队伍分组，每队保留单因子得分排名前 FACTOR_POOL_TOP_N 的因子；
        3. 已经落盘在旧因子池文件里的因子直接按列复用（跳过重新读取+重新对齐）；
           只有新增/此前未入池的因子才重新读取各提交产物；
        4. 全部因子按 date / instrument 对齐（concat 一次性 outer join），落盘为 parquet。

        返回入池因子数（供 on_tick 汇总成一行日志）；未落盘时返回 0。
        """
        self.log.info("pool.begin", msg="开始构建因子池")

        # 读取单因子得分，用于每个队伍内部的因子取舍
        sfa_scores = csv_to_map(read_csv(self.leaderboard_sfa_csv, logger=self.log), "id", "score")

        records = []
        for sid, submission, sub_dir in self._iter_submission_dirs():
            pf_path = os.path.join(sub_dir, self.process_factor_file)
            if not os.path.exists(pf_path):
                continue
            records.append({
                "sid": sid,
                "group": scoring.group_key(submission),
                "factor_name": sid,
                "score": sfa_scores.get(sid, float("nan")),
                "path": pf_path,
                "raw_path": os.path.join(sub_dir, self.raw_factor_file),
            })

        if not records:
            self.log.warning("pool.empty", msg="没有任何因子数据")
            return 0

        meta = pd.DataFrame(records)
        # 每个队伍内按单因子得分倒序，保留前 N（缺失得分排最后）
        meta["rank_in_group"] = (
            meta.groupby("group")["score"].rank(method="first", ascending=False, na_option="bottom")
        )
        kept = meta[meta["rank_in_group"] <= self.FACTOR_POOL_TOP_N]
        dropped = meta[meta["rank_in_group"] > self.FACTOR_POOL_TOP_N]
        self.log.info(
            "pool.candidates",
            candidates=len(meta),
            groups=meta["group"].nunique(),
            kept=len(kept),
            kept_ids=kept["sid"].tolist(),
            dropped=len(dropped),
            dropped_ids=dropped["sid"].tolist(),
            msg="已按每队 Top N 筛出入池候选因子",
        )

        # pool 与 raw_pool 各自独立判断「已落盘可复用」：两者来源不同的文件，复用范围可能不同
        # （比如某因子的 process_factor 已入过池，但 raw_factor 缺失/上次读取失败没入 raw_pool）。
        kept_sids = kept["sid"].tolist()
        kept_sid_set = set(kept_sids)
        existing_pool_cols = self._pool_existing_columns(self.factor_pool_path)
        existing_raw_cols = self._pool_existing_columns(self.factor_pool_raw_path)
        pool_new_sids = {s for s in kept_sids if s not in existing_pool_cols}
        raw_new_sids = {s for s in kept_sids if s not in existing_raw_cols}
        pool_reused_sids = [s for s in kept_sids if s not in pool_new_sids]
        raw_reused_sids = [s for s in kept_sids if s not in raw_new_sids]
        # 上一轮已落盘、但这一轮不再进 kept 的因子：不会进新池子文件，相当于被剔除。
        pool_evicted_sids = sorted(existing_pool_cols - kept_sid_set)
        raw_evicted_sids = sorted(existing_raw_cols - kept_sid_set)
        self.log.info(
            "pool.reuse",
            pool_reused=len(pool_reused_sids),
            pool_to_read=len(pool_new_sids),
            pool_new_ids=sorted(pool_new_sids),
            pool_evicted=len(pool_evicted_sids),
            pool_evicted_ids=pool_evicted_sids,
            raw_reused=len(raw_reused_sids),
            raw_to_read=len(raw_new_sids),
            raw_new_ids=sorted(raw_new_sids),
            raw_evicted=len(raw_evicted_sids),
            raw_evicted_ids=raw_evicted_sids,
            msg="已落盘因子按列复用，新增因子重新读取，掉出 kept 的因子将从池子剔除",
        )

        frames = []
        reused_pool = self._load_reused_pool_columns(self.factor_pool_path, pool_reused_sids)
        if reused_pool is not None:
            frames.append(reused_pool.set_index(["date", "instrument"]))

        raw_frames = []
        reused_raw_pool = self._load_reused_pool_columns(self.factor_pool_raw_path, raw_reused_sids)
        if reused_raw_pool is not None:
            raw_frames.append(reused_raw_pool.set_index(["date", "instrument"]))

        # 只对「新增/此前未入池」的因子重新读取各提交目录下的产物文件
        need_read = kept[kept["sid"].isin(pool_new_sids | raw_new_sids)]
        for _, r in need_read.iterrows():
            sid = r["sid"]
            name = r["factor_name"]
            self.log.info(
                "pool.reading_factor",
                submission_id=sid,
                read_pool=sid in pool_new_sids,
                read_raw=sid in raw_new_sids,
                msg="正在读取该因子数据",
            )

            if sid in pool_new_sids:
                try:
                    fdf = pd.read_parquet(r["path"])
                except Exception as e:
                    self.log.error("pool.read_failed", submission_id=sid, error=str(e), msg="无法读取因子数据")
                else:
                    if {"date", "instrument", "factor"}.issubset(fdf.columns):
                        fdf = fdf[["date", "instrument", "factor"]].rename(columns={"factor": name})
                        frames.append(fdf.set_index(["date", "instrument"]))

            # 同步把该因子的原始数据并入 raw_pool；原始数据缺失/格式不符不影响正常因子池落盘
            if sid in raw_new_sids:
                raw_path = r["raw_path"]
                if os.path.exists(raw_path):
                    try:
                        rdf = pd.read_parquet(raw_path)
                    except Exception as e:
                        self.log.error("pool.raw_read_failed", submission_id=sid, error=str(e), msg="无法读取原始因子数据")
                    else:
                        if {"date", "instrument", "factor"}.issubset(rdf.columns):
                            rdf = rdf[["date", "instrument", "factor"]].rename(columns={"factor": name})
                            raw_frames.append(rdf.set_index(["date", "instrument"]))

        # 一次性按 date/instrument 对齐（outer join），取代逐个因子的 merge
        pool = pd.concat(frames, axis=1, join="outer").reset_index() if frames else None
        raw_pool = pd.concat(raw_frames, axis=1, join="outer").reset_index() if raw_frames else None

        factor_cols = [] if pool is None else [c for c in pool.columns if c not in ("date", "instrument")]
        # 因子池回归要求至少 2 个因子，否则没有意义
        if pool is None or len(factor_cols) < 2:
            self.log.warning("pool.too_few", count=len(factor_cols), msg="因子池数量太少，无法进行回归")
            return len(factor_cols)

        os.makedirs(os.path.dirname(self.factor_pool_path), exist_ok=True)
        pool.to_parquet(self.factor_pool_path)
        self.log.info("pool.saved", factors=len(factor_cols), rows=len(pool), msg="保存因子池数据")

        # 原始因子池只作存档，不参与回归；有多少存多少，失败不影响主流程
        if raw_pool is not None:
            raw_cols = [c for c in raw_pool.columns if c not in ("date", "instrument")]
            raw_pool.to_parquet(self.factor_pool_raw_path)
            self.log.debug("pool.raw_saved", factors=len(raw_cols), msg="保存原始因子池数据")

        return len(factor_cols)

    # ---- 因子池回归 -------------------------------------------------------

    def run_regression(self) -> float:
        """跑一次因子池回归，产出 per_factor_scores（落盘到 leaderboard_reg_csv）。

        JUDGE_REG 模板不依赖任何用户代码，只读取因子池做回归，因此作为独立脚本直接运行，
        运行目录放在榜单目录下，与单因子分析的提交目录互不干扰。

        返回回归耗时（秒，供 on_tick 汇总成一行日志）。
        """
        runner = LocalProcessUserRunner(
            submission_id="_regression",
            files={"judge_runner.py": self.JUDGE_REG},
            cmd=["python3", "-c", "from judge_runner import judge_runner_main; judge_runner_main()"],
            runner_dir=self.regression_runner_dir,
        )
        # 回归子进程在主循环线程里同步跑，可能耗时数十秒到几分钟。把起始时刻挂到 self，
        # 心跳线程据此显示「回归已跑 N 秒」（实时进行中）；结束后清空 _reg_start 并把耗时
        # 存入 _reg_last，心跳空闲时显示上次耗时。见 heartbeat_fields。
        self._reg_start = time.time()
        try:
            with log_timer() as elapsed:
                runner.run(_raise=True)
        finally:
            self._reg_last = round(time.time() - self._reg_start, 1)
            self._reg_start = None
        self.log.debug("regression.done", elapsed_ms=elapsed(), msg="因子池回归完成")
        return elapsed() / 1000.0

    # ---- B 项得分 ---------------------------------------------------------

    def load_b_scores(self) -> dict[str, float]:
        """读取因子池回归产物并算出每个因子的 B 项得分（sid -> B_i）。

        回归产物（per_factor_scores）落盘在 leaderboard_reg_csv，列含
        factor / model_score / selection_rate。其中 factor 即提交 id。
        回归尚未产出（文件不存在/无法解析）时返回空 dict，调用方据此把 B_i 视为 0。
        具体的 ModelScore 百分位归一化逻辑见 scoring.compute_b_scores()。
        """
        if not os.path.exists(self.leaderboard_reg_csv):
            return {}

        # JUDGE_REG 以 to_csv 落盘，优先按 csv 读，兜底按 parquet（兼容历史旧文件）。
        reg = read_csv(self.leaderboard_reg_csv, logger=self.log)
        if reg is None:
            try:
                reg = pd.read_parquet(self.leaderboard_reg_csv)
            except Exception:
                return {}

        b_scores = scoring.compute_b_scores(reg)
        if not b_scores:
            self.log.warning("reg.bad_columns", columns=list(reg.columns), msg="回归得分缺少 factor/model_score 列")
        return b_scores
