"""单因子评估（A 项）

按 docs/因子挖掘_介绍_20260525.md "单因子得分（A 项）" 实现：

    FACTOR = 0.25 * Rank_IC_mean + 0.25 * Rank_IC_IR
           + 0.25 * Rank_SR      + 0.25 * Rank_Stress

输出原始指标（IC_mean、IC_IR、long-short SR、Stress IC_IR），
排名归一化由汇总层（todo.py）完成 —— 单因子场景下 Rank_X 退化为按指标本身归一化。

Stress 行情定义：以中证 1000（基准）月度收益作为分位标准，
取月度收益最低 20% 的月份作为压力期，计算压力期内的 IC_IR。
"""

import dai
import empyrical
import structlog
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = structlog.get_logger()


# 评估期间无风险年化（与原实现保持一致）
RISK_FREE_DAILY = 0.035 / 242
# 分组数量（多空收益用最高组 - 最低组）
GROUP_NUM = 5
# 基准指数（中证 1000）
BENCHMARK_CODE = "000852.SH"
# 压力期分位（基准月度收益的最低 20%）
STRESS_QUANTILE = 0.2


class FactorAnalyze:
    def __init__(self, start_date: str, end_date: str) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.group_num = GROUP_NUM

        # 中间产物（供绘图使用）
        self.merged: pd.DataFrame = pd.DataFrame()
        self.daily_ic: pd.Series = pd.Series(dtype=float)
        self.group_cumret: pd.DataFrame = pd.DataFrame()
        self.long_short_ret: pd.Series = pd.Series(dtype=float)

    # ---------- 数据准备 ----------

    def merge_return_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """合并 T+1 日频收益率。

        因子按文档为日频；下期收益取下一交易日的日收益（次日 close / 当日 close - 1）。
        """
        after_end_date = (
            datetime.strptime(self.end_date, "%Y-%m-%d") + timedelta(days=15)
        ).strftime("%Y-%m-%d")
        instruments = factor_data["instrument"].unique().tolist()

        sql = """
        WITH cte_status as (
            SELECT date as trading_day, instrument
            FROM cn_stock_status
            WHERE price_limit_status = 2
        ),
        cte_bar1d as (
            SELECT date as trading_day, instrument, close
            FROM cn_stock_bar1d
            WHERE volume > 0
        )
        SELECT
            trading_day,
            instrument,
            m_lead(close, 1) / close - 1 AS ret
        FROM cte_bar1d
        SEMI JOIN cte_status USING (trading_day, instrument)
        ORDER BY trading_day, instrument
        """
        ret_df = dai.query(
            sql,
            filters={
                "date": [f"{self.start_date} 00:00:00", f"{after_end_date} 23:59:59"],
                "instrument": instruments,
            },
        ).df()

        df = factor_data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["trading_day"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d"))
        ret_df["trading_day"] = pd.to_datetime(ret_df["trading_day"])

        merged = pd.merge(
            df[["trading_day", "instrument", "factor"]],
            ret_df[["trading_day", "instrument", "ret"]],
            how="left",
            on=["trading_day", "instrument"],
        )
        merged = merged.dropna(subset=["factor"])
        return merged

    # ---------- IC ----------

    def cpt_ic(self, merged: pd.DataFrame) -> pd.Series:
        """每日截面 Rank IC（Spearman），用 Pearson(rank(factor), rank(ret)) 等价实现。"""
        df = merged.dropna(subset=["factor", "ret"]).copy()
        if df.empty:
            return pd.Series(dtype=float)

        def _rank_ic(g: pd.DataFrame) -> float:
            if len(g) < 5:
                return np.nan
            return float(g["factor"].rank().corr(g["ret"].rank()))

        ic = df.groupby("trading_day").apply(_rank_ic)
        ic.index = pd.to_datetime(ic.index)
        ic.name = "ic"
        return ic.dropna()

    @staticmethod
    def cpt_ic_metrics(ic: pd.Series) -> tuple:
        """返回 (IC_mean, IC_IR)。"""
        if ic is None or ic.empty:
            return np.nan, np.nan
        ic_mean = float(ic.mean())
        ic_std = float(ic.std(ddof=1))
        ic_ir = ic_mean / ic_std * np.sqrt(252) if ic_std > 0 else np.nan
        return ic_mean, ic_ir

    # ---------- 分组 / 多空夏普 ----------

    def cpt_group_returns(self, merged: pd.DataFrame) -> tuple:
        """
        分组等权收益：将每日因子按 group_num 分组，组 0 为最低、组 group_num-1 为最高。
        多空收益 = 高组 - 低组（T+1 实现，整体 shift 1 期）。
        """
        df = merged.dropna(subset=["factor", "ret"]).copy()
        if df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        def _cut(g: pd.DataFrame) -> pd.DataFrame:
            if g["factor"].nunique() < self.group_num:
                return g.iloc[0:0]
            g = g.copy()
            g["group"] = pd.qcut(
                g["factor"], q=self.group_num, labels=False, duplicates="drop"
            )
            return g.dropna(subset=["group"])

        cut_df = df.groupby("trading_day", group_keys=False).apply(_cut)
        if cut_df.empty:
            return pd.DataFrame(), pd.Series(dtype=float)

        cut_df["group"] = cut_df["group"].astype(int).astype(str)
        group_ret = (
            cut_df.groupby(["trading_day", "group"])["ret"].mean().unstack("group")
        )
        # 因子用 T 日截面、收益已经是 T+1 → 不需要再 shift
        long_short = group_ret[str(self.group_num - 1)] - group_ret["0"]
        long_short.name = "ls"
        group_cum = (1 + group_ret.fillna(0)).cumprod()
        group_cum["ls"] = (1 + long_short.fillna(0)).cumprod()
        return group_cum, long_short

    @staticmethod
    def cpt_sharpe(ret: pd.Series) -> float:
        ret = pd.Series(ret).dropna()
        if len(ret) < 3:
            return np.nan
        return float(empyrical.sharpe_ratio(ret.values, risk_free=RISK_FREE_DAILY))

    # ---------- Stress ----------

    def _benchmark_monthly_ret(self) -> pd.Series:
        """中证 1000 的日收益率序列。"""
        sql = f"""
        SELECT date as trading_day, close
        FROM cn_stock_index_bar1d
        WHERE instrument = '{BENCHMARK_CODE}'
        ORDER BY trading_day
        """
        bench = dai.query(
            sql,
            filters={"date": [f"{self.start_date} 00:00:00", f"{self.end_date} 23:59:59"]},
        ).df()
        if bench.empty:
            return pd.Series(dtype=float)
        bench["trading_day"] = pd.to_datetime(bench["trading_day"])
        bench = bench.sort_values("trading_day").set_index("trading_day")
        bench["ret"] = bench["close"].pct_change()
        return bench["ret"].dropna()

    def cpt_stress_ic_ir(self, ic: pd.Series) -> float:
        """Stress IC_IR：基准月度收益最低 20% 月份内的 IC_IR。"""
        if ic is None or ic.empty:
            return np.nan
        bench_ret = self._benchmark_monthly_ret()
        if bench_ret.empty:
            return np.nan

        bench_monthly = (1 + bench_ret).resample("M").prod() - 1
        if bench_monthly.empty:
            return np.nan

        threshold = bench_monthly.quantile(STRESS_QUANTILE)
        stress_months = set(
            bench_monthly[bench_monthly <= threshold].index.to_period("M")
        )
        if not stress_months:
            return np.nan

        ic_periods = ic.index.to_period("M")
        mask = pd.Series([p in stress_months for p in ic_periods], index=ic.index)
        stress_ic = ic[mask]
        if len(stress_ic) < 5:
            return np.nan
        std = stress_ic.std(ddof=1)
        return float(stress_ic.mean() / std * np.sqrt(252)) if std > 0 else np.nan

    # ---------- 入口 ----------

    def validate(self, factor_data: pd.DataFrame, factor_name: str = "factor") -> dict:
        """计算 A 项四个原始指标。"""
        t0 = datetime.now()
        merged = self.merge_return_data(factor_data.rename(columns={factor_name: "factor"}))
        self.merged = merged
        t1 = datetime.now()
        logger.info(f"[A项] 合并下期收益, 耗时: {round((t1 - t0).total_seconds(), 4)} 秒")

        ic = self.cpt_ic(merged)
        self.daily_ic = ic
        ic_mean, ic_ir = self.cpt_ic_metrics(ic)
        t2 = datetime.now()
        logger.info(f"[A项] IC: mean={ic_mean:.4f}, IR={ic_ir:.4f}, 耗时: {round((t2 - t1).total_seconds(), 4)} 秒")

        group_cum, long_short = self.cpt_group_returns(merged)
        self.group_cumret = group_cum
        self.long_short_ret = long_short
        ls_sharpe = self.cpt_sharpe(long_short)
        t3 = datetime.now()
        logger.info(f"[A项] 多空 SR={ls_sharpe:.4f}, 耗时: {round((t3 - t2).total_seconds(), 4)} 秒")

        stress_ir = self.cpt_stress_ic_ir(ic)
        t4 = datetime.now()
        logger.info(f"[A项] Stress IC_IR={stress_ir:.4f}, 耗时: {round((t4 - t3).total_seconds(), 4)} 秒")

        return {
            "ic_mean": float(ic_mean) if pd.notna(ic_mean) else np.nan,
            "ic_ir": float(ic_ir) if pd.notna(ic_ir) else np.nan,
            "ls_sharpe": float(ls_sharpe) if pd.notna(ls_sharpe) else np.nan,
            "stress_ic_ir": float(stress_ir) if pd.notna(stress_ir) else np.nan,
        }
