"""15 分钟频率的单因子分析。

评估标签来自 ``cpt_gifp_2026_return``。每个截面以股票池内全部股票标签的
等权平均收益作为基准，个股标签先减去该截面均值，再用于 IC、分组收益
和多空组合分析。减去截面常数不改变 RankIC，并能隔离市场整体涨跌。
"""

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import structlog

from .data import load_stock_returns

logger = structlog.get_logger()

PERIODS_PER_YEAR = 16 * 242
# 兼容历史测试和调用方对 ``factoranalyze.analyzer`` 的 patch 路径。
analyzer = SimpleNamespace(load_stock_returns=load_stock_returns)


@dataclass(frozen=True)
class FactorScore:
    """单因子评估的五项原始指标（尚未做全场百分位排名）。"""

    ic_mean: float
    ic_ir: float
    sharpe_ratio: float
    max_drawdown: float
    turnover: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _safe_ir(values: Iterable[float]) -> float:
    """返回均值/样本标准差；退化序列返回 0，保证评估结果可序列化。"""
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return 0.0
    std = float(series.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return float(series.mean() / std)


class FactorAnalyze:
    """基于 15 分钟截面的单因子评估器。"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        factor_name: str = "factor",
        benchmark: str = "沪深300",
        group_number: int = 10,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.factor_name = factor_name
        self.benchmark = benchmark
        self.group_num = group_number

        self.merge_data = pd.DataFrame()
        self.group_data = pd.DataFrame()
        self.group_ret = pd.DataFrame()
        self.group_cumret = pd.DataFrame()
        self.section_ic = pd.Series(dtype="float64")
        self.section_volatility = pd.Series(dtype="float64")
        self.turnover_series = pd.DataFrame(
            columns=["low_turnover", "high_turnover", "turnover"]
        )

    def merge_related_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """合并股票标签，并以股票池截面平均收益生成超额收益。"""
        required = {"date", "instrument", self.factor_name}
        missing = required.difference(factor_data.columns)
        if missing:
            raise ValueError(f"因子数据缺少字段: {sorted(missing)}")

        factor = factor_data[["date", "instrument", self.factor_name]].copy()
        factor["date"] = pd.to_datetime(factor["date"], errors="coerce")
        stock_labels = analyzer.load_stock_returns(
            self.start_date, self.end_date
        ).copy()
        stock_labels["date"] = pd.to_datetime(
            stock_labels["date"], errors="coerce"
        )

        merged = factor.merge(
            stock_labels,
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        )
        # 基准收益来自完整股票标签池的截面均值。若某个时点缺失，应剔除
        # 完整截面，避免把股票绝对收益误当成超额收益。
        missing_benchmark_mask = merged["benchmark_return"].isna()
        missing_benchmark_dates = (
            merged.loc[missing_benchmark_mask, "date"]
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        if not missing_benchmark_dates.empty:
            logger.warning(
                "部分截面无法计算股票池平均收益，已剔除完整截面",
                sections=len(missing_benchmark_dates),
                rows=int(missing_benchmark_mask.sum()),
                dates=[timestamp.isoformat() for timestamp in missing_benchmark_dates],
            )
            merged = merged.loc[
                ~merged["date"].isin(missing_benchmark_dates)
            ].copy()
        merged["excess_return"] = merged["forward_return"] - merged["benchmark_return"]
        return merged.sort_values(["date", "instrument"]).reset_index(drop=True)

    def get_group_data(self, merged: pd.DataFrame) -> pd.DataFrame:
        """在每个 15 分钟截面内按因子秩等数量分组。"""
        frames = []
        for _, section in merged.groupby("date", sort=True):
            section = section.dropna(subset=[self.factor_name, "excess_return"]).copy()
            if len(section) < 2:
                continue
            ranks = section[self.factor_name].rank(method="first")
            bins = min(self.group_num, len(section))
            section["group"] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
            section = section.dropna(subset=["group"])
            section["group"] = section["group"].astype(int)
            section["max_group"] = int(section["group"].max())
            frames.append(section)
        if not frames:
            return merged.assign(group=np.nan, max_group=np.nan).iloc[0:0]
        return pd.concat(frames, ignore_index=True)

    def get_group_returns(self, group_data: pd.DataFrame) -> pd.DataFrame:
        """计算各组及最高组减最低组的同期收益。"""
        if group_data.empty:
            return pd.DataFrame(columns=["ls"], dtype="float64")
        grouped = (
            group_data.groupby(["date", "group"], observed=True)["excess_return"]
            .mean()
            .unstack("group")
            .sort_index()
        )
        low = group_data[group_data["group"].eq(0)].groupby("date")["excess_return"].mean()
        high = group_data[group_data["group"].eq(group_data["max_group"])].groupby("date")["excess_return"].mean()
        grouped["ls"] = high - low
        return grouped

    @staticmethod
    def get_group_cumulative_returns(group_returns: pd.DataFrame) -> pd.DataFrame:
        """所有分组及多空组合均按 ``(1 + r).cumprod() - 1`` 复利累计。"""
        if group_returns.empty:
            return group_returns.copy()
        return (1.0 + group_returns.fillna(0.0)).cumprod() - 1.0

    def get_section_ic(self, merged: pd.DataFrame) -> pd.Series:
        """每个 15 分钟截面的 Spearman RankIC。"""
        def rank_ic(section: pd.DataFrame) -> float:
            valid = section[[self.factor_name, "excess_return"]].dropna()
            if len(valid) < 2 or valid[self.factor_name].nunique() < 2 or valid["excess_return"].nunique() < 2:
                return np.nan
            return float(valid[self.factor_name].corr(valid["excess_return"], method="spearman"))

        values = {
            date: rank_ic(section)
            for date, section in merged.groupby("date", sort=True)
        }
        result = pd.Series(values, dtype="float64").dropna()
        result.index = pd.to_datetime(result.index)
        return result.rename("rank_ic")

    @staticmethod
    def _sharpe(returns: pd.Series) -> float:
        """按 16 期/日、242 日年化的零无风险利率夏普。"""
        values = pd.to_numeric(returns, errors="coerce").dropna()
        if len(values) < 2:
            return 0.0
        std = float(values.std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            return 0.0
        return float(values.mean() / std * np.sqrt(PERIODS_PER_YEAR))

    @staticmethod
    def get_max_drawdown(returns: pd.Series) -> float:
        """返回多空组合净值的最大回撤幅度，结果为非负数。"""
        values = pd.to_numeric(returns, errors="coerce").fillna(0.0)
        if values.empty:
            return 0.0
        nav = (1.0 + values).cumprod()
        running_peak = np.maximum.accumulate(
            np.concatenate(([1.0], nav.to_numpy(dtype=float)))
        )[1:]
        drawdown = nav.to_numpy(dtype=float) / running_peak - 1.0
        return float(-drawdown.min())

    def get_turnover_series(self, group_data: pd.DataFrame) -> pd.DataFrame:
        """计算相邻截面多空组合的单边换手率时序。

        每侧等权，单侧换手为 ``1 - 前后持仓交集权重``，最终取多头与空头
        换手的平均。隔夜边界不计入，以免把日间持仓变化混入日内指标。
        """
        if group_data.empty:
            return pd.DataFrame(
                columns=["low_turnover", "high_turnover", "turnover"],
                dtype="float64",
            )
        portfolios = []
        for date, section in group_data.groupby("date", sort=True):
            low = frozenset(section.loc[section["group"].eq(0), "instrument"])
            high = frozenset(section.loc[section["group"].eq(section["max_group"]), "instrument"])
            portfolios.append((pd.Timestamp(date), low, high))

        rows = []
        for previous, current in zip(portfolios, portfolios[1:]):
            prev_date, prev_low, prev_high = previous
            date, low, high = current
            if prev_date.normalize() != date.normalize():
                continue
            low_turnover = 1.0 - len(prev_low & low) / len(prev_low) if prev_low else 0.0
            high_turnover = 1.0 - len(prev_high & high) / len(prev_high) if prev_high else 0.0
            rows.append(
                {
                    "date": date,
                    "low_turnover": low_turnover,
                    "high_turnover": high_turnover,
                    "turnover": (low_turnover + high_turnover) / 2.0,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["low_turnover", "high_turnover", "turnover"],
                dtype="float64",
            )
        return pd.DataFrame(rows).set_index("date").sort_index()

    def get_turnover(self, group_data: pd.DataFrame) -> float:
        """计算相邻日内截面的平均单边换手率。"""
        turnover = self.get_turnover_series(group_data)
        return float(turnover["turnover"].mean()) if not turnover.empty else 0.0

    def plot(self, score: Optional[FactorScore] = None) -> None:
        """在 notebook 中展示 15 分钟单因子绩效报告。"""
        if self.group_cumret.empty or self.section_ic.empty:
            raise RuntimeError("请先调用 score(factor_data) 计算中间结果，再调用 plot()。")
        from . import render

        score_dict = score.to_dict() if score is not None else getattr(self, "_score_dict", {})
        render.render_report(
            group_cumret=self.group_cumret,
            group_ret=self.group_ret,
            section_ic=self.section_ic,
            section_volatility=self.section_volatility,
            turnover_series=self.turnover_series,
            factor_name=self.factor_name,
            score=score_dict,
        )

    def score(self, factor_data: pd.DataFrame, plot: bool = False) -> FactorScore:
        """返回 IC_mean、IC_IR、SR、turnover 和最大回撤五项原始值。"""
        self.merge_data = self.merge_related_data(factor_data)
        self.group_data = self.get_group_data(self.merge_data)
        self.group_ret = self.get_group_returns(self.group_data)
        self.group_cumret = self.get_group_cumulative_returns(self.group_ret)
        self.section_ic = self.get_section_ic(self.merge_data)
        self.section_volatility = (
            self.merge_data.groupby("date")["excess_return"].std().dropna()
        )
        self.section_volatility.index = pd.to_datetime(self.section_volatility.index)
        self.turnover_series = self.get_turnover_series(self.group_data)

        result = FactorScore(
            ic_mean=float(self.section_ic.mean()) if not self.section_ic.empty else 0.0,
            ic_ir=_safe_ir(self.section_ic),
            sharpe_ratio=self._sharpe(self.group_ret.get("ls", pd.Series(dtype=float))),
            max_drawdown=self.get_max_drawdown(
                self.group_ret.get("ls", pd.Series(dtype=float))
            ),
            turnover=(
                float(self.turnover_series["turnover"].mean())
                if not self.turnover_series.empty
                else 0.0
            ),
        )
        self._score_dict = result.to_dict()
        if plot:
            self.plot(result)
        logger.info("15 分钟单因子分析完成", **result.to_dict())
        return result
