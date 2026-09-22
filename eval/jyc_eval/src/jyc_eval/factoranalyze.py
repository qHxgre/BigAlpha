"""30 分钟频率的单因子分析。

评估标签来自 ``cpt_jyc_2026_vwap``。同一张表中的中证 1000 指数
（000852.SH）被用作基准，股票标签先减去同截面的指数标签，再用于
IC、分组收益和压力测试。减去截面常数不改变 RankIC，但可以让组合
收益和夏普反映超额收益。
"""

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import structlog

from .data import BM_DICT, load_evaluation_data

logger = structlog.get_logger()

PERIODS_PER_YEAR = 8 * 242
# 兼容历史测试和调用方对 ``factoranalyze.analyzer`` 的 patch 路径。
analyzer = SimpleNamespace(load_evaluation_data=load_evaluation_data)


@dataclass(frozen=True)
class FactorScore:
    """单因子评估的五项原始指标（尚未做全场百分位排名）。"""

    ic_mean: float
    ic_ir: float
    sharpe_ratio: float
    stress_stability: float
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
    """基于非重叠 30 分钟截面的单因子评估器。"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        factor_name: str = "factor",
        benchmark: str = "中证1000",
        group_number: int = 5,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.factor_name = factor_name
        self.benchmark_instrument = BM_DICT.get(benchmark, benchmark)
        self.group_num = group_number

        self.merge_data = pd.DataFrame()
        self.group_data = pd.DataFrame()
        self.group_ret = pd.DataFrame()
        self.group_cumret = pd.DataFrame()
        self.section_ic = pd.Series(dtype="float64")

    def merge_related_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """合并股票标签和同截面指数标签，并生成 30 分钟超额收益。"""
        required = {"date", "instrument", self.factor_name}
        missing = required.difference(factor_data.columns)
        if missing:
            raise ValueError(f"因子数据缺少字段: {sorted(missing)}")

        factor = factor_data[["date", "instrument", self.factor_name]].copy()
        factor["date"] = pd.to_datetime(factor["date"], errors="coerce")
        labels = analyzer.load_evaluation_data(self.start_date, self.end_date).copy()
        labels["date"] = pd.to_datetime(labels["date"], errors="coerce")

        benchmark = (
            labels.loc[
                labels["instrument"].eq(self.benchmark_instrument),
                ["date", "forward_return"],
            ]
            .drop_duplicates("date", keep="last")
            .rename(columns={"forward_return": "benchmark_return"})
        )
        stock_labels = labels.loc[
            ~labels["instrument"].eq(self.benchmark_instrument)
        ].drop_duplicates(["date", "instrument"], keep="last")

        merged = factor.merge(
            stock_labels,
            on=["date", "instrument"],
            how="left",
            validate="one_to_one",
        ).merge(benchmark, on="date", how="left", validate="many_to_one")
        # 在本地或旧数据缺少指数标签时，仍可计算股票绝对收益指标。
        missing_benchmark = int(merged["benchmark_return"].isna().sum())
        if missing_benchmark:
            logger.warning("部分截面缺少指数 VWAP 收益，按 0 处理", rows=missing_benchmark)
        merged["benchmark_return"] = merged["benchmark_return"].fillna(0.0)
        merged["excess_return"] = merged["forward_return"] - merged["benchmark_return"]
        return merged.sort_values(["date", "instrument"]).reset_index(drop=True)

    def get_group_data(self, merged: pd.DataFrame) -> pd.DataFrame:
        """在每个 30 分钟截面内按因子秩等数量分组。"""
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
        """计算各组及最高组减最低组的同期 30 分钟收益。"""
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

    def get_section_ic(self, merged: pd.DataFrame) -> pd.Series:
        """每个 30 分钟截面的 Spearman RankIC。"""
        def rank_ic(section: pd.DataFrame) -> float:
            valid = section[[self.factor_name, "excess_return"]].dropna()
            if len(valid) < 2 or valid[self.factor_name].nunique() < 2 or valid["excess_return"].nunique() < 2:
                return np.nan
            return float(valid[self.factor_name].corr(valid["excess_return"], method="spearman"))

        result = merged.groupby("date", sort=True).apply(rank_ic).dropna()
        result.index = pd.to_datetime(result.index)
        return result.astype(float).rename("rank_ic")

    @staticmethod
    def _sharpe(returns: pd.Series) -> float:
        """按约 8 期/日、242 日年化的零无风险利率夏普。"""
        values = pd.to_numeric(returns, errors="coerce").dropna()
        if len(values) < 2:
            return 0.0
        std = float(values.std(ddof=1))
        if not np.isfinite(std) or std <= 0:
            return 0.0
        return float(values.mean() / std * np.sqrt(PERIODS_PER_YEAR))

    def get_stress_stability(self, merged: pd.DataFrame, ic: pd.Series) -> float:
        """衡量高/低波动 regime 下 IC_IR 的一致性。

        先以每个截面股票超额收益的横截面标准差刻画市场波动，并按中位数
        分成高、低波动两组。得分为两组 IR 中较弱者，因而只有在两种市场
        状态下都稳定时才会较高。
        """
        section_vol = merged.groupby("date")["excess_return"].std().dropna()
        common = ic.index.intersection(section_vol.index)
        if len(common) < 2:
            return 0.0
        section_vol = section_vol.loc[common]
        aligned_ic = ic.loc[common]
        median = float(section_vol.median())
        low_ir = _safe_ir(aligned_ic[section_vol <= median])
        high_ir = _safe_ir(aligned_ic[section_vol > median])
        return float(min(low_ir, high_ir))

    def get_turnover(self, group_data: pd.DataFrame) -> float:
        """计算相邻截面多空组合的平均单边换手率。

        每侧等权，单侧换手为 ``1 - 前后持仓交集权重``，最终取多头与空头
        换手的平均。隔夜边界不计入，以免把日间持仓变化混入日内指标。
        """
        if group_data.empty:
            return 0.0
        portfolios = []
        for date, section in group_data.groupby("date", sort=True):
            low = frozenset(section.loc[section["group"].eq(0), "instrument"])
            high = frozenset(section.loc[section["group"].eq(section["max_group"]), "instrument"])
            portfolios.append((pd.Timestamp(date), low, high))

        turnovers = []
        for previous, current in zip(portfolios, portfolios[1:]):
            prev_date, prev_low, prev_high = previous
            date, low, high = current
            if prev_date.normalize() != date.normalize():
                continue
            low_turnover = 1.0 - len(prev_low & low) / len(prev_low) if prev_low else 0.0
            high_turnover = 1.0 - len(prev_high & high) / len(prev_high) if prev_high else 0.0
            turnovers.append((low_turnover + high_turnover) / 2.0)
        return float(np.mean(turnovers)) if turnovers else 0.0

    def score(self, factor_data: pd.DataFrame, plot: bool = False) -> FactorScore:
        """返回 IC_mean、IC_IR、SR、stress 和 turnover 五项原始值。"""
        self.merge_data = self.merge_related_data(factor_data)
        self.group_data = self.get_group_data(self.merge_data)
        self.group_ret = self.get_group_returns(self.group_data)
        self.group_cumret = self.group_ret.fillna(0.0).cumsum()
        self.section_ic = self.get_section_ic(self.merge_data)

        result = FactorScore(
            ic_mean=float(self.section_ic.mean()) if not self.section_ic.empty else 0.0,
            ic_ir=_safe_ir(self.section_ic),
            sharpe_ratio=self._sharpe(self.group_ret.get("ls", pd.Series(dtype=float))),
            stress_stability=self.get_stress_stability(self.merge_data, self.section_ic),
            turnover=self.get_turnover(self.group_data),
        )
        if plot:
            logger.warning("分钟级评估暂不提供图形报告，已返回五项指标")
        logger.info("30 分钟单因子分析完成", **result.to_dict())
        return result
