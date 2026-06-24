from dataclasses import asdict, dataclass


@dataclass
class ScoreMetrics:
    """端到端模型分数的四项核心评估指标。

    最终团队得分由这四项分别做全场百分位排名后等权相加：
        Score_final = 0.25 * Rank(ic_mean)
                    + 0.25 * Rank(ic_ir)
                    + 0.25 * Rank(sharpe_ratio)
                    + 0.25 * Rank(stress_ic_ir)
    """

    ic_mean: float       # 评估区间内截面 IC 均值
    ic_ir: float         # IC 序列的 IR（IC 均值 / IC 标准差）
    sharpe_ratio: float  # 多空 10 分组组合的年化夏普比率
    stress_ic_ir: float  # 分 regime（压力时段）评估的稳健性

    def to_dict(self) -> dict:
        return asdict(self)
