from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class FactorModelScore:
    """单因子在 Elastic Net 滚动回归下的得分。

    ModelScore = mean(|w|) / (std(|w|) + eps)
    """

    factor: str
    model_score: float
    abs_weight_mean: float
    abs_weight_std: float
    selection_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ElasticNetResult:
    """Elastic Net 滚动回归的完整结果。"""

    per_factor_scores: pd.DataFrame
    weights_history: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "per_factor_scores": self.per_factor_scores,
            "weights_history": self.weights_history,
        }
