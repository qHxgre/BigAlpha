from dataclasses import asdict, dataclass, fields

import pandas as pd


@dataclass
class FactorScore:
    ic_mean: float
    ic_ir: float
    sharpe_ratio: float
    stress_ic_ir: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnoverPerf:
    turnover: float


@dataclass
class ICPerf:
    ic: float
    ir: float
    ic_3: float
    ic_10: float
    ic_21: float
    ic_63: float
    ic_126: float
    ic_252: float


@dataclass
class BasicPerf:
    return_ratio: float
    annual_return_ratio: float
    ex_return_ratio: float
    ex_annual_return_ratio: float
    sharp_ratio: float
    return_volatility: float
    information_ratio: float
    max_drawdown: float
    win_percent: float
    trading_days: float
    ret_3: float
    ret_10: float
    ret_21: float
    ret_63: float
    ret_126: float
    ret_252: float


@dataclass
class Performance:
    return_ratio: list
    annual_return_ratio: list
    ex_return_ratio: list
    ex_annual_return_ratio: list
    sharp_ratio: list
    return_volatility: list
    max_drawdown: list
    win_percent: list
    trading_days: list

    def to_dataframe(self):
        data_dict = asdict(self)
        df = pd.DataFrame(data_dict)
        return df


@dataclass
class SummaryPerf:
    portfolio: str
    basic_perf: BasicPerf
    ic_perf: ICPerf
    turnover_perf: TurnoverPerf

    def __post_init__(self):
        for field in fields(BasicPerf):
            setattr(self, field.name, getattr(self.basic_perf, field.name))
        for field in fields(ICPerf):
            setattr(self, field.name, getattr(self.ic_perf, field.name))
        for field in fields(TurnoverPerf):
            setattr(self, field.name, getattr(self.turnover_perf, field.name))

    def to_dataframe(self):
        flat_data = {"portfolio": self.portfolio}
        flat_data.update(
            {
                f"{field.name}": getattr(self.basic_perf, field.name)
                for field in fields(BasicPerf)
            }
        )
        flat_data.update(
            {
                f"{field.name}": getattr(self.ic_perf, field.name)
                for field in fields(ICPerf)
            }
        )
        flat_data.update(
            {
                f"{field.name}": getattr(self.turnover_perf, field.name)
                for field in fields(TurnoverPerf)
            }
        )
        df = pd.DataFrame([flat_data])
        return df
