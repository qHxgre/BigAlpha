from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    seed: int = 20260701
    table: str = "auto"
    table_candidates: List[str] = field(
        default_factory=lambda: [
            "bigalpha_2026_stock_bar5m",
            "bigalpha_2026_stock_bar1m",
            "bigalpha_2026_stock_bar30m",
            "bigalpha_2026_stock_bar15m",
            "bigalpha_2026_cn_stock_bar30m",
            "bigalpha_2026_cn_stock_bar15m",
            "bigalpha_2026_cn_stock_bar5m",
            "bigalpha_2026_cn_stock_bar1m",
            "bigalpha_2026_bar30m",
            "bigalpha_2026_bar15m",
            "bigalpha_2026_bar5m",
            "bigalpha_2026_bar1m",
            "cn_stock_bar30m",
            "cn_stock_bar15m",
            "cn_stock_bar5m",
            "cn_stock_bar1m",
            "stock_bar30m",
            "stock_bar15m",
            "stock_bar5m",
            "stock_bar1m",
        ]
    )
    train_start: str = "2019-01-01"
    train_end: str = "2023-12-31"
    predict_start: str = "2024-01-01"
    predict_end: str = "2024-12-31"

    date_col: str = "date"
    instrument_col: str = "instrument"
    datetime_col: str = "datetime"
    time_col_candidates: List[str] = field(
        default_factory=lambda: [
            "datetime",
            "date_time",
            "timestamp",
            "time",
            "dt",
            "bar_time",
            "trade_time",
            "update_time",
        ]
    )

    raw_feature_candidates: List[str] = field(
        default_factory=lambda: [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
            "num_trades",
            "pre_close",
            "bid1",
            "ask1",
            "bid1_volume",
            "ask1_volume",
        ]
    )
    max_fields: int = 12
    seq_len: int = 32
    max_rows_per_query: int = 5000000
    fast_query: bool = True
    bars_per_day: int = 8

    batch_size: int = 1024
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    valid_ratio: float = 0.15
    grad_clip: float = 1.0
    num_workers: int = 0

    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1

    artifacts_dir: str = "artifacts"
    model_path: str = "artifacts/model.pt"
    scaler_path: str = "artifacts/scaler.json"
    meta_path: str = "artifacts/meta.json"
    fallback_index_path: str = "artifacts/fallback_index.csv"
    output_path: str = "submission.csv"


CFG = Config()
