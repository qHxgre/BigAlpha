"""本训练以聚合后的 2h 行情和盘口为主进行建模。

这样既降低分钟级噪音和数据量，也保留了每个交易日两根 bar 的短期状态。模型训练分三步：
(1) 将 1 分钟行情与盘口聚合为 2h bar，统一复权并构造慢变 context 状态
(2) 在四个市场状态区域中，以 17 条可解释交互信号做 Fama-French 横截面 OLS
(3) 由 4 个 regime context 驱动有界 MLP，对 17 条基座信号权重进行动态校准

注意：OLS 使用平台 10 个风格因子仅作为辅助，提前排除它们可解释的部分；
回归完成后这些风格因子被丢弃，未进入 MLP 校准流程，推理也不依赖这些因子，符合比赛规定。
训练时用 2020--2022 年选择基座与校准层、2023 年验证训练轮次；确定轮次后在
2019--2024 全样本重新拟合，得到最终提交模型。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import dai


START_DATE = "2019-01-01 00:00:00"
END_DATE = "2024-12-31 23:59:59"
SELECTION_START, SELECTION_END = "2020-10-01", "2022-12-31"
VALIDATION_START, VALIDATION_END = "2023-01-01", "2023-12-31"
OUTPUT_PATH = Path(__file__).with_name("fama_dynamic17_plus.json")
SEED = 20260802
ADJUSTMENT_CAP = 0.32

SIGNAL_COLS = [
    "direction", "uptrend_ma10_gap_score", "uptrend_extension_score",
    "uptrend_pullback_score", "uptrend_ma5up_instability_score",
    "uptrend_ma5up_high_close_score", "uptrend_ma5up_low_volume_score",
    "uptrend_ma5down_low_volume_score", "downtrend_ma5down_low_close_score",
    "downtrend_low_rsi10_score", "downtrend_ma5up_m10_down_score",
    "uptrend_ma5up_imbalance_score", "downtrend_ma5down_sell_pressure_score",
    "downtrend_ma5up_imbalance_score", "uptrend_ma5up_volatility_score",
    "uptrend_ma5down_volatility_score", "downtrend_ma5down_volatility_score",
]
POSITIVE_COLS = {
    "uptrend_ma5up_instability_score", "uptrend_ma5up_high_close_score",
    "uptrend_ma5up_low_volume_score", "uptrend_ma5down_low_volume_score",
    "downtrend_ma5down_low_close_score",
}
CONTEXT_COLS = ["ma30_relative", "ma60_relative", "had_limit_up_prev9", "prev9_limit_up_price_gap"]
STYLE_CONTROL_COLS = [
    "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL",
    "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE",
]


# 1. 读取平台行情与盘口，构造统一复权后的 2h bar 和四个慢变 context。
def fetch_training_data(start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch just the raw fields required by the 17 signals and four contexts."""
    filters = {"date": [start_date, end_date]}
    universe = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments", filters=filters,
    ).df()
    if universe.empty:
        raise ValueError("bigalpha_2026_instruments returned no rows")
    universe["trading_day"] = pd.to_datetime(universe["date"]).dt.normalize()
    universe = universe[["trading_day", "instrument"]].drop_duplicates()
    instruments = sorted(universe["instrument"].unique())

    def book_expression(side: str) -> str:
        return " + ".join(
            f"(1.0 * COALESCE({side}_volume{level}, 0) / {level})"
            for level in range(1, 6)
        )

    bar_filters = {"date": [start_date, end_date], "instrument": instruments}
    bars_sql = f"""
    SELECT
        MAX(date) AS date,
        instrument,
        ARG_MAX(adjust_factor, date) AS adjust_factor,
        ARG_MAX(pre_close, date) AS pre_close,
        ARG_MIN(open, date) AS open,
        MAX(high) AS high,
        MIN(low) AS low,
        ARG_MAX(close, date) AS close,
        SUM(volume) AS volume,
        AVG({book_expression('ask')}) AS ask_volume_w,
        AVG({book_expression('bid')}) AS bid_volume_w
    FROM (
        SELECT
            *,
            date_trunc('day', date)::DATE AS trading_day,
            FLOOR((ROW_NUMBER() OVER (
                PARTITION BY instrument, date_trunc('day', date)::DATE ORDER BY date
            ) - 1) / 120) AS bucket_id
        FROM bigalpha_2026_stock_bar1m
    )
    GROUP BY instrument, trading_day, bucket_id
    ORDER BY instrument, date
    """
    volatility_sql = """
    WITH bucketed AS (
        SELECT
            date,
            instrument,
            close,
            date_trunc('day', date)::DATE AS trading_day,
            FLOOR((ROW_NUMBER() OVER (
                PARTITION BY instrument, date_trunc('day', date)::DATE ORDER BY date
            ) - 1) / 120) AS bucket_id
        FROM bigalpha_2026_stock_bar1m
    ), lagged_close AS (
        SELECT
            date,
            instrument,
            trading_day,
            bucket_id,
            close,
            LAG(close) OVER (
                PARTITION BY instrument, trading_day, bucket_id ORDER BY date
            ) AS prev_close
        FROM bucketed
    )
    SELECT
        MAX(date) AS date,
        instrument,
        SQRT(SUM(CASE WHEN close > 0 AND prev_close > 0
            THEN LN(close / prev_close) * LN(close / prev_close) END)) AS realized_vol_2h
    FROM lagged_close
    GROUP BY instrument, trading_day, bucket_id
    ORDER BY instrument, date
    """
    bars = dai.query(bars_sql, filters=bar_filters).df()
    if bars.empty:
        raise ValueError("bigalpha_2026_stock_bar1m returned no 2h bars")
    volatility = dai.query(volatility_sql, filters=bar_filters).df()
    bars["date"] = pd.to_datetime(bars["date"])
    if volatility.empty:
        bars["realized_vol_2h"] = np.nan
    else:
        volatility["date"] = pd.to_datetime(volatility["date"])
        bars = bars.merge(
            volatility[["date", "instrument", "realized_vol_2h"]],
            on=["date", "instrument"], how="left", validate="one_to_one",
        )
    for col in bars.columns.difference(["date", "instrument"]):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")

    exposure = dai.query(
        f"SELECT date, instrument, {', '.join(STYLE_CONTROL_COLS)} FROM bigalpha_2026_exposure",
        filters=filters,
    ).df()
    if exposure.empty:
        raise ValueError("bigalpha_2026_exposure returned no rows")
    exposure["trading_day"] = pd.to_datetime(exposure["date"]).dt.normalize()
    exposure = exposure[["trading_day", "instrument", *STYLE_CONTROL_COLS]].drop_duplicates(
        ["trading_day", "instrument"]
    )
    return bars.replace([np.inf, -np.inf], np.nan), universe, exposure


def zscore(value: pd.Series) -> pd.Series:
    std = value.std(ddof=0)
    return pd.Series(0.0, index=value.index) if not np.isfinite(std) or std == 0 else (value - value.mean()) / std


def soft_clip(value: float, threshold: float) -> float:
    magnitude = abs(value)
    return value if magnitude <= threshold else np.sign(value) * (threshold + np.log1p(magnitude - threshold))


def add_limit_up_context(daily: pd.DataFrame) -> pd.DataFrame:
    """Prior-nine-day limit-up state from raw prices, with adjusted carry price."""
    daily = daily.sort_values(["instrument", "trading_day"]).copy()
    code = daily["instrument"].str.split(".", n=1).str[0]
    exchange = daily["instrument"].str.rsplit(".", n=1).str[-1]
    limit_bps = np.full(len(daily), 1000, dtype=np.int64)
    growth = code.str.startswith(("300", "301")) & (daily["trading_day"] >= pd.Timestamp("2020-08-24"))
    limit_bps[growth.to_numpy()] = 2000
    limit_bps[code.str.startswith(("688", "689")).to_numpy()] = 2000
    limit_bps[exchange.eq("BJ").to_numpy()] = 3000
    pre = np.rint(daily["pre_close"].fillna(0).to_numpy() * 100).astype(np.int64)
    close = np.rint(daily["close"].fillna(0).to_numpy() * 100).astype(np.int64)
    ceiling = np.maximum((pre * (10_000 + limit_bps) + 5_000) // 10_000, pre + 1)
    daily["limit_up"] = daily["pre_close"].gt(0) & daily["close"].gt(0) & (close == ceiling)
    daily["_n"] = daily.groupby("instrument", sort=False).cumcount()
    daily["_limit_n"] = daily["_n"].where(daily["limit_up"])
    daily["_limit_close"] = (daily["close"] * daily["adjust_factor"]).where(daily["limit_up"])
    prior_n = daily.groupby("instrument", sort=False)["_limit_n"].transform(lambda x: x.shift(1).ffill())
    prior_close = daily.groupby("instrument", sort=False)["_limit_close"].transform(lambda x: x.shift(1).ffill())
    daily["had_limit_up_prev9"] = daily.groupby("instrument", sort=False)["limit_up"].transform(
        lambda x: x.shift(1).rolling(9, min_periods=1).max()
    ).fillna(False).astype(float)
    daily["prev9_limit_up_price_gap"] = daily["close"] * daily["adjust_factor"] - prior_close.where(
        (daily["_n"] - prior_n).le(9), np.inf
    )
    return daily


def prepare_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply unified price adjustment and add the four dynamic contexts."""
    bars = raw.sort_values(["instrument", "date"], kind="stable").copy()
    bars["trading_day"] = pd.to_datetime(bars["date"]).dt.normalize()
    raw_daily = bars.groupby(["instrument", "trading_day"], as_index=False, sort=False).tail(1)[
        ["instrument", "trading_day", "pre_close", "close", "adjust_factor"]
    ]
    raw_daily = add_limit_up_context(raw_daily)
    bars = bars.merge(raw_daily[["instrument", "trading_day", "had_limit_up_prev9", "prev9_limit_up_price_gap"]],
                      on=["instrument", "trading_day"], how="left", validate="many_to_one")
    bars[["open", "high", "low", "close"]] = bars[["open", "high", "low", "close"]].multiply(
        bars["adjust_factor"], axis="index"
    )
    adjusted_daily = bars.groupby(["instrument", "trading_day"], as_index=False, sort=False).tail(1)[
        ["instrument", "trading_day", "close"]
    ].sort_values(["instrument", "trading_day"], kind="stable")
    for window, col in ((30, "ma30_relative"), (60, "ma60_relative")):
        average = adjusted_daily.groupby("instrument", sort=False)["close"].transform(
            lambda x: x.rolling(window, min_periods=window).mean()
        )
        adjusted_daily[col] = adjusted_daily["close"] / average - 1.0
    bars = bars.merge(
        adjusted_daily[["instrument", "trading_day", "ma30_relative", "ma60_relative"]],
        on=["instrument", "trading_day"], how="left", validate="many_to_one",
    )
    bars["imbalance_2h"] = (bars["bid_volume_w"] - bars["ask_volume_w"]) / (
        bars["bid_volume_w"] + bars["ask_volume_w"]
    )
    return bars.replace([np.inf, -np.inf], np.nan)


def regression_state(close: np.ndarray) -> tuple[float, float, float]:
    x = np.arange(len(close), dtype=float)
    centered_x = x - x.mean()
    mean_close = close.mean()
    slope = np.dot(centered_x, close - mean_close) / np.dot(centered_x, centered_x)
    intercept = mean_close - slope * x.mean()
    fitted = intercept + slope * x
    total = np.square(close - close.mean()).sum()
    r_squared = 1.0 if total == 0 else 1.0 - np.square(close - fitted).sum() / total
    instability = 1.0 / np.clip(r_squared, 1e-6, 1.0)
    move = slope * (len(close) - 1)
    deviation = (close[-1] - fitted[-1]) / move if move != 0 else 0.0
    return slope, instability, deviation


# 2. 构建 17 条可解释信号，并拟合逐日 Fama-French 横截面 OLS 权重。
# 信号以 G20 趋势方向和 MA5 上下划分四个区域；每条 signal 通过门控交互项只在其
# 已验证的区域发挥作用，从而提高线性模型对不同状态的拟合能力，并保证四个区域均有覆盖。
# 每日 OLS 同时加入 10 个风格因子作为辅助控制，平均后的 17 条信号系数构成基座；
# 风格因子随后完全丢弃，不进入动态校准或最终推理模型。
# 参数选择阶段使用 2020--2022 训练、2023 验证；确定训练轮次后，OLS 基座会在
# 2019--2024 全部可用样本上重新估计。
def stock_signals(stock: pd.DataFrame) -> pd.DataFrame:
    """The exact selected 17 Fama signals; no research candidates."""
    stock = stock.sort_values("date").reset_index(drop=True)
    rows = []
    close, high, low, volume = (stock[col].to_numpy(float) for col in ("close", "high", "low", "volume"))
    for i in range(20, len(stock)):
        slope, instability, deviation = regression_state(close[i - 19:i + 1])
        g20_up, g20_down = float(slope > 0), float(slope < 0)
        ma5 = close[i - 5:i].mean()
        g5_up, g5_down = float(close[i] >= ma5), float(close[i] < ma5)
        relative_volume = volume[i] / np.nanmean(volume[i - 20:i]) - 1.0
        low_volume = np.clip(-relative_volume, 0.0, 1.0)
        ma10 = close[i - 9:i + 1].mean()
        ma10_gap = (ma10 - close[i]) / ma10 if ma10 else np.nan
        m10 = np.log(close[i] / close[i - 10])
        high_close = (close[i] - low[i]) / (high[i] - low[i]) if high[i] > low[i] else 0.0
        low_close = (high[i] - close[i]) / (high[i] - low[i]) if high[i] > low[i] else 0.0
        rsi_delta = np.diff(close[i - 10:i + 1])
        gain, loss = np.clip(rsi_delta, 0, None).mean(), np.clip(-rsi_delta, 0, None).mean()
        rsi10 = 100.0 * gain / (gain + loss + 1e-12)
        deviation, instability = soft_clip(deviation, 5.0), soft_clip(instability, 10.0)
        imbalance, volatility = stock.at[i, "imbalance_2h"], max(stock.at[i, "realized_vol_2h"], 0.0)
        rows.append([
            stock.at[i, "date"], stock.at[i, "instrument"], np.sign(slope),
            g20_up * ma10_gap, g20_up * max(deviation, 0.0), g20_up * max(-deviation, 0.0),
            g20_up * g5_up * instability, g20_up * g5_up * high_close,
            g20_up * g5_up * low_volume, g20_up * g5_down * low_volume,
            g20_down * g5_down * low_close, g20_down * min((rsi10 - 40.0) / 40.0, 0.0),
            g20_down * g5_up * min(m10, 0.0), g20_up * g5_up * imbalance,
            g20_down * g5_down * min(imbalance, 0.0), g20_down * g5_up * imbalance,
            g20_up * g5_up * volatility, g20_up * g5_down * volatility, g20_down * g5_down * volatility,
        ])
    return pd.DataFrame(rows, columns=["date", "instrument", *SIGNAL_COLS])


def build_sample(bars: pd.DataFrame, universe: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    pieces = [stock_signals(x) for _, x in bars.groupby("instrument", sort=False)]
    pieces = [piece for piece in pieces if not piece.empty]
    if not pieces:
        raise ValueError("no stock had a continuous 20-bar history for feature construction")
    features = pd.concat(pieces, ignore_index=True)
    features["trading_day"] = pd.to_datetime(features["date"]).dt.normalize()
    features = features.sort_values("date").groupby(["instrument", "trading_day"], as_index=False).tail(1)
    contexts = bars.sort_values("date").groupby(["instrument", "trading_day"], as_index=False).tail(1)[
        ["date", "instrument", *CONTEXT_COLS]
    ]
    features = features.merge(contexts, on=["date", "instrument"], how="left", validate="one_to_one")
    daily = bars.sort_values(["instrument", "date"]).groupby(["instrument", "trading_day"], as_index=False).agg(
        open=("open", "first"), close=("close", "last")
    ).sort_values(["instrument", "trading_day"])
    market_day = {day: index for index, day in enumerate(sorted(universe["trading_day"].unique()))}
    daily["_market_day"] = daily["trading_day"].map(market_day)
    daily["next_open"] = daily.groupby("instrument", sort=False)["open"].shift(-1)
    daily["following_open"] = daily.groupby("instrument", sort=False)["open"].shift(-2)
    next_day = daily.groupby("instrument", sort=False)["_market_day"].shift(-1)
    following_day = daily.groupby("instrument", sort=False)["_market_day"].shift(-2)
    daily["target_return"] = (daily["following_open"] / daily["next_open"] - 1.0).where(
        next_day.eq(daily["_market_day"] + 1) & following_day.eq(daily["_market_day"] + 2)
    )
    sample = daily.merge(features, on=["instrument", "trading_day"], how="inner", validate="one_to_one")
    sample = sample.merge(universe, on=["instrument", "trading_day"], how="inner", validate="one_to_one")
    sample = sample.merge(exposure, on=["instrument", "trading_day"], how="inner", validate="one_to_one")
    return sample.replace([np.inf, -np.inf], np.nan)


def transform_day(day: pd.DataFrame):
    transformed = pd.DataFrame(index=day.index)
    for col in SIGNAL_COLS:
        raw = day[col].astype(float)
        if col in POSITIVE_COLS:
            transformed[col] = 0.0
            q95 = raw.loc[raw > 0].quantile(0.95)
            if np.isfinite(q95) and q95 > 0:
                transformed.loc[raw > 0, col] = (raw.loc[raw > 0] / q95).clip(0.0, 1.0)
        elif col == "direction":
            transformed[col] = raw
        else:
            transformed[col] = zscore(raw)
    controls = pd.DataFrame({col: zscore(day[col].astype(float)) for col in STYLE_CONTROL_COLS})
    context = np.column_stack([
        zscore(day["ma30_relative"]), zscore(day["ma60_relative"]),
        day["had_limit_up_prev9"].fillna(0.0),
        zscore(day["prev9_limit_up_price_gap"]).fillna(0.0),
    ]).astype(np.float32)
    target = zscore(day["target_return"])
    valid = transformed.notna().all(axis=1) & controls.notna().all(axis=1) & target.notna() & np.isfinite(context).all(axis=1)
    if valid.sum() < 50:
        return None
    return (
        transformed.loc[valid, SIGNAL_COLS].to_numpy(np.float32), context[valid],
        target.loc[valid].to_numpy(np.float32), controls.loc[valid].to_numpy(np.float32),
    )


def prepare_days(sample: pd.DataFrame) -> dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    return {day: item for day, frame in sample.groupby("trading_day", sort=True)
            if (item := transform_day(frame)) is not None}


def in_range(days, start, end):
    return [sample for day, sample in days.items() if pd.Timestamp(start) <= day <= pd.Timestamp(end)]


def fit_static_beta(samples):
    coefficients = []
    for features, _, target, controls in samples:
        design = np.column_stack([np.ones(len(features)), features, controls])
        if np.linalg.matrix_rank(design) == design.shape[1]:
            coefficients.append(np.linalg.lstsq(design, target, rcond=None)[0])
    if not coefficients:
        raise ValueError("no full-rank daily OLS cross-sections were available")
    mean = np.mean(coefficients, axis=0)
    return mean[1:18].astype(np.float32), float(mean[0])


# 3. 用 4 个经验 regime 信号作为 context control，训练一个 4 x 17 的小型 MLP。
# MLP 不直接预测收益，而是为 17 条 OLS 基座信号分别生成动态权重修正，并限制在
# 预设范围内。它在 2020--2022 样本上训练，按 2023 验证损失选择轮次；随后以该
# 轮次在 2019--2024 全样本重新训练，得到最终提交的动态校准矩阵。
class BoundedWeightMLP(nn.Module):
    """One-layer neural calibration: 4 context states -> 17 bounded multipliers."""
    def __init__(self, beta: np.ndarray, intercept: float):
        super().__init__()
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))
        self.weight_head = nn.Linear(4, 17, bias=False)
        nn.init.zeros_(self.weight_head.weight)

    def forward(self, features, context):
        adjustment = ADJUSTMENT_CAP * torch.tanh(self.weight_head(context))
        return self.intercept + (features * self.beta * (1.0 + adjustment)).sum(dim=1)


def loss_on(model, samples, device):
    model.eval()
    with torch.no_grad():
        return float(np.mean([
            F.smooth_l1_loss(model(torch.as_tensor(x, device=device), torch.as_tensor(z, device=device)), torch.as_tensor(y, device=device)).item()
            for x, z, y, _ in samples
        ]))


def train_selection(train, validation, beta, intercept, device):
    model = BoundedWeightMLP(beta, intercept).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-3)
    best, best_loss, best_epoch, stale = None, np.inf, 0, 0
    for epoch in range(1, 41):
        model.train()
        for index in np.random.permutation(len(train)):
            x, z, y, _ = train[index]
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(torch.as_tensor(x, device=device), torch.as_tensor(z, device=device)), torch.as_tensor(y, device=device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        value = loss_on(model, validation, device)
        print(f"selection epoch={epoch}; validation_huber={value:.6f}")
        if value < best_loss - 1e-5:
            best_loss, best_epoch, stale = value, epoch, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
    model.load_state_dict(best)
    return best_epoch


def refit(samples, beta, intercept, epochs, device):
    model = BoundedWeightMLP(beta, intercept).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=2e-3)
    model.train()
    for _ in range(epochs):
        for x, z, y, _ in samples:
            optimizer.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(model(torch.as_tensor(x, device=device), torch.as_tensor(z, device=device)), torch.as_tensor(y, device=device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return model


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    bars, universe, exposure = fetch_training_data(START_DATE, END_DATE)
    sample = build_sample(prepare_bars(bars), universe, exposure)
    days = prepare_days(sample)
    selection = in_range(days, SELECTION_START, SELECTION_END)
    validation = in_range(days, VALIDATION_START, VALIDATION_END)
    final = list(days.values())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 先用独立 2023 验证期选择校准层训练轮次，再以该轮次在完整样本 refit。
    beta, intercept = fit_static_beta(selection)
    epochs = train_selection(selection, validation, beta, intercept, device)
    final_beta, final_intercept = fit_static_beta(final)
    model = refit(final, final_beta, final_intercept, epochs, device)
    payload = {
        "feature_cols": SIGNAL_COLS, "context_cols": CONTEXT_COLS,
        "adjustment_mode": "bounded_tanh", "adjustment_cap": ADJUSTMENT_CAP,
        "rank_loss_weight": 0.0, "static_beta": final_beta.tolist(),
        "static_intercept": final_intercept, "best_selection_epoch": epochs,
        "dynamic_matrix": model.weight_head.weight.detach().cpu().numpy().tolist(),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {OUTPUT_PATH}; dynamic parameters={17 * 4}; refit days={len(final)}")


if __name__ == "__main__":
    main()
