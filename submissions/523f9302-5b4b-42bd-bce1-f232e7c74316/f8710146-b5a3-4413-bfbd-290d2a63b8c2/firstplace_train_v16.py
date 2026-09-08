def main(datasources, start_date, end_date):
    import gc
    import os
    import math
    import random
    import time

    import dai
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import structlog

    logger = structlog.get_logger()
    SEED = 20260723
    TRAIN_TABLE = "bigalpha_2026_stock_bar15m"
    INFER_TABLE = datasources["bar15m"]
    POOL_TABLE = "bigalpha_2026_instruments"
    TRAIN_START = "2022-01-01"
    PUBLIC_TRAIN_END = "2024-12-31 23:59:59"
    LOCAL_TRAIN_END = "2023-12-31 23:59:59"
    BARS_PER_DAY = 16
    LOOKBACK_DAYS = 10
    EPOCHS = 8
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    EMA_DECAY = 0.998

    price_fields = [
        "pre_close", "open", "high", "low", "close",
        "ask_price1", "ask_price2", "ask_price3",
        "bid_price1", "bid_price2", "bid_price3",
    ]
    log_fields = [
        "deal_number", "volume", "amount",
        "ask_volume1", "ask_volume2", "ask_volume3",
        "bid_volume1", "bid_volume2", "bid_volume3",
        "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
        "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
    ]
    features = price_fields + log_fields
    n_features = len(features)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    infer_end = pd.Timestamp(end_date)
    local_holdout = (
        INFER_TABLE == TRAIN_TABLE
        and infer_end <= pd.Timestamp("2024-12-31 23:59:59")
    )
    train_end = LOCAL_TRAIN_END if local_holdout else PUBLIC_TRAIN_END
    logger.info(
        "configuration",
        device=str(device),
        train_table=TRAIN_TABLE,
        infer_table=INFER_TABLE,
        train_end=train_end,
        local_holdout=local_holdout,
    )

    def pad_bars(value):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)[-BARS_PER_DAY:]
        if arr.size < BARS_PER_DAY:
            arr = np.pad(
                arr, (BARS_PER_DAY - arr.size, 0),
                mode="constant", constant_values=np.nan,
            )
        return arr

    def stack_field(series):
        values = series.to_numpy()
        try:
            arr = np.stack(values).astype(np.float32, copy=False)
            if arr.shape[1] == BARS_PER_DAY:
                return arr
        except (TypeError, ValueError):
            pass
        return np.stack([pad_bars(value) for value in values])

    def query_chunk(table, chunk_start, chunk_end):
        sql = f"""
            SELECT date, instrument, {', '.join(features)}
            FROM {table}
            ORDER BY instrument, date
        """
        frame = dai.query(
            sql,
            filters={"date": [str(chunk_start), str(chunk_end)]},
            compression=True,
        ).df()
        if frame.empty:
            return None
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(
            ["instrument", "date"], kind="stable"
        ).reset_index(drop=True)

        row_dates = frame["date"].dt.normalize().to_numpy(
            dtype="datetime64[D]"
        )
        row_instruments = frame["instrument"].astype(str).to_numpy()
        changes = np.r_[
            True,
            (row_dates[1:] != row_dates[:-1])
            | (row_instruments[1:] != row_instruments[:-1]),
            True,
        ]
        cuts = np.flatnonzero(changes)
        lefts, rights = cuts[:-1], cuts[1:]
        group_dates = row_dates[lefts]
        group_instruments = row_instruments[lefts]
        values = frame[features].to_numpy(np.float32)

        x = np.full(
            (len(lefts), BARS_PER_DAY, n_features),
            np.nan,
            dtype=np.float32,
        )
        for index, (left, right) in enumerate(zip(lefts, rights)):
            block = values[left:right][-BARS_PER_DAY:]
            x[index, -len(block):] = block

        pool = dai.query(
            f"SELECT date, instrument FROM {POOL_TABLE}",
            filters={"date": [str(chunk_start), str(chunk_end)]},
            compression=True,
        ).df()
        pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
        pool_index = pd.MultiIndex.from_frame(pool[["date", "instrument"]])
        row_index = pd.MultiIndex.from_arrays(
            [pd.to_datetime(group_dates), group_instruments]
        )
        in_pool = np.asarray(row_index.isin(pool_index), dtype=bool)

        close_seq = x[:, :, features.index("close")]
        close = np.array([
            row[np.flatnonzero(np.isfinite(row))[-1]]
            if np.isfinite(row).any() else np.nan
            for row in close_seq
        ], dtype=np.float32)
        result = (
            x,
            group_dates,
            group_instruments,
            in_pool,
            close,
        )
        logger.info(
            "query_chunk",
            start=str(chunk_start), end=str(chunk_end),
            rows=len(frame), daily_rows=len(group_dates),
            pool_rows=int(in_pool.sum()),
        )
        del frame, pool, pool_index, row_index, values, close_seq
        gc.collect()
        return result

    def load_daily(table, sd, ed):
        sd_ts, ed_ts = pd.Timestamp(sd), pd.Timestamp(ed)
        chunks = []
        month_start = sd_ts.to_period("M").start_time
        while month_start <= ed_ts:
            month_end = (
                month_start + pd.offsets.MonthEnd(0)
            ).replace(hour=23, minute=59, second=59)
            left = max(sd_ts, month_start)
            right = min(ed_ts, month_end)
            chunk = query_chunk(table, left, right)
            if chunk is not None:
                chunks.append(chunk)
            month_start = month_start + pd.offsets.MonthBegin(1)
        if not chunks:
            raise RuntimeError(f"No data in {table} for {sd} to {ed}")

        x = np.concatenate([c[0] for c in chunks])
        dates = np.concatenate([c[1] for c in chunks])
        instruments = np.concatenate([c[2] for c in chunks])
        in_pool = np.concatenate([c[3] for c in chunks])
        close = np.concatenate([c[4] for c in chunks])
        del chunks
        gc.collect()
        return x, dates, instruments, in_pool, close

    def add_targets(dates, instruments, close):
        codes, names = pd.factorize(instruments, sort=True)
        order = np.lexsort((dates.astype("int64"), codes))
        dates = dates[order]
        codes = codes[order].astype(np.int32)
        close = close[order]
        target = np.full(len(close), np.nan, dtype=np.float32)

        boundaries = np.flatnonzero(
            np.r_[True, codes[1:] != codes[:-1], True]
        )
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            current = close[left:right - 1].astype(np.float64)
            future = close[left + 1:right].astype(np.float64)
            gaps = (
                dates[left + 1:right] - dates[left:right - 1]
            ).astype("timedelta64[D]").astype(int)
            valid = (
                np.isfinite(current) & np.isfinite(future)
                & (current > 0) & (gaps <= 10)
            )
            returns = future / current - 1.0
            valid &= np.isfinite(returns) & (np.abs(returns) < 0.5)
            positions = np.arange(left, right - 1)[valid]
            target[positions] = returns[valid].astype(np.float32)
        return order, dates, codes, np.asarray(names, dtype=object), target

    def normalize(x, fit_rows, stats=None):
        base = x[:, :, 0].astype(np.float32, copy=True)
        close_raw = x[:, :, 4].astype(np.float32, copy=True)
        valid_base = np.isfinite(base) & (np.abs(base) > 1e-6)
        for col in range(1, 11):
            values = x[:, :, col].astype(np.float32, copy=False)
            transformed = np.full(values.shape, np.nan, dtype=np.float32)
            np.divide(values, base, out=transformed, where=valid_base)
            x[:, :, col] = transformed - 1.0
        previous = np.concatenate(
            [close_raw[:, :1], close_raw[:, :-1]], axis=1
        )
        valid_previous = np.isfinite(previous) & (np.abs(previous) > 1e-6)
        close_return = np.full(close_raw.shape, np.nan, dtype=np.float32)
        np.divide(close_raw, previous, out=close_return, where=valid_previous)
        x[:, :, 0] = close_return - 1.0

        for field in log_fields:
            col = features.index(field)
            values = np.log1p(np.clip(x[:, :, col], 0, None))
            finite = np.isfinite(values)
            clean = np.where(finite, values, 0.0)
            count_local = finite.sum(axis=1, keepdims=True)
            center = clean.sum(axis=1, keepdims=True) / np.maximum(count_local, 1)
            x[:, :, col] = values - center

        if stats is None:
            total = np.zeros(n_features, dtype=np.float64)
            total_sq = np.zeros(n_features, dtype=np.float64)
            count = np.zeros(n_features, dtype=np.int64)
            fit_idx = np.flatnonzero(fit_rows)
            for left in range(0, len(fit_idx), 100000):
                values = x[fit_idx[left:left + 100000]].astype(
                    np.float64, copy=False
                )
                finite = np.isfinite(values)
                clean = np.where(finite, values, 0.0)
                total += clean.sum(axis=(0, 1))
                total_sq += (clean * clean).sum(axis=(0, 1))
                count += finite.sum(axis=(0, 1))
            mean = total / np.maximum(count, 1)
            variance = total_sq / np.maximum(count, 1) - mean * mean
            scale = np.sqrt(np.maximum(variance, 1e-8))
        else:
            mean, scale = stats

        result = np.empty(x.shape, dtype=np.float16)
        for left in range(0, len(x), 100000):
            values = (
                x[left:left + 100000]
                - mean.reshape(1, 1, -1)
            ) / scale.reshape(1, 1, -1)
            result[left:left + 100000] = np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float16)
        return result, (
            np.asarray(mean, dtype=np.float32),
            np.asarray(scale, dtype=np.float32),
        )

    class TemporalBlock(nn.Module):
        def __init__(self, width, kernel):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(
                    width, width, kernel,
                    padding=kernel // 2, groups=width,
                ),
                nn.GELU(),
                nn.Conv1d(width, width, 1),
            )

        def forward(self, value):
            return self.net(value)

    class CrossSectionalRanker(nn.Module):
        def __init__(self, width=128):
            super().__init__()
            self.input_norm = nn.LayerNorm(n_features)
            self.feature_gate = nn.Parameter(torch.zeros(n_features))
            self.projection = nn.Sequential(
                nn.Linear(n_features, width),
                nn.LayerNorm(width),
                nn.GELU(),
            )
            self.position = nn.Parameter(
                torch.zeros(1, BARS_PER_DAY, width)
            )
            self.short = TemporalBlock(width, 3)
            self.long = TemporalBlock(width, 7)
            self.conv_norm = nn.LayerNorm(width)
            intraday_layer = nn.TransformerEncoderLayer(
                width, 8, 256, dropout=0.12,
                batch_first=True, norm_first=True,
                activation="gelu",
            )
            self.intraday = nn.TransformerEncoder(intraday_layer, 2)
            self.pool = nn.Linear(width, 1)
            self.day_position = nn.Parameter(
                torch.zeros(1, LOOKBACK_DAYS, width)
            )
            day_layer = nn.TransformerEncoderLayer(
                width, 8, 256, dropout=0.10,
                batch_first=True, norm_first=True,
                activation="gelu",
            )
            self.day_encoder = nn.TransformerEncoder(day_layer, 2)
            self.day_pool = nn.Linear(width, 1)
            cross_layer = nn.TransformerEncoderLayer(
                width, 8, 256, dropout=0.08,
                batch_first=True, norm_first=True,
                activation="gelu",
            )
            self.cross = nn.TransformerEncoder(cross_layer, 2)
            self.gate = nn.Sequential(
                nn.Linear(2 * width, width),
                nn.Sigmoid(),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Dropout(0.08),
                nn.Linear(width, 1),
            )

        def forward(self, value):
            stocks, days, bars, fields = value.shape
            h = value.reshape(stocks * days, bars, fields)
            h = h * (1.0 + 0.5 * torch.tanh(self.feature_gate))
            h = self.projection(self.input_norm(h)) + self.position
            conv_in = h.transpose(1, 2)
            conv = (
                self.short(conv_in) + self.long(conv_in)
            ).transpose(1, 2)
            h = self.intraday(self.conv_norm(h + conv))
            weights = torch.softmax(self.pool(h).squeeze(-1), dim=1)
            h = (h * weights.unsqueeze(-1)).sum(dim=1)
            h = h.reshape(stocks, days, -1) + self.day_position
            h = self.day_encoder(h)
            day_weights = torch.softmax(
                self.day_pool(h).squeeze(-1), dim=1
            )
            own = h[:, -1] + (
                h * day_weights.unsqueeze(-1)
            ).sum(dim=1)
            market = self.cross(own.unsqueeze(0)).squeeze(0)
            gate = self.gate(torch.cat([own, market], dim=-1))
            return self.head(own + gate * market).squeeze(-1)

    def rank_loss(prediction, target):
        prediction = prediction.float()
        target = target.float()
        pred_center = prediction - prediction.mean()
        target_center = target - target.mean()
        pred_std = pred_center.square().mean().sqrt().clamp_min(1e-5)
        target_std = target_center.square().mean().sqrt().clamp_min(1e-5)
        pred_z = pred_center / pred_std
        target_z = target_center / target_std
        correlation = (pred_z * target_z).mean()

        pair_count = min(len(target), 2048)
        left = torch.randperm(
            len(target), device=target.device
        )[:pair_count]
        right = torch.roll(left, 1)
        target_gap = target_z[left] - target_z[right]
        useful = target_gap.abs() >= 0.20
        if useful.any():
            prediction_gap = pred_z[left] - pred_z[right]
            pairwise = nn.functional.softplus(
                -target_gap[useful].sign() * prediction_gap[useful]
            ).mean()
        else:
            pairwise = prediction.new_zeros(())

        order = torch.argsort(target)
        tail = max(16, len(target) // 10)
        spread = (
            pred_z[order[-tail:]].mean()
            - pred_z[order[:tail]].mean()
        )
        loss = (
            -correlation
            + 0.05 * (pred_z - target_z).square().mean()
            + 0.08 * pairwise
            + 0.04 * nn.functional.softplus(0.75 - spread)
        )
        return loss, correlation.detach()

    WEIGHT_PATH = __file__.rsplit("/", 1)[0] + "/bigalpha_e2e_v4_weights.pt"
    json_mod = __import__("json")
    base64_mod = __import__("base64")
    io_mod = __import__("io")
    model_path = __file__.rsplit("/", 1)[0] + "/firstplace_model_v16.json"
    with open(model_path, "r", encoding="utf-8") as handle:
        packed = json_mod.load(handle)
    checkpoint_v4 = torch.load(io_mod.BytesIO(base64_mod.b64decode(packed["data_v4"])), map_location="cpu", weights_only=False)
    checkpoint_v10 = torch.load(io_mod.BytesIO(base64_mod.b64decode(packed["data_v10"])), map_location="cpu", weights_only=False)
    stats = checkpoint_v4["stats"]
    model_v4 = CrossSectionalRanker().to(device)
    model_v10 = CrossSectionalRanker().to(device)
    model_v4.load_state_dict(checkpoint_v4["state_dict"], strict=False)
    model_v10.load_state_dict(checkpoint_v10["state_dict"])
    del checkpoint_v4, checkpoint_v10
    logger.info("weights_loaded", path=model_path, ensemble="v4_v10_50_50")

    logger.info(
        "load_inference_data",
        table=INFER_TABLE, start=str(start_date), end=str(end_date),
    )
    raw_test, test_dates, test_instruments, test_pool, test_close = load_daily(
        INFER_TABLE, start_date, end_date
    )
    test_x, _ = normalize(
        raw_test,
        np.ones(len(raw_test), dtype=bool),
        stats=stats,
    )
    del raw_test, test_close
    gc.collect()

    test_codes, _ = pd.factorize(test_instruments, sort=True)
    test_order = np.lexsort(
        (test_dates.astype("int64"), test_codes)
    )
    test_x = test_x[test_order]
    test_dates = test_dates[test_order]
    test_instruments = test_instruments[test_order]
    test_pool = test_pool[test_order]
    test_codes = test_codes[test_order]

    test_boundaries = np.flatnonzero(
        np.r_[True, test_codes[1:] != test_codes[:-1], True]
    )
    test_window_parts, test_row_parts = [], []
    for left, right in zip(
        test_boundaries[:-1], test_boundaries[1:]
    ):
        local = np.arange(right - left, dtype=np.int64)
        rows = left + local
        rows = rows[test_pool[rows]]
        if not len(rows):
            continue
        relative = rows - left
        window = (
            relative[:, None]
            + np.arange(-LOOKBACK_DAYS + 1, 1)[None, :]
        )
        window = np.clip(window, 0, right - left - 1) + left
        test_window_parts.append(window)
        test_row_parts.append(rows)

    test_windows = np.concatenate(test_window_parts).astype(
        np.int32, copy=False
    )
    test_rows = np.concatenate(test_row_parts).astype(
        np.int32, copy=False
    )
    test_sort = np.argsort(test_dates[test_rows], kind="stable")
    test_windows = test_windows[test_sort]
    test_rows = test_rows[test_sort]
    scored_dates = test_dates[test_rows]
    test_cuts = np.flatnonzero(
        np.r_[True, scored_dates[1:] != scored_dates[:-1], True]
    )

    output_dates, output_instruments, output_scores = [], [], []
    model_v4.eval()
    model_v10.eval()
    with torch.no_grad():
        for left, right in zip(test_cuts[:-1], test_cuts[1:]):
            rows = test_rows[left:right]
            batch = torch.from_numpy(
                test_x[test_windows[left:right]].astype(
                    np.float32, copy=False
                )
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                score_v4 = model_v4(batch)
                score_v10 = model_v10(batch)
                score = 0.50 * score_v4 + 0.50 * score_v10
            score = score.float().cpu().numpy().astype(np.float64)
            output_dates.append(test_dates[rows])
            output_instruments.append(test_instruments[rows])
            output_scores.append(score)

    result = pd.DataFrame({
        "date": pd.to_datetime(np.concatenate(output_dates)),
        "instrument": np.concatenate(output_instruments),
        "score": np.concatenate(output_scores),
    })
    result = (
        result.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])
        [["date", "instrument", "score"]]
        .sort_values(["date", "instrument"])
        .reset_index(drop=True)
    )
    logger.info(
        "score_ready",
        rows=len(result),
        days=result["date"].nunique(),
        instruments=result["instrument"].nunique(),
    )
    return result


