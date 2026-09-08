import json
import logging
import os
from time import perf_counter
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import dai
from torch import nn
from tqdm.auto import tqdm


PRICE_FEATURES = [
    "pre_close",
    "open",
    "high",
    "low",
    "close",
    "bid_price1",
    "ask_price1",
]
VOLUME_FEATURES = [
    "volume",
    "amount",
    "deal_number",
    "bid_volume1",
    "ask_volume1",
    "bid_num_orders1",
    "ask_num_orders1",
]
FEATURES = [*PRICE_FEATURES, *VOLUME_FEATURES]
N_FEAT = len(FEATURES)
SEED = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_START = "2019-06-01"
TRAIN_END = "2023-11-30"
EPOCHS = 5
LR = 1e-3


LOOKBACK = 240
MINUTES = 1
BATCH = 1024
PREPROCESS_CHUNK_SIZE = 16_384
QUERY_TRADING_DAY_CHUNK_SIZE = 10
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model.json")
LOG_PATH = os.path.join(_HERE, "train.log")
logger = logging.getLogger(__name__)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging():
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file)


configure_logging()


class SelectiveStateSpaceBlock(nn.Module):
    def __init__(self, model_size, state_size=8, expansion=2):
        super().__init__()
        inner_size = model_size * expansion
        self.inner_size = inner_size
        self.state_size = state_size
        self.norm = nn.LayerNorm(model_size)
        self.input_projection = nn.Linear(model_size, inner_size * 2)
        self.local_conv = nn.Conv1d(
            inner_size,
            inner_size,
            kernel_size=4,
            padding=3,
            groups=inner_size,
        )
        self.delta_projection = nn.Linear(inner_size, inner_size)
        self.state_projection = nn.Linear(inner_size, state_size * 2)
        self.log_transition = nn.Parameter(
            torch.log(torch.arange(1, state_size + 1, dtype=torch.float32)).repeat(
                inner_size, 1
            )
        )
        self.skip = nn.Parameter(torch.ones(inner_size))
        self.output_projection = nn.Linear(inner_size, model_size)
        self.dropout = nn.Dropout(0.1)
        self.residual_scale = nn.Parameter(torch.full((model_size,), 0.1))
        nn.init.normal_(self.delta_projection.weight, std=0.02)
        nn.init.constant_(self.delta_projection.bias, -3.0)

    def forward(self, x):
        residual = x
        x, gate = self.input_projection(self.norm(x)).chunk(2, dim=-1)
        length = x.shape[1]
        x = self.local_conv(x.transpose(1, 2))[:, :, :length]
        x = nn.functional.silu(x).transpose(1, 2)
        gate = nn.functional.silu(gate)

        delta = nn.functional.softplus(self.delta_projection(x))
        state_parameters = self.state_projection(x)
        state_input, state_output = state_parameters.chunk(2, dim=-1)
        transition = -self.log_transition.exp()
        state = x.new_zeros(
            x.shape[0],
            self.inner_size,
            self.state_size,
        )
        outputs = []
        for step in range(length):
            step_delta = delta[:, step].unsqueeze(-1)
            decay = torch.exp(step_delta * transition)
            state = decay * state + step_delta * state_input[:, step].unsqueeze(1) * x[
                :, step
            ].unsqueeze(-1)
            output = (state * state_output[:, step].unsqueeze(1)).sum(dim=-1)
            output = output + self.skip * x[:, step]
            outputs.append(output)

        x = torch.stack(outputs, dim=1) * gate
        x = self.dropout(self.output_projection(x))
        return residual + self.residual_scale * x


class CompactChannelIndependentSelectiveStateSpaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_size = 24
        patch_size = 8
        self.patch = nn.Conv1d(
            1,
            model_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.field_position = nn.Parameter(torch.zeros(1, N_FEAT, 1, model_size))
        self.block = SelectiveStateSpaceBlock(
            model_size,
            state_size=4,
            expansion=2,
        )
        summary_size = N_FEAT * model_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(summary_size),
            nn.Linear(summary_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.transpose(1, 2).reshape(batch_size * N_FEAT, 1, LOOKBACK)
        x = self.patch(x).transpose(1, 2)
        x = x.reshape(batch_size, N_FEAT, x.shape[1], x.shape[2])
        x = x + self.field_position
        x = x.reshape(batch_size * N_FEAT, x.shape[2], x.shape[3])
        x = self.block(x)
        summary = torch.cat([x[:, -1], x.mean(dim=1)], dim=1)
        summary = summary.reshape(batch_size, -1)
        return self.head(summary).squeeze(-1)


class AlphaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CompactChannelIndependentSelectiveStateSpaceModel()

    def forward(self, x):
        return self.model(x)


def query_raw_features(sd, ed, instruments, table):
    # 获取原始特征
    stop = (pd.Timestamp(ed) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    f_str = ", ".join(FEATURES)
    ins_str = ", ".join([f"'{ins}'" for ins in instruments])
    frame = dai.query(
        f"""
        SELECT date, instrument, {f_str}
        FROM {table}
        WHERE date >= CAST('{sd}' AS TIMESTAMP)
          AND date < CAST('{stop}' AS TIMESTAMP)
          AND instrument IN ({ins_str})
        ORDER BY instrument, date
        """
    ).df()
    frame["date"] = pd.to_datetime(frame["date"])
    frame[VOLUME_FEATURES] = np.log1p(frame[VOLUME_FEATURES].clip(lower=0))
    return frame


def pool(sd, ed):
    frame = dai.query(
        f"""
        SELECT CAST(date AS DATE) AS date, instrument
        FROM bigalpha_2026_instruments
        WHERE CAST(date AS DATE) >= CAST('{sd}' AS DATE)
          AND CAST(date AS DATE) <= CAST('{ed}' AS DATE)
        ORDER BY date, instrument
        """
    ).df()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def trading_dates():
    frame = dai.query(
        """
        SELECT DISTINCT CAST(date AS DATE) AS date
        FROM bigalpha_2026_instruments
        ORDER BY date
        """,
        full_db_scan=True,
    ).df()
    return pd.to_datetime(frame["date"]).dt.normalize()


def source_trading_dates(table):
    frame = dai.query(
        f"""
        SELECT DISTINCT CAST(date AS DATE) AS date
        FROM {table}
        ORDER BY date
        """,
        full_db_scan=True,
    ).df()
    return pd.to_datetime(frame["date"]).dt.normalize()


def make_labels(sd, ed):
    """
    标签构建
    """
    samples = pool(sd, ed)
    dates = trading_dates()
    samples = samples.merge(
        pd.DataFrame(
            {
                "date": dates,
                "date_t1": dates.shift(-1),
                "date_t2": dates.shift(-2),
            }
        ),
        on="date",
        how="left",
        validate="many_to_one",
    )

    last_target_date = samples["date_t2"].max()
    if pd.isna(last_target_date):
        samples["label"] = np.nan
        return samples[["date", "instrument", "label"]]

    stop = (last_target_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices = dai.query(
        f"""
        SELECT
            CAST(date AS DATE) AS date,
            instrument,
            arg_min(open * adjust_factor, CAST(date AS TIMESTAMP)) AS open
        FROM bigalpha_2026_stock_bar30m
        WHERE CAST(date AS DATE) >= CAST('{sd}' AS DATE)
          AND CAST(date AS DATE) < CAST('{stop}' AS DATE)
        GROUP BY CAST(date AS DATE), instrument
        ORDER BY date, instrument
        """
    ).df()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

    for step in ("t1", "t2"):
        samples = samples.merge(
            prices.rename(columns={"date": f"date_{step}", "open": f"open_{step}"}),
            on=[f"date_{step}", "instrument"],
            how="left",
            validate="many_to_one",
        )

    valid = samples["open_t1"].gt(0) & samples["open_t2"].gt(0)
    samples["label"] = (samples["open_t2"] / samples["open_t1"] - 1).where(valid)
    # 在截面上做rank
    samples["label"] = samples.groupby("date")["label"].transform(
        lambda x: x.rank(pct=True, method="average")
    )
    return samples[["date", "instrument", "label"]]


def make_infer_meta(sd, ed):
    samples = pool(sd, ed)
    samples["label"] = 0.0
    return samples[["date", "instrument", "label"]]


def make_expected_times(sample_dates, dates, date_position, minutes, lookback):
    bars_per_day = 240 // minutes
    days_needed = (lookback + bars_per_day - 1) // bars_per_day
    offsets = np.concatenate(
        [
            np.arange(570 + minutes, 691, minutes),
            np.arange(780 + minutes, 901, minutes),
        ]
    ).astype("timedelta64[m]")

    result = {}
    for date in sample_dates:
        end = date_position[date] + 1
        start = end - days_needed
        if start < 0:
            continue
        days = dates.iloc[start:end].to_numpy(dtype="datetime64[D]")
        result[date] = (
            (days[:, None] + offsets[None, :])
            .reshape(-1)[-lookback:]
            .astype("datetime64[ns]")
        )
    return result


def _mean_std_from_chunks(chunks, feature_count, dtype, device):
    """合并若干数据块最后一个维度的总体均值和标准差。"""
    total_count = 0
    running_mean = torch.zeros(
        feature_count,
        dtype=torch.float64,
        device=device,
    )
    running_m2 = torch.zeros_like(running_mean)

    for chunk in chunks:
        reduce_dims = tuple(range(chunk.ndim - 1))
        chunk_variance, chunk_mean = torch.var_mean(
            chunk,
            dim=reduce_dims,
            correction=0,
        )
        chunk_count = chunk.numel() // chunk.shape[-1]
        chunk_mean = chunk_mean.to(torch.float64)
        chunk_m2 = chunk_variance.to(torch.float64) * chunk_count

        if total_count == 0:
            running_mean.copy_(chunk_mean)
            running_m2.copy_(chunk_m2)
        else:
            combined_count = total_count + chunk_count
            delta = chunk_mean - running_mean
            running_mean.add_(delta * (chunk_count / combined_count))
            running_m2.add_(
                chunk_m2 + delta.square() * (total_count * chunk_count / combined_count)
            )
        total_count += chunk_count

    if total_count == 0:
        raise ValueError("无法对空数据计算均值和标准差")

    mean = running_mean.to(dtype)
    std = (running_m2 / total_count).sqrt().to(dtype)
    return mean, std


def chunked_mean_std(data, chunk_size=PREPROCESS_CHUNK_SIZE):
    """分块计算最后一个维度的总体均值和标准差。"""
    return _mean_std_from_chunks(
        data.split(chunk_size, dim=0),
        data.shape[-1],
        data.dtype,
        data.device,
    )


def _iter_price_scaled_valid_chunks(
    data,
    valid_indices,
    chunk_size,
    desc,
):
    """逐块读取有效窗口并进行价格缩放，不复制全部有效数据。"""
    close_index = FEATURES.index("close")
    chunk_starts = range(0, len(valid_indices), chunk_size)
    with tqdm(
        chunk_starts,
        total=len(chunk_starts),
        desc=desc,
        unit="块",
        dynamic_ncols=True,
    ) as progress:
        for chunk_start in progress:
            indices = valid_indices[chunk_start : chunk_start + chunk_size]
            chunk = data.index_select(0, indices)
            # 必须复制锚点。price_anchor 若仍是 chunk 的视图，下面的原地除法会先把
            # 末端 close 改为 1，导致后续价格字段除以被改写的锚点。
            price_anchor = chunk[
                :,
                -1:,
                close_index : close_index + 1,
            ].clone()
            chunk[:, :, : len(PRICE_FEATURES)].div_(price_anchor)
            yield indices, chunk


def preprocess_valid_data(
    data,
    valid,
    stats=None,
    chunk_size=PREPROCESS_CHUNK_SIZE,
):
    """分块价格缩放、拟合或复用全局统计量，并将结果写回 data。"""
    valid_indices = valid.nonzero(as_tuple=False).flatten()
    fit_mode = stats is None

    if fit_mode:
        scaled_chunks = (
            chunk
            for _, chunk in _iter_price_scaled_valid_chunks(
                data,
                valid_indices,
                chunk_size,
                "统计全局均值/标准差",
            )
        )
        mean, std = _mean_std_from_chunks(
            scaled_chunks,
            data.shape[-1],
            data.dtype,
            data.device,
        )
        std = std + 1e-6
        stats = {
            "mean": mean,
            "std": std,
        }
    else:
        mean = torch.as_tensor(stats["mean"], dtype=data.dtype, device=data.device)
        std = torch.as_tensor(stats["std"], dtype=data.dtype, device=data.device)

    for indices, chunk in _iter_price_scaled_valid_chunks(
        data,
        valid_indices,
        chunk_size,
        "标准化有效窗口",
    ):
        chunk.sub_(mean).div_(std)
        data.index_copy_(0, indices, chunk)

    return stats


def build_data(sd, ed, table, stats=None):
    """
    输出
    1. data - tensor: (N * D, L, F)，和 meta 逐行对应
    2. meta - df: date, instrument, label, mask
    3. stats - dict: mean, std；stats=None 时拟合，否则复用
    训练模式 mask 同时检查标签和窗口；复用 stats 时 mask 只检查窗口。
    """
    started = perf_counter()
    fit_mode = stats is None
    logger.info(
        "构建%s数据 %s~%s source=%s",
        "训练" if fit_mode else "推理",
        sd,
        ed,
        table,
    )

    meta = make_labels(sd, ed) if fit_mode else make_infer_meta(sd, ed)
    meta = meta.sort_values(["date", "instrument"]).reset_index(drop=True)

    dates = source_trading_dates(table)
    date_position = {date: i for i, date in enumerate(dates)}

    data = torch.zeros(
        (len(meta), LOOKBACK, N_FEAT),
        dtype=torch.float32,
    )
    window_valid = np.zeros(len(meta), dtype=bool)
    bars_per_day = 240 // MINUTES
    days_needed = (LOOKBACK + bars_per_day - 1) // bars_per_day
    first_sample_position = date_position[meta["date"].min()]
    chunk_ids = (
        meta["date"].map(date_position) - first_sample_position
    ) // QUERY_TRADING_DAY_CHUNK_SIZE
    date_chunks = meta.groupby(chunk_ids, sort=True)
    with tqdm(
        date_chunks,
        total=date_chunks.ngroups,
        desc="查询并构建时间分片",
        unit="片",
        dynamic_ncols=True,
    ) as progress:
        for _, chunk_meta in progress:
            chunk_started = perf_counter()
            first_position = date_position[chunk_meta["date"].min()]
            query_position = max(0, first_position - days_needed + 1)
            frame = query_raw_features(
                dates.iloc[query_position],
                chunk_meta["date"].max(),
                chunk_meta["instrument"].unique().tolist(),
                table,
            )

            expected_times = make_expected_times(
                chunk_meta["date"].unique(),
                dates,
                date_position,
                MINUTES,
                LOOKBACK,
            )
            rows_by_instrument = chunk_meta.groupby(
                "instrument",
                sort=False,
            ).groups
            for instrument, part in frame.groupby("instrument", sort=False):
                row_indices = rows_by_instrument.get(instrument)
                if row_indices is None:
                    continue

                times = part["date"].to_numpy(dtype="datetime64[ns]")
                features = part[FEATURES].to_numpy(dtype=np.float32)
                for row_index in row_indices:
                    date = meta.at[row_index, "date"]
                    expected = expected_times.get(date)
                    if expected is None:
                        continue

                    end = np.searchsorted(
                        times,
                        np.datetime64(date) + np.timedelta64(1, "D"),
                    )
                    start = end - LOOKBACK
                    if start < 0 or not np.array_equal(times[start:end], expected):
                        continue

                    values = features[start:end]
                    ohlc = values[:, :5]
                    book_prices = values[:, 5 : len(PRICE_FEATURES)]
                    if (
                        np.isfinite(values).all()
                        and (ohlc > 0).all()
                        and (book_prices >= 0).all()
                    ):
                        # 买一/卖一为 0 表示该侧盘口缺失。用同一分钟的 close 中性
                        # 填充，避免把 0 当作真实价格并在标准化后制造极端值。
                        if (book_prices == 0).any():
                            values = values.copy()
                            close = values[:, FEATURES.index("close")]
                            for feature in ("bid_price1", "ask_price1"):
                                feature_index = FEATURES.index(feature)
                                missing = values[:, feature_index] == 0
                                values[missing, feature_index] = close[missing]
                        data[row_index] = torch.from_numpy(values.copy())
                        window_valid[row_index] = True

            chunk_rows = chunk_meta.index.to_numpy()
            chunk_windows = int(window_valid[chunk_rows].sum())
            chunk_labels = int(np.isfinite(chunk_meta["label"].to_numpy()).sum())
            chunk_raw_rows = len(frame)
            chunk_label = (
                f"{chunk_meta['date'].min():%Y-%m-%d}"
                f"~{chunk_meta['date'].max():%Y-%m-%d}"
            )
            del frame
            progress.set_postfix(
                interval=chunk_label,
                raw=f"{chunk_raw_rows:,}",
                windows=f"{chunk_windows:,}/{len(chunk_meta):,}",
            )
            logger.info(
                "%s raw=%s windows=%s/%s labels=%s/%s elapsed=%.1fs",
                chunk_label,
                f"{chunk_raw_rows:,}",
                f"{chunk_windows:,}",
                f"{len(chunk_meta):,}",
                f"{chunk_labels:,}",
                f"{len(chunk_meta):,}",
                perf_counter() - chunk_started,
            )

    label_valid = np.isfinite(meta["label"].to_numpy())
    valid = torch.from_numpy(window_valid & label_valid if fit_mode else window_valid)
    meta["mask"] = ~valid.numpy()

    # 分块处理有效窗口，避免 data[valid] 复制全部有效数据。
    stats = preprocess_valid_data(data, valid, stats)

    daily_coverage = meta.groupby("date")["mask"].agg(["sum", "count"])
    daily_coverage["valid"] = daily_coverage["count"] - daily_coverage["sum"]
    daily_coverage["coverage"] = daily_coverage["valid"] / daily_coverage["count"]
    worst_date = daily_coverage["coverage"].idxmin()
    worst = daily_coverage.loc[worst_date]

    logger.info(
        "构建完成 data_shape=%s dates=%s samples=%s valid=%s masked=%s elapsed=%.1fs",
        tuple(data.shape),
        meta["date"].nunique(),
        f"{len(meta):,}",
        f"{int(valid.sum()):,}",
        f"{int((~valid).sum()):,}",
        perf_counter() - started,
    )
    logger.info(
        "覆盖度最差 date=%s valid=%s/%s coverage=%.2f%%",
        worst_date.strftime("%Y-%m-%d"),
        f"{int(worst['valid']):,}",
        f"{int(worst['count']):,}",
        worst["coverage"] * 100,
    )
    return data, meta[["date", "instrument", "label", "mask"]], stats


class WindowDataset(Dataset):
    def __init__(self, data, meta):
        self.data = data
        self.label = torch.from_numpy(
            meta["label"].to_numpy(dtype=np.float32, copy=True)
        )
        self.mask = torch.from_numpy(meta["mask"].to_numpy(copy=True))
        self.meta = meta

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        m = self.mask[idx]
        return x, y, m


def masked_mse_loss(prediction, target, mask):
    return nn.functional.mse_loss(
        prediction[~mask],
        target[~mask],
    )


def save_model(checkpoint, model_path=MODEL_PATH):
    """把 checkpoint 保存为纯文本 JSON。"""
    state_dict = {}
    for name, value in checkpoint["state_dict"].items():
        tensor = value.detach().cpu()
        state_dict[name] = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": tensor.reshape(-1).tolist(),
        }

    payload = {
        name: value
        for name, value in checkpoint.items()
        if name != "state_dict"
    }
    payload["state_dict"] = state_dict
    with open(model_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    return model_path


def load_model(model_path=MODEL_PATH):
    """读取 save_model 生成的 JSON checkpoint。"""
    with open(model_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    state_dict = {}
    for name, metadata in payload["state_dict"].items():
        tensor = torch.tensor(
            metadata["data"],
            dtype=getattr(torch, metadata["dtype"]),
        )
        state_dict[name] = tensor.reshape(metadata["shape"])

    model = AlphaModel().to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, payload["stats"]


@torch.inference_mode()
def infer(model, loader):
    model.eval()
    predictions = []
    for features, _, mask in tqdm(
        loader,
        total=len(loader),
        desc="模型推理",
        unit="批",
        dynamic_ncols=True,
    ):
        prediction = model(features.to(DEVICE, non_blocking=True)).cpu()
        prediction[mask] = torch.nan
        predictions.append(prediction)
    return torch.cat(predictions)


def train():
    logger.info("训练开始, device=%s, seed=%s", DEVICE, SEED)
    set_seed(SEED)
    table = "bigalpha_2026_stock_bar1m"
    data, meta, stats = build_data(
        TRAIN_START,
        TRAIN_END,
        table,
    )
    dataset = WindowDataset(data, meta)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        pin_memory=DEVICE == "cuda",
    )

    model = AlphaModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    logger.info(
        "训练样本=%s batch=%s 参数量=%s",
        f"{len(dataset):,}",
        BATCH,
        f"{sum(parameter.numel() for parameter in model.parameters()):,}",
    )

    for epoch in range(1, EPOCHS + 1):
        started = perf_counter()
        loss_sum = 0.0
        batches = 0
        model.train()

        with tqdm(
            loader,
            total=len(loader),
            desc=f"训练 epoch {epoch}/{EPOCHS}",
            unit="批",
            dynamic_ncols=True,
        ) as progress:
            for features, target, mask in progress:
                features = features.to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                loss = masked_mse_loss(model(features), target, mask)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                batches += 1
                progress.set_postfix(loss=f"{loss_sum / batches:.8f}")

        logger.info(
            "epoch=%s/%s loss=%.8f elapsed=%.1fs",
            epoch,
            EPOCHS,
            loss_sum / batches,
            perf_counter() - started,
        )

    save_model(
        {
            "state_dict": model.state_dict(),
            "stats": {
                name: torch.as_tensor(value).detach().cpu().tolist()
                for name, value in stats.items()
            },
        },
        MODEL_PATH,
    )
    logger.info("训练完成 model=%s", MODEL_PATH)
    return MODEL_PATH


if __name__ == "__main__":
    # test_build_data()
    # test_window_dataset()
    train()
