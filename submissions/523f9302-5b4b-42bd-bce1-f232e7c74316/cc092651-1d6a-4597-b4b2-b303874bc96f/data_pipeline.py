"""
BigAlpha 2026 - 数据管道（性能优化版）
使用 BigQuant DAI 引擎查询原始数据，构建模型输入张量

原则:
  - 所有输入字段直接来源于原始数据，不做衍生特征
  - 仅做缺失填充、按字段归一化等允许的预处理
  - 归一化参数仅基于训练集统计
"""
import numpy as np
import pandas as pd
import torch
from config_mamba import SELECTED_FIELDS, FREQ_TABLES, MAX_BARS, NUM_FIELDS


class DataPipeline:
    """数据管道：查询 → 归一化 → 张量化"""

    def __init__(self, freq_names, lookback_days=90, normalization="standard",
                 use_log_volume=True):
        self.freq_names = freq_names
        self.lookback_days = lookback_days
        self.normalization = normalization
        self.use_log_volume = use_log_volume
        self.stats = {}                    # {freq: {field: (mean, std)}}
        self.all_trading_days: list = []    # 交易日列表
        self.day_to_idx: dict = {}          # 日期→索引
        self.volume_fields = {
            "volume", "amount", "deal_number",
            "ask_volume1", "ask_volume2", "ask_volume3",
            "bid_volume1", "bid_volume2", "bid_volume3",
            "ask_num_orders1", "ask_num_orders2", "ask_num_orders3",
            "bid_num_orders1", "bid_num_orders2", "bid_num_orders3",
        }

    # ---- 数据查询 ----
    def query_raw_data(self, freq, start, end):
        """查询单频率原始分钟数据"""
        import dai
        fields = ", ".join(SELECTED_FIELDS)
        table = FREQ_TABLES[freq]
        return dai.query(
            f"SELECT date, instrument, {fields} FROM {table} ORDER BY date, instrument",
            filters={"date": [f"{start} 00:00:00", f"{end} 23:59:59"]},
            compression=True,
        ).df()

    def query_all(self, start, end):
        """查询全部频率"""
        data = {}
        for freq in self.freq_names:
            print(f"  查询 {freq}...")
            data[freq] = self.query_raw_data(freq, start, end)
        return data

    # ---- 归一化统计量 ----
    def fit(self, start, end):
        """基于训练集计算归一化参数"""
        import dai
        print(f"[Pipeline] 计算统计量 ({start} ~ {end})...")

        for freq in self.freq_names:
            table = FREQ_TABLES[freq]
            agg_fields = []
            for f in SELECTED_FIELDS:
                agg_fields.append(f"AVG({f}) AS m_{f}")
                agg_fields.append(f"STDDEV({f}) AS s_{f}")

            try:
                df = dai.query(
                    f"SELECT {', '.join(agg_fields)} FROM {table}",
                    filters={"date": [f"{start} 00:00:00", f"{end} 23:59:59"]},
                    compression=True,
                ).df()

                self.stats[freq] = {}
                for f in SELECTED_FIELDS:
                    m = float(df[f"m_{f}"].iloc[0])
                    s = float(df[f"s_{f}"].iloc[0])
                    if np.isnan(m): m = 0.0
                    if np.isnan(s) or s < 1e-12: s = 1.0
                    self.stats[freq][f] = (m, s)
            except Exception as e:
                print(f"  ⚠ {freq} 统计量查询失败: {e}")
                self.stats[freq] = {f: (0.0, 1.0) for f in SELECTED_FIELDS}

        self._build_calendar(start, end)
        print(f"[Pipeline] 完成: {len(self.all_trading_days)} 交易日")

    def _build_calendar(self, start, end):
        import dai
        try:
            df = dai.query(
                "SELECT DISTINCT date::DATE::DATETIME AS td "
                "FROM bigalpha_2026_stock_bar1m ORDER BY td",
                filters={"date": [f"{start} 00:00:00", f"{end} 23:59:59"]},
                compression=True,
            ).df()
            self.all_trading_days = sorted(df["td"].unique())
            self.day_to_idx = {d: i for i, d in enumerate(self.all_trading_days)}
        except Exception as e:
            print(f"  ⚠ 交易日历获取失败: {e}")
            print(f"  → 使用降级方案：按自然日生成日历")
            dates = pd.date_range(start, end, freq="B")
            self.all_trading_days = list(dates)
            self.day_to_idx = {d: i for i, d in enumerate(self.all_trading_days)}

    # ---- 数据索引化 ----
    def index_data(self, data_by_freq):
        """
        将原始DataFrame转换为嵌套字典，O(1)查找。
        按天批量处理，大幅减少 groupby 调用次数。

        注意: 会修改 data_by_freq 的 DataFrame，调用后应 del 原始数据
        """
        indexed = {}
        import gc

        for freq in self.freq_names:
            df = data_by_freq[freq]
            # 转 str 避免 compression=True 导致的 category 类型问题
            df["instrument"] = df["instrument"].astype(str)
            df["_ds"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            T_max = MAX_BARS[freq]
            F = len(SELECTED_FIELDS)
            stats = self.stats.get(freq, {})
            vol_fields = self.volume_fields
            field_list = list(SELECTED_FIELDS)

            indexed[freq] = {}

            # 按天批量处理
            for ds, day_df in df.groupby("_ds", sort=False):
                # 直接用 groupby 的 key 跟踪索引，避免 category unique() 不一致
                inst_idx = {}
                next_idx = 0
                day_groups = list(day_df.groupby("instrument", sort=False))
                B = len(day_groups)

                tensor = np.zeros((B, T_max, F), dtype=np.float32)
                mask = np.zeros((B, T_max), dtype=np.bool_)

                for inst, inst_df in day_groups:
                    if inst not in inst_idx:
                        inst_idx[inst] = next_idx
                        next_idx += 1
                    i = inst_idx[inst]

                    inst_df = inst_df.sort_values("date").iloc[-T_max:]
                    n = min(len(inst_df), T_max)
                    if n == 0:
                        continue

                    # 批量取出所有字段 → [n, F] numpy
                    vals = inst_df[field_list].values.astype(np.float32)

                    # log1p 成交量字段
                    for j, field in enumerate(field_list):
                        if self.use_log_volume and field in vol_fields:
                            vals[:, j] = np.log1p(np.maximum(vals[:, j], 0))

                    # 归一化
                    for j, field in enumerate(field_list):
                        if field in stats:
                            m, s = stats[field]
                            vals[:, j] = (vals[:, j] - m) / s

                    tensor[i, :n, :] = vals
                    mask[i, :n] = True

                # 拆分为 per-instrument 条目
                for inst, _ in day_groups:
                    i = inst_idx[inst]
                    if mask[i].any():
                        indexed[freq].setdefault(inst, {})[ds] = (
                            tensor[i:i + 1].copy(),
                            mask[i:i + 1].copy(),
                        )

            data_by_freq[freq] = None
            gc.collect()

        return indexed

    # ---- 完整模型输入 ----
    def build_model_input(self, data_by_freq, instruments, date_idx, _=None):
        """
        为指定日期和股票列表构建模型输入。
        支持原始DataFrame或已索引化的dict。
        """
        B = len(instruments)
        D = self.lookback_days

        model_data = {
            freq: np.zeros((B, D, MAX_BARS[freq], NUM_FIELDS), dtype=np.float32)
            for freq in self.freq_names
        }
        day_mask = np.zeros((B, D), dtype=np.bool_)
        bar_masks = {
            freq: np.zeros((B, D, MAX_BARS[freq]), dtype=np.bool_)
            for freq in self.freq_names
        }

        lookback_start = max(0, date_idx - self.lookback_days + 1)
        lookback_dates = self.all_trading_days[lookback_start:date_idx + 1]
        lookback_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in lookback_dates]
        offset = D - len(lookback_dates)

        for freq in self.freq_names:
            store = data_by_freq[freq]
            T_max = MAX_BARS[freq]
            for i, inst in enumerate(instruments):
                inst_dict = store.get(inst)
                if inst_dict is None:
                    continue
                for d_idx, ds in enumerate(lookback_strs):
                    out_idx = offset + d_idx
                    if out_idx < 0 or out_idx >= D:
                        continue
                    day_mask[i, out_idx] = True
                    tup = inst_dict.get(ds)
                    if tup is None:
                        continue
                    t, m = tup
                    if t.shape[0] > 0:
                        nb = min(t.shape[1], T_max)
                        model_data[freq][i, out_idx, :nb, :] = t[0, :nb, :]
                        bar_masks[freq][i, out_idx, :nb] = m[0, :nb]

        return (
            {f: torch.from_numpy(model_data[f]) for f in self.freq_names},
            torch.from_numpy(day_mask),
            {f: torch.from_numpy(bar_masks[f]) for f in self.freq_names},
        )

    # ---- 标签构建 ----
    def build_labels(self, instruments, date_idx, horizon=1):
        """构建未来N日收益率标签"""
        import dai
        if date_idx + horizon >= len(self.all_trading_days):
            return None, None

        td = self.all_trading_days[date_idx + horizon]
        td_str = pd.to_datetime(td).strftime("%Y-%m-%d")
        insts = "', '".join(instruments)

        try:
            df = dai.query(
                f"""
                SELECT instrument,
                       (LAST(close) - FIRST(open)) / NULLIF(FIRST(open), 0) AS ret
                FROM bigalpha_2026_stock_bar1m
                WHERE instrument IN ('{insts}')
                GROUP BY instrument, date::DATE
                """,
                filters={"date": [f"{td_str} 09:00:00", f"{td_str} 17:00:00"]},
                compression=True,
            ).df()

            d = dict(zip(df["instrument"], df["ret"]))
            labels = np.array([d.get(inst, np.nan) for inst in instruments], dtype=np.float32)
            valid = ~np.isnan(labels)
            return torch.from_numpy(labels), torch.from_numpy(valid)
        except Exception as e:
            print(f"  ⚠ 标签失败: {e}")
            return None, None


class SimpleDataLoader:
    """按截面批量加载，使用索引化数据实现O(1)查找"""
    def __init__(self, pipeline, indexed_data, instruments, date_indices,
                 batch_size=128, forward_horizon=1, shuffle=True):
        self.pipeline = pipeline
        self.indexed = indexed_data
        self.instruments = instruments
        self.date_indices = date_indices
        self.batch_size = batch_size
        self.forward_horizon = forward_horizon
        self.shuffle = shuffle

    def __iter__(self):
        di_list = list(self.date_indices)
        if self.shuffle:
            np.random.shuffle(di_list)
        for di in di_list:
            insts = list(self.instruments)
            if self.shuffle:
                np.random.shuffle(insts)
            for s in range(0, len(insts), self.batch_size):
                batch = insts[s:s + self.batch_size]
                if len(batch) < 10:
                    continue
                try:
                    md, dm, bm = self.pipeline.build_model_input(
                        self.indexed, batch, di, None)
                    labels, vm = self.pipeline.build_labels(
                        batch, di, self.forward_horizon)
                    if labels is not None and vm is not None and vm.sum() >= 5:
                        yield md, dm, bm, labels, vm
                except Exception:
                    pass


# ============================================================
# 共享工具函数 (train + main 共用)
# ============================================================
def query_all_freqs(freq_names, start_str, end_str):
    """查询所有频率数据"""
    import time
    import dai
    from config_mamba import SELECTED_FIELDS, FREQ_TABLES
    data = {}
    fields = ", ".join(SELECTED_FIELDS)
    for freq in freq_names:
        t0 = time.time()
        data[freq] = dai.query(
            f"SELECT date, instrument, {fields} FROM {FREQ_TABLES[freq]} "
            f"ORDER BY date, instrument",
            filters={"date": [f"{start_str} 00:00:00", f"{end_str} 23:59:59"]},
            compression=True,
        ).df()
        print(f"    {freq}: {len(data[freq]):,} 行, "
              f"{data[freq]['instrument'].nunique()} 只, {time.time()-t0:.0f}s")
    return data


def build_lgb_features(pipeline, indexed, instruments, date_idx, config):
    """
    从 indexed 数据构建 LGB 扁平特征。
    每只股票: 回看 D 天，每天取 bar_sample_idx 个采样点的所有字段
    → n_fields × n_samples × D 维向量
    缺失填 0。
    返回: X [N, feat_dim], valid [N]
    """
    import numpy as np
    import pandas as pd
    from config_mamba import MAX_BARS

    freq = config.freq_names[0]
    bar_idx = config.bar_sample_idx
    n_fields = len(SELECTED_FIELDS)
    n_samples = len(bar_idx)
    D = config.lookback_days

    lookback_start = max(0, date_idx - D + 1)
    lookback_dates = pipeline.all_trading_days[lookback_start:date_idx + 1]
    lookback_strs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in lookback_dates]
    offset = D - len(lookback_dates)

    N = len(instruments)
    feat_dim = n_fields * n_samples * D
    X = np.zeros((N, feat_dim), dtype=np.float32)
    valid = np.zeros(N, dtype=np.bool_)

    store = indexed[freq]
    for i, inst in enumerate(instruments):
        inst_dict = store.get(inst)
        if inst_dict is None:
            continue
        has_data = False
        for d_idx, ds in enumerate(lookback_strs):
            out_day = offset + d_idx
            if out_day < 0 or out_day >= D:
                continue
            tup = inst_dict.get(ds)
            if tup is None:
                continue
            t, m = tup
            if t.shape[0] == 0:
                continue
            for s_idx, bar_pos in enumerate(bar_idx):
                if bar_pos < t.shape[1] and m[0, bar_pos]:
                    start_col = (out_day * n_samples + s_idx) * n_fields
                    X[i, start_col:start_col + n_fields] = t[0, bar_pos, :]
                    has_data = True
        valid[i] = has_data

    return X, valid
