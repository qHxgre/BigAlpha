"""BigAlpha 服务器：5m 五档 SQL 归一化、缓存、缩小六头 AttnLOB 训练。"""
from __future__ import annotations
import gc
import json
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from server5m_data import build_cache
from server5m_train_multitask import train_multitask
from server5m_factor_export import export_pair

TABLE="bigalpha_2026_stock_bar5m"
PRICE=[f"{side}_price{i}" for i in range(1,6) for side in ("ask","bid")]
VOLUME=[f"{side}_volume{i}" for i in range(1,6) for side in ("ask","bid")]
LOB=[x for i in range(1,6) for x in
     (f"ask_price{i}",f"ask_volume{i}",f"bid_price{i}",f"bid_volume{i}")]

LOGGER = logging.getLogger("server_5m_small")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


def _mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return -1.0


def _log_mem(tag):
    rss_mb = _mem_mb()
    gpu_text = ""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            gpu_text = f" cuda_allocated={allocated:.1f}MB cuda_reserved={reserved:.1f}MB"
    except Exception:
        pass
    if rss_mb >= 0:
        LOGGER.info("[mem] %s rss=%.1fMB%s", tag, rss_mb, gpu_text)
    else:
        LOGGER.info("[mem] %s%s", tag, gpu_text)

DEFAULT_CONFIG={
 "query_start":"2022-01-01 00:00:00","query_end":"2024-12-31 23:59:59",
 "normalized_parquet":"cache/server_5m_normalized_2022_2024.parquet",
 "cache_dir":"cache/server_5m_small_cache","force_query":False,
 "force_cache":False,"run_name":"server_5m_attn_small",
 "output_root":"training_results/server_5m_small","optimizer":"adamw",
 "lr":0.001,"weight_decay":0.0001,"batch_size":1024,"epochs":50,
 "feature_dim":32,"head_dim":16,"dropout":0.15,
 "regression_weights":[0.25,0.15,0.10],
 "classification_weights":[1.0,0.5,0.25],"ordinal_sigma":0.7,
 "selection_score":"cls_blend_532","early_stop_metric":"loss",
 "early_stop_patience":10,"grad_clip":1.0,"num_workers":0,
 "amp":True,"seed":42,
}

MODEL_FILE=Path(__file__).resolve().with_name("server_5m_attn_small_best.json")
MODEL_CONFIG_FILE=Path(__file__).resolve().with_name("server_5m_attn_small_config.json")
DEPLOY_METHOD="fixed532"

def _to_json_state_dict(state):
    """state_dict → 纯 Python JSON 结构（{dtype,shape,data}）。"""
    return {name: {"dtype": str(t.dtype).removeprefix("torch."),
                   "shape": list(t.shape),
                   "data": t.detach().cpu().flatten().tolist()}
            for name, t in state.items()}


def _from_json_state_dict(payload):
    """反向：JSON → state_dict 张量。"""
    import torch
    state = {}
    for name, meta in payload.items():
        dtype_name = str(meta["dtype"]).removeprefix("torch.")
        tensor = torch.tensor(meta["data"], dtype=getattr(torch, dtype_name))
        state[name] = tensor.reshape(meta["shape"])
    return state


def save_model(checkpoint, target):
    """以 JSON 格式保存模型。"""
    payload = {
        "model_type": checkpoint.get("model_type"),
        "model": checkpoint.get("model"),
        "epoch": checkpoint.get("epoch"),
        "selection_score": checkpoint.get("selection_score"),
        "val_metrics": checkpoint.get("val_metrics"),
        "state_dict": _to_json_state_dict(checkpoint["model_state"]),
    }
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf8")
    return target


def load_model(path, map_location=None):
    """以 JSON 格式加载模型。"""
    import torch
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    state = _from_json_state_dict(payload["state_dict"])
    if map_location is not None:
        state = {k: v.to(map_location) for k, v in state.items()}
    return {"model_state": state,
            "model_type": payload.get("model_type"),
            "model": payload.get("model"),
            "epoch": payload.get("epoch"),
            "selection_score": payload.get("selection_score"),
            "val_metrics": payload.get("val_metrics")}


def normalized_sql(table=TABLE):
    vmax="GREATEST("+",".join("b."+x for x in VOLUME)+")"
    base=["b.date AS datetime","CAST(b.date AS DATE) AS trade_date",
          "b.instrument AS stock_code","b.close",
          "FIRST_VALUE(b.close) OVER (PARTITION BY b.instrument, CAST(b.date AS DATE) ORDER BY b.date) AS open_px",
          f"{vmax} AS volume_max"]
    base += [f"b.{x}" for x in PRICE+VOLUME]
    rel=["datetime","trade_date","stock_code","close","volume_max"]
    rel += [f"({x}/NULLIF(open_px,0)-1.0) AS r_{x}" for x in PRICE]
    rel += VOLUME
    final=["datetime AS date","trade_date","stock_code","close"]
    for i in range(1,6):
      for side in ("ask","bid"):
        x=f"{side}_price{i}"; r=f"r_{x}"
        final.append(f"(({r}-AVG({r}) OVER (PARTITION BY datetime))/NULLIF(STDDEV_POP({r}) OVER (PARTITION BY datetime),0)) AS {x}")
        v=f"{side}_volume{i}"
        final.append(f"({v}/NULLIF(volume_max,0)) AS {v}")
    return ("WITH base AS (SELECT "+",".join(base)+f" FROM {table} b), "
            "rel AS (SELECT "+",".join(rel)+" FROM base) SELECT "+
            ",".join(final)+" FROM rel ORDER BY date, stock_code")

def query_data(table, start_date, end_date, instruments=None, compression=True):
    """DAI 一次拉取指定时间区间的归一化后 5m 数据。

    返回 pandas DataFrame，列：trade_date, datetime, stock_code, close, 20 维 LOB。
    SQL 已完成 open-relative 价格、截面 zscore、十量最大值归一化。
    """
    import dai
    LOGGER.info("[dai] query start table=%s range=[%s, %s] instruments=%s",
                table, start_date, end_date,
                None if instruments is None else len(instruments))
    _log_mem("before dai.query")
    t0 = time.time()
    filters = {"date": [str(start_date), str(end_date)]}
    if instruments is not None:
        filters["instrument"] = list(instruments)
    frame = dai.query(normalized_sql(table), filters=filters,
                      compression=compression).df()
    LOGGER.info("[dai] query done rows=%d cols=%d elapsed=%.1fs",
                len(frame), frame.shape[1], time.time() - t0)
    _log_mem("after dai.df")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame[["trade_date", "date", "stock_code", "close", *LOB]]


def query_and_cache(cfg=None):
    """逐月查询并追加到单个 parquet，避免三年数据同时驻留内存。"""
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    out = Path(cfg["normalized_parquet"])
    if out.exists() and not cfg.get("force_query", False):
        LOGGER.info("[query cache hit] %s", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(cfg["query_start"]), pd.Timestamp(cfg["query_end"])
    months = pd.date_range(start.normalize().replace(day=1), end, freq="MS")
    writer, total_rows = None, 0
    try:
        for number, month in enumerate(months, 1):
            chunk_start = max(start, month)
            chunk_end = min(end, month + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1))
            LOGGER.info("[query] month %d/%d %s ~ %s", number, len(months), chunk_start, chunk_end)
            frame = query_data(TABLE, chunk_start, chunk_end).rename(columns={"date": "datetime"})
            arrow_table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, arrow_table.schema, compression="snappy")
            writer.write_table(arrow_table)
            total_rows += len(frame)
            del frame, arrow_table
            gc.collect()
            _log_mem(f"month {number} cached")
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"查询区间没有数据: {start} ~ {end}")
    LOGGER.info("[normalized parquet] %s rows=%d", out, total_rows)
    return out

def build_data_cache(cfg=None):
    """读取 DAI 数据并紧接着构建训练缓存。"""
    cfg={**DEFAULT_CONFIG,**(cfg or {})}
    LOGGER.info("[cache] start cfg.run_name=%s", cfg.get("run_name"))
    t0 = time.time()
    source=query_and_cache(cfg)
    LOGGER.info("[cache] normalized parquet ready elapsed=%.1fs", time.time() - t0)
    _log_mem("before build_cache")
    t0 = time.time()
    cache=build_cache(source,cfg["cache_dir"],force=cfg.get("force_cache",False),
                      feature_mode="pre_normalized",price_transform="none",
                      volume_transform="none")
    LOGGER.info("[cache] build_cache done elapsed=%.1fs", time.time() - t0)
    _log_mem("after build_cache")
    return source,cache


def prepare_cache(cfg=None):
    return build_data_cache(cfg)[1]

def train(cfg=None):
    """严格 2022-23/2024H1/2024H2，按 val loss 早停，输出 5:3:2 与 Ridge。"""
    import torch
    cfg={**DEFAULT_CONFIG,**(cfg or {})}; prepare_cache(cfg)
    train_cfg={k:v for k,v in cfg.items() if k not in
      {"query_start","query_end","normalized_parquet","cache_dir",
       "force_query","force_cache"}}
    train_cfg.update(data={"cache_dir":cfg["cache_dir"]},model_type="attention_small",
                     rank_loss_weight=0.0)
    result=train_multitask(train_cfg)
    factors=export_pair(result,train_cfg,cfg["run_name"],years=(2022,2023,2024))
    MODEL_FILE.parent.mkdir(parents=True,exist_ok=True)
    src_pt=Path(result)/"best_model.pt"
    if src_pt.exists():
      checkpoint=torch.load(src_pt,map_location="cpu",weights_only=False)
      save_model(checkpoint,MODEL_FILE)
    MODEL_CONFIG_FILE.write_text(json.dumps(train_cfg,ensure_ascii=False,indent=2),encoding="utf8")
    summary={"result_dir":str(Path(result).resolve()),"factors":factors,
             "table":TABLE,"normalization":"SQL open-relative price + timestamp cross-sectional zscore; ten-volume max ratio"}
    Path(result,"server_summary.json").write_text(
      json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
    return summary

def _query_inference(table,start_date,end_date,instruments=None):
    return query_data(table,start_date,end_date,instruments=instruments)


def _half_year_chunks(start_date, end_date):
    """把查询区间切成自然半年，限制单次 DataFrame 和特征数组大小。"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        return []
    chunks = []
    for year in range(start.year, end.year + 1):
        periods = (
            (pd.Timestamp(year=year, month=1, day=1),
             pd.Timestamp(year=year, month=6, day=30, hour=23, minute=59, second=59)),
            (pd.Timestamp(year=year, month=7, day=1),
             pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)),
        )
        for period_start, period_end in periods:
            chunk_start = max(start, period_start)
            chunk_end = min(end, period_end)
            if chunk_start <= chunk_end:
                chunks.append((chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
                               chunk_end.strftime("%Y-%m-%d %H:%M:%S")))
    return chunks

def _inference_arrays(frame):
    """把 DataFrame 拼成 (N, 48, 20) 特征数组 + (date, instrument) 索引。

    优先用预计算 + sort+reshape 快路径；缺帧时回退到逐日 scatter。
    """
    t0 = time.time()
    n_total = len(frame)
    LOGGER.info("[arrange] rows=%d days=%d", n_total, frame["trade_date"].nunique())
    _log_mem("arrange start")

    # ---- 一次性预计算 ----
    day_codes, day_uniques = pd.factorize(frame["trade_date"])
    stock_str = frame["stock_code"].astype(str)
    stock_codes, stock_uniques = pd.factorize(stock_str)
    # 5min slot: 0-47
    dt = frame["date"]
    slot = (dt.dt.hour * 12 + dt.dt.minute // 5).to_numpy()
    lob = frame[LOB].to_numpy(np.float32)

    # ---- 检查完整性 ----
    gb_key = day_codes * (max(stock_codes) + 1) + stock_codes
    # 只统计实际存在的组合；不同交易日允许使用不同股票池。
    _, group_counts = np.unique(gb_key, return_counts=True)
    complete = bool(len(group_counts) and np.all(group_counts == 48))

    if complete:
        LOGGER.info("[arrange] complete groups=%d, fast reshape", len(group_counts))
        # 按 (day, stock, slot) 排序
        order = np.lexsort((slot, stock_codes, day_codes))
        lob_sorted = np.nan_to_num(lob[order], nan=0., posinf=0., neginf=0.)
        n_samples = len(order) // 48
        X = lob_sorted.reshape(n_samples, 48, -1)
        # index: 每组取首行
        idx = order[::48]
        dates = [pd.Timestamp(day_uniques[day_codes[i]]) for i in idx]
        instruments = [stock_uniques[stock_codes[i]] for i in idx]
        LOGGER.info("[arrange] X.shape=%s elapsed=%.1fs", X.shape, time.time() - t0)
        _log_mem("arrange done")
        return X, pd.DataFrame({"date": dates, "instrument": instruments})

    # ---- 降级：逐日 scatter（缺帧/重复帧） ----
    LOGGER.info("[arrange] incomplete groups, per-day scatter fallback")
    arrs = []
    dates_out = []
    insts_out = []
    uniq_days = np.unique(day_codes)
    total_days = len(uniq_days)
    for di, day_val in enumerate(uniq_days):
        mask = day_codes == day_val
        day_str = day_uniques[day_val]
        part_slot = slot[mask]
        part_stock = stock_codes[mask]
        part_stock_str = stock_str[mask]
        part_lob = lob[mask]
        times = np.sort(np.unique(part_slot))
        if len(times) != 48:
            # 旧代码 raise 异常；这里同样 raise
            raise ValueError(f"{pd.Timestamp(day_str).date()}: "
                             f"expected 48 5m bars, got {len(times)}")
        stocks_u = np.sort(np.unique(part_stock_str))
        si = {x: i for i, x in enumerate(stocks_u)}
        ti = {x: i for i, x in enumerate(times)}
        x = np.full((len(stocks_u), 48, 20), np.nan, dtype=np.float32)
        rows = part_stock_str.map(si).to_numpy()
        cols = pd.Series(part_slot).map(ti).to_numpy()
        x[rows, cols] = part_lob
        arrs.append(np.nan_to_num(x, nan=0., posinf=0., neginf=0.))
        dates_out.extend([pd.Timestamp(day_str)] * len(stocks_u))
        insts_out.extend(stocks_u)
        if (di + 1) % 50 == 0 or di + 1 == total_days:
            LOGGER.info("[arrange] day %d/%d stocks=%d",
                        di + 1, total_days, len(stocks_u))
    X = np.concatenate(arrs)
    LOGGER.info("[arrange] X.shape=%s elapsed=%.1fs", X.shape, time.time() - t0)
    _log_mem("arrange done")
    del arrs, lob, slot, stock_codes, day_codes; gc.collect()
    return X, pd.DataFrame({"date": dates_out, "instrument": insts_out})

def main(datasources,start_date,end_date):
    """平台入口：训练表固定；这里只读取注入的 5m 测试表并用最佳权重推理。

    固定按自然半年切分：每半年独立完成 DAI 读取、特征拼装、batch 推理，
    每块结束立即释放 X、DataFrame 和显存缓存。
    """
    import dai, torch
    from torch.utils.data import DataLoader
    from server5m_model import SmallMultiTaskAttentionLOB
    from server5m_train import InferenceDataset
    LOGGER.info("=" * 60)
    LOGGER.info("[main] start datasources=%s range=[%s, %s]",
                list(datasources.keys()), start_date, end_date)
    _log_mem("main start")
    if not MODEL_FILE.exists():
      raise FileNotFoundError("缺少最佳权重：{MODEL_FILE} 不存在；请确认已把 server_5m_attn_small_best.json 放在与 server_5m_small.py 同级目录")
    table=datasources.get("bar5m",TABLE)
    LOGGER.info("[main] table=%s", table)

    # ---- 先查股票池，缩小后续 bar5m 查询范围 ----
    LOGGER.info("[main] fetching instrument pool")
    t0 = time.time()
    pool = dai.query("SELECT date, instrument FROM bigalpha_2026_instruments",
                     filters={"date": [start_date, end_date]}).df()
    pool["date"] = pd.to_datetime(pool["date"]).dt.normalize()
    all_instruments = pool["instrument"].unique().tolist()
    LOGGER.info("[main] pool rows=%d unique_instruments=%d elapsed=%.1fs",
                len(pool), len(all_instruments), time.time() - t0)
    _log_mem("after pool")

    chunks = _half_year_chunks(start_date, end_date)
    use_chunked = len(chunks) > 1
    LOGGER.info("[main] half_year_chunks=%d use_chunked=%s",
                len(chunks), use_chunked)
    for i, (ys, ye) in enumerate(chunks, 1):
        LOGGER.info("[main] chunk %d/%d %s ~ %s", i, len(chunks), ys, ye)

    def _score_one(df_one):
        X_local, idx_local = _inference_arrays(df_one)
        loader_local = DataLoader(InferenceDataset(X_local), batch_size=1024,
                                  shuffle=False, num_workers=0,
                                  pin_memory=device.type == "cuda")
        outs = []
        with torch.no_grad():
            for batch in loader_local:
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    logits = model(batch.to(device, non_blocking=True))["classification"]
                outs.append((torch.softmax(logits.float(), dim=2) @ classes).cpu().numpy())
        expected_local = np.concatenate(outs)
        score_local = np.empty(len(expected_local), np.float64)
        day_local = idx_local["date"].to_numpy()
        for value in np.unique(day_local):
            mask = day_local == value
            section = expected_local[mask]
            section = (section - section.mean(0, keepdims=True)) / np.maximum(
                section.std(0, keepdims=True), 1e-8)
            score_local[mask] = section @ np.array([.5, .3, .2])
        idx_local["score"] = score_local
        del X_local, loader_local, outs, expected_local, score_local
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _log_mem("chunk end")
        return idx_local[["date", "instrument", "score"]]

    if not use_chunked:
        LOGGER.info("[main] single-pass path")
        t0 = time.time()
        frame = _query_inference(table, start_date, end_date, instruments=all_instruments)
        LOGGER.info("[main] dai rows=%d elapsed=%.1fs", len(frame), time.time() - t0)
        _log_mem("after dai")
        t0 = time.time()
        checkpoint = load_model(MODEL_FILE, map_location="cpu")
        model = SmallMultiTaskAttentionLOB(**checkpoint["model"])
        model.load_state_dict(checkpoint["model_state"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        classes = torch.arange(5, device=device, dtype=torch.float32)
        LOGGER.info("[main] model ready device=%s elapsed=%.1fs", device, time.time() - t0)
        _log_mem("after model")
        result = _score_one(frame)
        del frame
        gc.collect()
    else:
        LOGGER.info("[main] chunked path: load model once")
        t0 = time.time()
        checkpoint = load_model(MODEL_FILE, map_location="cpu")
        model = SmallMultiTaskAttentionLOB(**checkpoint["model"])
        model.load_state_dict(checkpoint["model_state"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        classes = torch.arange(5, device=device, dtype=torch.float32)
        LOGGER.info("[main] model ready device=%s elapsed=%.1fs", device, time.time() - t0)
        _log_mem("after model")
        pieces = []
        for i, (ys, ye) in enumerate(chunks, 1):
            t0 = time.time()
            LOGGER.info("[main] >> chunk %d/%d %s ~ %s", i, len(chunks), ys, ye)
            _log_mem(f"chunk {i} start")
            frame_year = _query_inference(table, ys, ye, instruments=all_instruments)
            LOGGER.info("[main] dai rows=%d elapsed=%.1fs",
                        len(frame_year), time.time() - t0)
            _log_mem(f"chunk {i} after dai")
            piece = _score_one(frame_year)
            pieces.append(piece)
            LOGGER.info("[main] << chunk %d/%d done rows=%d elapsed=%.1fs",
                        i, len(chunks), len(piece), time.time() - t0)
            del frame_year, piece
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result = pd.concat(pieces, ignore_index=True) if pieces else \
                 pd.DataFrame(columns=["date", "instrument", "score"])
        LOGGER.info("[main] concat rows=%d", len(result))
        _log_mem("after concat")
        del pieces
        gc.collect()

    LOGGER.info("[main] merging with instrument pool")
    result = (pd.merge(result, pool, on=["date", "instrument"], how="inner")
              .replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
              .drop_duplicates(["date", "instrument"])[["date", "instrument", "score"]]
              .sort_values(["date", "instrument"]).reset_index(drop=True))
    LOGGER.info("[main] result rows=%d days=%d", len(result), result.date.nunique())
    _log_mem("main end")
    LOGGER.info("=" * 60)
    return result
