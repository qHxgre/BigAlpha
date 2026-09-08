"""
数据下载脚本 —— 一次性把 BigAlpha 2026 端到端赛道的本地压缩表拉到本地并存为 parquet。

使用前提:
    1. pip install bigquant -i https://pypi.bigquant.com/simple/
    2. bq auth --apikey <AK.SK>   # 只需执行一次，认证信息存 ~/.bigquant/

注意:
    - SDK 有 quota 限制，运行一次后复用本地 parquet，不要反复拉取。
    - 本地表 bigalpha_2026_e2e_bar* 是压缩格式:
        价格单位"分"（整数）、instrument_id 整数、3 档盘口、deal_number 为成交笔数。
    - bigalpha_2026_instruments 表本地 SDK 无读取权限，故不下载；
      本地训练直接以 instrument_id（整数）作为股票分组键。
"""

import os
import sys
import time
import logging
import argparse
import tempfile
import glob
import re
from datetime import timedelta

import pandas as pd
import numpy_compat  # noqa: F401
import pyarrow as pa
import pyarrow.parquet as pq

# ---- 目录配置 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(SCRIPT_DIR, "local_data")
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# ---- 日期范围（训练集，写死） ----
TRAIN_START = os.environ.get("BQ_TRAIN_START", "2019-01-01")
TRAIN_END   = os.environ.get("BQ_TRAIN_END", "2023-12-31")

# ---- 端到端本地压缩表清单 ----
# 对应云端原始表 bigalpha_2026_stock_bar*m（格式不同，见 features.py）
E2E_TABLES = [
    "bigalpha_2026_e2e_bar1m",
    "bigalpha_2026_e2e_bar5m",
    "bigalpha_2026_e2e_bar15m",
    "bigalpha_2026_e2e_bar30m",
]

# instruments 映射表（本地 SDK 暂无权限，保留常量供云端推理侧使用）
INSTRUMENTS_TABLE = "bigalpha_2026_instruments"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _parquet_path(table: str, start: str, end: str) -> str:
    """返回对应表+日期区间的本地 parquet 文件路径。"""
    tag = f"{start}_{end}".replace("-", "")
    return os.path.join(LOCAL_DATA_DIR, f"{table}__{tag}.parquet")


def _parse_parquet_range(path: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Parse local cache filenames like table__20190101_20231231.parquet."""
    m = re.search(r"__(\d{8})_(\d{8})\.parquet$", os.path.basename(path))
    if not m:
        return None
    return pd.Timestamp(m.group(1)), pd.Timestamp(m.group(2))


def local_parquet_paths(table: str, start: str = TRAIN_START, end: str = TRAIN_END) -> list[str]:
    """
    Return local parquet caches that overlap the requested range.

    Prefer an exact full-range cache if present. Otherwise return sorted shard
    files such as 20190101_20231231 + 20240101_20241231. This lets training
    scripts use newly downloaded 2024 data without manually concatenating a
    multi-GB parquet file.
    """
    exact = _parquet_path(table, start, end)
    if os.path.exists(exact):
        return [exact]

    req_start = pd.Timestamp(start)
    req_end = pd.Timestamp(end)
    paths: list[tuple[pd.Timestamp, str]] = []
    for path in glob.glob(os.path.join(LOCAL_DATA_DIR, f"{table}__*.parquet")):
        parsed = _parse_parquet_range(path)
        if parsed is None:
            continue
        sd, ed = parsed
        if ed < req_start or sd > req_end:
            continue
        paths.append((sd, path))
    return [p for _, p in sorted(paths)]


def download_table(
    table: str,
    start: str,
    end: str,
    extra_filters: dict | None = None,
    *,
    force: bool = False,
    batch_size: int = 200_000,
) -> str:
    """
    下载一张表到本地 parquet，已存在且不强制刷新时直接返回路径。

    Args:
        table:         数据表名（BigQuant DAI 表）
        start / end:   日期区间，格式 "YYYY-MM-DD"
        extra_filters: 附加过滤条件，会与 date 过滤合并
        force:         True 则强制重新下载（忽略本地缓存）
        batch_size:    流式读取的每批行数，降低内存峰值
    Returns:
        本地 parquet 文件路径
    """
    path = _parquet_path(table, start, end)
    if os.path.exists(path) and not force:
        log.info("复用本地缓存: %s", path)
        return path

    try:
        from bigquant import dai  # 仅在需要下载时才导入
    except ImportError:
        log.error("未找到 bigquant 包，请先执行: pip install bigquant -i https://pypi.bigquant.com/simple/")
        sys.exit(1)

    filters: dict = {"date": [start, end]}
    if extra_filters:
        filters.update(extra_filters)

    log.info("开始下载: %s  [%s ~ %s]", table, start, end)
    t0 = time.time()

    def _is_size_limit_error(exc: Exception) -> bool:
        msg = str(exc)
        return "不能一次性读取超过" in msg or "200.0 MB" in msg

    def _query_batches(sd: pd.Timestamp, ed: pd.Timestamp):
        part_filters = dict(filters)
        part_filters["date"] = [sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d")]
        result = dai.query(f"SELECT * FROM {table}", filters=part_filters)
        reader = result.fetch_arrow_reader(batch_size=batch_size)
        yield from reader

    def _write_batch(writer: pq.ParquetWriter | None, batch: pa.RecordBatch, tmp_path: str) -> pq.ParquetWriter:
        table_arrow = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table_arrow.schema, compression="snappy")
        writer.write_table(table_arrow)
        return writer

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=f".{table}__",
        suffix=".parquet.tmp",
        dir=LOCAL_DATA_DIR,
    )
    os.close(tmp_fd)
    os.remove(tmp_path)

    writer: pq.ParquetWriter | None = None
    total_rows = 0

    def _download_range(sd: pd.Timestamp, ed: pd.Timestamp) -> None:
        nonlocal writer, total_rows
        if sd > ed:
            return

        sd_s = sd.strftime("%Y-%m-%d")
        ed_s = ed.strftime("%Y-%m-%d")
        rows_before = total_rows
        rows = 0
        try:
            for batch in _query_batches(sd, ed):
                if batch.num_rows == 0:
                    continue
                writer = _write_batch(writer, batch, tmp_path)
                rows += batch.num_rows
                total_rows += batch.num_rows
        except Exception as exc:
            if _is_size_limit_error(exc) and sd < ed and total_rows == rows_before:
                mid = sd + (ed - sd) / 2
                mid = mid.normalize()
                if mid < sd:
                    mid = sd
                left_ed = mid
                right_sd = mid + timedelta(days=1)
                log.info(
                    "区间过大，自动拆分: %s [%s ~ %s] -> [%s ~ %s] + [%s ~ %s]",
                    table,
                    sd_s, ed_s,
                    sd_s, left_ed.strftime("%Y-%m-%d"),
                    right_sd.strftime("%Y-%m-%d"), ed_s,
                )
                _download_range(sd, left_ed)
                _download_range(right_sd, ed)
                return
            raise

        if rows == 0:
            log.info("分片为空: %s  [%s ~ %s]", table, sd_s, ed_s)
            return

        log.info("分片完成: %s  [%s ~ %s]  行数=%d", table, sd_s, ed_s, rows)

    try:
        _download_range(start_ts, end_ts)
    except Exception:
        if writer is not None:
            writer.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"{table} 在 [{start} ~ {end}] 未下载到任何数据")

    os.replace(tmp_path, path)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(path) / 1024 / 1024
    log.info(
        "下载完成: %s  行数=%d  大小=%.1f MB  耗时=%.1fs",
        path, total_rows, size_mb, elapsed,
    )
    return path


def download_all(
    start: str = TRAIN_START,
    end: str = TRAIN_END,
    *,
    force: bool = False,
) -> dict[str, str]:
    """
    一次性下载所有 E2E 本地表。
    返回 {table_name: local_parquet_path} 字典。
    """
    paths: dict[str, str] = {}
    for table in E2E_TABLES:
        log.info("=== 下载 %s ===", table)
        paths[table] = download_table(table, start, end, force=force)

    log.info("=== 全部下载完成 ===")
    for t, p in paths.items():
        size_mb = os.path.getsize(p) / 1024 / 1024 if os.path.exists(p) else -1
        log.info("  %-45s  %.1f MB", t, size_mb)

    return paths


# ---- CLI 入口 ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigAlpha E2E 数据下载工具")
    parser.add_argument(
        "--table",
        choices=E2E_TABLES + ["all"],
        default="all",
        help="要下载的表名，默认 all（全部）",
    )
    parser.add_argument("--start", default=TRAIN_START, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end",   default=TRAIN_END,   help="结束日期 YYYY-MM-DD")
    parser.add_argument("--force", action="store_true",  help="强制重新下载（忽略本地缓存）")
    args = parser.parse_args()

    if args.table == "all":
        download_all(args.start, args.end, force=args.force)
    else:
        download_table(args.table, args.start, args.end, force=args.force)
