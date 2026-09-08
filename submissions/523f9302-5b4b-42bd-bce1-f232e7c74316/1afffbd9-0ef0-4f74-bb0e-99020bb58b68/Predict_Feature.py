# -*- coding: utf-8 -*-
"""分块顺序读取 DAI、逐日完整截面预测（整日 GPU 前向，无股票分块）。"""

import gc
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import dai
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(
    os.path.abspath(__file__)
)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ModelSet_Framework as framework_module  # noqa: E402

from ModelSet_Framework import (  # noqa: E402
    MODEL_PATH,
    PonyFramework,
    build_date_list,
    config_from_dict,
    load_model,
)
from ModelSet_Model import build_model  # noqa: E402
from Predict_Cache import PredictBaseCache  # noqa: E402
from Support_BaseData import (  # noqa: E402
    get_standard_details,
    instruments_table,
    set_datasources,
)
from Support_FeatureCal import get_batch_feature  # noqa: E402

logger = logging.getLogger("pony_predict")
_PREDICT_LOG_NAME = "predict.log"


class _FlushFileHandler(logging.FileHandler):
    """每次 emit 后立刻 flush，保证同步落盘。"""

    def emit(self, record):
        super().emit(record)
        self.flush()


def _setup_predict_logging() -> str:
    """控制台 + 同目录 predict.log；重复调用不会重复挂 handler。"""
    log_path = os.path.join(_HERE, _PREDICT_LOG_NAME)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_file = False
    has_stream = False
    abs_log = os.path.abspath(log_path)

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(h.baseFilename) == abs_log:
                has_file = True
        elif isinstance(h, logging.StreamHandler):
            has_stream = True

    if not has_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if not has_file:
        fh = _FlushFileHandler(
            log_path, mode="a", encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # 避免向 root 再冒泡一份
    logger.propagate = False
    return log_path


def _log_dai_summary(
    cache: PredictBaseCache,
    instruments_attempts: int,
    instruments_failures: int,
    instruments_seconds: float,
) -> None:
    bar = cache.get_dai_stats()
    logger.info(
        "  [DAI][SUMMARY] "
        "instruments_attempts=%s "
        "instruments_failures=%s "
        "instruments_seconds=%.3f "
        "bar_attempts=%s "
        "bar_successes=%s "
        "bar_failures=%s "
        "bar_cache_hits=%s "
        "bar_rows=%s "
        "bar_seconds=%.3f",
        instruments_attempts,
        instruments_failures,
        instruments_seconds,
        sum(v["attempts"] for v in bar.values()),
        sum(v["successes"] for v in bar.values()),
        sum(v["failures"] for v in bar.values()),
        sum(v["cache_hits"] for v in bar.values()),
        sum(v["rows"] for v in bar.values()),
        sum(v["seconds"] for v in bar.values()),
    )
    for delta in cache.SCALES:
        item = bar[delta]
        logger.info(
            "  [DAI][SUMMARY][SCALE] scale=%s "
            "attempts=%s successes=%s failures=%s "
            "cache_hits=%s rows=%s seconds=%.3f",
            delta,
            item["attempts"],
            item["successes"],
            item["failures"],
            item["cache_hits"],
            item["rows"],
            item["seconds"],
        )


def _build_inference_framework(
    model_path: str,
) -> PonyFramework:
    """只构建推理所需模型。

    不创建 loss、optimizer、scheduler 和 EMA 模型副本。
    权重先加载到 CPU，再把模型移动到推理设备。
    """
    requested_device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = load_model(
        model_path,
        map_location="cpu",
    )

    config = config_from_dict(
        checkpoint["config"],
        device=requested_device,
    )
    config.details_path = os.path.abspath(
        model_path
    )

    framework = PonyFramework(
        config=config,
        model_path=model_path,
    )

    framework.model = build_model(config)
    framework.model.load_state_dict(
        checkpoint["state_dict"],
        strict=False,
    )
    framework.model.eval()

    del checkpoint

    payload_key = os.path.abspath(model_path)
    framework_module._PAYLOAD_MEM.pop(
        payload_key,
        None,
    )

    gc.collect()

    actual_device = requested_device

    if requested_device == "cuda":
        try:
            framework.model.to("cuda")

        except RuntimeError as exc:
            is_oom = (
                "out of memory"
                in str(exc).lower()
            )

            if not is_oom:
                raise

            logger.warning(
                "[Predict_Feature] model loading CUDA OOM; "
                "falling back to CPU"
            )

            framework.model.to("cpu")
            actual_device = "cpu"

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()
    else:
        framework.model.to("cpu")

    framework.device = torch.device(
        actual_device
    )
    framework.config.device = actual_device
    framework.is_fitted = True

    amp_ok = (
        bool(
            getattr(
                framework.config,
                "use_bf16",
                False,
            )
        )
        and actual_device == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    framework._use_bf16 = amp_ok
    framework._autocast_dtype = (
        torch.bfloat16
        if amp_ok
        else torch.float32
    )
    framework._infer_cpu_fallback = (
        actual_device == "cpu"
    )

    framework.model.eval()

    return framework


def _forward_full_day(
    fw: PonyFramework,
    feature: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """整日截面一次前向，不按股票分块。"""
    beta = float(fw.config.score_beta)

    with fw._autocast():
        pred_rank, pred_ret, mask = fw._forward_batch(
            feature, None
        )
        score = pred_rank + beta * pred_ret

    scores = (
        score.detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    usable = (
        mask.detach()
        .cpu()
        .numpy()
        .astype(bool)
    )
    return scores, usable


def _switch_to_cpu(fw: PonyFramework, date: str) -> None:
    logger.warning(
        "[Predict_Feature] CUDA OOM day=%s; "
        "switching remaining dates to CPU",
        date,
    )
    fw.model.to("cpu")
    fw.device = torch.device("cpu")
    fw.config.device = "cpu"
    fw._use_bf16 = False
    fw._autocast_dtype = torch.float32
    fw._infer_cpu_fallback = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _predict_one_day(
    fw: PonyFramework,
    date: str,
    scale_details: dict,
    cache: PredictBaseCache,
    stock_list: List[str],
    code_to_idx: dict[str, int],
) -> Tuple[Optional[pd.DataFrame], int, int]:
    """完成单日完整截面预测（整批 GPU 前向）。"""
    n_pool = len(stock_list)

    if n_pool == 0:
        return None, 0, 0

    feature = get_batch_feature(
        date=date,
        seq_len=fw.config.seq_len,
        stock_list=stock_list,
        code_to_idx=code_to_idx,
        details_path=fw.config.details_path,
        scale_details=scale_details,
        store=cache,
    )

    usable_np = np.asarray(
        feature["usable"],
        dtype=bool,
    )
    n_usable = int(usable_np.sum())

    if n_usable == 0:
        del feature
        return None, n_pool, 0

    fw.model.eval()

    with torch.inference_mode():
        try:
            scores, usable_np = _forward_full_day(
                fw, feature
            )
        except RuntimeError as exc:
            is_cuda_oom = (
                "out of memory" in str(exc).lower()
                and str(fw.device).startswith("cuda")
            )
            if not is_cuda_oom:
                raise

            _switch_to_cpu(fw, date)
            scores, usable_np = _forward_full_day(
                fw, feature
            )

    n_usable = int(usable_np.sum())
    if n_usable == 0:
        del feature
        del scores
        return None, n_pool, 0

    codes = np.asarray(
        feature["code"],
        dtype=object,
    )

    day_df = pd.DataFrame(
        {
            "date": pd.Timestamp(date),
            "instrument": codes[usable_np],
            "score": scores[usable_np],
        }
    )

    del feature
    del codes
    del scores

    return day_df, n_pool, n_usable


def predict_scores_daily(
    datasources,
    start_date,
    end_date,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """正式预测入口。

    外部接口保持：
        predict_scores(datasources, start_date, end_date, model_path)

    返回列保持：
        date, instrument, score
    """
    log_path = _setup_predict_logging()
    logger.info(
        "[Predict_Feature] log file -> %s", log_path
    )

    set_datasources(datasources)

    dates: List[str] = build_date_list(
        start_date,
        end_date,
    )

    if not dates:
        raise RuntimeError(
            f"预测区间无交易日: "
            f"{start_date} ~ {end_date}"
        )

    # --------------------------------------------------------------
    # instruments 全区间只查询一次
    # --------------------------------------------------------------
    instruments_df = None
    instruments_error = None
    instruments_attempts = 0
    instruments_failures = 0
    instruments_seconds = 0.0

    for attempt in range(2):
        instruments_attempts += 1
        query_id = f"INSTRUMENTS-{attempt + 1:02d}"

        logger.info(
            "  [DAI][INSTRUMENTS][BEGIN] id=%s "
            "range=%s~%s attempt=%s/2",
            query_id,
            start_date,
            end_date,
            attempt + 1,
        )

        query_start = time.perf_counter()
        try:
            instruments_df = dai.query(
                f"""
                SELECT date, instrument
                FROM {instruments_table()}
                """,
                filters={
                    "date": [
                        start_date,
                        end_date,
                    ]
                },
            ).df()

            elapsed = (
                time.perf_counter() - query_start
            )
            instruments_seconds += elapsed
            instruments_error = None

            logger.info(
                "  [DAI][INSTRUMENTS][END] id=%s "
                "range=%s~%s rows=%s seconds=%.3f",
                query_id,
                start_date,
                end_date,
                len(instruments_df),
                elapsed,
            )
            break

        except Exception as exc:
            elapsed = (
                time.perf_counter() - query_start
            )
            instruments_seconds += elapsed
            instruments_failures += 1
            instruments_error = exc

            err = (
                str(exc)
                .replace("\n", " ")
                .replace("\r", " ")
            )[:300]
            logger.error(
                "  [DAI][INSTRUMENTS][ERROR] id=%s "
                "range=%s~%s seconds=%.3f "
                "error_type=%s error=%s",
                query_id,
                start_date,
                end_date,
                elapsed,
                type(exc).__name__,
                err,
            )

            if attempt == 0:
                logger.info(
                    "   [DAI][INSTRUMENTS][RETRY] "
                    "wait_seconds=1.0"
                )
                time.sleep(1.0)

    if instruments_error is not None:
        raise RuntimeError(
            "无法读取预测区间股票池"
        ) from instruments_error

    if (
        instruments_df is None
        or instruments_df.empty
    ):
        logger.info(
            "  [DAI][SUMMARY] "
            "instruments_attempts=%s "
            "instruments_failures=%s "
            "instruments_seconds=%.3f "
            "bar_attempts=0 bar_seconds=0.000",
            instruments_attempts,
            instruments_failures,
            instruments_seconds,
        )
        return pd.DataFrame(
            columns=[
                "date",
                "instrument",
                "score",
            ]
        )

    instruments_df["date"] = (
        pd.to_datetime(
            instruments_df["date"]
        ).dt.normalize()
    )
    instruments_df["instrument"] = (
        instruments_df["instrument"]
        .astype(str)
    )
    instruments_df = (
        instruments_df
        .drop_duplicates(
            ["date", "instrument"]
        )
        .reset_index(drop=True)
    )

    codes_by_date: dict[
        str, List[str]
    ] = {}

    grouped = instruments_df.groupby(
        instruments_df[
            "date"
        ].dt.strftime("%Y-%m-%d"),
        sort=False,
    )

    for date, frame in grouped:
        codes_by_date[str(date)] = sorted(
            frame["instrument"]
            .unique()
            .tolist()
        )

    # 全区间涉及股票并集：各步骤统一用该池提数，避免块间成分差触发补提
    global_pool = sorted(
        {
            code
            for codes in codes_by_date.values()
            for code in codes
        }
    )

    # --------------------------------------------------------------
    # 纯推理模型加载
    # --------------------------------------------------------------
    fw = _build_inference_framework(
        model_path
    )

    scale_details = get_standard_details(
        fw.config.details_path
    )

    cache = PredictBaseCache(
        seq_len=fw.config.seq_len,
        scale_details=scale_details,
    )

    prefetch_days = 30

    logger.info(
        "[Predict_Feature] start days=%s "
        "(%s -> %s) global_pools=%s device=%s "
        "bf16=%s dai_chunk_days=%s forward=full_day",
        len(dates),
        dates[0],
        dates[-1],
        len(global_pool),
        fw.device,
        fw._use_bf16,
        prefetch_days,
    )

    def prefetch_block(
        block_dates: List[str],
        retry: int = 0,
    ) -> None:
        """分块预取，失败时缩小日期块。始终使用全区间 global_pool。"""
        if not block_dates or not global_pool:
            return

        logger.info(
            "  [DAI][BLOCK][BEGIN] range=%s~%s "
            "days=%s global_pools=%s",
            block_dates[0],
            block_dates[-1],
            len(block_dates),
            len(global_pool),
        )

        try:
            cache.prefetch_dates(
                dates=block_dates,
                pools=global_pool,
            )

            logger.info(
                "  [DAI][BLOCK][END] range=%s~%s "
                "days=%s global_pools=%s",
                block_dates[0],
                block_dates[-1],
                len(block_dates),
                len(global_pool),
            )

        except Exception as exc:
            if len(block_dates) > 1:
                middle = len(block_dates) // 2
                left = block_dates[:middle]
                right = block_dates[middle:]

                logger.warning(
                    "  [DAI][BLOCK][SPLIT] range=%s~%s "
                    "left_days=%s right_days=%s "
                    "error_type=%s",
                    block_dates[0],
                    block_dates[-1],
                    len(left),
                    len(right),
                    type(exc).__name__,
                )

                gc.collect()
                prefetch_block(left)
                prefetch_block(right)
                return

            if retry < 1:
                logger.warning(
                    "  [DAI][BLOCK][RETRY] date=%s "
                    "error_type=%s",
                    block_dates[0],
                    type(exc).__name__,
                )
                gc.collect()
                time.sleep(1.0)

                prefetch_block(
                    block_dates,
                    retry=retry + 1,
                )
                return

            raise RuntimeError(
                "   [DAI] DAI 预取失败: "
                f"{block_dates[0]}"
            ) from exc

    result_parts: List[pd.DataFrame] = []
    all_start = time.perf_counter()

    for block_start in range(
        0,
        len(dates),
        prefetch_days,
    ):
        block_dates = dates[
            block_start:
            block_start + prefetch_days
        ]

        prefetch_start = time.perf_counter()
        prefetch_block(block_dates)
        prefetch_seconds = (
            time.perf_counter()
            - prefetch_start
        )

        logger.info(
            "[Predict_Feature] prefetched %s -> %s "
            "seconds=%.3f",
            block_dates[0],
            block_dates[-1],
            prefetch_seconds,
        )

        for index, date in enumerate(
            block_dates
        ):
            day_start = time.perf_counter()

            # 当日预测截面仍用当日成分；缺数靠 usable
            stock_list = codes_by_date.get(
                date, []
            )
            code_to_idx = {
                code: position
                for position, code
                in enumerate(stock_list)
            }

            day_df, n_pool, n_usable = (
                _predict_one_day(
                    fw=fw,
                    date=date,
                    scale_details=scale_details,
                    cache=cache,
                    stock_list=stock_list,
                    code_to_idx=code_to_idx,
                )
            )

            day_seconds = (
                time.perf_counter()
                - day_start
            )

            global_index = (
                block_start + index + 1
            )

            logger.info(
                "[Predict_Feature] day=%s (%s/%s) "
                "pool=%s usable=%s device=%s "
                "seconds=%.3f",
                date,
                global_index,
                len(dates),
                n_pool,
                n_usable,
                fw.device,
                day_seconds,
            )

            if (
                day_df is not None
                and not day_df.empty
            ):
                result_parts.append(day_df)

        gc.collect()

    elapsed = (
        time.perf_counter()
        - all_start
    )

    logger.info(
        "[Predict_Feature] all days done "
        "elapsed=%.1fs frames=%s final_device=%s",
        elapsed,
        len(result_parts),
        fw.device,
    )

    _log_dai_summary(
        cache=cache,
        instruments_attempts=instruments_attempts,
        instruments_failures=instruments_failures,
        instruments_seconds=instruments_seconds,
    )

    if not result_parts:
        return pd.DataFrame(
            columns=[
                "date",
                "instrument",
                "score",
            ]
        )

    scores_df = pd.concat(
        result_parts,
        ignore_index=True,
    )
    scores_df["date"] = (
        pd.to_datetime(
            scores_df["date"]
        ).dt.normalize()
    )

    result = (
        pd.merge(
            scores_df,
            instruments_df[
                ["date", "instrument"]
            ],
            on=[
                "date",
                "instrument",
            ],
            how="inner",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(subset=["score"])
        .drop_duplicates(
            ["date", "instrument"]
        )
        [
            [
                "date",
                "instrument",
                "score",
            ]
        ]
        .reset_index(drop=True)
    )

    return result


# 对外接口保持不变。
predict_scores = predict_scores_daily
