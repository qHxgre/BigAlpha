from Support_GetDates import get_next_days
from Support_BaseData import bar_table
import polars as pl
import numpy as np
import time
import dai

def get_batch_label(
    date: str = "2020-01-02",
    stock_list: list[str] = [],
    code_to_idx: dict[str, int] = {}
):
    date_list = get_next_days(date=date, N=2)

    label_rank = np.zeros(len(stock_list), dtype=np.float32)
    label_return = np.zeros(len(stock_list), dtype=np.float32)
    usable = np.zeros(len(stock_list), dtype=bool)

    q0 = time.perf_counter()

    use_data = dai.query(
        f"SELECT date, instrument, close FROM bigalpha_2026_bar1d", filters={
            "date": [f"{date_list[0]} 00:00:00", f"{date_list[-1]} 23:59:59"], "instrument": stock_list
        }
    ).pl().with_columns(
        pl.col("date").unique().len().alias("days"),
        pl.col("instrument").len().over("instrument").alias("count")
    ).filter(pl.col("days") == pl.col("count")).drop("days", "count").with_columns(
        pl.col("date").dt.date().alias("day")
    ).filter(
        pl.col("close").is_not_null() & (pl.col("close") > 0)
    ).sort("date", "instrument").group_by("day", "instrument").agg(
        pl.col("close").last().alias("close")
    ).sort("day", "instrument").with_columns(
        pl.col("day").rank().over("instrument").cast(pl.Int64).alias("day_rank")
    ).pivot(
        index="instrument", columns="day_rank", values="close"
    ).filter(pl.col("instrument").is_in(stock_list)).drop_nulls().sort("instrument").select(
        pl.col("instrument"),
        (pl.col("2") / pl.col("1")).log().alias("1d_return")
    ).with_columns([
        ((pl.col(f"{day}_return") - pl.col(f"{day}_return").mean()) / pl.col(f"{day}_return").std()).alias(f"{day}_return")
        for day in ["1d"]
    ]).with_columns([
        (2 * (pl.col(f"{day}_return").rank() / pl.col(f"{day}_return").len()) - 1).alias(f"{day}_rank")
        for day in ["1d"]
    ]).select(
        pl.col("instrument"), *[
            (pl.col(f"1d_{target}")).alias(f"{target}")
            for target in ["return", "rank"]
        ]
    )

    codes = use_data["instrument"].to_list()
    idx = np.fromiter((code_to_idx[c] for c in codes), dtype=np.int64, count=len(codes))

    label_return[idx] = use_data["return"].to_numpy().astype(np.float32)
    label_rank[idx] = use_data["rank"].to_numpy().astype(np.float32)
    usable[idx] = True

    return {
        "date": date,
        "code": stock_list,
        "usable": usable.tolist(),
        "label": {
            "rank": label_rank,
            "return": label_return,
        },
    }

# def get_batch_label(
#     date: str = "2020-01-02",
#     stock_list: list[str] = [],
#     code_to_idx: dict[str, int] = {}
# ):
#     date_list = get_next_days(date=date, N=3)

#     label_rank = np.zeros(len(stock_list), dtype=np.float32)
#     label_return = np.zeros(len(stock_list), dtype=np.float32)
#     usable = np.zeros(len(stock_list), dtype=bool)

#     q0 = time.perf_counter()

#     use_data = dai.query(
#         f"SELECT date, instrument, open FROM bigalpha_2026_bar1d", filters={
#             "date": [f"{date_list[-2]} 00:00:00", f"{date_list[-1]} 23:59:59"], "instrument": stock_list
#         }
#     ).pl().with_columns(
#         pl.col("date").unique().len().alias("days"),
#         pl.col("instrument").len().over("instrument").alias("count"),
#         pl.col("open").fill_nan(None),
#     ).filter(pl.col("days") == pl.col("count")).drop("days", "count").with_columns(
#         pl.col("date").dt.date().alias("day")
#     ).filter(
#         pl.col("open").is_not_null() & (pl.col("open") > 0)
#     ).sort("date", "instrument").with_columns(
#         pl.col("day").rank().over("instrument").cast(pl.Int64).alias("day_rank")
#     ).pivot(
#         index="instrument", columns="day_rank", values="open"
#     ).filter(pl.col("instrument").is_in(stock_list)).drop_nulls().sort("instrument").select(
#         pl.col("instrument"),
#         (pl.col("2") / pl.col("1")).log().alias("1d_return")
#     ).with_columns([
#         ((pl.col(f"{day}_return") - pl.col(f"{day}_return").mean()) / pl.col(f"{day}_return").std()).alias(f"{day}_return")
#         for day in ["1d"]
#     ]).with_columns([
#         (2 * (pl.col(f"{day}_return").rank() / pl.col(f"{day}_return").len()) - 1).alias(f"{day}_rank")
#         for day in ["1d"]
#     ]).select(
#         pl.col("instrument"), *[
#             (pl.col(f"1d_{target}")).alias(f"{target}")
#             for target in ["return", "rank"]
#         ]
#     )

#     codes = use_data["instrument"].to_list()
#     idx = np.fromiter((code_to_idx[c] for c in codes), dtype=np.int64, count=len(codes))

#     label_return[idx] = use_data["return"].to_numpy().astype(np.float32)
#     label_rank[idx] = use_data["rank"].to_numpy().astype(np.float32)
#     usable[idx] = True

#     return {
#         "date": date,
#         "code": stock_list,
#         "usable": usable.tolist(),
#         "label": {
#             "rank": label_rank,
#             "return": label_return,
#         },
#     }
