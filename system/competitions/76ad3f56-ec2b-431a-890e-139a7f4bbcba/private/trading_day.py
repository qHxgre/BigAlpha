"""私榜数据区间辅助函数。"""
from __future__ import annotations


def resolve_latest_trading_day(table_name: str) -> str:
    import dai

    result = dai.query(f"SELECT MAX(date) AS max_date FROM {table_name}").df()
    if result is None or result.empty or result.iloc[0]["max_date"] is None:
        raise RuntimeError(f"无法从 {table_name} 查询最新交易日")
    return f"{str(result.iloc[0]['max_date'])[:10]} 23:59:59"
