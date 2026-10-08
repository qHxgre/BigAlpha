"""评估所需的外部数据加载入口。"""

import dai
import pandas as pd

BM_DICT = {"中证1000": "000852.SH"}


def load_pool_pairs(start_date: str, end_date: str) -> pd.DataFrame:
    """加载官方分钟级股票池面板 (date, instrument)。"""
    sql = "SELECT date, instrument FROM cpt_jyc_2026_instruments"
    df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
    if df is None or df.empty:
        raise ValueError("无法获取中证 1000 股票池数据，无法进行评估，请联系官方解决")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["instrument"] = df["instrument"].astype(str)
    return df[["date", "instrument"]].drop_duplicates()


def load_evaluation_data(start_date: str, end_date: str) -> pd.DataFrame:
    """加载官方未来 30 分钟 VWAP 收益标签。"""
    sql = "SELECT date, instrument, vwap_return FROM cpt_jyc_2026_vwap_test"
    df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
    if df is None or df.empty:
        raise ValueError("无法获取未来 30 分钟 VWAP 收益标签")
    df = df.rename(columns={"vwap_return": "forward_return"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["instrument"] = df["instrument"].astype(str)
    df["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    return df


def load_stock_returns(start_date: str, end_date: str) -> pd.DataFrame:
    """加载股票收益标签，并计算同截面股票池平均基准收益。"""
    labels = load_evaluation_data(start_date, end_date)
    stock_labels = labels.loc[
        ~labels["instrument"].eq(BM_DICT["中证1000"])
    ].drop_duplicates(["date", "instrument"], keep="last").copy()
    stock_labels["benchmark_return"] = stock_labels.groupby("date")[
        "forward_return"
    ].transform("mean")
    return stock_labels[
        ["date", "instrument", "forward_return", "benchmark_return"]
    ]


def get_exposure(start_date: str, end_date: str) -> pd.DataFrame:
    """加载风格暴露和行业哑变量，供单因子风格中性化使用。"""
    sql = """
    SELECT
        date, instrument,
        SIZE, BETA, MOMENTUM, RESVOL, SIZENL, BTOP, LIQUIDTY, EARNYILD, GROWTH, LEVERAGE,
        AGRIFOREST, MINING, CHEM, IRONSTEEL, NONFERMETAL, ELECTRONICS, AUTO, HOUSEAPP,
        FOODBEVER, TEXTILE, LIGHTINDUS, HEALTH, UTILITIES, TRANSPORTATION, REALESTATE,
        COMMETRADE, LEISERVICE, BANK, NONBANKFINAN, CONGLOMERATES, CONMAT, BUILDDECO,
        ELECEQP, MACHIEQUIP, AERODEF, COMPUTER, MEDIA, TELECOM, COAL, PETRO, ENVP, BEAUTY
    FROM cpt_jyc_2026_exposure
    """
    return dai.query(sql, filters={"date": [start_date, end_date]}).df()
