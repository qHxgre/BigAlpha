"""评估所需的外部数据加载入口。"""

import dai
import pandas as pd
import structlog

logger = structlog.get_logger()

BM_DICT = {
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "沪深300": "000300.SH",
}


def load_pool_pairs(start_date: str, end_date: str) -> pd.DataFrame:
    """加载中证 1000 历史成分股面板 (date, instrument)。"""
    sql = "SELECT date, instrument FROM bigalpha_2026_instruments"
    df = dai.query(sql, filters={"date": [start_date, end_date]}).df()
    if df is None or df.empty:
        raise ValueError("无法获取中证 1000 股票池数据，无法进行评估，请联系官方解决")

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["instrument"] = df["instrument"].astype(str)
    return df[["date", "instrument"]].drop_duplicates()


def get_exposure(start_date: str, end_date: str) -> pd.DataFrame:
    """加载风格暴露和行业哑变量，供单因子风格中性化使用。"""
    sql = """
    SELECT
        date, instrument,
        SIZE, BETA, MOMENTUM, RESVOL, SIZENL, BTOP, LIQUIDTY, EARNYILD, GROWTH, LEVERAGE,
        AGRIFOREST, MINING, CHEM, IRONSTEEL, NONFERMETAL, ELECTRONICS, AUTO, HOUSEAPP,
        FOODBEVER, TEXTILE, LIGHTINDUS, HEALTH, UTILITIES, TRANSPORTATION, REALESTATE,
        COMMETRADE, LEISERVICE, BANK, NONBANKFINAN, CONGLOMERATES, CONMAT, BUILDDECO,
        ELECEQP, AERODEF, COMPUTER, MEDIA, TELECOM, COAL, PETRO, ENVP, BEAUTY
    FROM bigalpha_2026_exposure
    """
    return dai.query(sql, filters={"date": [start_date, end_date]}).df()


def get_daily_ret(start_date: str, end_date: str, instruments: list) -> pd.DataFrame:
    """加载 A 股每日收益数据。"""
    sql = f"""
        SELECT
            date,
            instrument,
            (m_lead(open, 2)/ m_lead(open, 1) - 1) AS daily_ret
        FROM bigalpha_2026_bar1d
        WHERE date BETWEEN DATE '{start_date}' - INTERVAL 10 DAY AND '{end_date}'
        ORDER BY date, instrument
    """
    daily_ret_data = dai.query(sql, filters={
        'instrument': instruments
    }).df()
    if daily_ret_data is None or daily_ret_data.empty:
        logger.warning("每日收益数据为空", start_date=start_date, end_date=end_date)
    return daily_ret_data


def get_bm_ret(start_date: str, end_date: str, benchmark: str) -> pd.DataFrame:
    """加载指定时间段的基准指数日收益率。"""
    sql = f"""
    SELECT
        date, instrument,
        (close - m_Lag(close,1)) / m_LAG(close, 1) as benchmark_ret
    FROM bigalpha_2026_bar1d
    WHERE date BETWEEN DATE '{start_date}' - INTERVAL 10 DAY AND '{end_date}'
    AND instrument = '{BM_DICT[benchmark]}'
    """
    bm_ret = dai.query(sql).df()
    if bm_ret is None or bm_ret.empty:
        logger.warning(
            "基准指数收益数据为空",
            benchmark=benchmark,
            instrument=BM_DICT.get(benchmark),
            start_date=start_date,
            end_date=end_date,
        )
    return bm_ret
