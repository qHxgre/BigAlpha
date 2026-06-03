import dai
import numpy as np
import pandas as pd
import structlog
from datetime import datetime
from joblib import Parallel, delayed

logger = structlog.get_logger()


def _solve_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS 残差。条件数差时用 SVD 伪逆。"""
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    def _solve_with_svd(X_, y_):
        U, s, Vt = np.linalg.svd(X_, full_matrices=False)
        eps = np.finfo(float).eps
        threshold = eps * max(X_.shape) * (np.max(s) if s.size else 1.0)
        s_inv = np.zeros_like(s)
        mask = s > threshold
        s_inv[mask] = 1.0 / s[mask]
        X_pinv = Vt.T @ np.diag(s_inv) @ U.T
        beta = X_pinv @ y_
        return y_ - X_ @ beta

    try:
        xtx = X.T @ X
        cond_number = np.linalg.cond(xtx)
        if not np.isfinite(cond_number) or cond_number > 1e12:
            return _solve_with_svd(X, y)
        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            return _solve_with_svd(X, y)
        beta = xtx_inv @ (X.T @ y)
        return y - X @ beta
    except np.linalg.LinAlgError:
        return _solve_with_svd(X, y)


def _neutralize_one_day(date_df: pd.DataFrame, factor_name: str, exposure_cols: list) -> pd.DataFrame:
    """单日截面回归取残差。"""
    out = date_df[["date", "instrument", factor_name]].copy()
    y = pd.to_numeric(date_df[factor_name], errors="coerce").to_numpy(dtype=float)

    if exposure_cols:
        X = date_df[exposure_cols].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(date_df), dtype=float), X])
    else:
        X = np.ones((len(date_df), 1), dtype=float)

    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if mask.sum() < 5:
        out[factor_name] = np.nan
        return out

    try:
        resid = _solve_normal_equation(X[mask], y[mask])
        result = np.full_like(y, np.nan, dtype=float)
        result[mask] = resid
        out[factor_name] = result
    except Exception:
        logger.exception("neutralize: 单日截面回归失败", factor_name=factor_name)
        out[factor_name] = np.nan
    return out


class DataProcess:
    """因子预处理：去极值 / 标准化 / 风格剔除（残差）。"""
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

    def drop_inf(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """把 inf/-inf 置为 NaN，不删行。"""
        df = factor_data.copy()
        x = pd.to_numeric(df['factor'], errors="coerce").to_numpy(dtype=float)
        inf_mask = np.isinf(x)
        if inf_mask.any():
            logger.warning("发现 inf/-inf，已置为 NaN", count=int(inf_mask.sum()))
            x[inf_mask] = np.nan
        df['factor'] = x
        return df

    def winsorize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """3 倍标准差去极值（按 date 截面）。"""
        sql = f"""
        SELECT
            date,
            instrument,
            clip(
                factor,
                c_avg(factor, pb:=date) - 3 * c_std(factor, pb:=date),
                c_avg(factor, pb:=date) + 3 * c_std(factor, pb:=date)
            ) AS factor
        FROM factor_data
        ORDER BY date, instrument
        """
        return dai.query(sql, bind_relations={"factor_data": factor_data}).df()

    def normalize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """截面 z-score 标准化（按 date 截面）。"""
        sql = f"""
        SELECT
            date,
            instrument,
            c_normalize(factor, pb:=date) AS factor
        FROM factor_data
        ORDER BY date, instrument
        """
        return dai.query(sql, bind_relations={"factor_data": factor_data}).df()

    def neutralize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """风格剔除：因子 ~ BARRA 风格暴露 + 行业哑变量，取残差。"""
        df = factor_data.copy()

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
        neutralize_df = dai.query(sql, filters={'date': [self.start_date, self.end_date]}).df()

        merge_df = pd.merge(df, neutralize_df, how="left", on=["date", "instrument"])

        exclude = {"date", "instrument", "factor"}
        exposure_cols = [c for c in merge_df.columns if c not in exclude]

        if exposure_cols:
            merge_df[exposure_cols] = merge_df[exposure_cols].fillna(0)

        parallel_result = Parallel(backend="threading", n_jobs=-1)(
            delayed(_neutralize_one_day)(group_df, 'factor', exposure_cols)
            for _, group_df in merge_df.groupby("date")
        )

        return pd.concat(parallel_result, ignore_index=True)

    def validate(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """完整预处理流程。"""
        t0 = datetime.now()
        factor_data = self.drop_inf(factor_data)
        t1 = datetime.now()
        logger.info(f"inf 处理, 耗时: {round((t1 - t0).total_seconds(), 4)} 秒")

        factor_data = self.winsorize(factor_data)
        t2 = datetime.now()
        logger.info(f"去极值, 耗时: {round((t2 - t1).total_seconds(), 4)} 秒")

        factor_data = self.normalize(factor_data)
        t3 = datetime.now()
        logger.info(f"标准化, 耗时: {round((t3 - t2).total_seconds(), 4)} 秒")

        factor_data = self.neutralize(factor_data)
        t4 = datetime.now()
        logger.info(f"风格剔除(取残差), 耗时: {round((t4 - t3).total_seconds(), 4)} 秒")

        return factor_data
