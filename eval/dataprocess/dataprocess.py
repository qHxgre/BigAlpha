"""数据预处理

按 docs/因子挖掘_介绍_20260525.md "数据校验与预处理" 章节实现：
- 去极值：截面 3 倍标准差
- 标准化：截面 z-score
- 风格剔除：与 BARRA 风险因子（含行业哑变量）回归，取残差作为新因子
"""

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

    def drop_inf(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """把 inf/-inf 置为 NaN，不删行。"""
        df = factor_data.copy()
        x = pd.to_numeric(df[factor_name], errors="coerce").to_numpy(dtype=float)
        inf_mask = np.isinf(x)
        if inf_mask.any():
            logger.warning("发现 inf/-inf，已置为 NaN", count=int(inf_mask.sum()))
            x[inf_mask] = np.nan
        df[factor_name] = x
        return df

    def winsorize(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """3 倍标准差去极值（按 date 截面）。"""
        sql = f"""
        SELECT
            date,
            instrument,
            clip(
                {factor_name},
                c_avg({factor_name}, pb:=date) - 3 * c_std({factor_name}, pb:=date),
                c_avg({factor_name}, pb:=date) + 3 * c_std({factor_name}, pb:=date)
            ) AS {factor_name}
        FROM factor_data
        ORDER BY date, instrument
        """
        return dai.query(sql, bind_relations={"factor_data": factor_data}).df()

    def normalize(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """截面 z-score 标准化（按 date 截面）。"""
        sql = f"""
        SELECT
            date,
            instrument,
            c_normalize({factor_name}, pb:=date) AS {factor_name}
        FROM factor_data
        ORDER BY date, instrument
        """
        return dai.query(sql, bind_relations={"factor_data": factor_data}).df()

    def neutralize(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """风格剔除：因子 ~ BARRA 风格暴露 + 行业哑变量，取残差。"""
        df = factor_data.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            raise ValueError("neutralize: factor 数据 date 列存在无法解析的值")
        df["trading_day"] = df["date"].dt.strftime("%Y-%m-%d")
        df["instrument"] = df["instrument"].astype(str)

        start_date = df["trading_day"].min()
        end_date = df["trading_day"].max()
        instruments = df["instrument"].unique().tolist()

        # 行业数据
        industry_df = dai.query(
            """
            SELECT date as trading_day, instrument, industry_level1_code as industry
            FROM cpt_dwc_2026_stock_industry_component
            """,
            filters={"date": [start_date, end_date], "instrument": instruments},
        ).df()
        industry_df["trading_day"] = pd.to_datetime(industry_df["trading_day"], errors="coerce").dt.strftime("%Y-%m-%d")
        industry_df["instrument"] = industry_df["instrument"].astype(str)
        industry_df["industry"] = industry_df["industry"].fillna("Unknown")

        industry_dummies = pd.get_dummies(industry_df["industry"], prefix="IND").astype(float)
        dummy_cols = list(industry_dummies.columns)
        industry_df = pd.concat(
            [industry_df[["trading_day", "instrument"]], industry_dummies], axis=1
        )

        # BARRA 风险因子
        style_factors_df = dai.query(
            """
            SELECT
                * exclude(date),
                date as trading_day
            FROM cpt_dwc_factorlib
            """,
            filters={"date": [start_date, end_date], "instrument": instruments},
        ).df()
        style_factors_df["trading_day"] = pd.to_datetime(
            style_factors_df["trading_day"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        style_factors_df["instrument"] = style_factors_df["instrument"].astype(str)

        merge_df = pd.merge(
            style_factors_df, industry_df, how="left", on=["trading_day", "instrument"]
        )
        merge_df = pd.merge(df, merge_df, how="left", on=["trading_day", "instrument"])

        exclude = {"date", "trading_day", "instrument", factor_name}
        exposure_cols = [c for c in merge_df.columns if c not in exclude]

        if exposure_cols:
            merge_df[exposure_cols] = merge_df[exposure_cols].fillna(0)

        parallel_result = Parallel(backend="threading", n_jobs=-1)(
            delayed(_neutralize_one_day)(group_df, factor_name, exposure_cols)
            for _, group_df in merge_df.groupby("date")
        )

        return pd.concat(parallel_result, ignore_index=True)

    def validate(self, factor_data: pd.DataFrame, factor_name: str='factor') -> pd.DataFrame:
        """完整预处理流程。"""
        t0 = datetime.now()
        factor_data = self.drop_inf(factor_data, factor_name)
        t1 = datetime.now()
        logger.info(f"inf 处理, 耗时: {round((t1 - t0).total_seconds(), 4)} 秒")

        factor_data = self.winsorize(factor_data, factor_name)
        t2 = datetime.now()
        logger.info(f"去极值, 耗时: {round((t2 - t1).total_seconds(), 4)} 秒")

        factor_data = self.normalize(factor_data, factor_name)
        t3 = datetime.now()
        logger.info(f"标准化, 耗时: {round((t3 - t2).total_seconds(), 4)} 秒")

        factor_data = self.neutralize(factor_data, factor_name)
        t4 = datetime.now()
        logger.info(f"风格剔除(取残差), 耗时: {round((t4 - t3).total_seconds(), 4)} 秒")

        return factor_data
