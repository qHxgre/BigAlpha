import dai
import numpy as np
import pandas as pd
import structlog
from datetime import datetime
from joblib import Parallel, delayed

logger = structlog.get_logger()


def _solve_normal_equation(X, y):
    """稳健最小二乘：条件数差时用 SVD 伪逆。"""

    def _solve_with_svd(X_, y_):
        U, s, Vt = np.linalg.svd(X_, full_matrices=False)
        eps = np.finfo(float).eps
        threshold = eps * max(X_.shape) * np.max(s)
        s_inv = np.zeros_like(s)
        mask = s > threshold
        s_inv[mask] = 1.0 / s[mask]
        X_pinv = Vt.T @ np.diag(s_inv) @ U.T
        beta = X_pinv @ y_
        y_pred = X_ @ beta
        return y_ - y_pred

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    try:
        xtx = X.T @ X
        xty = X.T @ y
        cond_number = np.linalg.cond(xtx)
        if cond_number > 1e12:
            return _solve_with_svd(X, y)

        try:
            xtx_inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(xtx)

        beta = xtx_inv @ xty
        y_pred = X @ beta
        return y - y_pred

    except np.linalg.LinAlgError:
        return _solve_with_svd(X, y)


def _parallel_process_single_date(date_df, factor_name, neutralize_cols):
    """
    单日截面回归取残差：
    - 加截距项
    - neutralize_cols 为空时仅用截距
    - 出错不再静默：logger.exception
    """
    try:
        y = pd.to_numeric(date_df[factor_name], errors="coerce").to_numpy(dtype=float)

        # X: 暴露矩阵 + 截距
        if neutralize_cols:
            X = date_df[neutralize_cols].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(date_df), dtype=float), X])
        else:
            X = np.ones((len(date_df), 1), dtype=float)

        # 仅对 y 有效样本做回归，避免 NaN/inf 影响
        mask = np.isfinite(y)
        if mask.sum() < 5:
            date_df[factor_name] = np.nan
            return date_df[["date", "instrument", factor_name]]

        resid = _solve_normal_equation(X[mask], y[mask])

        out = np.full_like(y, np.nan, dtype=float)
        out[mask] = resid
        date_df[factor_name] = out

    except Exception:
        dt = None
        try:
            dt = date_df["date"].iloc[0]
        except Exception:
            pass
        logger.exception(
            "neutralize: 单日截面回归失败，已将该日因子置为 NaN",
            date=str(dt) if dt is not None else None,
            factor_name=factor_name,
            n_rows=int(len(date_df)),
            n_cols=int(len(neutralize_cols)) if neutralize_cols is not None else None,
        )
        raise ValueError('因子正交化失败，请查看数据！')
        date_df[factor_name] = np.nan

    return date_df[["date", "instrument", factor_name]]

def _parallel_process_single_date(date_df, factor_name, neutralize_cols):
    """处理单个日期的函数"""
    try:
        y = date_df[factor_name].values
        X = date_df[neutralize_cols].values
        residuals = _solve_normal_equation(X, y)
        date_df[factor_name] = residuals
        return date_df[["date", "instrument", factor_name]]
    except Exception as e:
        error_msg = f'因子正交化失败，请检查：{e}'
        logger.error(error_msg)
        raise ValueError(error_msg)


class DataProcess:
    def drop_nan_and_inf(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """
        只把 inf/-inf 置为 NaN（不删行！）
        原实现会直接过滤行，会破坏“全量行数/覆盖度”口径。
        """
        df = factor_data.copy()

        # 先数值化，非数值 -> NaN
        x = pd.to_numeric(df[factor_name], errors="coerce").to_numpy(dtype=float)

        inf_mask = np.isinf(x)
        if inf_mask.any():
            logger.warning("发现 inf/-inf，已置为 NaN", factor_name=factor_name, count=int(inf_mask.sum()))
            x[inf_mask] = np.nan

        df[factor_name] = x
        return df

    def winsorize(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """
        去极值：三倍标准差法（显式按 date 截面）。
        这里用 DAI 的截面函数 window partition：pb:=date
        """
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
        """
        z-score 标准化（显式按 date 截面）。
        你们环境的 c_normalize 只能接收 1 个 positional 参数，分组键用 pb:= 传。
        """
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
        """
        中性化：因子 ~ 风格暴露 + 行业哑变量；取残差。
        修复点：
        - trading_day 全部统一成 YYYY-MM-DD 字符串，避免 merge 漏匹配
        - 不再 merge_df.fillna(0) 污染因子列，仅对暴露列填 0
        - neutralize_cols 排除 factor_name（不再硬编码 'factor'）
        - 回归加截距项
        - 单日异常可观测（logger.exception）
        """
        df = factor_data.copy()

        # 统一 date / trading_day
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().any():
            bad = df[df["date"].isna()].head(20)
            logger.error("neutralize: date 无法解析", sample=bad.to_dict(orient="records"))
            raise ValueError("neutralize: 因子数据 date 存在无法解析的值")

        df["trading_day"] = df["date"].dt.strftime("%Y-%m-%d")

        # 读取数据范围
        start_date = df["trading_day"].min()
        end_date = df["trading_day"].max()
        instruments = df["instrument"].astype(str).unique().tolist()
        df["instrument"] = df["instrument"].astype(str)

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
        industry_dummies = pd.get_dummies(industry_df["industry"], prefix="IND").astype(int)
        dummy_cols = [c for c in industry_dummies.columns if c.startswith("IND_")]
        industry_df = pd.concat([industry_df[["trading_day", "instrument"]], industry_dummies], axis=1)
        industry_df = industry_df[["trading_day", "instrument"] + dummy_cols]

        # 风格因子数据
        style_factors_df = dai.query(
            """
            SELECT
                * exclude(date),
                date as trading_day
            FROM cpt_dwc_factorlib
            """,
            filters={"date": [start_date, end_date], "instrument": instruments},
        ).df()
        style_factors_df["trading_day"] = pd.to_datetime(style_factors_df["trading_day"], errors="coerce").dt.strftime("%Y-%m-%d")
        style_factors_df["instrument"] = style_factors_df["instrument"].astype(str)

        # 合并暴露
        merge_df = pd.merge(style_factors_df, industry_df, how="left", on=["trading_day", "instrument"])
        merge_df = pd.merge(df, merge_df, how="left", on=["trading_day", "instrument"])

        # 排除 factor_name（而不是硬编码 'factor'）
        exclude = {"date", "trading_day", "instrument", factor_name}
        neutralize_cols = [c for c in merge_df.columns if c not in exclude]

        # 只对暴露列填 0，避免把因子列也填 0
        if neutralize_cols:
            merge_df[neutralize_cols] = merge_df[neutralize_cols].fillna(0)

        # 截面正交化取残差（按 date 分组）
        parallel_result = Parallel(backend="threading", n_jobs=-1)(
            delayed(_parallel_process_single_date)(group_df, factor_name, neutralize_cols)
            for _, group_df in merge_df.groupby("date")
        )

        return pd.concat(parallel_result, ignore_index=True)

    def validate(self, factor_data: pd.DataFrame, factor_name: str) -> pd.DataFrame:
        """执行所有的数据处理流程。"""
        t0 = datetime.now()

        factor_data = self.drop_nan_and_inf(factor_data, factor_name)
        t1 = datetime.now()
        logger.info(f"缺失/inf 处理, 耗时: {round((t1 - t0).total_seconds(), 4)} 秒")

        factor_data = self.winsorize(factor_data, factor_name)
        t2 = datetime.now()
        logger.info(f"去极值, 耗时: {round((t2 - t1).total_seconds(), 4)} 秒")

        factor_data = self.normalize(factor_data, factor_name)
        t3 = datetime.now()
        logger.info(f"标准化, 耗时: {round((t3 - t2).total_seconds(), 4)} 秒")

        factor_data = self.neutralize(factor_data, factor_name)
        t4 = datetime.now()
        logger.info(f"中性化(取残差), 耗时: {round((t4 - t3).total_seconds(), 4)} 秒")

        return factor_data
