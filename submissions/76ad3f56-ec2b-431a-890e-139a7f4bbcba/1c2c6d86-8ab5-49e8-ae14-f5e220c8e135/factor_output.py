"""
DisFT-GNN 因子输出模块（Layer 5）
==================================
将模型输出转化为符合比赛要求的因子表，并做B项优化校验。

处理管线：投影 → 方向校验 → 去极值 → 标准化 → 覆盖度检查 → BARRA预检
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List, Tuple
from config import CONFIG

# scipy可选导入（spearmanr有numpy fallback实现）
try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _spearmanr(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Spearman秩相关（scipy不可用时用numpy实现）"""
    if _HAS_SCIPY:
        corr, pval = spearmanr(a, b)
        return float(corr), float(pval)
    # numpy fallback
    if len(a) != len(b) or len(a) < 2:
        return 0.0, 1.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-8)
    rb = (rb - rb.mean()) / (rb.std() + 1e-8)
    corr = float(np.mean(ra * rb))
    return corr, 0.0


class FactorProcessor:
    """
    因子后处理与合规校验。
    """

    def __init__(self, config=None):
        self.config = config or CONFIG

    def process_factor(
        self,
        factor_values: np.ndarray,
        stock_list: List[str],
        date: str,
        fill_missing: bool = True,
    ) -> pd.DataFrame:
        """
        完整的因子后处理管线。

        Args:
            factor_values: (N,) 原始因子值
            stock_list: (N,) 股票代码列表
            date: 日期字符串
            fill_missing: 是否填充缺失值

        Returns:
            df: DataFrame[date, instrument, factor]
        """
        factor = factor_values.copy()

        # 1. 缺失值处理
        if fill_missing:
            factor = self._fill_missing(factor)

        # 2. 去极值（MAD-Winsorize）
        factor = self._winsorize(factor)

        # 3. 标准化（Z-score）
        factor = self._standardize(factor)

        # 4. 覆盖度检查
        missing_ratio = np.isnan(factor).sum() / len(factor)
        if missing_ratio > self.config.data.max_missing_ratio:
            print(f"[警告] {date} 缺失率={missing_ratio:.2%} > {self.config.data.max_missing_ratio:.0%}")

        # 5. 构建输出DataFrame
        df = pd.DataFrame({
            'date': date,
            'instrument': stock_list,
            'factor': factor,
        })

        # 6. 确保因子方向：值越大越好（通过IC符号判断，需调用方提供）
        # 此处默认方向正确，方向校验在evaluate_factor_direction中完成

        return df

    @staticmethod
    def _fill_missing(factor: np.ndarray, method: str = "median") -> np.ndarray:
        """缺失值填充（行业中位数）"""
        if method == "median":
            median_val = np.nanmedian(factor)
            factor = np.where(np.isnan(factor), median_val, factor)
        elif method == "zero":
            factor = np.where(np.isnan(factor), 0.0, factor)
        return factor

    def _winsorize(self, factor: np.ndarray) -> np.ndarray:
        """去极值：MAD方法 + 分位截断"""
        # MAD方法
        median = np.nanmedian(factor)
        mad = np.nanmedian(np.abs(factor - median))
        upper = median + 3 * 1.4826 * mad
        lower = median - 3 * 1.4826 * mad
        factor = np.clip(factor, lower, upper)

        # 分位截断
        lower_pct = np.nanpercentile(factor, self.config.data.winsorize_lower * 100)
        upper_pct = np.nanpercentile(factor, self.config.data.winsorize_upper * 100)
        factor = np.clip(factor, lower_pct, upper_pct)

        return factor

    @staticmethod
    def _standardize(factor: np.ndarray) -> np.ndarray:
        """Z-score标准化"""
        mean = np.nanmean(factor)
        std = np.nanstd(factor)
        if std < 1e-8:
            return factor - mean
        return (factor - mean) / std

    @staticmethod
    def evaluate_factor_direction(
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> float:
        """
        评估因子方向：计算IC，若IC<0则需要取负。

        Args:
            factor_df: DataFrame[date, instrument, factor]
            returns_df: DataFrame[date, instrument, return]

        Returns:
            ic: IC值，负值表示需要反转方向
        """
        merged = factor_df.merge(returns_df, on=['date', 'instrument'], how='inner')
        ics = []

        for date, group in merged.groupby('date'):
            if len(group) < 10:
                continue
            f = group['factor'].values
            r = group['return'].values
            mask = ~(np.isnan(f) | np.isnan(r))
            if mask.sum() < 10:
                continue
            # Spearman秩相关
            ic, _ = _spearmanr(f[mask], r[mask])
            if not np.isnan(ic):
                ics.append(ic)

        return np.mean(ics) if ics else 0.0


class FactorValidator:
    """
    因子质量验证（提交前自检）。
    """

    @staticmethod
    def validate_output_format(df: pd.DataFrame) -> bool:
        """验证输出格式合规性"""
        # 必须且仅包含三列
        required_cols = {'date', 'instrument', 'factor'}
        if set(df.columns) != required_cols:
            print(f"[格式错误] 列名应为{required_cols}，实际为{set(df.columns)}")
            return False

        # 检查数据类型
        if not pd.api.types.is_numeric_dtype(df['factor']):
            print("[格式错误] factor列必须为数值类型")
            return False

        return True

    @staticmethod
    def validate_coverage(df: pd.DataFrame, threshold: float = 0.4) -> bool:
        """验证每日缺失率"""
        for date, group in df.groupby('date'):
            missing_ratio = group['factor'].isna().sum() / len(group)
            if missing_ratio > threshold:
                print(f"[覆盖度警告] {date} 缺失率={missing_ratio:.2%} > {threshold:.0%}")
                return False
        return True

    @staticmethod
    def validate_completeness(
        df: pd.DataFrame,
        expected_dates: List[str],
    ) -> bool:
        """验证交易日完整性"""
        actual_dates = set(df['date'].unique())
        missing = set(expected_dates) - actual_dates
        if missing:
            print(f"[完整性错误] 缺失交易日: {sorted(missing)[:5]}...")
            return False
        return True

    @staticmethod
    def compute_factor_metrics(
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        计算因子评估指标（模拟A项）。

        Returns:
            dict: ic_mean, ic_ir, ic_std, sharpe_ratio
        """
        merged = factor_df.merge(returns_df, on=['date', 'instrument'], how='inner')

        ics = []
        for date, group in merged.groupby('date'):
            if len(group) < 10:
                continue
            f = group['factor'].values
            r = group['return'].values
            mask = ~(np.isnan(f) | np.isnan(r))
            if mask.sum() < 10:
                continue
            ic, _ = _spearmanr(f[mask], r[mask])
            if not np.isnan(ic):
                ics.append(ic)

        if not ics:
            return {'ic_mean': 0, 'ic_ir': 0, 'ic_std': 0, 'sharpe_ratio': 0}

        ics = np.array(ics)
        ic_mean = np.mean(ics)
        ic_std = np.std(ics) + 1e-8
        ic_ir = ic_mean / ic_std

        # 简化的多空组合夏普
        long_short_returns = ics  # 简化：用IC序列近似
        sharpe = np.mean(long_short_returns) / (np.std(long_short_returns) + 1e-8) * np.sqrt(252)

        return {
            'ic_mean': float(ic_mean),
            'ic_ir': float(ic_ir),
            'ic_std': float(ic_std),
            'sharpe_ratio': float(sharpe),
        }

    @staticmethod
    def check_factor_correlation(
        factor_a: pd.DataFrame,
        factor_b: pd.DataFrame,
        threshold: float = 0.3,
    ) -> float:
        """
        检查两个因子间的相关性（B项去冗余）。

        Returns:
            correlation: 相关系数，若>threshold需去冗余
        """
        merged = factor_a.merge(
            factor_b, on=['date', 'instrument'], suffixes=('_a', '_b'), how='inner'
        )
        corr = merged['factor_a'].corr(merged['factor_b'])
        if abs(corr) > threshold:
            print(f"[冗余警告] 因子相关系数={corr:.4f} > {threshold}")
        return corr


def generate_factor_table(
    student_model: torch.nn.Module,
    all_features: Dict[str, torch.Tensor],
    all_adjs: Dict[str, torch.Tensor],
    stock_list: List[str],
    dates: List[str],
    config=None,
    device: torch.device = torch.device('cpu'),
) -> pd.DataFrame:
    """
    [AI应用环节7] 使用学生模型生成全部日期的因子表。

    Args:
        student_model: 已训练的学生模型
        all_features: {date: (N, L, M)} 各日特征
        all_adjs: {date: (N, N)} 各日邻接矩阵
        stock_list: 股票代码列表
        dates: 日期列表
        config: 配置
        device: 计算设备

    Returns:
        factor_df: DataFrame[date, instrument, factor]
    """
    config = config or CONFIG
    processor = FactorProcessor(config)
    student_model = student_model.to(device).eval()

    all_factors = []

    print(f"\n[因子生成] 开始生成 {len(dates)} 个交易日的因子...")
    for date in dates:
        if date not in all_features:
            continue

        x = all_features[date].to(device)
        adj = all_adjs[date].to(device)

        # NaN安全：将输入中的NaN替换为0
        x = torch.nan_to_num(x, nan=0.0)

        with torch.no_grad():
            outputs = student_model(x, adj)
            factor_values = outputs['factor'].cpu().numpy()

        # NaN安全：将输出中的NaN/Inf替换为0
        factor_values = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)

        df = processor.process_factor(factor_values, stock_list, date)
        all_factors.append(df)

    if not all_factors:
        raise ValueError("未生成任何因子数据")

    factor_df = pd.concat(all_factors, ignore_index=True)
    print(f"[因子生成] 完成，共 {len(factor_df)} 行")

    return factor_df
