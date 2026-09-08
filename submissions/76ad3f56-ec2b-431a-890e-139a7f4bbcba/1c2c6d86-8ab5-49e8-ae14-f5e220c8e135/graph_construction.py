"""
DisFT-GNN 图构建模块（Layer 1）
================================
构建多层动态邻接矩阵（行业+收益相关性+市值相似性），
执行严格的时间序列划分，生成图序列数据。

合规要点：相关性邻接矩阵严格使用截止当前时刻的历史数据，
窗口端点不含t及之后数据，禁止前向窥探。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from config import CONFIG


class GraphBuilder:
    """
    构建多层动态邻接矩阵并生成图序列数据。

    层A（行业关系）：静态，同行业连边
    层B（收益相关性）：动态，滚动20日分钟级收益相关系数
    层C（市值相似性）：半动态，对数市值差 < 0.5 连边
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.industry_map: Optional[Dict[str, str]] = None
        self.market_cap: Optional[Dict[str, float]] = None

    def set_static_info(self, industry_map: Dict[str, str], market_cap: Dict[str, float]):
        """
        设置静态信息。

        Args:
            industry_map: {stock: industry} 股票到行业的映射
            market_cap: {stock: log_market_cap} 股票到对数市值的映射
        """
        self.industry_map = industry_map
        self.market_cap = market_cap

    def build_industry_adj(self, stock_list: List[str]) -> np.ndarray:
        """层A：行业邻接矩阵（静态）"""
        N = len(stock_list)
        A = np.zeros((N, N), dtype=np.float32)

        if self.industry_map is None:
            return A

        industries = [self.industry_map.get(s, "unknown") for s in stock_list]
        for i in range(N):
            for j in range(i + 1, N):
                if industries[i] == industries[j] and industries[i] != "unknown":
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        # 自环
        np.fill_diagonal(A, 1.0)
        return A

    def build_correlation_adj(
        self,
        returns: np.ndarray,
        stock_list: List[str],
        window: int = 20,
    ) -> np.ndarray:
        """
        层B：收益相关性邻接矩阵（动态，严格无前向窥探）。

        Args:
            returns: (T, N) 历史日频收益率矩阵，T为截至前一日的交易日数
            stock_list: 股票列表
            window: 滚动窗口（默认20日）

        Returns:
            A: (N, N) 相关性邻接矩阵

        合规说明：使用截至t-1日的历史收益计算，窗口不含t日及以后数据。
        """
        N = len(stock_list)
        A = np.zeros((N, N), dtype=np.float32)

        if returns.shape[0] < 2:
            np.fill_diagonal(A, 1.0)
            return A

        # 取最近window日的收益（不含当日）
        recent_returns = returns[-window:]
        if recent_returns.shape[0] < 5:
            np.fill_diagonal(A, 1.0)
            return A

        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(recent_returns.T)

        # 处理NaN（停牌股票）
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 阈值过滤
        threshold = self.config.data.corr_threshold
        mask = np.abs(corr_matrix) > threshold
        A = np.where(mask, corr_matrix, 0).astype(np.float32)

        # 自环
        np.fill_diagonal(A, 1.0)
        return A

    def build_size_adj(self, stock_list: List[str]) -> np.ndarray:
        """层C：市值相似性邻接矩阵"""
        N = len(stock_list)
        A = np.zeros((N, N), dtype=np.float32)

        if self.market_cap is None:
            np.fill_diagonal(A, 1.0)
            return A

        log_caps = np.array([self.market_cap.get(s, 0.0) for s in stock_list])
        threshold = self.config.data.size_threshold

        for i in range(N):
            for j in range(i + 1, N):
                if abs(log_caps[i] - log_caps[j]) < threshold:
                    A[i, j] = 1.0
                    A[j, i] = 1.0

        np.fill_diagonal(A, 1.0)
        return A

    def build_fused_adj(
        self,
        stock_list: List[str],
        returns: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建融合邻接矩阵及各层矩阵。

        Returns:
            A_fused: (N, N) 融合邻接矩阵
            A_industry: (N, N) 行业层
            A_corr: (N, N) 相关性层
            A_size: (N, N) 市值层

        注意：融合权重alpha/beta/gamma由LearnableAdjFusion学习（AI主导）。
        """
        A_industry = self.build_industry_adj(stock_list)
        A_corr = self.build_correlation_adj(returns, stock_list) if returns is not None else np.eye(len(stock_list), dtype=np.float32)
        A_size = self.build_size_adj(stock_list)

        # 初始融合（后续由LearnableAdjFusion优化）
        alpha = self.config.model.adj_alpha_init
        beta = self.config.model.adj_beta_init
        gamma = self.config.model.adj_gamma_init

        A_fused = alpha * A_industry + beta * A_corr + gamma * A_size
        # 行归一化
        row_sum = A_fused.sum(axis=1, keepdims=True)
        A_fused = A_fused / (row_sum + 1e-8)

        return A_fused, A_industry, A_corr, A_size


class LearnableAdjFusion(nn.Module):
    """
    [AI应用环节 - AI主导] 可学习邻接矩阵融合。

    邻接矩阵融合权重α,β,γ由AI学习而非人工设定，
    满足AI赛道"核心因子构建逻辑由AI主导"要求。
    """

    def __init__(self, alpha_init: float = 0.4, beta_init: float = 0.4, gamma_init: float = 0.2):
        super().__init__()
        # 使用softmax参数化，保证权重非负且和为1
        raw_init = torch.tensor([alpha_init, beta_init, gamma_init])
        self.raw_weights = nn.Parameter(torch.log(raw_init + 1e-8))

    def forward(self, A_industry: torch.Tensor, A_corr: torch.Tensor, A_size: torch.Tensor) -> torch.Tensor:
        """融合三层邻接矩阵"""
        weights = torch.softmax(self.raw_weights, dim=0)
        A_fused = weights[0] * A_industry + weights[1] * A_corr + weights[2] * A_size
        # 行归一化
        row_sum = A_fused.sum(dim=-1, keepdim=True)
        return A_fused / (row_sum + 1e-8)

    def get_weights(self) -> Tuple[float, float, float]:
        """获取当前融合权重"""
        with torch.no_grad():
            w = torch.softmax(self.raw_weights, dim=0)
        return w[0].item(), w[1].item(), w[2].item()


class TimeSeriesSplitter:
    """
    严格时间序列划分（改进6）。

    训练集：2019-01-01 ~ 2022-12-31
    验证集：2023-01-01 ~ 2024-06-30（模拟公榜分布偏移）
    测试集：2024-07-01 ~ 2024-12-31（模拟私榜）
    """

    def __init__(self, config=None):
        self.config = config or CONFIG

    def split_dates(self, all_dates: List[str]) -> Dict[str, List[str]]:
        """
        按时间轴划分日期。

        Returns:
            dict: {'train': [...], 'val': [...], 'test': [...]}
        """
        train_start = self.config.data.train_start
        train_end = self.config.data.train_end
        val_start = self.config.data.val_start
        val_end = self.config.data.val_end
        test_start = self.config.data.test_start
        test_end = self.config.data.test_end

        train_dates = [d for d in all_dates if train_start <= d <= train_end]
        val_dates = [d for d in all_dates if val_start <= d <= val_end]
        test_dates = [d for d in all_dates if test_start <= d <= test_end]

        return {
            "train": train_dates,
            "val": val_dates,
            "test": test_dates,
        }

    @staticmethod
    def create_rolling_windows(
        dates: List[str],
        lookback: int = 5,
        future_horizon: int = 1,
    ) -> List[Dict]:
        """
        创建滚动窗口样本。

        每个样本包含：
        - history_dates: [t-L+1, ..., t] 历史窗口
        - future_date: t+1 未来标签日（仅教师使用）

        Args:
            dates: 日期列表（已排序）
            lookback: 历史窗口长度 L
            future_horizon: 未来标签天数 T

        Returns:
            samples: [{'history': [...], 'future': [...], 't': date}, ...]
        """
        samples = []
        for i in range(lookback, len(dates) - future_horizon + 1):
            history = dates[i - lookback: i]
            future = dates[i: i + future_horizon]
            samples.append({
                "history": history,
                "future": future,
                "t": dates[i - 1],  # 当前日
            })
        return samples


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """
    对称归一化邻接矩阵：D^{-1/2} A D^{-1/2}

    用于GCN的标准化处理。
    """
    row_sum = adj.sum(axis=1)
    d_inv_sqrt = np.power(row_sum, -0.5, where=row_sum > 0)
    d_inv_sqrt = np.nan_to_num(d_inv_sqrt, nan=0.0)
    D_inv = np.diag(d_inv_sqrt)
    return D_inv @ adj @ D_inv


def sparse_adj_to_tensor(adj: np.ndarray, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """将邻接矩阵转为稀疏张量（节省内存）"""
    adj_t = torch.tensor(adj, dtype=torch.float32, device=device)
    return adj_t
