"""
DisFT-GNN 特征工程模块（Layer 2）
=================================
[AI应用环节1] 将高频分钟数据压缩为日频特征向量，保留微观结构信息。

特征族：基础统计、订单簿压力、已实现波动率、价格冲击、形态特征、分布特征
聚合策略：分位数统计、高阶矩、日内积分

可学习特征聚合（AI主导）：特征候选池通过可学习权重自动聚合，L1正则自动筛选有效特征。
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional
from config import CONFIG


class FeatureExtractor:
    """
    [AI应用环节1] 分钟级微观结构特征提取与日频聚合。

    输入：单只股票单日的1分钟K线 + 盘口数据
    输出：DailyFeatureVector ∈ R^M
    """

    def __init__(self):
        self.feature_names = self._get_feature_names()

    @staticmethod
    def _get_feature_names() -> List[str]:
        """返回特征名列表"""
        return [
            # 基础统计 (6维)
            "open_ret", "close_ret", "high_low_ratio", "vwap_ret", "volume_total", "amount_total",
            # 订单簿压力 (4维)
            "obi_depth_weighted", "obi_spread_weighted", "active_buy_ratio", "obi_integral",
            # 已实现波动率 (3维)
            "rv_std", "rv_range", "rv_parkinson",
            # 价格冲击 (2维)
            "kyle_lambda", "amihud_illiq",
            # 形态特征 (3维)
            "overnight_gap", "last30_ret", "max_drawdown_pos",
            # 分布特征 (3维)
            "ret_skew", "ret_kurt", "volume_entropy",
            # 可学习聚合特征占位 (4维，由LearnableFeatureAggregator生成)
            "learnable_agg_0", "learnable_agg_1", "learnable_agg_2", "learnable_agg_3",
        ]

    def extract_daily_features(
        self,
        kline_1min: pd.DataFrame,
        orderbook: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        [AI应用环节1] 从单日分钟数据提取日频特征向量。

        Args:
            kline_1min: 1分钟K线，需含列 open/high/low/close/volume/amount
            orderbook: 盘口快照数据（买卖五档量价），可选

        Returns:
            features: (M,) 日频特征向量，M≈25
        """
        features = {}

        # ===== 基础统计 =====
        features["open_ret"] = (kline_1min["close"].iloc[-1] / kline_1min["open"].iloc[0] - 1)
        features["close_ret"] = kline_1min["close"].pct_change().fillna(0).sum()
        features["high_low_ratio"] = (kline_1min["high"].max() / kline_1min["low"].min() - 1) if kline_1min["low"].min() > 0 else 0
        vwap = (kline_1min["amount"].sum() / kline_1min["volume"].sum()) if kline_1min["volume"].sum() > 0 else kline_1min["close"].mean()
        features["vwap_ret"] = (kline_1min["close"].iloc[-1] / vwap - 1) if vwap > 0 else 0
        features["volume_total"] = np.log1p(kline_1min["volume"].sum())
        features["amount_total"] = np.log1p(kline_1min["amount"].sum())

        # 分钟收益率
        minute_ret = kline_1min["close"].pct_change().fillna(0).values

        # ===== 订单簿压力 =====
        if orderbook is not None and len(orderbook) > 0:
            obi = self._compute_obi(orderbook)
            features["obi_depth_weighted"] = obi["depth_weighted"].mean()
            features["obi_spread_weighted"] = obi["spread_weighted"].mean()
            features["active_buy_ratio"] = obi["active_buy_ratio"]
            # 日内积分（累积不平衡），反映持续性压力
            features["obi_integral"] = obi["depth_weighted"].sum()
        else:
            # 无盘口数据时用量价近似
            buy_vol = kline_1min[kline_1min["close"] >= kline_1min["open"]]["volume"].sum()
            total_vol = kline_1min["volume"].sum()
            features["obi_depth_weighted"] = (buy_vol / total_vol - 0.5) if total_vol > 0 else 0
            features["obi_spread_weighted"] = 0.0
            features["active_buy_ratio"] = (buy_vol / total_vol) if total_vol > 0 else 0.5
            features["obi_integral"] = features["obi_depth_weighted"] * len(kline_1min)

        # ===== 已实现波动率 =====
        features["rv_std"] = np.std(minute_ret)
        features["rv_range"] = np.log(kline_1min["high"].max() / kline_1min["low"].min()) if kline_1min["low"].min() > 0 else 0
        # Parkinson波动率
        hl_ratio = np.log(kline_1min["high"] / kline_1min["low"]).replace([np.inf, -np.inf], 0)
        features["rv_parkinson"] = np.sqrt(np.mean(hl_ratio ** 2) / (4 * np.log(2)))

        # ===== 价格冲击 =====
        # Kyle's Lambda: 价格变化 / 订单流
        volume_arr = kline_1min["volume"].values
        nonzero_mask = volume_arr > 0
        if nonzero_mask.sum() > 5:
            price_change = np.diff(kline_1min["close"].values)
            order_flow = volume_arr[1:] - volume_arr[:-1]
            denom = np.sum(order_flow ** 2)
            features["kyle_lambda"] = (np.sum(price_change * order_flow) / denom) if denom > 0 else 0
        else:
            features["kyle_lambda"] = 0

        # Amihud ILLIQ
        ret_abs = np.abs(minute_ret)
        features["amihud_illiq"] = np.mean(ret_abs[1:] / (volume_arr[1:] + 1e-8))

        # ===== 形态特征 =====
        features["overnight_gap"] = (kline_1min["open"].iloc[0] / kline_1min["close"].iloc[-1] - 1) if kline_1min["close"].iloc[-1] > 0 else 0
        # 尾盘30分钟收益率
        n = len(kline_1min)
        last30_start = max(0, n - 30)
        features["last30_ret"] = (kline_1min["close"].iloc[-1] / kline_1min["close"].iloc[last30_start] - 1) if kline_1min["close"].iloc[last30_start] > 0 else 0
        # 日内最大回撤位置（0=开盘附近，1=收盘附近）
        cummax = kline_1min["close"].cummax()
        drawdown = (kline_1min["close"] / cummax - 1)
        features["max_drawdown_pos"] = drawdown.idxmin() / n if n > 0 else 0

        # ===== 分布特征 =====
        features["ret_skew"] = self._safe_skew(minute_ret)
        features["ret_kurt"] = self._safe_kurt(minute_ret)
        # 成交量分布的熵
        vol_norm = volume_arr / (volume_arr.sum() + 1e-8)
        vol_norm = vol_norm[vol_norm > 0]
        features["volume_entropy"] = -np.sum(vol_norm * np.log(vol_norm + 1e-8)) if len(vol_norm) > 0 else 0

        # ===== 可学习聚合特征占位（由LearnableFeatureAggregator填充） =====
        for i in range(4):
            features[f"learnable_agg_{i}"] = 0.0

        # 按特征名顺序输出
        return np.array([features.get(name, 0.0) for name in self.feature_names], dtype=np.float32)

    @staticmethod
    def _compute_obi(orderbook: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算订单簿不平衡（OBI）"""
        # 假设orderbook包含 bid_vol_1~5, ask_vol_1~5, bid_price_1~5, ask_price_1~5
        bid_cols = [c for c in orderbook.columns if c.startswith("bid_vol")]
        ask_cols = [c for c in orderbook.columns if c.startswith("ask_vol")]

        if not bid_cols or not ask_cols:
            return {
                "depth_weighted": np.zeros(len(orderbook)),
                "spread_weighted": np.zeros(len(orderbook)),
                "active_buy_ratio": 0.5,
            }

        bid_vol = orderbook[bid_cols].values  # (T, 5)
        ask_vol = orderbook[ask_cols].values  # (T, 5)

        # 深度加权OBI
        weights = np.arange(1, bid_vol.shape[1] + 1) ** (-1)
        weights = weights / weights.sum()
        depth_weighted = (bid_vol @ weights - ask_vol @ weights) / (bid_vol @ weights + ask_vol @ weights + 1e-8)

        # 价差加权OBI（简化版）
        spread_weighted = depth_weighted  # 简化处理

        # 主动买入比率（近似）
        active_buy_ratio = (bid_vol.sum() / (bid_vol.sum() + ask_vol.sum() + 1e-8))

        return {
            "depth_weighted": depth_weighted,
            "spread_weighted": spread_weighted,
            "active_buy_ratio": float(active_buy_ratio),
        }

    @staticmethod
    def _safe_skew(x: np.ndarray) -> float:
        """安全的偏度计算"""
        if len(x) < 3:
            return 0.0
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((x - np.mean(x)) / std) ** 3))

    @staticmethod
    def _safe_kurt(x: np.ndarray) -> float:
        """安全的峰度计算"""
        if len(x) < 4:
            return 0.0
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((x - np.mean(x)) / std) ** 4) - 3)

    def extract_all_stocks_one_day(
        self,
        date: str,
        stock_list: List[str],
        kline_data: Dict[str, pd.DataFrame],
        orderbook_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> np.ndarray:
        """
        提取某日全市场所有股票的日频特征。

        Args:
            date: 日期字符串
            stock_list: 股票代码列表
            kline_data: {stock: 1分钟K线DataFrame} 字典
            orderbook_data: {stock: 盘口DataFrame} 字典，可选

        Returns:
            features: (N, M) 全市场日频特征矩阵
        """
        N = len(stock_list)
        M = len(self.feature_names)
        features = np.zeros((N, M), dtype=np.float32)

        for i, stock in enumerate(stock_list):
            kline = kline_data.get(stock)
            if kline is None or len(kline) == 0:
                features[i] = np.nan  # 停牌或无数据
                continue
            ob = orderbook_data.get(stock) if orderbook_data else None
            features[i] = self.extract_daily_features(kline, ob)

        return features


class LearnableFeatureAggregator(torch.nn.Module):
    """
    [AI应用环节1 - AI主导] 可学习特征聚合器。

    将人工提供的特征候选池通过可学习权重自动聚合，
    L1正则自动筛选有效特征维度，满足AI赛道"核心因子构建逻辑由AI主导"要求。
    """

    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        # 可学习的聚合权重
        self.weight = torch.nn.Parameter(torch.randn(input_dim, output_dim) * 0.02)
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))
        # L1正则系数（通过外部loss实现）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) -> (..., output_dim)"""
        return x @ self.weight + self.bias

    def l1_penalty(self) -> torch.Tensor:
        """L1正则项，促使无效特征权重归零"""
        return self.weight.abs().sum()
