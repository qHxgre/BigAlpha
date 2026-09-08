"""
DisFT-GNN 模型模块（Layer 3）
==============================
教师模型：历史GNN + 未来趋势编码 + 多通道双线性融合（低秩）
学生模型：轻量历史GNN（仅历史数据）

[AI应用环节2/3/7] 标注AI技术应用的各个环节。

合规说明：
- 教师模型使用未来标签（LUPI范式），仅本地/Notebook内训练，不在main中直接调用推理
- 学生模型严格只使用历史数据
- 教师代码与学生代码独立隔离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from config import CONFIG


# ============================================================================
# 公共组件
# ============================================================================

class BiLSTMEncoder(nn.Module):
    """
    [AI应用环节7] 时序编码器：Bi-LSTM（主方案备选，稳定无需额外依赖）。
    若Mamba可用则可替换，但LSTM更适合3小时Notebook内训练。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., L, M) 节点特征序列，支持 (N, L, M) 或 (B, N, L, M)
        Returns:
            h: (..., D_p) 时序嵌入
        """
        orig_shape = x.shape
        # 展平到 (B*N, L, M)
        if x.dim() == 4:
            B, N, L, M = x.shape
            x = x.reshape(B * N, L, M)
        elif x.dim() == 3:
            B = 1
            N, L, M = x.shape
            x = x.reshape(N, L, M)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

        out, _ = self.lstm(x)  # (B*N, L, 2*hidden)
        h = self.proj(out[:, -1, :])  # 取最后时刻 (B*N, D_p)

        # 恢复原始形状
        if len(orig_shape) == 4:
            h = h.reshape(orig_shape[0], orig_shape[1], -1)
        return h


class GCNLayer(nn.Module):
    """单层GCN：A*X*W"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim)
            adj: (N, N) 归一化邻接矩阵
        Returns:
            out: (N, out_dim)
        """
        support = self.linear(x)
        return adj @ support


class STGNNEncoder(nn.Module):
    """
    [AI应用环节7] 时空GNN编码器：Bi-LSTM(时序) + GCN(空间)

    教师/学生共享此架构，但教师额外接入未来趋势编码。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 gnn_hidden: int = 32, gnn_layers: int = 2, num_layers: int = 1):
        super().__init__()
        self.temporal_encoder = BiLSTMEncoder(input_dim, hidden_dim, hidden_dim, num_layers)

        self.gnn_layers = nn.ModuleList()
        prev_dim = hidden_dim
        for _ in range(gnn_layers):
            self.gnn_layers.append(GCNLayer(prev_dim, gnn_hidden))
            prev_dim = gnn_hidden

        self.proj = nn.Linear(gnn_hidden, output_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, L, M) 或 (B, N, L, M) 节点特征序列
            adj: (N, N) 或 (B, N, N) 邻接矩阵
        Returns:
            h: (N, D_p) 或 (B, N, D_p) 时空嵌入
        """
        is_batched = x.dim() == 4

        # 时序编码
        h = self.temporal_encoder(x)  # (N, hidden) 或 (B, N, hidden)

        # 空间编码（图卷积）
        if is_batched:
            B = h.shape[0]
            for gnn_layer in self.gnn_layers:
                h_list = [F.relu(gnn_layer(h[b], adj[b])) for b in range(B)]
                h = torch.stack(h_list, dim=0)
        else:
            for gnn_layer in self.gnn_layers:
                h = F.relu(gnn_layer(h, adj))

        # 投影到输出维度
        return self.proj(h)


class MultiChannelBilinearFusion(nn.Module):
    """
    [AI应用环节3] 多通道双线性融合（Multi-Channel Bilinear Fusion）

    命名修正（改进4）：原"VMV多通道注意力"实际是多通道双线性融合。
    低秩分解（改进4）：F^[d] ≈ U_d · V_d^T，参数量从O(D·D_p·D_f)降至O(D·r·(D_p+D_f))。
    """

    def __init__(self, d_p: int, d_f: int, d_out: int, rank: int = 8, tau: float = 0.5):
        super().__init__()
        self.d_out = d_out
        self.rank = rank
        self.tau = tau

        # 低秩分解参数
        self.U = nn.Parameter(torch.randn(d_out, d_p, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(d_out, d_f, rank) * 0.02)

        # 缩放点积注意力（真正的注意力，作用于融合后结果）
        self.W_Q = nn.Linear(d_f, d_out, bias=False)
        self.W_K = nn.Linear(d_p, d_out, bias=False)

    def forward(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: (N, D_p) 历史时空嵌入
            q: (N, D_f) 未来趋势嵌入
        Returns:
            h: (N, D_out) 未来感知时空表征
        """
        # 低秩双线性融合
        # p: (N, D_p), U: (D_out, D_p, rank) -> pU: (N, D_out, rank)
        pU = torch.einsum('np,dpr->ndr', p, self.U)
        # q: (N, D_f), V: (D_out, D_f, rank) -> qV: (N, D_out, rank)
        qV = torch.einsum('nf,dfr->ndr', q, self.V)
        # 逐元素乘积后对rank维求和 -> bilinear_vals: (N, D_out)
        bilinear_vals = (pU * qV).sum(dim=-1)

        # 缩放点积注意力
        Q = self.W_Q(q)
        K = self.W_K(p)
        attn_scores = self.tau * (Q @ K.T) / (self.d_out ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        h = attn_weights @ bilinear_vals
        return h


# ============================================================================
# 教师模型（[AI应用环节2] 仅本地/Notebook内训练，不部署）
# ============================================================================

# ============================================================
# [合规声明] 教师模型代码仅用于训练，不在main函数中调用推理
# 教师模型使用未来标签(f^{[t+1,t+T]})是LUPI范式的设计要求
# 提交的因子仅由学生模型(仅用历史数据)生成
# 详见AI技术应用说明文档第1.3节
# ============================================================

class FutureTrendEncoder(nn.Module):
    """
    [AI应用环节2] 未来趋势编码器：将未来涨跌标签编码为高层嵌入。
    仅教师模型使用，学生模型不接触此模块。
    """

    def __init__(self, future_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(future_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, future_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            future_labels: (N, T) 未来T天涨跌标签（二值）
        Returns:
            q: (N, D_f) 未来趋势嵌入
        """
        return self.encoder(future_labels.float())


class TeacherModel(nn.Module):
    """
    [AI应用环节2/3] 教师模型：历史GNN + 未来趋势编码 + 多通道双线性融合 + 预测头。

    教师模型拥有"特权信息"（未来标签），通过蒸馏将知识传递给学生。
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or CONFIG
        m = self.config.model

        # 组件A：历史时空编码器
        self.history_encoder = STGNNEncoder(
            input_dim=m.input_dim,
            hidden_dim=m.hidden_dim,
            output_dim=m.d_p,
            gnn_hidden=m.gnn_hidden,
            gnn_layers=m.gnn_layers,
            num_layers=m.num_layers,
        )

        # 组件B：未来趋势编码器
        self.future_encoder = FutureTrendEncoder(
            future_dim=self.config.train.future_horizon,
            output_dim=m.d_f,
        )

        # 组件C：多通道双线性融合
        self.fusion = MultiChannelBilinearFusion(
            d_p=m.d_p, d_f=m.d_f, d_out=m.d_out, rank=m.fusion_rank, tau=m.tau,
        )

        # 组件D：预测头
        self.pred_head = nn.Sequential(
            nn.Linear(m.d_out, m.projection_hidden),
            nn.ReLU(),
            nn.Linear(m.projection_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        future_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        教师前向传播。支持 (N, L, M) 或 (B, N, L, M) 输入。

        Args:
            x: (N, L, M) 或 (B, N, L, M) 历史特征序列
            adj: (N, N) 或 (B, N, N) 邻接矩阵
            future_labels: (N, T) 或 (B, N, T) 未来标签

        Returns:
            dict: logits, fused_repr, history_repr, future_repr
        """
        is_batched = x.dim() == 4

        if is_batched:
            B = x.shape[0]
            all_logits, all_fused, all_p, all_q = [], [], [], []
            for b in range(B):
                p = self.history_encoder(x[b], adj[b])
                q = self.future_encoder(future_labels[b])
                h_fused = self.fusion(p, q)
                logits = self.pred_head(h_fused).squeeze(-1)
                all_logits.append(logits)
                all_fused.append(h_fused)
                all_p.append(p)
                all_q.append(q)
            return {
                'logits': torch.stack(all_logits),
                'fused_repr': torch.stack(all_fused),
                'history_repr': torch.stack(all_p),
                'future_repr': torch.stack(all_q),
            }
        else:
            # 历史编码
            p = self.history_encoder(x, adj)  # (N, D_p)
            # 未来编码
            q = self.future_encoder(future_labels)  # (N, D_f)
            # 融合
            h_fused = self.fusion(p, q)  # (N, D_out)
            # 预测
            logits = self.pred_head(h_fused).squeeze(-1)  # (N,)

            return {
                'logits': logits,
                'fused_repr': h_fused,
                'history_repr': p,
                'future_repr': q,
            }

    def extract_fused_repr(self, x, adj, future_labels):
        """提取融合表征h^{t+}（供可蒸馏性验证使用）"""
        return self.forward(x, adj, future_labels)['fused_repr']


# ============================================================================
# 学生模型（[AI应用环节7] 部署到比赛环境）
# ============================================================================

class StudentModel(nn.Module):
    """
    [AI应用环节7] 学生模型：轻量历史GNN，仅使用历史数据。

    提取隐藏表征ĥ^t作为因子，不接触任何未来信息。
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or CONFIG
        m = self.config.model

        # 时空GNN编码器（与教师同构或更轻量）
        self.encoder = STGNNEncoder(
            input_dim=m.input_dim,
            hidden_dim=m.hidden_dim,
            output_dim=m.d_p,
            gnn_hidden=m.gnn_hidden,
            gnn_layers=m.gnn_layers,
            num_layers=m.num_layers,
        )

        # 可学习特征聚合器（AI主导特征选择）
        from feature_engineering import LearnableFeatureAggregator
        self.feat_aggregator = LearnableFeatureAggregator(
            input_dim=m.input_dim,
            output_dim=4,  # 4个可学习聚合特征
        )

        # 因子投影层：D_p -> 1（标量因子值）
        self.factor_projection = nn.Sequential(
            nn.Linear(m.d_p, m.projection_hidden),
            nn.ReLU(),
            nn.Linear(m.projection_hidden, 1),
        )

        # 分类头（辅助，用于CrossEntropy正则）
        self.cls_head = nn.Linear(m.d_p, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        学生前向传播（仅历史数据）。支持 (N, L, M) 或 (B, N, L, M) 输入。

        Args:
            x: (N, L, M) 或 (B, N, L, M) 历史特征序列
            adj: (N, N) 或 (B, N, N) 邻接矩阵

        Returns:
            dict: factor, hidden_repr, cls_logits
        """
        is_batched = x.dim() == 4

        if is_batched:
            B = x.shape[0]
            all_factor, all_hidden, all_cls = [], [], []
            for b in range(B):
                h = self.encoder(x[b], adj[b])
                f = self.factor_projection(h).squeeze(-1)
                c = self.cls_head(h).squeeze(-1)
                all_factor.append(f)
                all_hidden.append(h)
                all_cls.append(c)
            return {
                'factor': torch.stack(all_factor),
                'hidden_repr': torch.stack(all_hidden),
                'cls_logits': torch.stack(all_cls),
            }
        else:
            # 时空GNN编码
            hidden_repr = self.encoder(x, adj)  # (N, D_p)
            # 因子投影
            factor = self.factor_projection(hidden_repr).squeeze(-1)  # (N,)
            # 分类logits（辅助正则）
            cls_logits = self.cls_head(hidden_repr).squeeze(-1)  # (N,)

            return {
                'factor': factor,
                'hidden_repr': hidden_repr,
                'cls_logits': cls_logits,
            }

    def get_l1_penalty(self) -> torch.Tensor:
        """获取可学习聚合器的L1正则"""
        return self.feat_aggregator.l1_penalty()

    @torch.no_grad()
    def predict_factor(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """推理模式：直接返回因子值"""
        self.eval()
        return self.forward(x, adj)['factor']
