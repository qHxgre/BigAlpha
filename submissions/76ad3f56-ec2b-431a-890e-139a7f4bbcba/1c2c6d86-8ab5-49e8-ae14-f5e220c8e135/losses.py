"""
DisFT-GNN 损失函数模块（改进1、改进5）
======================================
整合已有核心组件：
- ICRankLoss: IC排序损失（主损失，对齐评估A项Rank IC）
- HSICLoss: HSIC蒸馏损失（可学习核带宽+稳定性保护）
- StudentTotalLoss: 学生三重损失组合（排序+CE+蒸馏）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


class ICRankLoss(nn.Module):
    """
    [AI应用环节5] IC排序损失：按日分组计算Spearman秩相关，取负作为损失。
    直接优化截面排序质量，对齐评估A项中的Rank_IC_mean和Rank_IC_IR。
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, factor_pred: torch.Tensor, future_return: torch.Tensor) -> torch.Tensor:
        """
        Args:
            factor_pred: (N,) 单日截面因子预测值
            future_return: (N,) 对应未来收益率
        Returns:
            loss: 负Spearman秩相关系数
        """
        mask = ~(torch.isnan(factor_pred) | torch.isnan(future_return))
        if mask.sum() < 10:
            return torch.tensor(0.0, device=factor_pred.device, requires_grad=True)

        pred = factor_pred[mask]
        ret = future_return[mask]

        pred_rank = pred.argsort().argsort().float()
        ret_rank = ret.argsort().argsort().float()

        pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + self.eps)
        ret_rank = (ret_rank - ret_rank.mean()) / (ret_rank.std() + self.eps)

        ic = (pred_rank * ret_rank).mean()
        return -ic


class BatchICRankLoss(nn.Module):
    """批量IC排序损失：对batch中多个交易日的IC取均值"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.ic_loss = ICRankLoss(eps)

    def forward(self, factor_pred: torch.Tensor, future_return: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            factor_pred: (B, N) B个交易日，N只股票
            future_return: (B, N)
        Returns:
            loss: 平均IC损失, ic_mean: 平均IC值
        """
        batch_size = factor_pred.shape[0]
        losses, ics = [], []

        for b in range(batch_size):
            loss = self.ic_loss(factor_pred[b], future_return[b])
            losses.append(loss)
            ics.append(-loss.detach())

        loss_mean = torch.stack(losses).mean()
        ic_mean = torch.stack(ics).mean()
        return loss_mean, ic_mean


class HSICLoss(nn.Module):
    """
    [AI应用环节4] HSIC蒸馏损失：最大化学生与教师表征的非线性统计依赖。

    工程加固（改进5）：
    1. 节点维度(n)计算，统计有效性更好
    2. 可学习核带宽σ，随特征分布漂移自适应
    3. 特征值下限阈值检测，不稳定时自动切换CKA
    4. CKA备选方案内置
    """

    def __init__(self, sigma_init: float = 1.0, eigenvalue_threshold: float = 1e-6, use_cka_fallback: bool = True):
        super().__init__()
        self.log_sigma_student = nn.Parameter(torch.tensor(sigma_init).log())
        self.log_sigma_teacher = nn.Parameter(torch.tensor(sigma_init).log())
        self.eigenvalue_threshold = eigenvalue_threshold
        self.use_cka_fallback = use_cka_fallback
        self._cka_mode = False

    def _rbf_kernel(self, x: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = log_sigma.exp()
        x_sq = (x ** 2).sum(dim=-1, keepdim=True)
        dist_sq = x_sq + x_sq.T - 2 * x @ x.T
        dist_sq = dist_sq.clamp(min=0)
        return torch.exp(-dist_sq / (2 * sigma ** 2 + 1e-8))

    def _compute_hsic(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - 1.0 / n
        Kc = K @ H
        Lc = L @ H
        return (Kc * Lc).sum() / ((n - 1) ** 2)

    def _compute_cka(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        hsic_kl = self._compute_hsic(K, L)
        hsic_kk = self._compute_hsic(K, K)
        hsic_ll = self._compute_hsic(L, L)
        return hsic_kl / (torch.sqrt(hsic_kk * hsic_ll + 1e-8) + 1e-8)

    def _check_stability(self, K: torch.Tensor, L: torch.Tensor) -> bool:
        try:
            eigvals_K = torch.linalg.eigvalsh(K)
            eigvals_L = torch.linalg.eigvalsh(L)
            min_eig = min(eigvals_K.min().item(), eigvals_L.min().item())
            if min_eig < self.eigenvalue_threshold:
                self._cka_mode = True
                return False
        except Exception:
            self._cka_mode = True
            return False
        return True

    def forward(self, student_repr: torch.Tensor, teacher_repr: torch.Tensor) -> torch.Tensor:
        """计算HSIC蒸馏损失（取负，最大化HSIC）"""
        if self.training and not self._cka_mode:
            self._check_stability(
                self._rbf_kernel(student_repr, self.log_sigma_student),
                self._rbf_kernel(teacher_repr, self.log_sigma_teacher),
            )

        K = self._rbf_kernel(student_repr, self.log_sigma_student)
        L = self._rbf_kernel(teacher_repr, self.log_sigma_teacher)

        if self._cka_mode and self.use_cka_fallback:
            dependency = self._compute_cka(K, L)
        else:
            dependency = self._compute_hsic(K, L)

        return -dependency

    def get_stats(self) -> Dict[str, float]:
        return {
            "sigma_student": self.log_sigma_student.exp().item(),
            "sigma_teacher": self.log_sigma_teacher.exp().item(),
            "cka_mode": self._cka_mode,
        }


@dataclass
class StudentLossConfig:
    """学生损失配置"""
    alpha: float = 1.0
    beta: float = 0.15
    lambda_distill: float = 0.3
    use_distillation: bool = True
    l1_lambda: float = 0.001  # 可学习聚合器的L1正则


class StudentTotalLoss(nn.Module):
    """
    [AI应用环节5] 学生三重损失（改进1：训练目标对齐评估指标）

    L = α·L_rank + β·L_ce + λ·L_distill + l1_lambda·L_l1
    """

    def __init__(self, config: StudentLossConfig):
        super().__init__()
        self.config = config
        self.ic_loss = BatchICRankLoss()
        self.hsic_loss = HSICLoss() if config.use_distillation else None

    def forward(
        self,
        factor_pred: torch.Tensor,
        future_return: torch.Tensor,
        student_repr: torch.Tensor,
        teacher_repr: Optional[torch.Tensor] = None,
        cls_logits: Optional[torch.Tensor] = None,
        cls_labels: Optional[torch.Tensor] = None,
        l1_penalty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算学生总损失及各项分量"""
        rank_loss, ic_mean = self.ic_loss(factor_pred, future_return)

        ce_loss = torch.tensor(0.0, device=factor_pred.device)
        if cls_logits is not None and cls_labels is not None:
            ce_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_labels.float())

        distill_loss = torch.tensor(0.0, device=factor_pred.device)
        if self.hsic_loss is not None and teacher_repr is not None:
            B, N = factor_pred.shape
            student_flat = student_repr.reshape(B, N, -1) if student_repr.dim() == 2 else student_repr
            teacher_flat = teacher_repr.reshape(B, N, -1) if teacher_repr.dim() == 2 else teacher_repr
            distill_losses = []
            for b in range(B):
                dl = self.hsic_loss(student_flat[b], teacher_flat[b])
                distill_losses.append(dl)
            distill_loss = torch.stack(distill_losses).mean()

        l1_loss = l1_penalty if l1_penalty is not None else torch.tensor(0.0, device=factor_pred.device)

        total = (
            self.config.alpha * rank_loss
            + self.config.beta * ce_loss
            + self.config.lambda_distill * distill_loss
            + self.config.l1_lambda * l1_loss
        )

        return {
            'total_loss': total,
            'rank_loss': rank_loss,
            'ce_loss': ce_loss,
            'distill_loss': distill_loss,
            'l1_loss': l1_loss,
            'ic_mean': ic_mean,
        }

    def update_lambda(self, epoch: int, total_epochs: int):
        """蒸馏权重λ从0.1逐步增至0.5"""
        if self.config.use_distillation:
            progress = min(epoch / total_epochs, 1.0)
            self.config.lambda_distill = self.config.lambda_distill_start + (self.config.lambda_distill_end - self.config.lambda_distill_start) * progress
