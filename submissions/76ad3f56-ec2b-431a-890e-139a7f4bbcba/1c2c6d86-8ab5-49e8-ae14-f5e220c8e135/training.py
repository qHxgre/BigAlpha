"""
DisFT-GNN 训练管线（Layer 3/4）
================================
包含完整的训练流程：
1. 教师模型训练（使用历史+未来标签）
2. 可蒸馏性前置验证（R²检验）
3. 蒸馏+学生训练（仅历史数据，IC排序损失为主）
4. 学生推理

[AI应用环节6] 可蒸馏性验证标注。
合规：全部在Notebook内完成，随机种子固定，训练日志可审计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time

from config import CONFIG
from models import TeacherModel, StudentModel
from losses import (
    StudentTotalLoss, StudentLossConfig, HSICLoss,
    ICRankLoss, BatchICRankLoss,
)
from graph_construction import GraphBuilder, LearnableAdjFusion, TimeSeriesSplitter, normalize_adj


class DistillabilityValidator:
    """
    [AI应用环节6] 可蒸馏性前置验证（改进3）

    验证学生仅用历史数据能否还原教师的未来感知表征。
    若R²过低，说明未来信息不可从历史预测，蒸馏路线失效，应切换纯时序方案。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 4):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def validate(
        self,
        teacher_model: nn.Module,
        train_features: torch.Tensor,
        train_adjs: torch.Tensor,
        train_future_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_adjs: torch.Tensor,
        val_future_labels: torch.Tensor,
        device: torch.device = torch.device('cpu'),
    ) -> Dict[str, float]:
        """
        执行可蒸馏性验证。

        Args:
            teacher_model: 已训练的教师模型
            train_features: (T_train, N, L, M) 训练集历史特征
            train_adjs: (T_train, N, N) 训练集邻接矩阵
            train_future_labels: (T_train, N, T_future) 训练集未来标签
            val_*: 验证集对应数据
            device: 计算设备

        Returns:
            dict: r2_train, r2_val, decision('proceed'/'cautious'/'abort')
        """
        config = CONFIG
        self.mlp = self.mlp.to(device)
        teacher_model = teacher_model.to(device).eval()

        print("[可蒸馏性验证] 提取教师表征...")
        train_teacher_repr = self._extract_teacher_repr(
            teacher_model, train_features, train_adjs, train_future_labels, device
        )
        val_teacher_repr = self._extract_teacher_repr(
            teacher_model, val_features, val_adjs, val_future_labels, device
        )

        # 用历史特征的最后时刻值拟合教师表征
        # train_features: (T, N, L, M) -> 取最后时刻 (T*N, M)
        train_flat = train_features[:, :, -1, :].reshape(-1, train_features.shape[-1]).to(device)
        train_target = train_teacher_repr.reshape(-1, train_teacher_repr.shape[-1])
        val_flat = val_features[:, :, -1, :].reshape(-1, val_features.shape[-1]).to(device)
        val_target = val_teacher_repr.reshape(-1, val_teacher_repr.shape[-1])

        print("[可蒸馏性验证] 训练MLP拟合器...")
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=1e-3)

        for epoch in range(30):
            self.mlp.train()
            optimizer.zero_grad()
            pred = self.mlp(train_flat)
            loss = F.mse_loss(pred, train_target)
            loss.backward()
            optimizer.step()

        self.mlp.eval()
        with torch.no_grad():
            train_pred = self.mlp(train_flat)
            val_pred = self.mlp(val_flat)
            r2_train = self._compute_r2(train_pred, train_target)
            r2_val = self._compute_r2(val_pred, val_target)

        if r2_val > config.train.distill_r2_cautious:
            decision = 'proceed'
        elif r2_val > config.train.distill_r2_threshold:
            decision = 'cautious'
        else:
            decision = 'abort'

        print(f"[可蒸馏性验证] R²_train={r2_train:.4f}, R²_val={r2_val:.4f}, 决策={decision}")
        return {'r2_train': r2_train, 'r2_val': r2_val, 'decision': decision}

    @torch.no_grad()
    def _extract_teacher_repr(self, teacher_model, features, adjs, future_labels, device):
        """提取教师模型的融合表征h^{t+}"""
        reprs = []
        for t in range(features.shape[0]):
            x = features[t].to(device)
            adj = adjs[t].to(device)
            fl = future_labels[t].to(device)
            h = teacher_model.extract_fused_repr(x, adj, fl)
            reprs.append(h.cpu())
        return torch.stack(reprs)

    @staticmethod
    def _compute_r2(pred, target):
        ss_res = ((pred - target) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        return (1 - ss_res / (ss_tot + 1e-8)).item()


class TeacherTrainer:
    """
    [AI应用环节2] 教师模型训练器。
    教师使用历史+未来标签训练，优化CrossEntropy。
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.model = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train(
        self,
        train_features: torch.Tensor,   # (T, N, L, M)
        train_adjs: torch.Tensor,       # (T, N, N)
        train_future_labels: torch.Tensor,  # (T, N, T_future)
        train_future_returns: torch.Tensor,  # (T, N) 用于构建标签
        val_features: torch.Tensor,
        val_adjs: torch.Tensor,
        val_future_labels: torch.Tensor,
        val_future_returns: torch.Tensor,
        device: torch.device = torch.device('cpu'),
    ) -> TeacherModel:
        """训练教师模型"""
        t_config = self.config.train
        m_config = self.config.model

        self.model = TeacherModel(self.config).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=t_config.learning_rate)

        print(f"\n{'='*60}")
        print(f"[教师训练] 开始训练 (epochs={t_config.teacher_epochs})")
        print(f"{'='*60}")

        for epoch in range(t_config.teacher_epochs):
            # 训练
            self.model.train()
            train_losses = []
            n_samples = train_features.shape[0]

            indices = torch.randperm(n_samples)
            for i in range(0, n_samples, t_config.batch_size):
                batch_idx = indices[i:i + t_config.batch_size]
                if len(batch_idx) == 0:
                    continue

                x = train_features[batch_idx].to(device)
                adj = train_adjs[batch_idx].to(device)
                fl = train_future_labels[batch_idx].to(device)
                # 二分类标签：未来收益>0为1
                y = (train_future_returns[batch_idx] > t_config.future_threshold).float().to(device)

                optimizer.zero_grad()
                outputs = self.model(x, adj, fl)
                loss = F.binary_cross_entropy_with_logits(outputs['logits'], y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # 验证
            val_loss = self._validate(val_features, val_adjs, val_future_labels, val_future_returns, device)

            avg_train = np.mean(train_losses) if train_losses else 0
            print(f"  Epoch {epoch+1}/{t_config.teacher_epochs} | train_loss={avg_train:.4f} | val_loss={val_loss:.4f}")

            # 早停
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= t_config.teacher_patience:
                    print(f"  早停：{t_config.teacher_patience}轮无改善")
                    break

        print(f"[教师训练] 完成，最佳验证损失={self.best_val_loss:.4f}")
        return self.model

    @torch.no_grad()
    def _validate(self, val_features, val_adjs, val_future_labels, val_future_returns, device):
        self.model.eval()
        losses = []
        t_config = self.config.train
        n_samples = val_features.shape[0]

        for i in range(0, n_samples, t_config.batch_size):
            batch = slice(i, min(i + t_config.batch_size, n_samples))
            x = val_features[batch].to(device)
            adj = val_adjs[batch].to(device)
            fl = val_future_labels[batch].to(device)
            y = (val_future_returns[batch] > t_config.future_threshold).float().to(device)

            outputs = self.model(x, adj, fl)
            loss = F.binary_cross_entropy_with_logits(outputs['logits'], y)
            losses.append(loss.item())

        return np.mean(losses) if losses else float('inf')


class StudentTrainer:
    """
    [AI应用环节5] 学生模型训练器（蒸馏+排序损失）。

    学生训练使用三重损失：IC排序损失(主) + CrossEntropy(辅) + HSIC蒸馏。
    若蒸馏被禁用（可蒸馏性验证未通过），则纯排序损失训练。
    """

    def __init__(self, config=None, use_distillation: bool = True):
        self.config = config or CONFIG
        self.use_distillation = use_distillation
        self.model = None
        self.loss_fn = None
        self.best_val_ic = -float('inf')
        self.patience_counter = 0
        self.training_log = []  # 训练日志（供审计）

    def train(
        self,
        train_features: torch.Tensor,
        train_adjs: torch.Tensor,
        train_future_returns: torch.Tensor,
        val_features: torch.Tensor,
        val_adjs: torch.Tensor,
        val_future_returns: torch.Tensor,
        teacher_model: Optional[TeacherModel] = None,
        teacher_train_repr: Optional[torch.Tensor] = None,
        teacher_val_repr: Optional[torch.Tensor] = None,
        device: torch.device = torch.device('cpu'),
    ) -> StudentModel:
        """训练学生模型"""
        t_config = self.config.train

        # 初始化学生模型
        self.model = StudentModel(self.config).to(device)

        # 初始化损失函数
        loss_config = StudentLossConfig(
            alpha=t_config.alpha_rank,
            beta=t_config.beta_ce,
            lambda_distill=t_config.lambda_distill_start,
            use_distillation=self.use_distillation,
        )
        self.loss_fn = StudentTotalLoss(loss_config).to(device)

        # 教师模型（冻结）
        if self.use_distillation and teacher_model is not None:
            teacher_model = teacher_model.to(device).eval()
            for p in teacher_model.parameters():
                p.requires_grad = False

        optimizer = torch.optim.Adam(self.model.parameters(), lr=t_config.student_lr)

        distill_status = "启用" if self.use_distillation else "禁用（纯时序方案）"
        print(f"\n{'='*60}")
        print(f"[学生训练] 开始训练 (epochs={t_config.student_epochs}, 蒸馏={distill_status})")
        print(f"{'='*60}")

        for epoch in range(t_config.student_epochs):
            # 更新蒸馏权重λ
            self.loss_fn.update_lambda(epoch, t_config.student_epochs)

            # 训练
            self.model.train()
            train_losses = []
            train_ics = []
            n_samples = train_features.shape[0]

            indices = torch.randperm(n_samples)
            for i in range(0, n_samples, t_config.batch_size):
                batch_idx = indices[i:i + t_config.batch_size]
                if len(batch_idx) == 0:
                    continue

                x = train_features[batch_idx].to(device)
                adj = train_adjs[batch_idx].to(device)
                y_ret = train_future_returns[batch_idx].to(device)
                y_cls = (y_ret > t_config.future_threshold).float()

                optimizer.zero_grad()

                # 学生前向
                outputs = self.model(x, adj)

                # 教师表征（蒸馏目标）
                teacher_repr = None
                if self.use_distillation and teacher_model is not None:
                    with torch.no_grad():
                        # 教师需要未来标签，这里用预提取的表征
                        if teacher_train_repr is not None:
                            teacher_repr = teacher_train_repr[batch_idx].to(device)

                # L1正则
                l1_penalty = self.model.get_l1_penalty()

                # 计算损失
                losses = self.loss_fn(
                    factor_pred=outputs['factor'],
                    future_return=y_ret,
                    student_repr=outputs['hidden_repr'],
                    teacher_repr=teacher_repr,
                    cls_logits=outputs['cls_logits'],
                    cls_labels=y_cls,
                    l1_penalty=l1_penalty,
                )

                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(losses['total_loss'].item())
                train_ics.append(losses['ic_mean'].item())

            # 验证
            val_ic = self._validate(val_features, val_adjs, val_future_returns, device)

            avg_loss = np.mean(train_losses) if train_losses else 0
            avg_ic = np.mean(train_ics) if train_ics else 0

            log_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_ic': avg_ic,
                'val_ic': val_ic,
                'lambda': self.loss_fn.config.lambda_distill,
            }
            self.training_log.append(log_entry)

            print(f"  Epoch {epoch+1}/{t_config.student_epochs} | "
                  f"loss={avg_loss:.4f} | train_IC={avg_ic:.4f} | "
                  f"val_IC={val_ic:.4f} | λ={self.loss_fn.config.lambda_distill:.2f}")

            # 早停（基于验证IC）
            if val_ic > self.best_val_ic:
                self.best_val_ic = val_ic
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= t_config.student_patience:
                    print(f"  早停：{t_config.student_patience}轮无改善")
                    break

        print(f"[学生训练] 完成，最佳验证IC={self.best_val_ic:.4f}")
        return self.model

    @torch.no_grad()
    def _validate(self, val_features, val_adjs, val_future_returns, device):
        self.model.eval()
        ics = []
        n_samples = val_features.shape[0]

        for i in range(n_samples):
            x = val_features[i:i+1].to(device)
            adj = val_adjs[i:i+1].to(device)
            y_ret = val_future_returns[i].to(device)

            outputs = self.model(x, adj)
            factor = outputs['factor'].squeeze(0)

            mask = ~(torch.isnan(factor) | torch.isnan(y_ret))
            if mask.sum() < 10:
                continue

            pred = factor[mask]
            ret = y_ret[mask]
            pred_rank = pred.argsort().argsort().float()
            ret_rank = ret.argsort().argsort().float()
            pred_rank = (pred_rank - pred_rank.mean()) / (pred_rank.std() + 1e-8)
            ret_rank = (ret_rank - ret_rank.mean()) / (ret_rank.std() + 1e-8)
            ic = (pred_rank * ret_rank).mean().item()
            ics.append(ic)

        return np.mean(ics) if ics else 0.0


@torch.no_grad()
def extract_teacher_representations(
    teacher_model: TeacherModel,
    features: torch.Tensor,
    adjs: torch.Tensor,
    future_labels: torch.Tensor,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """提取教师模型在所有时间点的融合表征（供学生蒸馏使用）"""
    teacher_model = teacher_model.to(device).eval()
    reprs = []

    for t in range(features.shape[0]):
        x = features[t].to(device)
        adj = adjs[t].to(device)
        fl = future_labels[t].to(device)
        h = teacher_model.extract_fused_repr(x, adj, fl)
        reprs.append(h.cpu())

    return torch.stack(reprs)


def set_seed(seed: int = 42):
    """固定随机种子，确保审计可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
