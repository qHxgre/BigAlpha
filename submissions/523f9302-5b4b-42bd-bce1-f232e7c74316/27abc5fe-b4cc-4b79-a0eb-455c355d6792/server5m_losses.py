"""分类与回归损失。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class OrdinalCELoss(nn.Module):
    def __init__(self, num_classes: int, sigma: float = 0.7):
        super().__init__()
        true_class = torch.arange(num_classes).float().unsqueeze(1)
        predicted_class = torch.arange(num_classes).float().unsqueeze(0)
        distance_sq = (true_class - predicted_class) ** 2
        self.register_buffer(
            "soft_target",
            F.softmax(-distance_sq / (2 * sigma ** 2), dim=1),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        soft_target = self.soft_target[target]
        return -(soft_target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def build_loss(name: str, task: str, num_classes: int) -> nn.Module:
    if task == "regression":
        if name != "mse":
            raise ValueError("Regression requires loss=mse")
        return nn.MSELoss()
    if name == "celoss":
        return nn.CrossEntropyLoss()
    if name == "ordinal_ce":
        return OrdinalCELoss(num_classes=num_classes, sigma=0.7)
    raise ValueError(f"Unknown loss: {name}")
