"""本地实验优化器。"""

from __future__ import annotations

import torch
from torch.nn.utils import parameters_to_vector
from torch.optim import Optimizer


class PVRAdamW(Optimizer):
    """PVR 梯度子空间投影 + AdamW。

    ``project_ratio=0`` 时退化为 AdamW；先用较小投影强度烟测，再做超参搜索。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        sketch_rank: int = 8,
        oja_lr: float = 1e-3,
        qr_freq: int = 100,
        project_ratio: float = 0.25,
    ):
        if not 0 <= project_ratio <= 1:
            raise ValueError("project_ratio must be in [0, 1]")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            sketch_rank=sketch_rank,
            oja_lr=oja_lr,
            qr_freq=qr_freq,
            project_ratio=project_ratio,
        )
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError("PVRAdamW currently supports one parameter group")
        self._basis = None
        self._layout = None
        self._global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        parameters = [p for p in group["params"] if p.grad is not None]
        if not parameters:
            return loss
        if any(p.grad.is_sparse for p in parameters):
            raise RuntimeError("PVRAdamW does not support sparse gradients")

        gradient = parameters_to_vector([p.grad for p in parameters])
        layout = tuple((id(p), p.numel()) for p in parameters)
        rank = min(int(group["sketch_rank"]), gradient.numel())
        if (
            self._basis is None
            or self._layout != layout
            or self._basis.shape != (gradient.numel(), rank)
            or self._basis.device != gradient.device
        ):
            basis = torch.randn(
                gradient.numel(), rank, device=gradient.device, dtype=gradient.dtype
            )
            self._basis, _ = torch.linalg.qr(basis, mode="reduced")
            self._layout = layout

        projection = gradient @ self._basis
        oja_lr = float(group["oja_lr"])
        if oja_lr:
            update = torch.outer(gradient, projection)
            self._basis.add_(
                update - self._basis @ (update.transpose(0, 1) @ self._basis),
                alpha=oja_lr,
            )
        self._global_step += 1
        if self._global_step % int(group["qr_freq"]) == 0:
            self._basis, _ = torch.linalg.qr(self._basis, mode="reduced")

        ratio = float(group["project_ratio"])
        if ratio:
            gradient = gradient - ratio * (self._basis @ (gradient @ self._basis))

        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.grad.copy_(gradient[offset:offset + size].view_as(parameter))
            offset += size

        beta1, beta2 = group["betas"]
        for parameter in parameters:
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            if group["weight_decay"]:
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
            state["exp_avg"].mul_(beta1).add_(parameter.grad, alpha=1 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(
                parameter.grad, parameter.grad, value=1 - beta2
            )
            bias1 = 1 - beta1 ** state["step"]
            bias2 = 1 - beta2 ** state["step"]
            denominator = state["exp_avg_sq"].sqrt().div_(bias2 ** 0.5)
            denominator.add_(group["eps"])
            parameter.addcdiv_(
                state["exp_avg"],
                denominator,
                value=-(group["lr"] / bias1),
            )
        return loss


def build_optimizer(model: torch.nn.Module, config: dict) -> Optimizer:
    name = config["optimizer"].lower()
    common = dict(
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)
    if name == "pvr_adamw":
        return PVRAdamW(
            model.parameters(),
            **common,
            **(config.get("optimizer_params") or {}),
        )
    if name == "pvradam":
        from train.pvradam import PVRAdam

        return PVRAdam(
            model.parameters(),
            lr=common["lr"],
            **(config.get("optimizer_params") or {}),
        )
    raise ValueError(
        f"Unknown optimizer {name!r}; use adamw, pvradam, or pvr_adamw"
    )
