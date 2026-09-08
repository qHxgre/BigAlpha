# ModelSet_Loss.py
# PonyOracle EVO: 加权成对 logistic 排序损失（零超参）
# L = Σ (r_i-r_j)^+ · softplus(f_j-f_i)  /  Σ (r_i-r_j)^+
# f = 原始分数（不做 z-score）；r = rank / return / van der Waerden
from typing import Dict, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LOG2 = math.log(2.0)  # 常数预测器损失 ≈ 0.693；L < LOG2 才算有技能


# ======================================================================
# 截面工具（监控 / 验证用，不进主损失）
# ======================================================================

def masked_z(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """沿股票维 masked z-score；x/m: (N,)"""
    m = m.float()
    n = m.sum().clamp_min(1.0)
    mu = (x * m).sum() / n
    var = (((x - mu) ** 2) * m).sum() / n
    return (x - mu) / (var.sqrt() + eps)


def daily_ic(score: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """可微截面 Pearson IC（仅监控；训练主损失不用）。"""
    s = masked_z(score, m)
    t = masked_z(y, m)
    return (s * t * m.float()).sum() / m.float().sum().clamp_min(1.0)


def van_der_waerden(r: torch.Tensor) -> torch.Tensor:
    """排名 → Φ^{-1}(k/(N+1))（变体 B，零超参尾部加强）。"""
    n = r.numel()
    # 1..N 名次（越大越好与 r 同序）
    k = r.argsort().argsort().to(torch.float32) + 1.0
    p = k / (n + 1.0)
    return math.sqrt(2.0) * torch.erfinv((2.0 * p - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def pairwise_logistic_loss(score: torch.Tensor, rank_label: torch.Tensor) -> torch.Tensor:
    """
    加权成对 logistic 排序。score / rank_label 均为有效子集上的 1D 向量。
    强制 fp32，避免 bf16 下 N×N 求和丢精度。
    """
    f = score.float()
    r = rank_label.float()
    g = (r[:, None] - r[None, :]).clamp_(min=0.0)  # g[i,j]>0：i 应高于 j
    d = f[None, :] - f[:, None]                    # d[i,j]=f_j-f_i；>0 即排错
    den = g.sum()
    if float(den.item()) < 1e-12:
        return f.sum() * 0.0
    return (g * F.softplus(d)).sum() / den


# ======================================================================
# 主损失模块
# ======================================================================

class PonyOracleLoss(nn.Module):
    """
    单日加权成对 logistic。无温度 / 门槛 / 项间配比 / 锚定 / 退火。

    典型用法（Framework：K 天梯度累积，逐日释放图）::

        opt.zero_grad()
        for day in K_days:
            loss_day, info = criterion.daily(pred, y_rank, y_ret, mask)
            (loss_day / K).backward()
        opt.step()

    pair_label:
      - "rank"   : 默认，稳健顺序监督
      - "return" : 变体 A，P&L 量纲对齐（需 winsorize）
      - "vdw"    : 变体 B，van der Waerden 尾部加强
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.direction = int(getattr(config, "rank_direction", -1))
        self.pair_label = str(getattr(config, "pair_label", "rank")).lower()
        self.min_valid = int(getattr(config, "min_valid_stocks", 50))
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long))

    def step(self):
        """每个优化步后由 Framework 调用。"""
        self._train_step += 1

    @torch.no_grad()
    def current_weights(self) -> dict:
        return {
            "pair_label": self.pair_label,
            "min_valid_stocks": self.min_valid,
            "meta_batch_k": int(getattr(self.config, "meta_batch_k", 12)),
            "log2_ref": LOG2,
        }

    # ----- 标签预处理 -----
    def _prep_labels(
        self,
        y_rank: torch.Tensor,
        y_ret: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = mask.bool()
        if self.direction == 1:
            y_rank = -y_rank
            y_ret = -y_ret
        clamp = float(getattr(self.config, "ret_clamp", 4.0) or 4.0)
        y_ret = y_ret.clamp(-clamp, clamp)
        y_rank = y_rank.masked_fill(~m, 0.0)
        y_ret = y_ret.masked_fill(~m, 0.0)
        return y_rank, y_ret, m

    def _pair_target(self, y_rank: torch.Tensor, y_ret: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """按 pair_label 构造成对损失用的 r（仅有效位）。"""
        if self.pair_label == "return":
            return y_ret[m].float()
        if self.pair_label in ("vdw", "waerden", "van_der_waerden"):
            return van_der_waerden(y_rank[m].float())
        # 默认 rank
        return y_rank[m].float()

    # ----- 单日 -----
    def daily(
        self,
        pred_rank: torch.Tensor,
        y_rank: torch.Tensor,
        y_ret: torch.Tensor,
        mask: torch.Tensor,
        aux: Optional[dict] = None,
        pred_ret: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        loss : 标量（保梯度，供 /K 后 backward）
        info : 监控量（已 detach），供日志；不含跨天图
        """
        _ = aux, pred_ret
        if pred_rank.dim() != 1:
            raise ValueError("daily() 期望单日截面向量 (N,)")

        y_rank, y_ret, m = self._prep_labels(y_rank, y_ret, mask)
        n_valid = int(m.sum().item())
        zero = pred_rank.new_zeros(())

        if n_valid < self.min_valid:
            info = {
                "L_pair": zero.detach(),
                "ic": zero.detach(),
                "ic_ret": zero.detach(),
                "r_ls": zero.detach(),
                "score_std": zero.detach(),
                "bound": zero.detach(),
                "L_daily": zero.detach(),
                "n_valid": n_valid,
            }
            return pred_rank.sum() * 0.0, info

        # 原始分数：不做 z-score，尺度由网络自学
        f = pred_rank
        r = self._pair_target(y_rank, y_ret, m)
        loss = pairwise_logistic_loss(f[m], r)

        # ----- 监控（detach，不进反传）-----
        with torch.no_grad():
            f_m = f[m].float()
            score_std = f_m.std(unbiased=False)
            # 认证下界：Π/C ≥ 1 - 2L/log2
            bound = 1.0 - 2.0 * float(loss.detach().item()) / LOG2
            ic = daily_ic(f, y_rank, m)
            ic_ret = daily_ic(f, y_ret, m)
            # 硬十分位多空（评估用代理，非损失项）
            r_ls = f.new_tensor(
                hard_long_short_return(f, y_ret, m, frac=0.1)
            )

        info = {
            "L_pair": loss.detach(),
            "ic": ic.detach(),
            "ic_ret": ic_ret.detach(),
            "r_ls": r_ls.detach(),
            "score_std": score_std.detach(),
            "bound": f.new_tensor(bound).detach(),
            "L_daily": loss.detach(),
            "n_valid": n_valid,
        }
        return loss, info

    # ----- K 天日志归约（无图，仅汇总）-----
    def reduce_meta(self, infos: Sequence[dict]) -> Tuple[torch.Tensor, dict]:
        """
        兼容旧接口：对已逐日 backward 的 infos 做日志平均。
        返回的 loss 无梯度，不可再 backward（训练请在 daily 上 /K 反传）。
        """
        if not infos:
            raise ValueError("reduce_meta: infos 为空")

        def _mean_key(key: str) -> float:
            vals = []
            for info in infos:
                v = info.get(key)
                if v is None:
                    continue
                if torch.is_tensor(v):
                    vals.append(float(v.detach().float().item()))
                else:
                    vals.append(float(v))
            if not vals:
                return float("nan")
            return float(sum(vals) / len(vals))

        L = _mean_key("L_pair")
        logs = {
            "loss": L,
            "L_pair": L,
            "L_daily": L,
            "ic_mean": _mean_key("ic"),
            "r_mean": _mean_key("r_ls"),
            "score_std": _mean_key("score_std"),
            "bound": _mean_key("bound"),
            "K": len(infos),
        }
        # 占位张量（无梯度）；真正反传已在 Framework 逐日完成
        ref = infos[0]["L_pair"]
        loss_tensor = ref.new_tensor(L if L == L else 0.0)
        return loss_tensor, logs

    # ----- 验证 / 兼容入口 -----
    def forward(
        self,
        pred_rank: torch.Tensor,
        pred_ret: torch.Tensor,
        y_rank: torch.Tensor,
        y_ret: torch.Tensor,
        mask: torch.Tensor,
        aux: Optional[dict] = None,
    ):
        """
        单日入口。返回 (loss, L_a, L_b)：
          L_a = L_pair，L_b = 认证下界 bound
        """
        _ = pred_ret
        loss, info = self.daily(
            pred_rank, y_rank, y_ret, mask, aux=aux, pred_ret=pred_ret
        )
        L_a = info["L_pair"]
        L_b = info["bound"]
        return loss, L_a, L_b


# ======================================================================
# 验证代理总分（硬分组，不进反传）
# ======================================================================

@torch.no_grad()
def hard_long_short_return(
    score: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    frac: float = 0.1,
) -> float:
    """真实硬十分位多空收益（评估用）。"""
    m = mask.bool()
    if int(m.sum().item()) < 10:
        return float("nan")
    s = score[m]
    yt = y[m]
    n = s.numel()
    k = max(int(n * frac), 1)
    order = s.argsort(descending=True)
    long = yt[order[:k]].mean()
    short = yt[order[-k:]].mean()
    return float((long - short).item())


@torch.no_grad()
def proxy_score_from_series(
    ics: List[float],
    rets: List[float],
    regime_size: int = 20,
) -> Dict[str, float]:
    """
    由验证期逐日 IC / 多空收益序列构造代理指标。
    Stress ≈ 按时间块 IC 均值的最小值。
    """
    import numpy as np

    ic = np.asarray([x for x in ics if x == x], dtype=np.float64)
    r = np.asarray([x for x in rets if x == x], dtype=np.float64)
    out = {
        "IC_mean": float("nan"),
        "IC_IR": float("nan"),
        "SR": float("nan"),
        "Stress": float("nan"),
        "Proxy": float("nan"),
    }
    if ic.size == 0:
        return out
    out["IC_mean"] = float(ic.mean())
    out["IC_IR"] = float(ic.mean() / (ic.std() + 1e-12))
    if r.size > 1:
        out["SR"] = float(r.mean() / (r.std() + 1e-12) * math.sqrt(244))
    rs = max(int(regime_size), 1)
    block_means = []
    for i in range(0, ic.size, rs):
        block = ic[i: i + rs]
        if block.size > 0:
            block_means.append(block.mean())
    out["Stress"] = float(min(block_means)) if block_means else out["IC_mean"]
    parts = [out["IC_mean"], out["IC_IR"], out["SR"], out["Stress"]]
    vals = [p for p in parts if p == p]
    out["Proxy"] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out
