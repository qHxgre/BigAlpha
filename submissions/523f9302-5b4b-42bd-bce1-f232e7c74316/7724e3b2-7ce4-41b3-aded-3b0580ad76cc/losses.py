from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def rank_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    order = torch.argsort(torch.argsort(x))
    ranks = order.float()
    return zscore(ranks, eps=eps)


def base_loss(pred: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(pred, target)
    if name == "huber":
        return F.huber_loss(pred, target, delta=0.01)
    if name == "daily_z_mse":
        pred_z = (pred - pred.mean()) / (pred.std(unbiased=False) + 1e-6)
        target_z = (target - target.mean()) / (target.std(unbiased=False) + 1e-6)
        return F.mse_loss(pred_z, target_z)
    if name == "neg_ic":
        pred_z = zscore(pred)
        target_z = zscore(target)
        return -(pred_z * target_z).mean()
    if name == "rank_corr":
        return rank_corr_surrogate_loss(pred, target)
    if name == "spearman_aligned":
        return spearman_aligned_loss(pred, target)
    raise ValueError(f"Unknown loss {name}")


def rank_corr_surrogate_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    return -(zscore(pred) * rank_zscore(target)).mean()


def group_rank_corr_surrogate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_groups: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Between-group rank correlation.

    Splits the day's cross-section into ``n_groups`` deciles by target rank and
    correlates the group-mean standardized score with the group-mean target.
    This measures decile monotonicity (between-group ordering) directly and is
    insensitive to intra-group noise, which is what a top-K / long-only use
    cares about.
    """
    n = pred.numel()
    if n < 2 * n_groups:
        return rank_corr_surrogate_loss(pred, target)
    pred_z = zscore(pred)
    target_z = zscore(target)
    order = torch.argsort(target)
    group_size = n // n_groups
    pred_means: list[torch.Tensor] = []
    target_means: list[torch.Tensor] = []
    for g in range(n_groups):
        lo = g * group_size
        hi = n if g == n_groups - 1 else (g + 1) * group_size
        idx = order[lo:hi]
        pred_means.append(pred_z[idx].mean())
        target_means.append(target_z[idx].mean())
    gp = torch.stack(pred_means)
    gt = torch.stack(target_means)
    gp = (gp - gp.mean()) / (gp.std(unbiased=False) + eps)
    gt = (gt - gt.mean()) / (gt.std(unbiased=False) + eps)
    return -(gp * gt).mean()


def pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_pairs: int = 8192,
    temperature: float = 1.0,
) -> torch.Tensor:
    n = pred.numel()
    if n < 2:
        return pred.sum() * 0.0
    idx_i = torch.randint(0, n, (max_pairs,), device=pred.device)
    idx_j = torch.randint(0, n, (max_pairs,), device=pred.device)
    sign = torch.sign(target[idx_i] - target[idx_j])
    keep = sign != 0
    if keep.sum() == 0:
        return pred.sum() * 0.0
    margin = sign[keep] * (pred[idx_i[keep]] - pred[idx_j[keep]]) / max(temperature, 1e-6)
    return F.softplus(-margin).mean()


def spearman_aligned_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    daily_z_weight: float = 0.2,
    rank_corr_weight: float = 0.5,
    pairwise_weight: float = 0.3,
    pairwise_max_pairs: int = 8192,
    pairwise_temperature: float = 0.05,
) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    daily_z = F.mse_loss(zscore(pred), zscore(target))
    rank_corr = rank_corr_surrogate_loss(pred, target)
    pairwise = pairwise_rank_loss(
        pred,
        target,
        max_pairs=pairwise_max_pairs,
        temperature=pairwise_temperature,
    )
    return daily_z_weight * daily_z + rank_corr_weight * rank_corr + pairwise_weight * pairwise


class SpearmanAlignedLoss(nn.Module):
    """Composite ranking loss with learnable uncertainty weights."""

    def __init__(
        self,
        learnable: bool = True,
        pairwise_max_pairs: int = 8192,
        pairwise_temperature: float = 0.05,
    ):
        super().__init__()
        self.learnable = bool(learnable)
        self.pairwise_max_pairs = int(pairwise_max_pairs)
        self.pairwise_temperature = float(pairwise_temperature)
        if self.learnable:
            self.log_sigma_daily_z = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_rank = nn.Parameter(torch.tensor(0.0))
            self.log_sigma_pairwise = nn.Parameter(torch.tensor(0.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.numel() < 2:
            return pred.sum() * 0.0
        daily_z = F.mse_loss(zscore(pred), zscore(target))
        rank_corr = rank_corr_surrogate_loss(pred, target)
        pairwise = pairwise_rank_loss(
            pred,
            target,
            max_pairs=self.pairwise_max_pairs,
            temperature=self.pairwise_temperature,
        )
        if not self.learnable:
            return daily_z + rank_corr + pairwise
        return (
            0.5 * torch.exp(-self.log_sigma_daily_z) * daily_z
            + 0.5 * self.log_sigma_daily_z
            + 0.5 * torch.exp(-self.log_sigma_rank) * rank_corr
            + 0.5 * self.log_sigma_rank
            + 0.5 * torch.exp(-self.log_sigma_pairwise) * pairwise
            + 0.5 * self.log_sigma_pairwise
        )

    @torch.no_grad()
    def effective_weights(self) -> dict[str, float]:
        if not self.learnable:
            return {"daily_z": 1.0, "rank_corr": 1.0, "pairwise": 1.0}
        return {
            "daily_z": float(
                (0.5 * torch.exp(-self.log_sigma_daily_z)).detach().cpu()
            ),
            "rank_corr": float(
                (0.5 * torch.exp(-self.log_sigma_rank)).detach().cpu()
            ),
            "pairwise": float(
                (0.5 * torch.exp(-self.log_sigma_pairwise)).detach().cpu()
            ),
        }


def stratified_pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_pairs: int = 8192,
    temperature: float = 0.05,
    extreme_fraction: float = 0.25,
    local_fraction: float = 0.50,
    local_window_fraction: float = 0.10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mix extreme, nearby-rank, and random pairs without crossing dates."""
    n = pred.numel()
    if n < 2 or max_pairs <= 0:
        return pred.sum() * 0.0
    if extreme_fraction < 0.0 or local_fraction < 0.0:
        raise ValueError("Pair fractions must be non-negative")
    if extreme_fraction + local_fraction > 1.0:
        raise ValueError("Extreme and local pair fractions must sum to at most one")

    n_extreme = int(round(max_pairs * extreme_fraction))
    n_local = int(round(max_pairs * local_fraction))
    n_random = max_pairs - n_extreme - n_local
    order = torch.argsort(target)
    pair_i: list[torch.Tensor] = []
    pair_j: list[torch.Tensor] = []

    if n_extreme > 0:
        quartile = max(1, n // 4)
        bottom_pos = torch.randint(
            0, quartile, (n_extreme,), device=pred.device, generator=generator
        )
        top_pos = torch.randint(
            n - quartile,
            n,
            (n_extreme,),
            device=pred.device,
            generator=generator,
        )
        pair_i.append(order[top_pos])
        pair_j.append(order[bottom_pos])

    if n_local > 0:
        window = max(1, int(round(n * local_window_fraction)))
        offset = torch.randint(
            1, window + 1, (n_local,), device=pred.device, generator=generator
        )
        uniform = torch.rand(
            n_local, device=pred.device, generator=generator
        )
        left_pos = torch.floor(uniform * (n - offset).to(uniform.dtype)).to(
            torch.long
        )
        right_pos = left_pos + offset
        pair_i.append(order[right_pos])
        pair_j.append(order[left_pos])

    if n_random > 0:
        pair_i.append(
            torch.randint(0, n, (n_random,), device=pred.device, generator=generator)
        )
        pair_j.append(
            torch.randint(0, n, (n_random,), device=pred.device, generator=generator)
        )

    idx_i = torch.cat(pair_i)
    idx_j = torch.cat(pair_j)
    sign = torch.sign(target[idx_i] - target[idx_j])
    keep = sign != 0
    if not bool(keep.any()):
        return pred.sum() * 0.0
    margin = (
        sign[keep] * (pred[idx_i[keep]] - pred[idx_j[keep]])
        / max(temperature, 1e-6)
    )
    return F.softplus(-margin).mean()


def head_aware_pair_sampling(
    pred: torch.Tensor,
    target: torch.Tensor,
    head_target: torch.Tensor | None = None,
    max_pairs: int = 16384,
    top_fraction: float = 0.10,
    boundary_fraction: float = 0.25,
    intra_top_fraction: float = 0.125,
    false_positive_fraction: float = 0.125,
    local_fraction: float = 0.25,
    local_window_fraction: float = 0.10,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Sample head-aware pairs and return (idx_i, idx_j, counts).

    Replaces the easy top-quartile-vs-bottom-quartile pairs with pairs that
    directly train (a) adjacent decile boundaries (group-vs-group ordering),
    (b) the top-decile membership boundary (true top vs currently predicted
    false positives, selected with detached predictions), and (c) ordering
    inside the top decile. Local pairs no longer clamp at the best rank.
    ``head_target`` optionally defines the head bands and pair sign on a
    different label (e.g. raw return) than the optimization target.
    """
    n = pred.numel()
    if n < 2 or max_pairs <= 0:
        return (
            torch.empty(0, dtype=torch.long, device=pred.device),
            torch.empty(0, dtype=torch.long, device=pred.device),
            {},
        )
    fractions = {
        "boundary": float(boundary_fraction),
        "intra_top": float(intra_top_fraction),
        "false_positive": float(false_positive_fraction),
        "local": float(local_fraction),
    }
    if any(value < 0.0 for value in fractions.values()):
        raise ValueError("Pair fractions must be non-negative")
    if sum(fractions.values()) > 1.0:
        raise ValueError("Pair fractions must sum to at most one")
    counts = {
        name: int(round(max_pairs * value)) for name, value in fractions.items()
    }
    counts["random"] = max_pairs - sum(counts.values())

    rank_source = head_target if head_target is not None else target
    order = torch.argsort(rank_source)
    top_count = max(1, min(int(round(n * top_fraction)), n - 1))
    top_start = n - top_count

    pair_i: list[torch.Tensor] = []
    pair_j: list[torch.Tensor] = []

    def randint(low, high, count: int) -> torch.Tensor:
        return torch.randint(
            low, high, (count,), device=pred.device, generator=generator
        )

    def rand_span(low: int, span: torch.Tensor, count: int) -> torch.Tensor:
        """Sample integers uniformly in [low, low + span) with per-sample span."""
        uniform = torch.rand(
            count, device=pred.device, generator=generator
        )
        return low + torch.floor(uniform * span.to(uniform.dtype)).to(
            torch.long
        )

    # (a) Adjacent decile boundaries: upper decile vs the decile directly below.
    if counts["boundary"] > 0:
        decile = max(1, n // 10)
        n_boundaries = min(9, (n // decile) - 1)
        per_boundary = counts["boundary"] // n_boundaries
        remainder = counts["boundary"] % n_boundaries
        for b in range(n_boundaries):
            cnt = per_boundary + (1 if b < remainder else 0)
            if cnt <= 0:
                continue
            upper_start = (b + 1) * decile
            lower_start = b * decile
            upper_end = min(upper_start + decile, n)
            lower_end = min(lower_start + decile, upper_start)
            if upper_end - upper_start < 1 or lower_end - lower_start < 1:
                continue
            pair_i.append(order[randint(upper_start, upper_end, cnt)])
            pair_j.append(order[randint(lower_start, lower_end, cnt)])

    # (b) Intra-top ordering: both members inside the top decile, i better than j.
    if counts["intra_top"] > 0:
        hi = randint(top_start + 1, n, counts["intra_top"])
        lo = rand_span(top_start, hi - top_start, counts["intra_top"])
        pair_i.append(order[hi])
        pair_j.append(order[lo])

    # (c) Membership boundary: true top decile vs predicted-top false positives.
    if counts["false_positive"] > 0:
        pred_order = torch.argsort(pred.detach(), descending=True)
        pred_top = pred_order[:top_count]
        true_top = order[top_start:]
        is_true_top = torch.zeros(n, dtype=torch.bool, device=pred.device)
        is_true_top[true_top] = True
        fp = pred_top[~is_true_top[pred_top]]
        if fp.numel() > 0:
            i_idx = randint(0, top_count, counts["false_positive"])
            j_idx = randint(0, fp.numel(), counts["false_positive"])
            pair_i.append(order[top_start:][i_idx])
            pair_j.append(fp[j_idx])

    # (d) Local neighbors without clamping: g ~ U{1..w}, l ~ U{0..n-g-1}, r = l+g.
    if counts["local"] > 0:
        window = max(1, int(round(n * local_window_fraction)))
        g = randint(1, window + 1, counts["local"])
        l = rand_span(0, n - g, counts["local"])
        pair_i.append(order[l + g])
        pair_j.append(order[l])

    # (e) Uniform random pairs for global coverage.
    if counts["random"] > 0:
        pair_i.append(randint(0, n, counts["random"]))
        pair_j.append(randint(0, n, counts["random"]))

    idx_i = torch.cat(pair_i)
    idx_j = torch.cat(pair_j)
    return idx_i, idx_j, counts


def head_aware_pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    head_target: torch.Tensor | None = None,
    max_pairs: int = 16384,
    temperature: float = 0.05,
    top_fraction: float = 0.10,
    boundary_fraction: float = 0.25,
    intra_top_fraction: float = 0.125,
    false_positive_fraction: float = 0.125,
    local_fraction: float = 0.25,
    local_window_fraction: float = 0.10,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Head-aware pairwise RankNet loss (see ``head_aware_pair_sampling``)."""
    idx_i, idx_j, _counts = head_aware_pair_sampling(
        pred,
        target,
        head_target=head_target,
        max_pairs=max_pairs,
        top_fraction=top_fraction,
        boundary_fraction=boundary_fraction,
        intra_top_fraction=intra_top_fraction,
        false_positive_fraction=false_positive_fraction,
        local_fraction=local_fraction,
        local_window_fraction=local_window_fraction,
        generator=generator,
    )
    if idx_i.numel() == 0:
        return pred.sum() * 0.0
    rank_source = head_target if head_target is not None else target
    sign = torch.sign(rank_source[idx_i] - rank_source[idx_j])
    keep = sign != 0
    if not bool(keep.any()):
        return pred.sum() * 0.0
    margin = (
        sign[keep] * (pred[idx_i[keep]] - pred[idx_j[keep]])
        / max(temperature, 1e-6)
    )
    return F.softplus(-margin).mean()


def quantile_weighted_pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    head_target: torch.Tensor | None = None,
    max_pairs: int = 8192,
    temperature: float = 0.05,
    tail_fraction: float = 0.10,
    top_pair_fraction: float = 0.40,
    bottom_pair_fraction: float = 0.20,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 3.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Weight same-date pairs by the empirical quantiles of their labels."""
    n = pred.numel()
    if n < 2 or max_pairs <= 0:
        return pred.sum() * 0.0
    if not 0.0 < tail_fraction < 0.5:
        raise ValueError("tail_fraction must be in (0, 0.5)")
    if top_pair_fraction < 0.0 or bottom_pair_fraction < 0.0:
        raise ValueError("Tail pair fractions must be non-negative")
    if top_pair_fraction + bottom_pair_fraction > 1.0:
        raise ValueError("Top and bottom pair fractions must sum to at most one")
    if alpha < 0.0 or beta < 0.0 or gamma <= 0.0:
        raise ValueError("alpha and beta must be non-negative and gamma positive")

    n_top = int(round(max_pairs * top_pair_fraction))
    n_bottom = int(round(max_pairs * bottom_pair_fraction))
    n_global = max_pairs - n_top - n_bottom
    rank_source = head_target if head_target is not None else target
    order = torch.argsort(rank_source)
    tail_count = min(max(1, math.ceil(n * tail_fraction)), n - 1)
    bottom = order[:tail_count]
    top = order[-tail_count:]
    non_bottom = order[tail_count:]
    non_top = order[:-tail_count]
    pair_i: list[torch.Tensor] = []
    pair_j: list[torch.Tensor] = []

    def sample(pool: torch.Tensor, count: int) -> torch.Tensor:
        positions = torch.randint(
            0,
            pool.numel(),
            (count,),
            device=pred.device,
            generator=generator,
        )
        return pool[positions]

    if n_top > 0:
        pair_i.append(sample(top, n_top))
        pair_j.append(sample(non_top, n_top))
    if n_bottom > 0:
        pair_i.append(sample(non_bottom, n_bottom))
        pair_j.append(sample(bottom, n_bottom))
    if n_global > 0:
        pair_i.append(
            torch.randint(
                0,
                n,
                (n_global,),
                device=pred.device,
                generator=generator,
            )
        )
        pair_j.append(
            torch.randint(
                0,
                n,
                (n_global,),
                device=pred.device,
                generator=generator,
            )
        )

    idx_i = torch.cat(pair_i)
    idx_j = torch.cat(pair_j)
    swap = rank_source[idx_i] < rank_source[idx_j]
    higher = torch.where(swap, idx_j, idx_i)
    lower = torch.where(swap, idx_i, idx_j)
    keep = rank_source[higher] > rank_source[lower]
    if not bool(keep.any()):
        return pred.sum() * 0.0
    higher = higher[keep]
    lower = lower[keep]

    ranks = torch.empty(n, dtype=pred.dtype, device=pred.device)
    ranks[order] = torch.arange(n, dtype=pred.dtype, device=pred.device)
    quantiles = (ranks + 0.5) / n
    pair_weights = (
        1.0
        + alpha * quantiles[higher].pow(gamma)
        + beta * (1.0 - quantiles[lower]).pow(gamma)
    )
    margin = (
        pred[higher] - pred[lower]
    ) / max(temperature, 1e-6)
    losses = F.softplus(-margin)
    return (pair_weights * losses).sum() / pair_weights.sum().clamp_min(1e-12)


class RankingLoss(nn.Module):
    """Non-negative date-wise ranking objective with constrained weights."""

    def __init__(
        self,
        learnable: bool = True,
        initial_weights: tuple[float, float, float] = (0.15, 0.55, 0.30),
        min_weight: float = 0.05,
        pairwise_max_pairs: int = 8192,
        pairwise_temperature: float = 0.05,
        pairwise_extreme_fraction: float = 0.25,
        pairwise_local_fraction: float = 0.50,
        pairwise_local_window_fraction: float = 0.10,
        pairwise_mode: str = "stratified",
        pairwise_top_fraction: float = 0.10,
        pairwise_boundary_fraction: float = 0.0,
        pairwise_intra_top_fraction: float = 0.0,
        pairwise_false_positive_fraction: float = 0.0,
        pairwise_head_on_raw: bool = False,
        rank_corr_mode: str = "full",
        rank_corr_groups: int = 10,
        rank_corr_head_on_raw: bool = False,
        quantile_pairwise_enabled: bool = False,
        quantile_tail_fraction: float = 0.10,
        quantile_top_pair_fraction: float = 0.40,
        quantile_bottom_pair_fraction: float = 0.20,
        quantile_alpha: float = 1.0,
        quantile_beta: float = 0.5,
        quantile_gamma: float = 3.0,
    ):
        super().__init__()
        self.learnable = bool(learnable)
        self.min_weight = float(min_weight)
        self.pairwise_max_pairs = int(pairwise_max_pairs)
        self.pairwise_temperature = float(pairwise_temperature)
        self.pairwise_extreme_fraction = float(pairwise_extreme_fraction)
        self.pairwise_local_fraction = float(pairwise_local_fraction)
        self.pairwise_local_window_fraction = float(pairwise_local_window_fraction)
        if pairwise_mode not in ("stratified", "head_aware"):
            raise ValueError("pairwise_mode must be 'stratified' or 'head_aware'")
        self.pairwise_mode = pairwise_mode
        self.pairwise_top_fraction = float(pairwise_top_fraction)
        self.pairwise_boundary_fraction = float(pairwise_boundary_fraction)
        self.pairwise_intra_top_fraction = float(pairwise_intra_top_fraction)
        self.pairwise_false_positive_fraction = float(
            pairwise_false_positive_fraction
        )
        self.pairwise_head_on_raw = bool(pairwise_head_on_raw)
        if rank_corr_mode not in ("full", "group"):
            raise ValueError("rank_corr_mode must be 'full' or 'group'")
        self.rank_corr_mode = rank_corr_mode
        self.rank_corr_groups = max(2, int(rank_corr_groups))
        self.rank_corr_head_on_raw = bool(rank_corr_head_on_raw)
        self.quantile_pairwise_enabled = bool(quantile_pairwise_enabled)
        self.quantile_tail_fraction = float(quantile_tail_fraction)
        self.quantile_top_pair_fraction = float(quantile_top_pair_fraction)
        self.quantile_bottom_pair_fraction = float(quantile_bottom_pair_fraction)
        self.quantile_alpha = float(quantile_alpha)
        self.quantile_beta = float(quantile_beta)
        self.quantile_gamma = float(quantile_gamma)

        initial = torch.tensor(initial_weights, dtype=torch.float32)
        if initial.numel() != 3:
            raise ValueError("initial_weights must contain three values")
        if self.learnable:
            if bool((initial <= 0).any()):
                raise ValueError(
                    "initial_weights must be positive for learnable weights"
                )
        elif bool((initial < 0).any()):
            raise ValueError(
                "initial_weights must be non-negative for fixed weights"
            )
        initial = initial / initial.sum()
        if not 0.0 <= self.min_weight < 1.0 / 3.0:
            raise ValueError("min_weight must be in [0, 1/3)")
        if self.learnable:
            available = 1.0 - 3.0 * self.min_weight
            adjusted = (initial - self.min_weight) / available
            if bool((adjusted <= 0).any()):
                raise ValueError("Every initial weight must exceed min_weight")
            self.weight_logits = nn.Parameter(adjusted.log())
        else:
            self.register_buffer("fixed_weights", initial, persistent=True)

    def weights(self) -> torch.Tensor:
        if not self.learnable:
            return self.fixed_weights
        available = 1.0 - 3.0 * self.min_weight
        return self.min_weight + available * torch.softmax(self.weight_logits, dim=0)

    def component_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        head_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pred.numel() < 2:
            zero = pred.sum() * 0.0
            return {"daily_z": zero, "rank_corr": zero, "pairwise": zero}
        daily_z = F.mse_loss(zscore(pred), zscore(target))
        if self.rank_corr_mode == "group":
            group_source = (
                head_target
                if (self.rank_corr_head_on_raw and head_target is not None)
                else target
            )
            rank_corr_value = group_rank_corr_surrogate_loss(
                pred, group_source, self.rank_corr_groups
            )
        else:
            rank_corr_value = rank_corr_surrogate_loss(pred, target)
        rank_corr = (1.0 + rank_corr_value).clamp_min(0.0)
        generator = None
        if not self.training:
            generator = torch.Generator(device=pred.device)
            generator.manual_seed(0)
        if self.quantile_pairwise_enabled:
            pairwise = quantile_weighted_pairwise_rank_loss(
                pred,
                target,
                head_target=(
                    head_target if self.pairwise_head_on_raw else None
                ),
                max_pairs=self.pairwise_max_pairs,
                temperature=self.pairwise_temperature,
                tail_fraction=self.quantile_tail_fraction,
                top_pair_fraction=self.quantile_top_pair_fraction,
                bottom_pair_fraction=self.quantile_bottom_pair_fraction,
                alpha=self.quantile_alpha,
                beta=self.quantile_beta,
                gamma=self.quantile_gamma,
                generator=generator,
            )
        elif self.pairwise_mode == "head_aware":
            pairwise = head_aware_pairwise_rank_loss(
                pred,
                target,
                head_target=(
                    head_target if self.pairwise_head_on_raw else None
                ),
                max_pairs=self.pairwise_max_pairs,
                temperature=self.pairwise_temperature,
                top_fraction=self.pairwise_top_fraction,
                boundary_fraction=self.pairwise_boundary_fraction,
                intra_top_fraction=self.pairwise_intra_top_fraction,
                false_positive_fraction=self.pairwise_false_positive_fraction,
                local_fraction=self.pairwise_local_fraction,
                local_window_fraction=self.pairwise_local_window_fraction,
                generator=generator,
            )
        else:
            pairwise = stratified_pairwise_rank_loss(
                pred,
                target,
                max_pairs=self.pairwise_max_pairs,
                temperature=self.pairwise_temperature,
                extreme_fraction=self.pairwise_extreme_fraction,
                local_fraction=self.pairwise_local_fraction,
                local_window_fraction=self.pairwise_local_window_fraction,
                generator=generator,
            )
        return {"daily_z": daily_z, "rank_corr": rank_corr, "pairwise": pairwise}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        head_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        components = self.component_losses(pred, target, head_target)
        weights = self.weights()
        return sum(
            weights[index] * components[name]
            for index, name in enumerate(("daily_z", "rank_corr", "pairwise"))
        )

    @torch.no_grad()
    def effective_weights(self) -> dict[str, float]:
        values = self.weights().detach().cpu()
        return {
            name: float(values[index])
            for index, name in enumerate(("daily_z", "rank_corr", "pairwise"))
        }
