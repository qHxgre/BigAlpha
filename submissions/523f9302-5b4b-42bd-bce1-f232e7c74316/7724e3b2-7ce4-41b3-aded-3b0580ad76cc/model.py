# ===================================================================
# AlphaModel (dynamic-top2-v1) -- model_dynamic1 native-grid Top-2 routing
# with fusion_mass_bias balance and load EMA control. Extracted from
# the inference notebook so training and inference share one model.
# ===================================================================
"""BigQuant inference for the epoch-14 native-token model_dynamic1 checkpoint.

Upload this notebook together with ``model.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURE = "dynamic-top2-v1"
BALANCE_MODES = {
    "fusion_mass_bias",
    "contribution_bias",
    "count_bias",
    "auxiliary_loss",
    "none",
}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class TradingMinuteRoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        angles = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        return x * angles.cos().to(x.dtype) + _rotate_half(x) * angles.sin().to(x.dtype)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, rope_base: float):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = int(nhead)
        self.head_dim = int(d_model // nhead)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = float(dropout)
        self.rope = TradingMinuteRoPE(self.head_dim, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, n_tokens, width = x.shape
        qkv = self.qkv(x).reshape(batch, n_tokens, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.rope(q.transpose(1, 2), positions)
        k = self.rope(k.transpose(1, 2), positions)
        v = v.transpose(1, 2)
        attention_mask = None
        if valid_tokens is not None:
            if valid_tokens.shape != (batch, n_tokens):
                raise ValueError("valid_tokens must have shape [B, tokens]")
            attention_mask = valid_tokens[:, None, None, :]
        h = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        h = self.out(h.transpose(1, 2).reshape(batch, n_tokens, width))
        if valid_tokens is not None:
            h = h * valid_tokens.unsqueeze(-1).to(h.dtype)
        return h


class RoPEEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float, rope_base: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model, nhead, dropout, rope_base)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), positions, valid_tokens))
        x = x + self.dropout(self.ff(self.norm2(x)))
        if valid_tokens is not None:
            x = x * valid_tokens.unsqueeze(-1).to(x.dtype)
        return x


class ChannelIndependentPatchProjection(nn.Module):
    """Project each feature separately before parameter-matched aggregation."""

    def __init__(
        self,
        patch_len: int,
        n_features: int,
        d_model: int,
        joint_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_len = int(patch_len)
        self.n_features = int(n_features)
        self.channel_hidden = max(1, int(joint_hidden) // 2)
        input_width = 2 * self.patch_len
        self.input_norm = nn.LayerNorm(input_width)
        self.input_weight = nn.Parameter(
            torch.empty(self.n_features, input_width, self.channel_hidden)
        )
        self.input_bias = nn.Parameter(
            torch.zeros(self.n_features, self.channel_hidden)
        )
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.channel_hidden, d_model),
            nn.LayerNorm(d_model),
        )
        for weight in self.input_weight:
            nn.init.xavier_uniform_(weight)

    def forward(
        self,
        patches: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        if patches.shape[:-2] != observed.shape[:-1]:
            raise ValueError("Patch and observation batch axes must agree")
        if patches.shape[-2:] != (self.patch_len, self.n_features):
            raise ValueError("Unexpected channel-independent patch shape")
        if observed.shape[-1] != self.patch_len:
            raise ValueError("Unexpected observation-mask patch length")
        channel_values = patches.transpose(-1, -2)
        channel_observed = observed.unsqueeze(-2).expand_as(channel_values)
        channel_input = self.input_norm(
            torch.cat((channel_values, channel_observed.to(patches.dtype)), dim=-1)
        )
        channel_hidden = torch.einsum(
            "...fw,fwh->...fh",
            channel_input,
            self.input_weight,
        ) + self.input_bias
        return self.output(channel_hidden).mean(dim=-2)


class ResolutionPatchExpert(nn.Module):
    """Jointly project every non-overlapping [patch_len, F] patch to one token."""

    def __init__(
        self,
        patch_len: int,
        n_features: int,
        d_model: int,
        hidden: int,
        dropout: float,
        joint_multivariate_embedding: bool = True,
    ):
        super().__init__()
        self.patch_len = int(patch_len)
        self.n_features = int(n_features)
        self.joint_multivariate_embedding = bool(joint_multivariate_embedding)
        if self.joint_multivariate_embedding:
            input_width = self.patch_len * (self.n_features + 1)
            self.net = nn.Sequential(
                nn.LayerNorm(input_width),
                nn.Linear(input_width, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.independent_projection = ChannelIndependentPatchProjection(
                self.patch_len,
                self.n_features,
                d_model,
                hidden,
                dropout,
            )

    def forward(
        self,
        sessions: torch.Tensor,
        observed_minutes: torch.Tensor,
    ) -> torch.Tensor:
        if sessions.ndim != 3:
            raise ValueError("Resolution experts expect [N, session, features]")
        if sessions.shape[:2] != observed_minutes.shape:
            raise ValueError("observed_minutes must match the expert time axes")
        if sessions.shape[-1] != self.n_features:
            raise ValueError("Unexpected feature dimension for resolution expert")
        if sessions.shape[1] % self.patch_len:
            raise ValueError("A resolution patch may not cross a session boundary")
        n_patches = sessions.shape[1] // self.patch_len
        if not self.joint_multivariate_embedding:
            patch_values = sessions.reshape(
                sessions.shape[0],
                n_patches,
                self.patch_len,
                self.n_features,
            )
            patch_observed = observed_minutes.reshape(
                sessions.shape[0],
                n_patches,
                self.patch_len,
            )
            return self.independent_projection(patch_values, patch_observed)
        values = torch.cat(
            (sessions, observed_minutes.unsqueeze(-1).to(sessions.dtype)),
            dim=-1,
        )
        patches = values.reshape(
            sessions.shape[0],
            n_patches,
            self.patch_len * (self.n_features + 1),
        )
        return self.net(patches)


class RegionMarketRouter(nn.Module):
    """Infer one joint multivariate market state for each routing region."""

    def __init__(
        self,
        region_len: int,
        router_atom_len: int,
        n_features: int,
        d_model: int,
        nhead: int,
        dim_ff: int,
        n_experts: int,
        dropout: float,
        rope_base: float,
        joint_multivariate_embedding: bool = True,
    ):
        super().__init__()
        self.region_len = int(region_len)
        self.router_atom_len = int(router_atom_len)
        self.n_features = int(n_features)
        self.joint_multivariate_embedding = bool(joint_multivariate_embedding)
        self.n_atoms = self.region_len // self.router_atom_len
        if self.joint_multivariate_embedding:
            input_width = self.router_atom_len * (self.n_features + 1)
            self.base_projection = nn.Sequential(
                nn.LayerNorm(input_width),
                nn.Linear(input_width, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
        else:
            self.independent_base_projection = ChannelIndependentPatchProjection(
                self.router_atom_len,
                self.n_features,
                d_model,
                d_model,
                dropout,
            )
        self.position_embedding = nn.Embedding(self.n_atoms, d_model)
        self.state_block = RoPEEncoderBlock(
            d_model,
            nhead,
            dim_ff,
            dropout,
            rope_base,
        )
        self.state_norm = nn.LayerNorm(d_model)
        self.logit_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_experts),
        )
        nn.init.zeros_(self.logit_head[-1].weight)
        nn.init.zeros_(self.logit_head[-1].bias)

    def forward(
        self,
        sessions: torch.Tensor,
        observed_minutes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sessions.ndim != 4:
            raise ValueError("Router expects [B, regions, minutes, features]")
        batch, n_regions, region_len, n_features = sessions.shape
        if (region_len, n_features) != (self.region_len, self.n_features):
            raise ValueError("Router received an incompatible region tensor")
        if observed_minutes.shape != sessions.shape[:-1]:
            raise ValueError("observed_minutes must match router time axes")
        if self.joint_multivariate_embedding:
            values = torch.cat(
                (sessions, observed_minutes.unsqueeze(-1).to(sessions.dtype)),
                dim=-1,
            )
            atoms = values.reshape(
                batch * n_regions,
                self.n_atoms,
                self.router_atom_len * (self.n_features + 1),
            )
            h = self.base_projection(atoms)
        else:
            atoms = sessions.reshape(
                batch * n_regions,
                self.n_atoms,
                self.router_atom_len,
                self.n_features,
            )
            atom_observed = observed_minutes.reshape(
                batch * n_regions,
                self.n_atoms,
                self.router_atom_len,
            )
            h = self.independent_base_projection(atoms, atom_observed)
        positions = torch.arange(self.n_atoms, device=sessions.device)
        h = h + self.position_embedding(positions).unsqueeze(0)
        minute_positions = (
            positions.mul(self.router_atom_len)
            .unsqueeze(0)
            .expand(batch * n_regions, -1)
        )
        h = self.state_block(h, minute_positions)
        mean_state = h.mean(dim=1)
        max_state = h.amax(dim=1)
        state = self.state_norm(mean_state + max_state)
        logits = self.logit_head(torch.cat((mean_state, max_state), dim=-1))
        return (
            state.reshape(batch, n_regions, -1),
            logits.reshape(batch, n_regions, -1),
        )


@dataclass
class RoutingState:
    clean_logits: torch.Tensor
    clean_probabilities: torch.Tensor
    calibrated_logits: torch.Tensor
    routing_probabilities: torch.Tensor
    selected: torch.Tensor
    fusion_weights: torch.Tensor
    local_counts: torch.Tensor
    local_gate_mass: torch.Tensor
    local_regions: torch.Tensor
    patch_starts: torch.Tensor
    patch_lengths: torch.Tensor
    patch_counts: torch.Tensor
    valid_tokens: torch.Tensor
    token_counts: torch.Tensor
    native_attention_pairs: torch.Tensor
    executed_attention_pairs: torch.Tensor
    fixed_grid_attention_pairs: torch.Tensor


class AlphaModel(nn.Module):
    """Joint multivariate Top-2 patch routing with fixed-prior balancing."""

    def __init__(self, config: dict, n_features: int):
        super().__init__()
        model_cfg = config["model"]
        chrono_cfg = config["routing"]
        freq_cfg = config["frequency"]["bar1m"]

        self.n_features = int(n_features)
        self.d_model = int(model_cfg["d_model"])
        self.nhead = int(model_cfg["nhead"])
        self.seq_len = int(freq_cfg["seq_len"])
        self.lookback_days = int(freq_cfg["lookback_days"])
        self.bars_per_day = int(freq_cfg["bars_per_day"])
        self.session_lengths = tuple(
            int(value) for value in freq_cfg.get("session_lengths", [self.bars_per_day])
        )
        self.router_atom_len = int(
            chrono_cfg.get(
                "router_atom_len",
                chrono_cfg.get("alignment_grid_len", chrono_cfg.get("base_patch_len", 5)),
            )
        )
        self.alignment_grid_len = self.router_atom_len
        self.base_patch_len = self.router_atom_len
        self.routing_region_len = int(chrono_cfg.get("routing_region_len", 60))
        self.patch_lens = tuple(
            int(value) for value in chrono_cfg.get("patch_lens", [5, 15, 30, 60])
        )
        self.n_experts = int(chrono_cfg.get("n_routed_experts", len(self.patch_lens)))
        self.top_k = int(chrono_cfg.get("top_k", 2))
        self.base_tokens_per_day = self.bars_per_day // self.router_atom_len
        self.n_sessions = self.lookback_days * len(self.session_lengths)
        self.session_len = self.session_lengths[0]
        self.n_regions = self.lookback_days * sum(
            length // self.routing_region_len for length in self.session_lengths
        )
        self.max_tokens_per_region = self.routing_region_len // min(self.patch_lens)
        self.max_dynamic_tokens = self.n_regions * self.max_tokens_per_region
        self.n_dynamic_tokens = self.max_dynamic_tokens
        self.attention_bucket_size = int(chrono_cfg.get("attention_bucket_size", 64))
        self.mask_span_lens = tuple(
            int(value) for value in chrono_cfg.get("mask_span_lens", self.patch_lens)
        )
        self.routing_temperature = float(chrono_cfg.get("routing_temperature", 1.0))
        joint_embedding = chrono_cfg.get("joint_multivariate_embedding", True)
        if not isinstance(joint_embedding, bool):
            raise ValueError("joint_multivariate_embedding must be boolean")
        self.joint_multivariate_embedding = joint_embedding
        self.balance_mode = str(
            chrono_cfg.get("balance_mode", "fusion_mass_bias")
        ).lower()
        self.aux_balance_weight = float(chrono_cfg.get("aux_balance_weight", 0.01))
        self.bias_update_speed = float(chrono_cfg.get("bias_update_speed", 0.01))
        self.load_ema_decay = float(chrono_cfg.get("load_ema_decay", 0.99))
        self.bias_clip = float(chrono_cfg.get("bias_clip", 0.50))

        if self.seq_len != self.lookback_days * self.bars_per_day:
            raise ValueError("seq_len must equal lookback_days * bars_per_day")
        if sum(self.session_lengths) != self.bars_per_day:
            raise ValueError("session_lengths must sum to bars_per_day")
        if len(set(self.session_lengths)) != 1:
            raise ValueError("Resolution routing currently requires equal session lengths")
        if self.n_experts != len(self.patch_lens):
            raise ValueError("Each routed expert must correspond to exactly one patch length")
        if self.top_k <= 0 or self.top_k > self.n_experts:
            raise ValueError("top_k must lie in [1, n_routed_experts]")
        if self.routing_temperature <= 0.0:
            raise ValueError("routing_temperature must be positive")
        if self.balance_mode not in BALANCE_MODES:
            raise ValueError(
                f"balance_mode must be one of {sorted(BALANCE_MODES)}"
            )
        if self.aux_balance_weight < 0.0:
            raise ValueError("aux_balance_weight must be non-negative")
        if self.bias_update_speed < 0.0:
            raise ValueError("bias_update_speed must be non-negative")
        if not 0.0 <= self.load_ema_decay < 1.0:
            raise ValueError("load_ema_decay must lie in [0, 1)")
        if self.bias_clip <= 0.0:
            raise ValueError("bias_clip must be positive")
        if self.router_atom_len <= 0:
            raise ValueError("router_atom_len must be positive")
        if self.routing_region_len <= 0:
            raise ValueError("routing_region_len must be positive")
        if self.attention_bucket_size < 0:
            raise ValueError("attention_bucket_size must be non-negative")
        if tuple(sorted(set(self.patch_lens))) != self.patch_lens:
            raise ValueError("patch_lens must be unique and strictly increasing")
        if self.patch_lens[-1] != self.routing_region_len:
            raise ValueError("The largest patch length must equal routing_region_len")
        if any(length % self.router_atom_len for length in self.patch_lens):
            raise ValueError("Every patch length must align to router_atom_len")
        if any(session % self.router_atom_len for session in self.session_lengths):
            raise ValueError("Every trading session must align to router_atom_len")
        for session in self.session_lengths:
            if session % self.routing_region_len:
                raise ValueError(
                    "routing_region_len must divide every trading session exactly"
                )
        for finer_index, finer in enumerate(self.patch_lens):
            if any(coarser % finer for coarser in self.patch_lens[finer_index + 1 :]):
                raise ValueError(
                    "Every coarser candidate must be divisible by each finer candidate"
                )

        dropout = float(model_cfg.get("dropout", 0.1))
        dim_ff = int(model_cfg["dim_ff"])
        expert_hidden = int(chrono_cfg.get("expert_hidden", max(self.d_model, dim_ff // 2)))
        rope_base = float(chrono_cfg.get("rope_base", 10000.0))
        router_dim_ff = int(chrono_cfg.get("router_dim_ff", self.d_model * 2))

        self.router = RegionMarketRouter(
            self.routing_region_len,
            self.router_atom_len,
            self.n_features,
            self.d_model,
            self.nhead,
            router_dim_ff,
            self.n_experts,
            dropout,
            rope_base,
            self.joint_multivariate_embedding,
        )
        self.experts = nn.ModuleList(
            ResolutionPatchExpert(
                patch_len,
                self.n_features,
                self.d_model,
                expert_hidden,
                dropout,
                self.joint_multivariate_embedding,
            )
            for patch_len in self.patch_lens
        )
        self.duration_embedding = nn.Embedding(self.n_experts, self.d_model)
        self.market_state_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )

        self.embedding_dropout = nn.Dropout(dropout)
        self.backbone = nn.ModuleList(
            RoPEEncoderBlock(self.d_model, self.nhead, dim_ff, dropout, rope_base)
            for _ in range(int(model_cfg["nlayers"]))
        )
        self.final_norm = nn.LayerNorm(self.d_model)

        self.pool_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.temporal_pool = nn.MultiheadAttention(
            self.d_model,
            self.nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ranking_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )
        self.reconstruction_heads = nn.ModuleDict(
            {
                str(patch_len): nn.Linear(
                    self.d_model,
                    patch_len * self.n_features,
                )
                for patch_len in self.patch_lens
            }
        )
        self.raw_mask_token = nn.Parameter(torch.zeros(1, 1, self.n_features))

        layout = self._build_region_layout()
        for name, value in layout.items():
            self.register_buffer(name, value, persistent=False)

        expert_prior = torch.tensor(
            chrono_cfg.get("expert_prior", [1.0 / self.n_experts] * self.n_experts),
            dtype=torch.float32,
        )
        if expert_prior.shape != (self.n_experts,):
            raise ValueError("expert_prior must contain one value per expert")
        if not torch.isfinite(expert_prior).all() or bool((expert_prior < 0.0).any()):
            raise ValueError("expert_prior must be finite and non-negative")
        if float(expert_prior.sum()) <= 0.0:
            raise ValueError("expert_prior must have positive mass")
        expert_prior = expert_prior / expert_prior.sum()
        target_load = torch.tensor(
            chrono_cfg.get("target_load", expert_prior.tolist()),
            dtype=torch.float32,
        )
        if target_load.shape != (self.n_experts,):
            raise ValueError("target_load must contain one value per expert")
        if not torch.isfinite(target_load).all() or bool((target_load < 0.0).any()):
            raise ValueError("target_load must be finite and non-negative")
        if float(target_load.sum()) <= 0.0:
            raise ValueError("target_load must have positive mass")
        target_load = target_load / target_load.sum()
        self.register_buffer(
            "expert_bias", torch.zeros(self.n_experts), persistent=True
        )
        self.register_buffer(
            "expert_prior", expert_prior, persistent=True
        )
        self.register_buffer(
            "target_load", target_load, persistent=True
        )
        self.register_buffer(
            "load_ema", target_load.clone(), persistent=True
        )
        self.register_buffer(
            "bias_updates",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        # Break exact Top-k ties reproducibly while leaving initial affinities near uniform.
        with torch.no_grad():
            tie_break = 1.0e-3 * expert_prior
            tie_break = tie_break + 1.0e-7 * torch.arange(self.n_experts)
            self.router.logit_head[-1].bias.copy_(tie_break)

    def _session_specs(self) -> list[tuple[int, int, int, int]]:
        specs = []
        for day in range(self.lookback_days):
            offset = day * self.bars_per_day
            for session_id, session_len in enumerate(self.session_lengths):
                specs.append((offset, session_len, day, session_id))
                offset += session_len
        return specs

    def _region_specs(self) -> list[tuple[int, int, int, int, int]]:
        specs = []
        for session_start, session_len, day, session_id in self._session_specs():
            for region_id, local_start in enumerate(
                range(0, session_len, self.routing_region_len)
            ):
                specs.append(
                    (
                        session_start + local_start,
                        self.routing_region_len,
                        day,
                        session_id,
                        region_id,
                    )
                )
        return specs

    def _build_region_layout(self) -> dict[str, torch.Tensor]:
        specs = self._region_specs()
        return {
            "region_starts_template": torch.tensor(
                [start for start, _length, _day, _session, _region in specs],
                dtype=torch.long,
            ),
        }

    def base_patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, F], got {tuple(x.shape)}")
        if x.shape[1:] != (self.seq_len, self.n_features):
            raise ValueError(
                f"Expected [B, {self.seq_len}, {self.n_features}], got {tuple(x.shape)}"
            )
        days = x.reshape(
            x.shape[0],
            self.lookback_days,
            self.bars_per_day,
            self.n_features,
        )
        return days.reshape(
            x.shape[0],
            self.lookback_days,
            self.base_tokens_per_day,
            self.alignment_grid_len,
            self.n_features,
        )

    def sessionize(self, x: torch.Tensor) -> torch.Tensor:
        self.base_patchify(x)
        return torch.stack(
            [x[:, start : start + length] for start, length, _day, _sid in self._session_specs()],
            dim=1,
        )

    def regionize(self, x: torch.Tensor) -> torch.Tensor:
        self.base_patchify(x)
        return torch.stack(
            [
                x[:, start : start + length]
                for start, length, _day, _session, _region in self._region_specs()
            ],
            dim=1,
        )

    def resolution_patchify(self, x: torch.Tensor, patch_len: int) -> torch.Tensor:
        if patch_len not in self.patch_lens:
            raise ValueError(f"Unsupported resolution {patch_len}")
        regions = self.regionize(x)
        return regions.reshape(
            x.shape[0],
            self.n_regions,
            self.routing_region_len // patch_len,
            patch_len,
            self.n_features,
        )

    def _route(
        self,
        clean_logits: torch.Tensor,
        dense_routing: bool = False,
        forced_expert: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Bounded clean affinities keep scale contributions numerically stable.
        clean_probabilities = torch.sigmoid(
            clean_logits / self.routing_temperature
        )
        # The balance bias changes expert selection only; task fusion remains clean.
        calibrated_logits = clean_probabilities + self.expert_bias.to(
            clean_logits.dtype
        )
        routing_probabilities = torch.softmax(
            calibrated_logits / self.routing_temperature,
            dim=-1,
        )
        if forced_expert is not None:
            if not 0 <= int(forced_expert) < self.n_experts:
                raise ValueError("forced_expert must index a routed expert")
            selected = torch.zeros_like(routing_probabilities, dtype=torch.bool)
            selected[..., int(forced_expert)] = True
            fusion_weights = selected.to(routing_probabilities.dtype)
        elif dense_routing:
            selected = torch.ones_like(routing_probabilities, dtype=torch.bool)
            fusion_weights = torch.full_like(
                routing_probabilities,
                1.0 / self.n_experts,
            )
        else:
            selected_indices = torch.topk(
                calibrated_logits,
                k=self.top_k,
                dim=-1,
            ).indices
            selected = torch.zeros_like(routing_probabilities, dtype=torch.bool).scatter(
                dim=-1,
                index=selected_indices,
                value=True,
            )
            sparse_gates = clean_probabilities * selected.to(
                clean_probabilities.dtype
            )
            fusion_weights = sparse_gates / sparse_gates.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(sparse_gates.dtype).eps)
        counts = selected.sum(dim=tuple(range(selected.ndim - 1))).to(clean_logits.dtype)
        gate_mass = fusion_weights.sum(dim=tuple(range(fusion_weights.ndim - 1)))
        return (
            clean_probabilities,
            calibrated_logits,
            routing_probabilities,
            selected,
            fusion_weights,
            counts,
            gate_mass,
        )

    def _encode_and_fuse(
        self,
        regions: torch.Tensor,
        observed_regions: torch.Tensor,
        selected: torch.Tensor,
        fusion_weights: torch.Tensor,
        market_state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, n_regions, region_len, _ = regions.shape
        flat_regions = regions.reshape(-1, region_len, self.n_features)
        flat_observed = observed_regions.reshape(-1, region_len)
        flat_selected = selected.reshape(-1, self.n_experts)
        flat_weights = fusion_weights.reshape(-1, self.n_experts)
        patch_lens = torch.tensor(
            self.patch_lens,
            device=regions.device,
            dtype=torch.long,
        )
        inactive = torch.full_like(patch_lens, self.routing_region_len + 1)
        finest_lens = torch.where(
            flat_selected,
            patch_lens.unsqueeze(0),
            inactive.unsqueeze(0),
        ).amin(dim=-1)
        if bool((finest_lens > self.routing_region_len).any()):
            raise RuntimeError("Every routing region must activate at least one expert")
        region_token_counts = self.routing_region_len // finest_lens
        local_index = torch.arange(
            self.max_tokens_per_region,
            device=regions.device,
        )
        flat_valid = local_index.unsqueeze(0) < region_token_counts.unsqueeze(1)
        fused = regions.new_zeros(
            batch * n_regions,
            self.max_tokens_per_region,
            self.d_model,
        )
        for expert_index, (patch_len, expert) in enumerate(zip(self.patch_lens, self.experts)):
            active_indices = flat_selected[:, expert_index].nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                continue
            expert_tokens = expert(
                flat_regions.index_select(0, active_indices),
                flat_observed.index_select(0, active_indices),
            )
            active_finest = finest_lens.index_select(0, active_indices)
            ancestor_indices = torch.div(
                local_index.unsqueeze(0) * active_finest.unsqueeze(1),
                patch_len,
                rounding_mode="floor",
            ).clamp_max(expert_tokens.shape[1] - 1)
            aligned = expert_tokens.gather(
                1,
                ancestor_indices.unsqueeze(-1).expand(-1, -1, self.d_model),
            )
            aligned = aligned + self.duration_embedding.weight[expert_index].view(1, 1, -1)
            weights = flat_weights.index_select(0, active_indices)[:, expert_index]
            fused.index_add_(0, active_indices, aligned * weights.view(-1, 1, 1))
        fused = fused + self.market_state_projection(
            market_state.reshape(-1, self.d_model)
        ).unsqueeze(1)

        region_starts = self.region_starts_template.repeat(batch)
        flat_starts = region_starts.unsqueeze(1) + (
            local_index.unsqueeze(0) * finest_lens.unsqueeze(1)
        )
        flat_lengths = finest_lens.unsqueeze(1).expand_as(flat_starts)
        fused = fused.reshape(batch, n_regions * self.max_tokens_per_region, self.d_model)
        valid = flat_valid.reshape(batch, -1)
        starts = flat_starts.reshape(batch, -1)
        lengths = flat_lengths.reshape(batch, -1)

        slots = valid.shape[1]
        original_order = torch.arange(slots, device=regions.device).unsqueeze(0)
        compact_order = torch.argsort(
            torch.where(valid, original_order, original_order + slots),
            dim=1,
        )
        fused = fused.gather(
            1,
            compact_order.unsqueeze(-1).expand(-1, -1, self.d_model),
        )
        starts = starts.gather(1, compact_order)
        lengths = lengths.gather(1, compact_order)
        token_counts = valid.sum(dim=1)
        max_tokens = int(token_counts.max().item())
        fused = fused[:, :max_tokens]
        starts = starts[:, :max_tokens]
        lengths = lengths[:, :max_tokens]
        valid = (
            torch.arange(max_tokens, device=regions.device).unsqueeze(0)
            < token_counts.unsqueeze(1)
        )
        starts = torch.where(valid, starts, torch.zeros_like(starts))
        lengths = torch.where(valid, lengths, torch.zeros_like(lengths))
        return fused, valid, starts, lengths, token_counts

    def _encode_bucketed(
        self,
        tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, global_max_tokens, _ = tokens.shape
        bucket_size = self.attention_bucket_size or batch
        bucket_size = min(bucket_size, batch)
        sorted_indices = torch.argsort(token_counts)
        encoded_chunks = []
        executed_pairs = 0
        for bucket_start in range(0, batch, bucket_size):
            indices = sorted_indices[bucket_start : bucket_start + bucket_size]
            local_max_tokens = int(token_counts.index_select(0, indices).max().item())
            h = tokens.index_select(0, indices)[:, :local_max_tokens]
            mask = valid_tokens.index_select(0, indices)[:, :local_max_tokens]
            local_starts = starts.index_select(0, indices)[:, :local_max_tokens]
            local_lengths = lengths.index_select(0, indices)[:, :local_max_tokens]
            positions = (
                local_starts.to(h.dtype)
                + 0.5 * (local_lengths.to(h.dtype) - 1.0).clamp_min(0.0)
            ) / float(self.router_atom_len)
            h = self.embedding_dropout(h)
            for block in self.backbone:
                h = block(h, positions, mask)
            h = self.final_norm(h) * mask.unsqueeze(-1).to(h.dtype)
            if local_max_tokens < global_max_tokens:
                h = F.pad(h, (0, 0, 0, global_max_tokens - local_max_tokens))
            encoded_chunks.append(h)
            executed_pairs += int(indices.numel()) * local_max_tokens * local_max_tokens
        encoded_sorted = torch.cat(encoded_chunks, dim=0)
        inverse_order = torch.argsort(sorted_indices)
        encoded = encoded_sorted.index_select(0, inverse_order)
        native_pairs = token_counts.to(torch.long).square().sum()
        executed_pairs_tensor = torch.tensor(
            executed_pairs,
            device=tokens.device,
            dtype=torch.long,
        )
        fixed_pairs = torch.tensor(
            batch * self.max_dynamic_tokens * self.max_dynamic_tokens,
            device=tokens.device,
            dtype=torch.long,
        )
        return encoded, native_pairs, executed_pairs_tensor, fixed_pairs

    def routed_features(
        self,
        x: torch.Tensor,
        observed_minutes: torch.Tensor | None = None,
        dense_routing: bool = False,
        forced_expert: int | None = None,
    ) -> tuple[torch.Tensor, RoutingState]:
        regions = self.regionize(x)
        if observed_minutes is None:
            observed_minutes = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if observed_minutes.shape != x.shape[:2]:
            raise ValueError("observed_minutes must have shape [B, T]")
        observed_regions = torch.stack(
            [
                observed_minutes[:, start : start + length]
                for start, length, _day, _session, _region in self._region_specs()
            ],
            dim=1,
        )
        market_state, clean_logits = self.router(regions, observed_regions)
        (
            clean_probabilities,
            calibrated_logits,
            routing_probabilities,
            selected,
            weights,
            counts,
            gate_mass,
        ) = self._route(
            clean_logits,
            dense_routing=dense_routing,
            forced_expert=forced_expert,
        )
        tokens, valid_tokens, starts, lengths, token_counts = self._encode_and_fuse(
            regions,
            observed_regions,
            selected,
            weights,
            market_state,
        )
        h, native_pairs, executed_pairs, fixed_pairs = self._encode_bucketed(
            tokens,
            valid_tokens,
            starts,
            lengths,
            token_counts,
        )
        routing = RoutingState(
            clean_logits=clean_logits,
            clean_probabilities=clean_probabilities,
            calibrated_logits=calibrated_logits,
            routing_probabilities=routing_probabilities,
            selected=selected,
            fusion_weights=weights,
            local_counts=counts,
            local_gate_mass=gate_mass,
            local_regions=torch.tensor(
                weights.shape[0] * weights.shape[1],
                device=weights.device,
                dtype=weights.dtype,
            ),
            patch_starts=starts,
            patch_lengths=lengths,
            patch_counts=counts.clone(),
            valid_tokens=valid_tokens,
            token_counts=token_counts,
            native_attention_pairs=native_pairs,
            executed_attention_pairs=executed_pairs,
            fixed_grid_attention_pairs=fixed_pairs,
        )
        return h, routing

    def forward(
        self,
        x: torch.Tensor,
        return_routing: bool = False,
    ):
        h, routing = self.routed_features(x)
        query = self.pool_query.expand(h.shape[0], -1, -1)
        pooled, _ = self.temporal_pool(
            query,
            h,
            h,
            key_padding_mask=~routing.valid_tokens,
            need_weights=False,
        )
        score = self.ranking_head(pooled.squeeze(1)).squeeze(-1)
        if return_routing:
            return score, routing
        return score

    def sample_base_mask(
        self,
        batch_size: int,
        device: torch.device,
        mask_ratio: float,
    ) -> torch.Tensor:
        total_atoms = self.seq_len // self.alignment_grid_len
        mask = torch.zeros(batch_size, total_atoms, dtype=torch.bool, device=device)
        target = max(1, int(round(total_atoms * float(mask_ratio))))
        span_atoms = torch.tensor(
            [span // self.alignment_grid_len for span in self.mask_span_lens],
            dtype=torch.long,
            device=device,
        )
        session_specs = self._session_specs()
        session_starts = torch.tensor(
            [start // self.alignment_grid_len for start, _length, _day, _sid in session_specs],
            device=device,
        )
        session_atoms = torch.tensor(
            [length // self.alignment_grid_len for _start, length, _day, _sid in session_specs],
            device=device,
        )
        max_span = int(span_atoms.max().item())
        offsets = torch.arange(max_span, device=device).view(1, -1)
        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        while True:
            active = (counts < target).nonzero(as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                break
            span = span_atoms[
                torch.randint(span_atoms.numel(), (active.numel(),), device=device)
            ]
            session = torch.randint(len(session_specs), (active.numel(),), device=device)
            span = torch.minimum(span, session_atoms[session])
            max_start = session_atoms[session] - span + 1
            local_start = torch.floor(
                torch.rand(active.numel(), device=device) * max_start
            ).long()
            indices = session_starts[session].unsqueeze(1) + local_start.unsqueeze(1) + offsets
            valid = offsets < span.unsqueeze(1)
            mask[active.unsqueeze(1), indices.clamp_max(total_atoms - 1)] |= valid
            counts = mask.sum(dim=1)
        return mask

    def _reconstruct_minutes(
        self,
        h: torch.Tensor,
        routing: RoutingState,
    ) -> torch.Tensor:
        reconstruction = h.new_zeros(h.shape[0], self.seq_len, self.n_features)
        coverage = torch.zeros(
            h.shape[0],
            self.seq_len,
            dtype=torch.bool,
            device=h.device,
        )
        for patch_len in self.patch_lens:
            active = routing.valid_tokens & routing.patch_lengths.eq(patch_len)
            token_indices = active.nonzero(as_tuple=False)
            if token_indices.numel() == 0:
                continue
            batch_indices = token_indices[:, 0]
            sequence_indices = token_indices[:, 1]
            decoded = self.reconstruction_heads[str(patch_len)](
                h[batch_indices, sequence_indices]
            ).reshape(-1, patch_len, self.n_features)
            starts = routing.patch_starts[batch_indices, sequence_indices]
            minute_indices = starts.unsqueeze(1) + torch.arange(
                patch_len,
                device=h.device,
            ).unsqueeze(0)
            reconstruction[batch_indices.unsqueeze(1), minute_indices] = decoded
            coverage[batch_indices.unsqueeze(1), minute_indices] = True
        if not bool(coverage.all()):
            raise RuntimeError("Native dynamic tokens did not cover every input minute")
        return reconstruction

    def masked_reconstruction_loss(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        dense_routing: bool = False,
        forced_expert: int | None = None,
    ) -> tuple[torch.Tensor, RoutingState]:
        self.base_patchify(x)
        base_mask = self.sample_base_mask(x.shape[0], x.device, mask_ratio)
        minute_mask = base_mask.repeat_interleave(self.alignment_grid_len, dim=1)
        masked_x = torch.where(
            minute_mask.unsqueeze(-1),
            self.raw_mask_token.to(x.dtype),
            x,
        )
        h, routing = self.routed_features(
            masked_x,
            observed_minutes=~minute_mask,
            dense_routing=dense_routing,
            forced_expert=forced_expert,
        )
        reconstruction = self._reconstruct_minutes(h, routing)
        return F.mse_loss(reconstruction[minute_mask], x[minute_mask]), routing

    @torch.no_grad()
    def update_load_balance(
        self,
        global_load: torch.Tensor,
        update_speed: float | None = None,
    ) -> None:
        mass = global_load.to(
            self.load_ema.device,
            dtype=self.load_ema.dtype,
        )
        if mass.shape != self.load_ema.shape:
            raise ValueError("Observed load must contain one value per expert")
        total = mass.sum()
        if bool(total <= 0.0):
            raise ValueError("Global gate mass must be positive")
        load = mass / total
        if not torch.isfinite(load).all():
            raise ValueError("Observed load must be finite")
        speed = (
            self.bias_update_speed
            if update_speed is None
            else float(update_speed)
        )
        if speed < 0.0:
            raise ValueError("update_speed must be non-negative")
        self.load_ema.mul_(self.load_ema_decay).add_(
            load, alpha=1.0 - self.load_ema_decay
        )
        if speed == 0.0:
            return
        # Low-variance stochastic approximation toward a fixed, data-independent prior.
        self.expert_bias.add_(self.target_load - self.load_ema, alpha=speed)
        self.expert_bias.sub_(self.expert_bias.mean())
        self.expert_bias.clamp_(-self.bias_clip, self.bias_clip)
        self.expert_bias.sub_(self.expert_bias.mean())
        self.bias_updates.add_(1)

    @property
    def uses_bias_control(self) -> bool:
        return self.balance_mode in {
            "fusion_mass_bias",
            "contribution_bias",
            "count_bias",
        }

    def select_balance_load(
        self,
        counts: torch.Tensor,
        gate_mass: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.balance_mode in {"fusion_mass_bias", "contribution_bias"}:
            return gate_mass
        if self.balance_mode == "count_bias":
            return counts
        return None

    def auxiliary_load_balance_loss(
        self,
        clean_logits: torch.Tensor,
        global_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Switch-style load loss with detached, globally aggregated assignments."""
        if clean_logits.shape[-1] != self.n_experts:
            raise ValueError("clean_logits must contain one score per expert")
        counts = global_counts.to(clean_logits.device, clean_logits.dtype)
        if counts.shape != (self.n_experts,):
            raise ValueError("global_counts must contain one value per expert")
        assignment_share = counts / counts.sum().clamp_min(1.0)
        clean_router_share = torch.softmax(
            clean_logits / self.routing_temperature,
            dim=-1,
        ).mean(dim=tuple(range(clean_logits.ndim - 1)))
        return self.n_experts * torch.sum(
            assignment_share.detach() * clean_router_share
        )

    def routing_state(self) -> dict[str, torch.Tensor]:
        return {
            "expert_bias": self.expert_bias.detach().cpu().clone(),
            "expert_prior": self.expert_prior.detach().cpu().clone(),
            "target_load": self.target_load.detach().cpu().clone(),
            "load_ema": self.load_ema.detach().cpu().clone(),
            "bias_updates": self.bias_updates.detach().cpu().clone(),
        }

    def finetune_parameter_groups(self, training_cfg: dict) -> list[dict]:
        expert_parameters = list(self.experts.parameters())
        expert_parameters += list(self.duration_embedding.parameters())
        backbone_parameters = list(self.market_state_projection.parameters())
        backbone_parameters += list(self.backbone.parameters())
        backbone_parameters += list(self.final_norm.parameters())
        backbone_parameters += [self.pool_query, self.raw_mask_token]
        backbone_parameters += list(self.temporal_pool.parameters())
        backbone_parameters += list(self.reconstruction_heads.parameters())
        return [
            {
                "name": "head",
                "params": list(self.ranking_head.parameters()),
                "lr": float(training_cfg["head_lr"]),
            },
            {
                "name": "router",
                "params": list(self.router.parameters()),
                "lr": float(training_cfg["router_lr"]),
            },
            {
                "name": "experts",
                "params": expert_parameters,
                "lr": float(training_cfg["experts_lr"]),
            },
            {
                "name": "backbone",
                "params": backbone_parameters,
                "lr": float(training_cfg["backbone_lr"]),
            },
        ]


__all__ = [
    "ARCHITECTURE",
    "BALANCE_MODES",
    "AlphaModel",
    "RoutingState",
]

import base64
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_FORMAT = "bigalpha_model_json_v1"
MODEL_PATH = Path("model.json")
DEFAULT_BATCH_SIZE = 128
MIN_BATCH_SIZE = 8
WARMUP_DAYS = 45
INSTRUMENT_CHUNK_SIZE = 250
MONTH_INSTRUMENT_BATCH_SIZE = 100
INSTRUMENT_TABLE = "bigalpha_2026_instruments"
