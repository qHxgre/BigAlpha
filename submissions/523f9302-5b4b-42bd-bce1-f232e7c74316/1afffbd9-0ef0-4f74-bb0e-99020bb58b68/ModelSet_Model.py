from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ModelSet_Config import Config
from ModelSet_Modules import (
    FTAxialEncoder,
    BookAxialEncoder,
    ScaleIntraFusionV2,
    CrossScaleFusion,
    CrossSectionBlock,
    PredictHead,
    FiLM,
    scale_decorr_loss,
)


class ScaleTower(nn.Module):
    """单尺度塔：B1 (+B2) → B3。可在多尺度间共享或独立。"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vector = FTAxialEncoder(config)
        self.book = BookAxialEncoder(config) if config.book_stream else None
        self.intra = ScaleIntraFusionV2(config)

    def forward(
        self,
        vector: torch.Tensor,
        matrix: Optional[torch.Tensor],
        scale_id: int,
        film: Optional[FiLM] = None,
        return_cnn: bool = False,
    ):
        if return_cnn:
            Hv, v, cnn_feat = self.vector(
                vector, scale_id=scale_id, film=film, return_cnn=True
            )
        else:
            Hv, v = self.vector(vector, scale_id=scale_id, film=film)
            cnn_feat = None

        Hb = None
        if self.book is not None and matrix is not None:
            Hb = self.book(matrix, scale_id=scale_id, film=film)

        H, z = self.intra(Hv, Hb)
        z = z + 0.5 * v
        if return_cnn:
            return H, z, cnn_feat
        return H, z


class MAEDecoder(nn.Module):
    """轻量线性解码：从 H 重建 B1a CNN 特征的 patch 均值（合规自监督）。"""

    def __init__(self, d_model: int, conv_channels: int, n_fields: int):
        super().__init__()
        self.n_fields = n_fields
        self.conv_channels = conv_channels
        self.proj = nn.Linear(d_model, n_fields * conv_channels)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.proj(H)


class PonyOracleModule(nn.Module):
    """
    EVO 端到端模型。

    forward(feature_dict, mask) -> (pred_rank, pred_ret)
    forward(..., return_aux=True) -> (pred_rank, pred_ret, aux)
      aux 含: z_scales, moe_gate, decile_logits, mae_loss, style_feat, decorr_loss
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.active_scales: List[str] = list(config.active_scales)
        for k in self.active_scales:
            if k not in config.scale_id_map:
                raise ValueError(f"active_scales 含未知尺度: {k}")

        n_scales = len(config.scale_keys)
        n_blocks = int(config.n_blocks)
        scale_cond = str(config.scale_cond).lower()
        share = bool(config.share_backbone_across_scales)

        if (not share) or scale_cond == "separate_weights":
            self.towers = nn.ModuleDict(
                {k: ScaleTower(config) for k in config.scale_keys}
            )
            self.shared_tower = None
        else:
            self.shared_tower = ScaleTower(config)
            self.towers = None

        if share and scale_cond == "film":
            self.film = FiLM(n_scales, n_blocks, config.d_model)
        else:
            self.film = None

        self.fusion = CrossScaleFusion(config)
        self.cs = CrossSectionBlock(config)
        self.head = PredictHead(
            in_dim=self.cs.out_dim,
            d_model=config.d_model,
            dropout=config.dropout,
            use_decile=config.use_decile_head,
            use_quantile=config.use_quantile_head,
            n_quantiles=config.n_quantiles,
        )

        # 可选 MAE（旧 ckpt 若仍带 w_aux_mae>0 则可开启）
        self.mae_decoder = None
        if float(getattr(config, "w_aux_mae", 0.0) or 0.0) > 0:
            self.mae_decoder = MAEDecoder(
                config.d_model, config.conv_channels, config.n_fields
            )

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def assert_param_budget(self) -> int:
        n = self.count_trainable_params()
        if not (1e5 < n < 1e8):
            raise AssertionError(f"可训练参数量越界: {n}")
        return n

    def _get_tower(self, scale_key: str) -> ScaleTower:
        if self.shared_tower is not None:
            return self.shared_tower
        return self.towers[scale_key]

    def _maybe_noise(self, x: torch.Tensor) -> torch.Tensor:
        std = float(getattr(self.config, "input_noise_std", 0.0) or 0.0)
        if self.training and std > 0:
            return x + torch.randn_like(x) * std
        return x

    def _maybe_time_crop(
        self,
        vector: torch.Tensor,
        matrix: Optional[torch.Tensor],
    ):
        """训练时随机左裁剪，保持右端对齐、长度仍为 seq_len（左侧 pad 0）。"""
        max_crop = int(getattr(self.config, "time_crop_max", 0) or 0)
        if (not self.training) or max_crop <= 0:
            return vector, matrix
        crop = int(torch.randint(0, max_crop + 1, (1,)).item())
        if crop <= 0:
            return vector, matrix
        T = vector.size(1)
        keep = T - crop
        v = torch.zeros_like(vector)
        v[:, -keep:] = vector[:, crop:]
        m = None
        if matrix is not None:
            m = torch.zeros_like(matrix)
            m[:, -keep:] = matrix[:, crop:]
        return v, m

    # ------------------------------------------------------------------
    # 逐股票：B0–B4（整截面一次前向）
    # ------------------------------------------------------------------
    def _encode_scales(
        self,
        feature_dict: dict,
        need_mae: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict]:
        """
        Returns
        -------
        tokens: {scale: (N, L, d)}
        summary: {scale: (N, d)}
        local_aux: mae 相关
        """
        tokens, summary = {}, {}
        local_aux = {"mae_loss": None, "cnn_targets": {}}
        mae_losses = []

        for key in self.active_scales:
            sid = self.config.scale_id_map[key]
            vec = self._maybe_noise(feature_dict[key]["vector"])
            mat = feature_dict[key].get("matrix", None)
            if mat is not None:
                mat = self._maybe_noise(mat)
            vec, mat = self._maybe_time_crop(vec, mat)

            tower = self._get_tower(key)
            if need_mae and self.mae_decoder is not None:
                out = tower(vec, mat, scale_id=sid, film=self.film, return_cnn=True)
            else:
                out = tower(vec, mat, scale_id=sid, film=self.film, return_cnn=False)

            if need_mae and self.mae_decoder is not None:
                H, z, cnn_feat = out
                P = int(self.config.patch_len)
                S = int(self.config.patch_stride)
                N, Ff, T, c = cnn_feat.shape
                cf = cnn_feat.permute(0, 1, 3, 2)
                pooled = F.avg_pool1d(
                    cf.reshape(N * Ff * c, 1, T), kernel_size=P, stride=S
                )
                L = pooled.size(-1)
                target = (
                    pooled.view(N, Ff, c, L)
                    .permute(0, 3, 1, 2)
                    .reshape(N, L, Ff * c)
                )
                pred = self.mae_decoder(H[:, :L])
                mask_p = (torch.rand(N, L, device=H.device) < 0.15).float().unsqueeze(-1)
                if mask_p.sum() < 1:
                    mae = H.new_zeros(())
                else:
                    mae = ((pred - target.detach()) ** 2 * mask_p).sum() / mask_p.sum().clamp_min(1.0)
                mae_losses.append(mae)
            else:
                H, z = out

            tokens[key] = H
            summary[key] = z

        if mae_losses:
            local_aux["mae_loss"] = torch.stack(mae_losses).mean()
        return tokens, summary, local_aux

    def encode_per_stock(
        self,
        feature_dict: dict,
        usable: Optional[torch.Tensor] = None,
        need_mae: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """B0–B4 → u (N, d)"""
        tokens, summary, local_aux = self._encode_scales(
            feature_dict, need_mae=need_mae
        )
        u, aux_c = self.fusion(tokens, summary, usable=usable)

        aux = {
            "z_scales": aux_c.get("z_scales") or [],
            "moe_gate": aux_c.get("moe_gate"),
            "mae_loss": local_aux.get("mae_loss"),
            "decorr_loss": None,
        }
        if aux["z_scales"] and len(aux["z_scales"]) >= 2:
            aux["decorr_loss"] = scale_decorr_loss(aux["z_scales"])
        return u, aux

    def _style_feat(self, feature_dict: dict) -> Optional[torch.Tensor]:
        if float(getattr(self.config, "w_style", 0.0) or 0.0) <= 0:
            return None
        idx = int(getattr(self.config, "style_field_idx", 5))
        key = "m01" if "m01" in feature_dict else self.active_scales[0]
        return feature_dict[key]["vector"][:, -1, idx]

    def forward(
        self,
        feature_dict: dict,
        mask: torch.Tensor,
        return_aux: bool = False,
    ):
        """
        feature_dict: batch['feature']，已是 Tensor
          {m01: {vector, matrix}, ...}
        mask: (N,) bool，有效样本
        """
        need_mae = (
            self.training
            and self.mae_decoder is not None
            and float(getattr(self.config, "w_aux_mae", 0.0) or 0.0) > 0
        )

        u, stock_aux = self.encode_per_stock(
            feature_dict, usable=mask, need_mae=need_mae
        )
        z = self.cs(u, mask)
        pred_rank, pred_ret, head_aux = self.head(z)

        if not return_aux:
            return pred_rank, pred_ret

        aux = {
            "z_scales": stock_aux.get("z_scales"),
            "moe_gate": stock_aux.get("moe_gate"),
            "mae_loss": stock_aux.get("mae_loss"),
            "decorr_loss": stock_aux.get("decorr_loss"),
            "style_feat": self._style_feat(feature_dict),
            "repr": z,
        }
        aux.update(head_aux)
        return pred_rank, pred_ret, aux


def build_model(config: Config) -> PonyOracleModule:
    """工厂：构建并校验参数量。"""
    model = PonyOracleModule(config)
    n = model.assert_param_budget()
    print(f"[PonyOracle EVO] trainable params = {n:,}")
    return model
