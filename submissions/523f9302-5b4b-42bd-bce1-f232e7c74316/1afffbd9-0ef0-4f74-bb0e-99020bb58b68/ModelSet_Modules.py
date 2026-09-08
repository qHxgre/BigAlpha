# B0 NormGate → B1 FT-Axial → B2 Book-Axial → B3 Intra → B4 CrossScale → B5 CS
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 通用工具
# ======================================================================

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
    return x * mask


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.p, self.training)


class MLP(nn.Module):
    def __init__(self, d_model: int, ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _alibi_slopes(n_heads: int) -> torch.Tensor:
    def _slopes(n: int) -> List[float]:
        ratio = 2 ** (-(2 ** -(math.log2(max(n, 2)) - 3)))
        return [ratio ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        slopes = _slopes(closest) + _slopes(2 * closest)[0::2][: n_heads - closest]
    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(n_heads: int, seq_len: int, device, dtype) -> torch.Tensor:
    """(H, L, L) ALiBi 相对距离偏置。"""
    slopes = _alibi_slopes(n_heads).to(device=device, dtype=dtype)
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    rel = pos[None, :] - pos[:, None]
    return -slopes[:, None, None] * rel.abs()[None, :, :]


class MultiHeadAttn(nn.Module):
    """Self / Cross Attention，内部走 scaled_dot_product_attention。"""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dropout: float = 0.0,
        use_alibi: bool = False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} 不能整除 nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.use_alibi = bool(use_alibi)
        self.dropout = float(dropout)

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _shape(self, t: torch.Tensor, B: int, L: int) -> torch.Tensor:
        return t.view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # (B,H,L,Dh)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query: (B, Lq, d)
        key/value: (B, Lk, d)；缺省则自注意力
        key_padding_mask: (B, Lk) True=有效
        attn_bias: (H, Lq, Lk) 或 (1,H,Lq,Lk) 或 (B,H,Lq,Lk)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        B, Lq, _ = query.shape
        Lk = key.size(1)

        q = self._shape(self.wq(query), B, Lq)
        k = self._shape(self.wk(key), B, Lk)
        v = self._shape(self.wv(value), B, Lk)

        # 组装加性 mask（SDPA float mask）
        mask = None
        if self.use_alibi and Lq == Lk and attn_bias is None:
            mask = build_alibi_bias(self.nhead, Lq, query.device, query.dtype)
            mask = mask.unsqueeze(0)  # (1,H,L,L)
        if attn_bias is not None:
            b = attn_bias
            if b.dim() == 3:
                b = b.unsqueeze(0)
            mask = b if mask is None else mask + b

        if key_padding_mask is not None:
            # True=valid → 无效位置 -inf
            pad = torch.zeros(B, 1, 1, Lk, device=query.device, dtype=query.dtype)
            pad = pad.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
            mask = pad if mask is None else mask + pad

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B,H,Lq,Dh)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out(out)


class AttnPool(nn.Module):
    """可学习 query 注意力池化: (B, L, d) → (B, d)"""

    def __init__(self, d_model: int, nhead: int = 1, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = MultiHeadAttn(d_model, nhead=nhead, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        return self.attn(q, x, x, key_padding_mask=key_padding_mask).squeeze(1)


class FiLM(nn.Module):
    """尺度条件调制：每尺度每层一组 γ, β。"""

    def __init__(self, n_scales: int, n_layers: int, d_model: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_scales, n_layers, d_model))
        self.beta = nn.Parameter(torch.zeros(n_scales, n_layers, d_model))

    def forward(self, x: torch.Tensor, scale_id: int, layer_id: int) -> torch.Tensor:
        return x * self.gamma[scale_id, layer_id] + self.beta[scale_id, layer_id]


# ======================================================================
# B0 — NormGate
# ======================================================================

class NormGate(nn.Module):
    """
    序列内实例归一化（RevIN 式）+ 水平分支。
    revin_mode:
      - off:      仅水平分支（管道全局标准化值）
      - dyn_only: 仅动态分支
      - dual:     两分支并存，由上层投影融合
    不做 mean/std 回注通道。
    """

    def __init__(
        self,
        n_channels: int,
        mode: str = "dual",
        affine: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.mode = str(mode).lower()
        self.eps = float(eps)
        if self.mode not in ("off", "dyn_only", "dual"):
            raise ValueError(f"未知 revin_mode={mode}")
        if affine and self.mode in ("dyn_only", "dual"):
            self.gamma = nn.Parameter(torch.ones(n_channels))
            self.beta = nn.Parameter(torch.zeros(n_channels))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def _dyn(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, C) 沿 T 做 per-sample per-channel z-score
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = var.sqrt().clamp_min(self.eps)
        y = (x - mean) / std
        if self.gamma is not None:
            y = y * self.gamma.view(1, 1, -1) + self.beta.view(1, 1, -1)
        return torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_d, x_l : 动态分支 / 水平分支，形状同 x
                   关闭的分支返回零张量（保持接口统一）
        """
        if self.mode == "off":
            return torch.zeros_like(x), x
        if self.mode == "dyn_only":
            return self._dyn(x), torch.zeros_like(x)
        return self._dyn(x), x


# ======================================================================
# B1 — 向量流 FT-Axial
# ======================================================================

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, groups=channels, bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = self.dw(x)
        # 因果/窗口保护：若 dilation padding 导致长度变化，裁回
        if y.size(-1) != x.size(-1):
            y = y[..., : x.size(-1)]
        return self.act(self.norm(self.pw(y)))


class FieldTemporalCNN(nn.Module):
    """
    B1a: 每字段 Linear(1→c)（不共享）+ 跨字段共享深度可分离 1D-CNN。
    输入 (N,T,F) → (N,F,T,c)
    """

    def __init__(
        self,
        n_fields: int,
        channels: int = 32,
        n_layers: int = 3,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2, 4),
        share_across_fields: bool = True,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.channels = channels
        self.share_across_fields = bool(share_across_fields)

        self.in_proj = nn.ModuleList(
            [nn.Linear(1, channels) for _ in range(n_fields)]
        )
        dils = list(dilations)[:n_layers]
        while len(dils) < n_layers:
            dils.append(dils[-1] if dils else 1)

        if share_across_fields:
            self.convs = nn.ModuleList(
                [DepthwiseSeparableConv1d(channels, kernel_size, d) for d in dils]
            )
        else:
            self.convs = nn.ModuleList(
                [
                    nn.ModuleList(
                        [DepthwiseSeparableConv1d(channels, kernel_size, d) for d in dils]
                    )
                    for _ in range(n_fields)
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, F)
        N, T, F = x.shape
        feats = []
        for i in range(F):
            feats.append(self.in_proj[i](x[:, :, i: i + 1]))  # (N,T,c)
        h = torch.stack(feats, dim=1)  # (N,F,T,c)

        if self.share_across_fields:
            # 合并字段跑共享卷积
            y = h.reshape(N * F, T, self.channels).transpose(1, 2)  # (N*F,c,T)
            for conv in self.convs:
                y = y + conv(y)
            h = y.transpose(1, 2).reshape(N, F, T, self.channels)
        else:
            outs = []
            for i in range(F):
                y = h[:, i].transpose(1, 2)  # (N,c,T)
                for conv in self.convs[i]:
                    y = y + conv(y)
                outs.append(y.transpose(1, 2))
            h = torch.stack(outs, dim=1)
        return h


class PatchEmbed(nn.Module):
    """B1b: Conv1d(kernel=P, stride=S) 把时间压成 L 个 token。"""

    def __init__(self, in_ch: int, d_model: int, patch_len: int, stride: int):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, L, d)
        y = self.proj(x)
        return y.transpose(1, 2).contiguous()


class AxialBlock(nn.Module):
    """B1c: TimeAttn → FieldAttn → FFN，Pre-LN + 残差 + DropPath。"""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        droppath: float = 0.0,
        use_field_attn: bool = True,
        use_alibi: bool = True,
    ):
        super().__init__()
        self.use_field_attn = bool(use_field_attn)

        self.n1 = nn.LayerNorm(d_model)
        self.time_attn = MultiHeadAttn(
            d_model, nhead, dropout=attn_dropout, use_alibi=use_alibi
        )
        self.dp1 = DropPath(droppath)

        if self.use_field_attn:
            self.n2 = nn.LayerNorm(d_model)
            self.field_attn = MultiHeadAttn(
                d_model, nhead, dropout=attn_dropout, use_alibi=False
            )
            self.dp2 = DropPath(droppath)
        else:
            self.n2 = None
            self.field_attn = None
            self.dp2 = None

        self.n3 = nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, ratio=ffn_ratio, dropout=dropout)
        self.dp3 = DropPath(droppath)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, F, L, d)
        N, F, L, d = x.shape

        # TimeAttn over L
        t = x.reshape(N * F, L, d)
        t = t + self.dp1(self.time_attn(self.n1(t)))
        x = t.reshape(N, F, L, d)

        # FieldAttn over F
        if self.use_field_attn:
            f = x.permute(0, 2, 1, 3).contiguous().reshape(N * L, F, d)
            f = f + self.dp2(self.field_attn(self.n2(f)))
            x = f.reshape(N, L, F, d).permute(0, 2, 1, 3).contiguous()

        # FFN
        x = x + self.dp3(self.ffn(self.n3(x)))
        return x


class BiGRUReadout(nn.Module):
    """
    B1d: 字段维 AttnPool → BiGRU → 末态与序列 AttnPool 门控融合。
    输入 (N,F,L,d) → Hv (N,L,d), v (N,d)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, use_bigru: bool = True):
        super().__init__()
        self.use_bigru = bool(use_bigru)
        self.field_pool = AttnPool(d_model, nhead=1, dropout=dropout)

        if self.use_bigru:
            self.gru = nn.GRU(
                d_model, d_model // 2, batch_first=True, bidirectional=True
            )
            self.seq_pool = AttnPool(d_model, nhead=1, dropout=dropout)
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            self.proj_h = nn.Linear(d_model, d_model)
            self.proj_p = nn.Linear(d_model, d_model)
        else:
            self.gru = None
            self.seq_pool = AttnPool(d_model, nhead=1, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, F, L, d = x.shape
        # 字段池化
        flat = x.permute(0, 2, 1, 3).reshape(N * L, F, d)
        tok = self.field_pool(flat).view(N, L, d)  # (N,L,d)

        if self.use_bigru:
            hv, h_n = self.gru(tok)  # hv:(N,L,d); h_n:(2,N,d/2)
            h_last = torch.cat([h_n[0], h_n[1]], dim=-1)  # (N,d)
            p = self.seq_pool(hv)
            g = self.gate(torch.cat([h_last, p], dim=-1))
            v = g * self.proj_h(h_last) + (1.0 - g) * self.proj_p(p)
            return self.norm(hv), v

        v = self.seq_pool(tok)
        return self.norm(tok), v


class FTAxialEncoder(nn.Module):
    """
    B1 完整向量流。
    输入 vector (N,T,F) → Hv (N,L,d), v (N,d)
    可选返回 CNN 特征供 MAE 重建。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        F = config.n_fields
        d = config.d_model
        P = int(config.patch_len)
        S = int(config.patch_stride)
        c = int(config.conv_channels)

        self.norm_gate = NormGate(
            F, mode=config.revin_mode, affine=config.revin_affine, eps=config.revin_eps
        )
        # 双分支投影（水平分支参数名含 level_proj，便于独立 wd）
        self.dyn_proj = nn.ModuleList([nn.Linear(c, d, bias=False) for _ in range(F)])
        self.level_proj = nn.ModuleList([nn.Linear(c, d, bias=False) for _ in range(F)])

        self.cnn = FieldTemporalCNN(
            n_fields=F,
            channels=c,
            n_layers=config.conv_layers,
            kernel_size=config.conv_kernel,
            dilations=config.conv_dilations,
            share_across_fields=config.share_across_fields,
        )
        # 每字段独立 patch（输入已是 c 通道）
        self.patch = nn.ModuleList(
            [PatchEmbed(c, d, P, S) for _ in range(F)]
        )

        self.field_emb = nn.Parameter(torch.randn(F, d) * 0.02)
        # L 在首次 forward 时确定；先按理论长度注册
        L_expect = (config.seq_len - P) // S + 1
        self.pos_emb = nn.Parameter(torch.randn(L_expect, d) * 0.02)
        self.scale_emb = nn.Embedding(len(config.scale_keys), d)

        use_alibi = str(getattr(config, "time_pos", "alibi")).lower() == "alibi"
        self.blocks = nn.ModuleList(
            [
                AxialBlock(
                    d_model=d,
                    nhead=config.nhead,
                    ffn_ratio=config.ffn_ratio,
                    dropout=config.dropout,
                    attn_dropout=config.attn_dropout,
                    droppath=config.droppath,
                    use_field_attn=config.use_field_attn,
                    use_alibi=use_alibi,
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.readout = BiGRUReadout(d, dropout=config.dropout, use_bigru=config.use_bigru)
        self.film = None  # 由外层按需注入 / 或在 ScaleEncoder 里调

    def _embed_branches(
        self,
        h_d: torch.Tensor,
        h_l: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_*: (N,F,T,c) → 字段投影后相加再 patch。
        实际流程：先 CNN 得到共享局部特征，再分别对 dyn/level 原值投影意义不大；
        按文档：在字段嵌入处 W_d·x_d + W_l·x_l。
        这里对 CNN 输出做双分支线性（dyn/level 来自 NormGate 后再 CNN）。
        """
        # 简化且符合文档：对 dyn/level 分别 CNN 成本翻倍；采用文档推荐的嵌入处相加。
        # 实现对齐：先对原始 dyn/level 做 per-field 升维，相加后走共享 CNN+patch。
        raise RuntimeError("内部不应调用")

    def forward(
        self,
        vector: torch.Tensor,
        scale_id: int = 0,
        film: Optional[FiLM] = None,
        film_offset: int = 0,
        return_cnn: bool = False,
    ):
        # vector: (N, T, F)
        x_d, x_l = self.norm_gate(vector)  # (N,T,F)

        # 嵌入处双分支：升到 c 后相加，再共享 CNN
        N, T, F = vector.shape
        c = self.cnn.channels
        toks = []
        for i in range(F):
            # 直接用 NormGate 输出做 1→c（复用 cnn.in_proj）
            e_d = self.cnn.in_proj[i](x_d[:, :, i: i + 1])
            e_l = self.cnn.in_proj[i](x_l[:, :, i: i + 1])
            # 再用 d 维 level/dyn 投影会破坏通道；保持 c 通道相加
            toks.append(e_d + e_l)
        h0 = torch.stack(toks, dim=1)  # (N,F,T,c)

        # 共享 CNN（跳过 in_proj，直接从 h0 开始）
        if self.cnn.share_across_fields:
            y = h0.reshape(N * F, T, c).transpose(1, 2)
            for conv in self.cnn.convs:
                y = y + conv(y)
            h = y.transpose(1, 2).reshape(N, F, T, c)
        else:
            outs = []
            for i in range(F):
                y = h0[:, i].transpose(1, 2)
                for conv in self.cnn.convs[i]:
                    y = y + conv(y)
                outs.append(y.transpose(1, 2))
            h = torch.stack(outs, dim=1)

        # Patch + 嵌入
        patches = []
        for i in range(F):
            p = self.patch[i](h[:, i].transpose(1, 2))  # (N,L,d)
            # 双分支残余：对 patch 再加 level/dyn 的线性修正（可选增强）
            patches.append(p)
        x = torch.stack(patches, dim=1)  # (N,F,L,d)
        L = x.size(2)

        x = x + self.field_emb.view(1, F, 1, -1)
        pos = self.pos_emb[:L].view(1, 1, L, -1)
        x = x + pos
        x = x + self.scale_emb.weight[scale_id].view(1, 1, 1, -1)

        for li, blk in enumerate(self.blocks):
            x = blk(x)
            if film is not None:
                x = film(x, scale_id, film_offset + li)

        Hv, v = self.readout(x)
        if return_cnn:
            return Hv, v, h
        return Hv, v


# ======================================================================
# B2 — 盘口流 Book-Axial
# ======================================================================

class BookAxialEncoder(nn.Module):
    """
    (N,T,4,5) → Hb (N,L,d)
    B2a 组别混合 → B2b 时间×档位 2D-CNN（末层 stride 完成 patch）→ B2c 档位注意力池化
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        db = config.d_book
        P = int(config.patch_len)
        T = int(config.seq_len)
        n_ch = config.n_book_channels
        n_lv = config.n_book_levels

        self.norm_gate = NormGate(
            n_ch * n_lv,
            mode=config.revin_mode,
            affine=config.revin_affine,
            eps=config.revin_eps,
        )
        # 组别混合：每个 (t,level) 上 Linear(4→db)
        self.group_mix_d = nn.Linear(n_ch, db)
        self.level_proj = nn.Linear(n_ch, db)  # 水平分支，独立 wd

        # 2D CNN: 输入 (N, db, T, 5)
        layers = []
        in_c = db
        n_conv = int(config.book_conv_layers)
        kernels = [(3, 3), (5, 3), (3, 3)]
        for i in range(n_conv):
            kt, kl = kernels[min(i, len(kernels) - 1)]
            is_last = i == n_conv - 1
            stride = (P, 1) if is_last else (1, 1)
            pad = (kt // 2, kl // 2)
            # depthwise-separable 近似：分组卷积 + pointwise
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_c, in_c, kernel_size=(kt, kl),
                        stride=stride, padding=pad, groups=in_c, bias=False,
                    ),
                    nn.Conv2d(in_c, db, kernel_size=1, bias=False),
                    nn.BatchNorm2d(db),
                    nn.GELU(),
                )
            )
            in_c = db
        self.conv2d = nn.ModuleList(layers)

        self.level_emb = nn.Parameter(torch.randn(n_lv, db) * 0.02)
        self.level_rel_bias = bool(config.level_rel_bias)
        if self.level_rel_bias:
            # 相对档位偏置表 (2*n_lv-1,)
            self.rel_bias_table = nn.Parameter(torch.zeros(2 * n_lv - 1))
            coords = torch.arange(n_lv)
            rel = coords[None, :] - coords[:, None] + (n_lv - 1)
            self.register_buffer("rel_index", rel.long(), persistent=False)

        self.level_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "n1": nn.LayerNorm(db),
                        "attn": MultiHeadAttn(db, nhead=max(1, db // 32), dropout=config.attn_dropout),
                        "n2": nn.LayerNorm(db),
                        "ffn": MLP(db, ratio=config.ffn_ratio, dropout=config.dropout),
                    }
                )
                for _ in range(config.level_attn_blocks)
            ]
        )
        self.level_pool_mode = str(config.level_pool).lower()
        if self.level_pool_mode == "attn":
            self.level_pool = AttnPool(db, nhead=1, dropout=config.dropout)
        else:
            self.level_pool = None
        self.out_proj = nn.Linear(db, d)
        self.norm = nn.LayerNorm(d)
        self.scale_emb = nn.Embedding(len(config.scale_keys), d)

    def _level_bias(self, nhead: int) -> Optional[torch.Tensor]:
        if not self.level_rel_bias:
            return None
        # (Lvl,Lvl) → (H,Lvl,Lvl)
        b = self.rel_bias_table[self.rel_index]  # (5,5)
        return b.unsqueeze(0).expand(nhead, -1, -1)

    def forward(
        self,
        matrix: torch.Tensor,
        scale_id: int = 0,
        film: Optional[FiLM] = None,
        film_offset: int = 0,
    ) -> torch.Tensor:
        # matrix: (N, T, 4, 5)
        N, T, C, Lv = matrix.shape
        flat = matrix.permute(0, 1, 3, 2).reshape(N, T, C * Lv)  # (N,T,20)
        x_d, x_l = self.norm_gate(flat)
        x_d = x_d.view(N, T, Lv, C)
        x_l = x_l.view(N, T, Lv, C)

        # 组别混合 + 双分支
        h = self.group_mix_d(x_d) + self.level_proj(x_l)  # (N,T,Lv,db)
        h = h.permute(0, 3, 1, 2).contiguous()  # (N,db,T,Lv)

        for conv in self.conv2d:
            h = conv(h)
            # 档位维保护
            if h.size(-1) != Lv:
                h = h[..., :Lv]

        # (N, db, L, Lv)
        db, L = h.size(1), h.size(2)
        x = h.permute(0, 2, 3, 1).contiguous()  # (N,L,Lv,db)
        x = x + self.level_emb.view(1, 1, Lv, -1)

        for bi, blk in enumerate(self.level_blocks):
            y = x.reshape(N * L, Lv, db)
            bias = self._level_bias(blk["attn"].nhead)
            if bias is not None:
                bias = bias.to(dtype=y.dtype, device=y.device)
            y2 = blk["attn"](blk["n1"](y), attn_bias=bias)
            y = y + y2
            y = y + blk["ffn"](blk["n2"](y))
            x = y.view(N, L, Lv, db)
            if film is not None:
                # FiLM 在 d_book 上；投影到 d 后再调也可，这里跳过 book 维 film
                pass

        if self.level_pool_mode == "attn":
            pooled = self.level_pool(x.reshape(N * L, Lv, db)).view(N, L, db)
        else:
            pooled = x.mean(dim=2)

        out = self.norm(self.out_proj(pooled))
        out = out + self.scale_emb.weight[scale_id].view(1, 1, -1)
        return out  # (N,L,d)


# ======================================================================
# B3 — 尺度内 Vector ⇄ Book 双向交叉注意力
# ======================================================================

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_ratio: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.n_q = nn.LayerNorm(d_model)
        self.n_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttn(d_model, nhead, dropout=attn_dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, ratio=ffn_ratio, dropout=dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        y = q + self.attn(self.n_q(q), self.n_kv(kv), self.n_kv(kv))
        return y + self.ffn(self.n2(y))


class ScaleIntraFusion(nn.Module):
    """
    双向交叉注意力 + [SCALE_CLS] 读出。
    book_stream=False 时退化为仅向量流 + CLS。
    """

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.bidirectional = bool(config.bidirectional_cross)
        self.book_stream = bool(config.book_stream)
        self.n_layers = int(config.cross_layers)

        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.v2b = nn.ModuleList(
            [
                CrossAttnBlock(d, config.nhead, config.ffn_ratio, config.dropout, config.attn_dropout)
                for _ in range(self.n_layers)
            ]
        )
        if self.bidirectional and self.book_stream:
            self.b2v = nn.ModuleList(
                [
                    CrossAttnBlock(d, config.nhead, config.ffn_ratio, config.dropout, config.attn_dropout)
                    for _ in range(self.n_layers)
                ]
            )
        else:
            self.b2v = None

        self.merge = nn.Linear(d * 2, d) if self.book_stream else nn.Identity()
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        Hv: torch.Tensor,
        Hb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hv/Hb: (N,L,d)
        Returns: H (N,L,d), z (N,d)
        """
        N, L, d = Hv.shape
        if (not self.book_stream) or Hb is None:
            H = Hv
        else:
            hv, hb = Hv, Hb
            for i in range(self.n_layers):
                hv = self.v2b[i](hv, hb)
                if self.b2v is not None:
                    hb = self.b2v[i](hb, hv)
            H = self.merge(torch.cat([hv, hb], dim=-1))

        cls = self.cls.expand(N, -1, -1)
        seq = torch.cat([cls, H], dim=1)  # (N, L+1, d)
        # 用 CLS attend 全序列做读出
        # 简单：CLS 位置经 LayerNorm 后取第 0 token；再与 mean 混合也可
        z = self.norm(seq[:, 0])
        # 增强：让 CLS 做一次池化注意力
        return self.norm(H), z


class ScaleIntraFusionV2(nn.Module):
    """带 CLS 交叉注意力读出的尺度内融合（推荐使用）。"""

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.book_stream = bool(config.book_stream)
        self.bidirectional = bool(config.bidirectional_cross)
        n = int(config.cross_layers)

        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.layers_v = nn.ModuleList(
            [CrossAttnBlock(d, config.nhead, config.ffn_ratio, config.dropout, config.attn_dropout) for _ in range(n)]
        )
        self.layers_b = nn.ModuleList(
            [CrossAttnBlock(d, config.nhead, config.ffn_ratio, config.dropout, config.attn_dropout) for _ in range(n)]
        ) if (self.book_stream and self.bidirectional) else None

        self.merge = nn.Linear(2 * d, d) if self.book_stream else nn.Identity()
        self.cls_attn = MultiHeadAttn(d, nhead=1, dropout=config.attn_dropout)
        self.norm_h = nn.LayerNorm(d)
        self.norm_z = nn.LayerNorm(d)

    def forward(self, Hv: torch.Tensor, Hb: Optional[torch.Tensor] = None):
        N = Hv.size(0)
        if (not self.book_stream) or Hb is None:
            H = Hv
        else:
            hv, hb = Hv, Hb
            for i, layer in enumerate(self.layers_v):
                hv = layer(hv, hb)
                if self.layers_b is not None:
                    hb = self.layers_b[i](hb, hv)
            H = self.merge(torch.cat([hv, hb], dim=-1))

        H = self.norm_h(H)
        cls = self.cls.expand(N, -1, -1)
        z = self.cls_attn(cls, H, H).squeeze(1)
        return H, self.norm_z(z)


# ======================================================================
# B4 — 跨尺度融合
# ======================================================================

class TimeOffsetEncoding(nn.Module):
    """log(1+Δ物理交易日) → MLP → d。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, delta_days: torch.Tensor) -> torch.Tensor:
        # delta_days: (L,) or (1,L,1)
        if delta_days.dim() == 1:
            delta_days = delta_days.view(1, -1, 1)
        elif delta_days.dim() == 2:
            delta_days = delta_days.unsqueeze(-1)
        x = torch.log1p(delta_days.clamp_min(0.0))
        return self.mlp(x)


class CoarseToFineFusion(nn.Module):
    """以细尺度(m01) token 为 Q，粗尺度为 K/V。"""

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.layers = nn.ModuleList(
            [
                CrossAttnBlock(d, config.nhead, config.ffn_ratio, config.dropout, config.attn_dropout)
                for _ in range(config.c2f_layers)
            ]
        )

    def forward(self, fine: torch.Tensor, coarse_list: List[torch.Tensor]) -> torch.Tensor:
        kv = torch.cat(coarse_list, dim=1) if len(coarse_list) > 1 else coarse_list[0]
        x = fine
        for layer in self.layers:
            x = layer(x, kv)
        return x


class JointScaleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=d * config.ffn_ratio,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.joint_blocks)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokens: (N, 4L, d) 已含各种嵌入
        N = tokens.size(0)
        seq = torch.cat([self.cls.expand(N, -1, -1), tokens], dim=1)
        y = self.encoder(seq)
        return y[:, 1:], self.norm(y[:, 0])


class ScaleMoE(nn.Module):
    """g = softmax(MLP([z_s]) / τ); u = Σ g_s · Proj_s(z_s)"""

    def __init__(self, d_model: int, n_scales: int, temp: float = 1.0):
        super().__init__()
        self.temp = float(temp)
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_scales, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_scales),
        )
        self.projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_scales)])

    def forward(self, zs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat(zs, dim=-1)
        g = torch.softmax(self.gate(cat) / max(self.temp, 1e-4), dim=-1)  # (N,S)
        stacked = torch.stack([self.projs[i](zs[i]) for i in range(len(zs))], dim=1)
        u = (g.unsqueeze(-1) * stacked).sum(dim=1)
        return u, g


class ScaleGRUFusion(nn.Module):
    """沿 m30→m15→m05→m01 单向 GRU。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, zs_coarse_to_fine: List[torch.Tensor]) -> torch.Tensor:
        x = torch.stack(zs_coarse_to_fine, dim=1)  # (N,S,d)
        _, h = self.gru(x)
        return h.squeeze(0)


class CrossScaleFusion(nn.Module):
    """
    fusion_mode 解析：c2f / joint / moe / gru 的组合，最后相加。
    输入: scale_tokens dict[str,(N,L,d)], scale_summary dict[str,(N,d)]
    输出: u (N,d), aux(dict)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        mode = str(config.fusion_mode).lower()
        self.use_c2f = "c2f" in mode
        self.use_joint = "joint" in mode
        self.use_moe = "moe" in mode
        self.use_gru = ("gru" in mode) or bool(config.use_scale_gru)

        n_scales = len(config.scale_keys)
        self.scale_emb = nn.Embedding(n_scales, d)
        self.patch_pos = None  # lazy: 按 L 建
        self.time_enc = TimeOffsetEncoding(d) if config.time_offset_encoding else None

        if self.use_c2f:
            self.c2f = CoarseToFineFusion(config)
            self.c2f_pool = AttnPool(d)
        if self.use_joint:
            self.joint = JointScaleTransformer(config)
        if self.use_moe:
            self.moe = ScaleMoE(d, n_scales=n_scales, temp=config.moe_temp)
        if self.use_gru:
            self.sgru = ScaleGRUFusion(d)

        self.usable_emb = nn.Linear(1, d)
        self.out_norm = nn.LayerNorm(d)
        self._pos_cache = nn.ParameterDict()

    def _pos(self, L: int, device) -> torch.Tensor:
        key = str(L)
        if key not in self._pos_cache:
            self._pos_cache[key] = nn.Parameter(torch.randn(L, self.config.d_model) * 0.02)
        return self._pos_cache[key].to(device)

    def _physical_delta(self, scale_key: str, L: int, device, dtype) -> torch.Tensor:
        # 右端对齐：patch i 距当前的物理交易日数
        P = float(self.config.patch_len)
        bar_min = float(self.config.scale_bar_minutes.get(scale_key, 1))
        # 每 patch 覆盖 P 根 bar → 交易日 ≈ P*bar_min/240
        days_per_patch = P * bar_min / 240.0
        idx = torch.arange(L, device=device, dtype=dtype)
        return (L - 1 - idx) * days_per_patch

    def forward(
        self,
        scale_tokens: dict,
        scale_summary: dict,
        usable: Optional[torch.Tensor] = None,
    ):
        cfg = self.config
        active = [k for k in cfg.active_scales if k in scale_tokens]
        if not active:
            raise ValueError("active_scales 为空或与输入不匹配")

        parts = []
        aux = {"moe_gate": None, "z_scales": []}

        zs_ordered = []
        for k in cfg.scale_keys:
            if k in scale_summary and k in active:
                zs_ordered.append(scale_summary[k])
            elif k in active:
                zs_ordered.append(scale_tokens[k].mean(dim=1))
        aux["z_scales"] = zs_ordered

        # --- c2f ---
        if self.use_c2f and "m01" in scale_tokens:
            fine = scale_tokens["m01"]
            coarse = [
                scale_tokens[k] for k in ("m30", "m15", "m05")
                if k in scale_tokens and k in active
            ]
            if coarse:
                fine2 = self.c2f(fine, coarse)
                parts.append(self.c2f_pool(fine2))
            else:
                parts.append(fine.mean(dim=1))

        # --- joint ---
        if self.use_joint:
            tok_list = []
            for k in active:
                t = scale_tokens[k]
                N, L, d = t.shape
                sid = cfg.scale_id_map[k]
                te = t + self.scale_emb.weight[sid].view(1, 1, -1)
                te = te + self._pos(L, t.device).view(1, L, -1)
                if self.time_enc is not None:
                    delta = self._physical_delta(k, L, t.device, t.dtype)
                    te = te + self.time_enc(delta)
                tok_list.append(te)
            cat = torch.cat(tok_list, dim=1)
            _, cls = self.joint(cat)
            parts.append(cls)

        # --- moe ---
        if self.use_moe and zs_ordered:
            # MoE 按 config.scale_keys 对齐长度
            zs_full = []
            for k in cfg.scale_keys:
                if k in scale_summary:
                    zs_full.append(scale_summary[k])
                else:
                    # 占位零（被 gate 学成忽略）
                    ref = zs_ordered[0]
                    zs_full.append(torch.zeros_like(ref))
            u_moe, gate = self.moe(zs_full)
            parts.append(u_moe)
            aux["moe_gate"] = gate

        # --- gru ---
        if self.use_gru:
            zs_c2f = []
            for k in cfg.fusion_order:  # m30→…→m01
                if k in scale_summary:
                    zs_c2f.append(scale_summary[k])
            if zs_c2f:
                parts.append(self.sgru(zs_c2f))

        if not parts:
            # fallback
            u = zs_ordered[0]
        else:
            u = parts[0]
            for p in parts[1:]:
                u = u + p

        if usable is not None:
            u = u + self.usable_emb(usable.float().view(-1, 1))

        return self.out_norm(u), aux


# ======================================================================
# B5 — 截面块 ISAB / full-attn / off
# ======================================================================

class CSNorm(nn.Module):
    """沿股票维 masked z-score。"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (N,d); mask: (N,) True=有效
        m = mask.float().unsqueeze(-1)
        n = m.sum(dim=0).clamp_min(1.0)
        mean = (x * m).sum(dim=0, keepdim=True) / n
        var = (((x - mean) ** 2) * m).sum(dim=0, keepdim=True) / n
        y = (x - mean) / (var.sqrt().clamp_min(self.eps))
        y = y * self.gamma + self.beta
        return torch.nan_to_num(y) * m + x * (1 - m)


class MAB(nn.Module):
    """Set Transformer MAB(Q, K)。"""

    def __init__(self, d_model: int, nhead: int, ffn_ratio: int, dropout: float, attn_dropout: float):
        super().__init__()
        self.n_q = nn.LayerNorm(d_model)
        self.n_k = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttn(d_model, nhead, dropout=attn_dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, ratio=ffn_ratio, dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = q + self.attn(self.n_q(q), self.n_k(k), self.n_k(k), key_padding_mask=key_padding_mask)
        return y + self.ffn(self.n2(y))


class ISAB(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_inducing: int,
        nhead: int,
        ffn_ratio: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, nhead, ffn_ratio, dropout, attn_dropout)
        self.mab2 = MAB(d_model, nhead, ffn_ratio, dropout, attn_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (1, N, d); mask: (N,)
        B = x.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, x, key_padding_mask=mask.unsqueeze(0).expand(B, -1))
        return self.mab2(x, H, key_padding_mask=None)


class PMA(nn.Module):
    """Pooling by Multihead Attention → 市场向量。"""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = MultiHeadAttn(d_model, nhead=nhead, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        q = self.seed.expand(B, -1, -1)
        return self.attn(q, x, x, key_padding_mask=mask.unsqueeze(0).expand(B, -1)).squeeze(1)


class CrossSectionBlock(nn.Module):
    """
    cs_block: off | ISAB | full-attn
    输出:
      - market_residual=True: (N, 2d)
      - 否则: (N, d)
    """

    def __init__(self, config):
        super().__init__()
        self.mode = str(config.cs_block).lower()
        d = config.d_model
        self.stock_dropout = float(config.stock_dropout)
        self.market_residual = bool(config.market_residual)
        self.use_cs_norm = bool(config.cs_norm)

        self.cs_norm = CSNorm(d) if self.use_cs_norm else None

        if self.mode == "isab":
            self.layers = nn.ModuleList(
                [
                    ISAB(
                        d, config.n_inducing, config.nhead,
                        config.ffn_ratio, config.dropout, config.attn_dropout,
                    )
                    for _ in range(config.cs_layers)
                ]
            )
        elif self.mode in ("full-attn", "full", "full_attn"):
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.nhead,
                dim_feedforward=d * config.ffn_ratio,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.layers = nn.TransformerEncoder(layer, num_layers=max(config.cs_layers, 1))
        elif self.mode in ("off", "none", "0"):
            self.layers = None
        else:
            raise ValueError(f"未知 cs_block={config.cs_block}")

        self.pma = PMA(d, nhead=1, dropout=config.dropout) if self.market_residual else None
        self.market_proj = nn.Linear(d, d) if self.market_residual else None
        self.norm = nn.LayerNorm(d)
        self.out_dim = 2 * d if (self.market_residual and self.layers is not None) else d

    def forward(self, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # u: (N,d); mask: (N,) True=有效
        if self.layers is None:
            return u

        m = mask
        if self.training and self.stock_dropout > 0:
            keep = torch.rand_like(m.float()) > self.stock_dropout
            m = m & keep
            # 至少一个有效
            if not bool(m.any()):
                m = mask

        x = u
        if self.cs_norm is not None:
            x = self.cs_norm(x, m)

        x = x.unsqueeze(0)  # (1,N,d)
        if self.mode == "isab":
            for layer in self.layers:
                x = layer(x, m)
            x = x.squeeze(0)
        else:
            key_pad = ~m.unsqueeze(0)
            x = self.layers(x, src_key_padding_mask=key_pad).squeeze(0)

        x = self.norm(x)
        if self.market_residual:
            mvec = self.pma(x.unsqueeze(0), m).squeeze(0)  # (d,)
            # 扩到 (N,d)
            mvec = mvec.unsqueeze(0).expand_as(x)
            resid = x - self.market_proj(mvec)
            return torch.cat([x, resid], dim=-1)
        return x


# ======================================================================
# B6 头（放 Modules 便于 Model 组装；也可只在 Model 里定义）
# ======================================================================

class PredictHead(nn.Module):
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1,
                 use_decile: bool = True, use_quantile: bool = False, n_quantiles: int = 9):
        super().__init__()
        self.rank_head = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.ret_head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1),
        )
        self.decile_head = nn.Linear(in_dim, 10) if use_decile else None
        self.quantile_head = nn.Linear(in_dim, n_quantiles) if use_quantile else None

    def forward(self, z: torch.Tensor):
        pred_rank = self.rank_head(z).squeeze(-1)
        pred_ret = self.ret_head(z).squeeze(-1)
        aux = {}
        if self.decile_head is not None:
            aux["decile_logits"] = self.decile_head(z)
        if self.quantile_head is not None:
            aux["quantile_pred"] = self.quantile_head(z)
        return pred_rank, pred_ret, aux


# ======================================================================
# 跨尺度去冗余（Barlow 式，供 Loss 调用也可在此计算）
# ======================================================================

def scale_decorr_loss(zs: List[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """L_dec = Σ_{s≠s'} ||offdiag(corr(z_s, z_s'))||^2"""
    if len(zs) < 2:
        return torch.zeros((), device=zs[0].device if zs else "cpu")
    loss = zs[0].new_zeros(())
    pair = 0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            a = zs[i] - zs[i].mean(dim=0, keepdim=True)
            b = zs[j] - zs[j].mean(dim=0, keepdim=True)
            a = a / (a.norm(dim=0, keepdim=True).clamp_min(eps))
            b = b / (b.norm(dim=0, keepdim=True).clamp_min(eps))
            corr = (a.T @ b) / max(a.size(0), 1)  # (d,d)
            off = corr - torch.diag(torch.diag(corr))
            loss = loss + off.pow(2).sum()
            pair += 1
    return loss / max(pair, 1)


__all__ = [
    "DropPath",
    "MLP",
    "MultiHeadAttn",
    "AttnPool",
    "FiLM",
    "NormGate",
    "FieldTemporalCNN",
    "FTAxialEncoder",
    "BookAxialEncoder",
    "ScaleIntraFusion",
    "ScaleIntraFusionV2",
    "CrossScaleFusion",
    "CrossSectionBlock",
    "PredictHead",
    "scale_decorr_loss",
]
