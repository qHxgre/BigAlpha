# -*- coding: utf-8 -*-
"""M3：plain+SiLU+残差 / FixedOpStem 主路（对齐本地 plan5 silu_res2 / fixedop）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_conv_act(kind: str) -> nn.Module:
    k = (kind or "gelu").lower()
    if k == "silu":
        return nn.SiLU()
    if k == "relu":
        return nn.ReLU()
    return nn.GELU()


def _apply_conv_act(kind: str, x: torch.Tensor) -> torch.Tensor:
    k = (kind or "gelu").lower()
    if k == "silu":
        return F.silu(x)
    if k == "relu":
        return F.relu(x)
    return F.gelu(x)


class TemporalResBlock(nn.Module):
    """同分辨率残差：h + Conv→Act→Conv。"""

    def __init__(self, channels: int, kernel_size: int, conv_act: str = "gelu") -> None:
        super().__init__()
        k = int(kernel_size)
        pad = k // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, stride=1, padding=pad)
        self.act = _make_conv_act(conv_act)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, stride=1, padding=pad)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.conv2(self.act(self.conv1(h)))


class FixedOpStem(nn.Module):
    """零参数固定算子 bank + 1×1 融合。"""

    def __init__(
        self,
        n_feat: int,
        out_ch: int,
        scales: tuple[int, ...] = (5, 15, 60),
    ) -> None:
        super().__init__()
        self.scales = tuple(int(s) for s in scales)
        self.log_tau = nn.Parameter(torch.zeros(1, n_feat, 1))
        n_ops = 1 + 1 + len(self.scales) + 1 + 1
        self.fuse = nn.Conv1d(n_feat * n_ops, out_ch, kernel_size=1)

    @staticmethod
    def _causal_avg(h: torch.Tensor, w: int) -> torch.Tensor:
        return F.avg_pool1d(F.pad(h, (w - 1, 0), mode="replicate"), w, stride=1)

    @staticmethod
    def _lag_diff(h: torch.Tensor, lag: int = 1) -> torch.Tensor:
        return h - F.pad(h, (lag, 0), mode="replicate")[..., :-lag]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        d = self._lag_diff(h, 1)
        ops: list[torch.Tensor] = [h, d]
        ops += [h - self._causal_avg(h, w) for w in self.scales]
        w0 = self.scales[0]
        m = self._causal_avg(h, w0)
        var = (self._causal_avg(h * h, w0) - m * m).clamp_min(0.0)
        ops.append(var.sqrt())
        tau = self.log_tau.exp().clamp_min(1e-3)
        ops.append(torch.tanh(d / tau))
        return self.fuse(torch.cat(ops, dim=1))


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(h), dim=1)
        return (h * w).sum(dim=1)


class TemporalConvEncoder(nn.Module):
    """maxpool 下采样；支持 plain(+res) / fixedop 主路。"""

    def __init__(
        self,
        n_feat: int = 25,
        out_dim: int = 128,
        t_compressed: int = 240,
        t_intraday: int = 480,
        kernel_size: int = 5,
        mid_channels: int = 64,
        conv_act: str = "silu",
        stem_kind: str = "plain",
        n_res_blocks: int = 0,
        downsample_mode: str = "maxpool",
        **_kwargs,
    ) -> None:
        super().__init__()
        del downsample_mode
        self.t_intraday = int(t_intraday)
        self.t_compressed = int(t_compressed)
        self.stem_kind = (stem_kind or "plain").lower()
        self.conv_act = (conv_act or "gelu").lower()
        self.n_res_blocks = max(0, int(n_res_blocks))
        k = int(kernel_size)
        pad = k // 2
        mid = int(mid_channels)
        self.out_dim = out_dim

        if self.stem_kind == "fixedop":
            self.fixed_ops: FixedOpStem | None = FixedOpStem(n_feat, mid)
            self.stem_conv: nn.Conv1d | None = None
        else:
            self.fixed_ops = None
            self.stem_conv = nn.Conv1d(n_feat, mid, kernel_size=k, stride=1, padding=pad)
        self.stem_act = _make_conv_act(self.conv_act)
        self.res_blocks = (
            nn.ModuleList(
                [TemporalResBlock(mid, k, self.conv_act) for _ in range(self.n_res_blocks)]
            )
            if self.n_res_blocks > 0
            else None
        )
        # 480→240 或 240→120：一级 2× MaxPool
        ratio = self.t_intraday // self.t_compressed
        n_stages = 0
        r = ratio
        while r > 1:
            r //= 2
            n_stages += 1
        if n_stages < 1:
            raise ValueError(
                f"需要下采样 t_intraday={t_intraday}→t_compressed={t_compressed}"
            )
        self.pools = nn.ModuleList(
            [nn.MaxPool1d(kernel_size=2, stride=2) for _ in range(n_stages)]
        )
        self.proj = nn.Sequential(
            nn.Conv1d(mid, out_dim, kernel_size=k, stride=1, padding=pad),
            _make_conv_act(self.conv_act),
        )

    def _stem(self, h: torch.Tensor) -> torch.Tensor:
        if self.stem_kind == "fixedop":
            assert self.fixed_ops is not None
            return self.stem_act(self.fixed_ops(h))
        assert self.stem_conv is not None
        return self.stem_act(self.stem_conv(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] → [B, T', D]
        h = x.transpose(1, 2)
        h = self._stem(h)
        if self.res_blocks is not None:
            for blk in self.res_blocks:
                h = blk(h)
        for pool in self.pools:
            h = pool(h)
        h = self.proj(h)
        return h.transpose(1, 2)


class TemporalConvGRUModel(nn.Module):
    """按日卷积压缩 → GRU → last/mean/max/attn readout → 标量分数。"""

    needs_industry = False

    def __init__(
        self,
        n_feat: int = 25,
        n_day: int = 1,
        t_intraday: int = 480,
        t_compressed: int = 240,
        conv_out: int = 128,
        conv_mid: int = 64,
        conv_kernel: int = 5,
        downsample_mode: str = "maxpool",
        stem_kind: str = "plain",
        conv_act: str = "silu",
        n_res_blocks: int = 0,
        gru_hidden: int = 192,
        gru_layers: int = 3,
        gru_dropout: float = 0.10,
        bidirectional: bool = False,
        readout_kind: str = "last_mean_max_attn",
        head_dropout: float = 0.10,
        head_hidden: int = 64,
        head_kind: str = "shallow",
        **_kwargs,
    ) -> None:
        super().__init__()
        del head_kind, bidirectional, readout_kind
        self.n_day = max(1, int(n_day))
        self.t_intraday = int(t_intraday)
        self.t_compressed = int(t_compressed)
        self.conv = TemporalConvEncoder(
            n_feat=n_feat,
            out_dim=conv_out,
            t_compressed=self.t_compressed,
            t_intraday=self.t_intraday,
            kernel_size=conv_kernel,
            mid_channels=conv_mid,
            conv_act=conv_act,
            stem_kind=stem_kind,
            n_res_blocks=n_res_blocks,
            downsample_mode=downsample_mode,
        )
        self.gru = nn.GRU(
            input_size=conv_out,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        h_out = int(gru_hidden)
        self.time_attn = TemporalAttentionPool(h_out)
        head_in = h_out * 4
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def _encode_days(self, x: torch.Tensor) -> torch.Tensor:
        n, t_total, f = x.shape
        expect = self.n_day * self.t_intraday
        if t_total != expect:
            raise ValueError(f"期望 T={expect}，收到 T={t_total}")
        flat = x.reshape(n * self.n_day, self.t_intraday, f)
        enc = self.conv(flat)
        if enc.shape[1] != self.t_compressed:
            raise ValueError(f"卷积输出长度 {enc.shape[1]} != {self.t_compressed}")
        return enc.reshape(n, self.n_day * self.t_compressed, -1)

    def _readout(self, out: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                out[:, -1, :],
                out.mean(dim=1),
                out.max(dim=1).values,
                self.time_attn(out),
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor, industry: torch.Tensor | None = None) -> torch.Tensor:
        del industry
        h = self._encode_days(x.float())
        out, _ = self.gru(h)
        return self.head(self._readout(out)).squeeze(-1)


def build_m3(family: str, cfg: dict) -> TemporalConvGRUModel:
    return TemporalConvGRUModel(**cfg)
