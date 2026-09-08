"""适配五档、20 维盘口输入的 Attention LOB。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (1, kernel), stride=(1, stride)
            ),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 192,
        num_heads: int = 8,
        head_dim: int = 8,
        output_dim: int = 64,
    ):
        super().__init__()
        embed_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(input_dim, embed_dim)
        self.k = nn.Linear(input_dim, embed_dim)
        self.v = nn.Linear(input_dim, embed_dim)
        self.out = nn.Linear(embed_dim, output_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        batch = query.size(0)
        q = self.q(query).view(
            batch, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k(key).view(
            batch, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v(value).view(
            batch, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        weights = F.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        result = (weights @ v).transpose(1, 2).contiguous()
        result = result.view(batch, -1, self.num_heads * self.head_dim)
        return self.out(result)


class AttentionLOB(nn.Module):
    """五档 Attention LOB。

    输入字段必须按每档 ``ask_price, ask_volume, bid_price, bid_volume`` 排列。
    特征宽度依次为 20 → 10 → 5 → 1。
    """

    def __init__(self, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.block1 = ConvBlock(1, 32, kernel=2, stride=2)
        self.block2 = ConvBlock(32, 32, kernel=2, stride=2)
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 5)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, (4, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.inception3 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, (3, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.inception5 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, (5, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.inception_pool = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
        )
        self.attention = MultiHeadAttention()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 20:
            raise ValueError(f"AttentionLOB expects (B,T,20), got {tuple(x.shape)}")
        x = x.unsqueeze(1)
        x = self.block3(self.block2(self.block1(x)))
        x = torch.cat(
            (self.inception3(x), self.inception5(x), self.inception_pool(x)),
            dim=1,
        )
        sequence = x.squeeze(-1).permute(0, 2, 1)
        context = self.attention(sequence[:, -1:, :], sequence, sequence).squeeze(1)
        return self.head(self.dropout(context))


class MultiTaskAttentionLOB(nn.Module):
    """共享完整 AttentionLOB 表征层，外接多 horizon 回归与分类头。"""

    def __init__(
        self,
        feature_dim: int = 64,
        head_dim: int = 32,
        dropout: float = 0.1,
        num_horizons: int = 3,
        num_classes: int = 5,
    ):
        super().__init__()
        self.feature_extractor = AttentionLOB(
            output_dim=feature_dim,
            dropout=dropout,
        )
        self.regression_heads = nn.ModuleList(
            [
                self._head(feature_dim, head_dim, 1, dropout)
                for _ in range(num_horizons)
            ]
        )
        self.classification_heads = nn.ModuleList(
            [
                self._head(feature_dim, head_dim, num_classes, dropout)
                for _ in range(num_horizons)
            ]
        )

    @staticmethod
    def _head(
        feature_dim: int,
        head_dim: int,
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        regression = torch.cat(
            [head(features) for head in self.regression_heads],
            dim=1,
        )
        classification = torch.stack(
            [head(features) for head in self.classification_heads],
            dim=1,
        )
        return {
            "features": features,
            "regression": regression,
            "classification": classification,
        }


class CompactAttentionLOB(nn.Module):
    """通道减半的五档 AttnLOB 主干，供真正的小号模型使用。"""
    def __init__(self, output_dim=32, dropout=0.15):
        super().__init__()
        self.block1=ConvBlock(1,16,2,2); self.block2=ConvBlock(16,16,2,2)
        self.block3=nn.Sequential(
            nn.Conv2d(16,16,(1,5)),nn.LeakyReLU(.01),
            nn.Conv2d(16,16,(4,1),padding="same"),nn.LeakyReLU(.01),
            nn.Conv2d(16,16,(4,1),padding="same"),nn.LeakyReLU(.01))
        def branch(kernel):
            return nn.Sequential(nn.Conv2d(16,32,(1,1),padding="same"),
              nn.LeakyReLU(.01),nn.Conv2d(32,32,(kernel,1),padding="same"),
              nn.LeakyReLU(.01))
        self.inc3=branch(3); self.inc5=branch(5)
        self.pool=nn.Sequential(nn.MaxPool2d((3,1),stride=1,padding=(1,0)),
          nn.Conv2d(16,32,(1,1),padding="same"),nn.LeakyReLU(.01))
        self.attention=MultiHeadAttention(96,4,8,32)
        self.head=nn.Linear(32,output_dim); self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        x=x.unsqueeze(1); x=self.block3(self.block2(self.block1(x)))
        x=torch.cat((self.inc3(x),self.inc5(x),self.pool(x)),dim=1)
        seq=x.squeeze(-1).permute(0,2,1)
        z=self.attention(seq[:,-1:,:],seq,seq).squeeze(1)
        return self.head(self.dropout(z))


class SmallMultiTaskAttentionLOB(nn.Module):
    def __init__(self,feature_dim=32,head_dim=16,dropout=.15,
                 num_horizons=3,num_classes=5):
        super().__init__(); self.feature_extractor=CompactAttentionLOB(feature_dim,dropout)
        make=MultiTaskAttentionLOB._head
        self.regression_heads=nn.ModuleList([make(feature_dim,head_dim,1,dropout) for _ in range(num_horizons)])
        self.classification_heads=nn.ModuleList([make(feature_dim,head_dim,num_classes,dropout) for _ in range(num_horizons)])
    def forward(self,x):
        f=self.feature_extractor(x)
        return {"features":f,
          "regression":torch.cat([h(f) for h in self.regression_heads],dim=1),
          "classification":torch.stack([h(f) for h in self.classification_heads],dim=1)}


class FusionMultiTaskAttentionLOB(nn.Module):
    """完整 AttnLOB 分支 + 双向 GRU 日内时序分支。"""

    def __init__(
        self,
        feature_dim: int = 128,
        head_dim: int = 64,
        dropout: float = 0.1,
        gru_hidden: int = 48,
        gru_layers: int = 2,
        num_horizons: int = 3,
        num_classes: int = 5,
    ):
        super().__init__()
        self.attention_lob = AttentionLOB(output_dim=64, dropout=dropout)
        self.gru = nn.GRU(
            input_size=20, hidden_size=gru_hidden, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        temporal_dim = gru_hidden * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(64 + temporal_dim * 3),
            nn.Linear(64 + temporal_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression_heads = nn.ModuleList(
            [MultiTaskAttentionLOB._head(feature_dim, head_dim, 1, dropout)
             for _ in range(num_horizons)]
        )
        self.classification_heads = nn.ModuleList(
            [MultiTaskAttentionLOB._head(feature_dim, head_dim, num_classes, dropout)
             for _ in range(num_horizons)]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        attention_features = self.attention_lob(x)
        sequence, hidden = self.gru(x)
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        temporal = torch.cat(
            (final, sequence.mean(dim=1), sequence.amax(dim=1)), dim=1
        )
        features = self.fusion(torch.cat((attention_features, temporal), dim=1))
        regression = torch.cat(
            [head(features) for head in self.regression_heads], dim=1
        )
        classification = torch.stack(
            [head(features) for head in self.classification_heads], dim=1
        )
        return {
            "features": features,
            "regression": regression,
            "classification": classification,
        }


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class TCNFusionMultiTaskLOB(nn.Module):
    """完整 AttnLOB 分支 + 多尺度膨胀 TCN 时序分支。"""

    def __init__(
        self, feature_dim: int = 128, head_dim: int = 64, dropout: float = 0.08,
        tcn_channels: int = 64, tcn_bins: int = 1,
        num_horizons: int = 3, num_classes: int = 5,
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    ):
        super().__init__()
        if tcn_bins < 1:
            raise ValueError("tcn_bins must be at least 1")
        self.tcn_bins = int(tcn_bins)
        self.attention_lob = AttentionLOB(output_dim=64, dropout=dropout)
        self.tcn_stem = nn.Sequential(
            nn.Conv1d(20, tcn_channels, 1), nn.GroupNorm(8, tcn_channels), nn.GELU()
        )
        self.tcn = nn.Sequential(
            *[TemporalResidualBlock(tcn_channels, dilation, dropout)
              for dilation in tcn_dilations]
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(64 + tcn_channels * (self.tcn_bins + 2)),
            nn.Linear(
                64 + tcn_channels * (self.tcn_bins + 2), feature_dim
            ),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.regression_heads = nn.ModuleList(
            [MultiTaskAttentionLOB._head(feature_dim, head_dim, 1, dropout)
             for _ in range(num_horizons)]
        )
        self.classification_heads = nn.ModuleList(
            [MultiTaskAttentionLOB._head(feature_dim, head_dim, num_classes, dropout)
             for _ in range(num_horizons)]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        attention_features = self.attention_lob(x)
        temporal = self.tcn(self.tcn_stem(x.transpose(1, 2)))
        time_bins = F.adaptive_avg_pool1d(
            temporal, self.tcn_bins
        ).flatten(start_dim=1)
        pooled = torch.cat(
            (temporal[:, :, -1], time_bins, temporal.amax(dim=2)), dim=1
        )
        features = self.fusion(torch.cat((attention_features, pooled), dim=1))
        regression = torch.cat(
            [head(features) for head in self.regression_heads], dim=1
        )
        classification = torch.stack(
            [head(features) for head in self.classification_heads], dim=1
        )
        return {
            "features": features, "regression": regression,
            "classification": classification,
        }


class SmallTCNFusionMultiTaskLOB(nn.Module):
    """小号 AttnLOB 主干 + 32 通道 TCN。"""
    def __init__(self,feature_dim=64,head_dim=32,dropout=.15,
                 tcn_channels=32,tcn_bins=1,num_horizons=3,num_classes=5,
                 tcn_dilations=(1,2,4,8,16)):
        super().__init__(); self.tcn_bins=int(tcn_bins)
        self.attention_lob=CompactAttentionLOB(32,dropout)
        self.tcn_stem=nn.Sequential(nn.Conv1d(20,tcn_channels,1),
          nn.GroupNorm(8,tcn_channels),nn.GELU())
        self.tcn=nn.Sequential(*[TemporalResidualBlock(tcn_channels,d,dropout)
                                 for d in tcn_dilations])
        width=32+tcn_channels*(self.tcn_bins+2)
        self.fusion=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,feature_dim),
                                  nn.GELU(),nn.Dropout(dropout))
        make=MultiTaskAttentionLOB._head
        self.regression_heads=nn.ModuleList([make(feature_dim,head_dim,1,dropout) for _ in range(num_horizons)])
        self.classification_heads=nn.ModuleList([make(feature_dim,head_dim,num_classes,dropout) for _ in range(num_horizons)])
    def forward(self,x):
        a=self.attention_lob(x); t=self.tcn(self.tcn_stem(x.transpose(1,2)))
        bins=F.adaptive_avg_pool1d(t,self.tcn_bins).flatten(start_dim=1)
        f=self.fusion(torch.cat((a,t[:,:,-1],bins,t.amax(dim=2)),dim=1))
        return {"features":f,
          "regression":torch.cat([h(f) for h in self.regression_heads],dim=1),
          "classification":torch.stack([h(f) for h in self.classification_heads],dim=1)}


class StyleRobustTCNFusionLOB(nn.Module):
    """AttnLOB + TCN，针对残差化后截面区分度的四头模型。"""

    def __init__(self, feature_dim=128, head_dim=64, dropout=0.10,
                 tcn_channels=64, tcn_bins=4):
        super().__init__()
        self.backbone = TCNFusionMultiTaskLOB(
            feature_dim=feature_dim, head_dim=head_dim, dropout=dropout,
            tcn_channels=tcn_channels, tcn_bins=tcn_bins,
        )
        del self.backbone.regression_heads
        del self.backbone.classification_heads
        make_head = MultiTaskAttentionLOB._head
        self.rank_day1 = make_head(feature_dim, head_dim, 1, dropout)
        self.cls10_day1 = make_head(feature_dim, head_dim, 10, dropout)
        self.top20_rank_day1 = make_head(feature_dim, head_dim, 1, dropout)
        self.rank_day2 = make_head(feature_dim, head_dim, 1, dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        attention = self.backbone.attention_lob(x)
        temporal = self.backbone.tcn(
            self.backbone.tcn_stem(x.transpose(1, 2))
        )
        bins = F.adaptive_avg_pool1d(
            temporal, self.backbone.tcn_bins
        ).flatten(start_dim=1)
        pooled = torch.cat(
            (temporal[:, :, -1], bins, temporal.amax(dim=2)), dim=1
        )
        features = self.backbone.fusion(torch.cat((attention, pooled), dim=1))
        return {
            "rank_day1": self.rank_day1(features).squeeze(1),
            "cls10_day1": self.cls10_day1(features),
            "top20_rank_day1": self.top20_rank_day1(features).squeeze(1),
            "rank_day2": self.rank_day2(features).squeeze(1),
        }


class TransformerFusionMultiTaskLOB(nn.Module):
    """完整 AttnLOB 分支 + 带绝对时点编码的降采样 Transformer。"""

    def __init__(
        self,
        feature_dim: int = 128,
        head_dim: int = 64,
        dropout: float = 0.12,
        transformer_dim: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 3,
        max_tokens: int = 128,
        num_horizons: int = 3,
        num_classes: int = 5,
    ):
        super().__init__()
        self.attention_lob = AttentionLOB(output_dim=64, dropout=dropout)
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(20, transformer_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, transformer_dim),
            nn.GELU(),
            nn.Conv1d(
                transformer_dim, transformer_dim, 5, stride=2, padding=2
            ),
            nn.GroupNorm(8, transformer_dim),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.position = nn.Parameter(
            torch.zeros(1, max_tokens, transformer_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(transformer_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        temporal_features = transformer_dim * 3
        self.fusion = nn.Sequential(
            nn.LayerNorm(64 + temporal_features),
            nn.Linear(64 + temporal_features, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression_heads = nn.ModuleList(
            [
                MultiTaskAttentionLOB._head(
                    feature_dim, head_dim, 1, dropout
                )
                for _ in range(num_horizons)
            ]
        )
        self.classification_heads = nn.ModuleList(
            [
                MultiTaskAttentionLOB._head(
                    feature_dim, head_dim, num_classes, dropout
                )
                for _ in range(num_horizons)
            ]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        attention_features = self.attention_lob(x)
        sequence = self.temporal_stem(x.transpose(1, 2)).transpose(1, 2)
        cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat((cls, sequence), dim=1)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError(
                f"Transformer tokens {tokens.shape[1]} exceed "
                f"max_tokens={self.position.shape[1]}"
            )
        encoded = self.temporal_norm(
            self.transformer(tokens + self.position[:, : tokens.shape[1]])
        )
        temporal = torch.cat(
            (
                encoded[:, 0],
                encoded[:, 1:].mean(dim=1),
                encoded[:, 1:].amax(dim=1),
            ),
            dim=1,
        )
        features = self.fusion(
            torch.cat((attention_features, temporal), dim=1)
        )
        regression = torch.cat(
            [head(features) for head in self.regression_heads], dim=1
        )
        classification = torch.stack(
            [head(features) for head in self.classification_heads], dim=1
        )
        return {
            "features": features,
            "regression": regression,
            "classification": classification,
        }


class _Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        for p in self.net.modules():
            if isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight, gain=0.5)
                nn.init.zeros_(p.bias)
    def forward(self, x): return self.net(x)


class PureTransformerMultiTaskLOB(nn.Module):
    """纯 Transformer（无 AttnLOB 分支），用于与 AttnLOB/TCN 对比。"""
    def __init__(self, feature_dim=96, head_dim=48, dropout=0.12,
                 transformer_dim=96, transformer_heads=4, transformer_layers=3,
                 max_seq=64, num_horizons=3, num_classes=5):
        super().__init__()
        self.stem = nn.Linear(20, transformer_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.position = nn.Parameter(torch.zeros(1, max_seq + 1, transformer_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(transformer_dim)
        pool_dim = transformer_dim * 3
        self.projector = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, feature_dim), nn.GELU(), nn.Dropout(dropout))
        self.regression_heads = nn.ModuleList(
            [_Head(feature_dim, head_dim, 1, dropout) for _ in range(num_horizons)])
        self.classification_heads = nn.ModuleList(
            [_Head(feature_dim, head_dim, num_classes, dropout) for _ in range(num_horizons)])

    def forward(self, x):
        B, T = x.shape[:2]
        x = self.stem(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.norm(self.transformer(x + self.position[:, :T + 1]))
        pooled = torch.cat([x[:, 0], x[:, 1:].mean(dim=1), x[:, 1:].amax(dim=1)], dim=1)
        feat = self.projector(pooled)
        regression = torch.cat([h(feat) for h in self.regression_heads], dim=1)
        classification = torch.stack([h(feat) for h in self.classification_heads], dim=1)
        return {'features': feat, 'regression': regression, 'classification': classification}
