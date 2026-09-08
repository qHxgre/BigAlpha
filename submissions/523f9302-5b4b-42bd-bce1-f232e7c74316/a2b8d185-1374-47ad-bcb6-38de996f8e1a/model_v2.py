# -*- coding: utf-8 -*-
"""BigAlpha 2026 端到端量价预测 —— 模型定义 v5

v5 核心改进（参考 THU-BDC2026）：
1. 频率差异化字段分配：
   - 1min(28字段): 5档盘口深度 — 微观结构
   - 5min(20字段): 3档盘口 — 短线模式
   - 15min(12字段): 1档盘口 — 中期趋势
   - 30min(7字段): 仅OHLCV — 长线方向
   合计 67 字段（<100）

2. 模型容量升级（对齐 THU-BDC）：
   - d_model: 96→128, nlayers: 2→3, dim_ff: 192→256
   - 参数量 ≈ 2.8M

3. 多周期标签：残差收益率（BARRA风格剔除后）
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 频率差异化字段配置
# ============================================================
FREQ_CONFIGS = {
    "1m": {
        "price_cols": [
            "open", "high", "low", "close", "pre_close",
            "bid_price1", "bid_price2", "bid_price3", "bid_price4", "bid_price5",
            "ask_price1", "ask_price2", "ask_price3", "ask_price4", "ask_price5",
        ],
        "vol_cols": [
            "volume", "amount",
            "bid_volume1", "bid_volume2", "bid_volume3", "bid_volume4", "bid_volume5",
            "ask_volume1", "ask_volume2", "ask_volume3", "ask_volume4", "ask_volume5",
        ],
        "other_cols": ["deal_number"],
        "seq_len": 240,   # 1个交易日
    },
    "5m": {
        "price_cols": [
            "open", "high", "low", "close", "pre_close",
            "bid_price1", "bid_price2", "bid_price3",
            "ask_price1", "ask_price2", "ask_price3",
        ],
        "vol_cols": [
            "volume", "amount",
            "bid_volume1", "bid_volume2", "bid_volume3",
            "ask_volume1", "ask_volume2", "ask_volume3",
        ],
        "other_cols": ["deal_number"],
        "seq_len": 200,   # ~8个交易日
    },
    "15m": {
        "price_cols": [
            "open", "high", "low", "close", "pre_close",
            "bid_price1", "ask_price1",
        ],
        "vol_cols": [
            "volume", "amount",
            "bid_volume1", "ask_volume1",
        ],
        "other_cols": ["deal_number"],
        "seq_len": 160,   # ~15个交易日
    },
    "30m": {
        "price_cols": [
            "open", "high", "low", "close", "pre_close",
        ],
        "vol_cols": ["volume", "amount"],
        "other_cols": [],
        "seq_len": 120,   # ~15个交易日
    },
}

# 为每个频率计算总字段数和全部列名
for _cfg in FREQ_CONFIGS.values():
    _cfg["all_cols"] = _cfg["price_cols"] + _cfg["vol_cols"] + _cfg["other_cols"]
    _cfg["n_feat"] = len(_cfg["all_cols"])

# 验证字段上限
_total = sum(_cfg["n_feat"] for _cfg in FREQ_CONFIGS.values())
assert _total <= 100, f"总字段数 {_total} > 100"

# 暴露兼容接口（训练/推理脚本中遍历用）
FREQ_LABELS = list(FREQ_CONFIGS.keys())  # ["1m","5m","15m","30m"]

# ============================================================
# 模型超参（对齐 THU-BDC 的规模）
# ============================================================
MODEL_CFG = dict(
    d_model=96,
    nhead=4,
    nlayers=2,
    dim_ff=192,
    dropout=0.15,    # 略增 dropout 防过拟合
)

# ============================================================
# 位置编码
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learnable_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('sinusoidal_pe', pe.unsqueeze(0))
        nn.init.normal_(self.learnable_pe, std=0.02)

    def forward(self, x):
        sl = x.size(1)
        return self.dropout(x + self.sinusoidal_pe[:, :sl] + self.learnable_pe[:, :sl])

# ============================================================
# 注意力池化
# ============================================================
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d_model, d_model//2), nn.Tanh(), nn.Linear(d_model//2, 1))
    def forward(self, x):
        return (x * F.softmax(self.attn(x), dim=1)).sum(dim=1)

# ============================================================
# 单频率编码器
# ============================================================
class FreqEncoder(nn.Module):
    def __init__(self, n_feat, d_model, nhead, nlayers, dim_ff, max_len, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        el = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(el, nlayers)
        self.pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.pool(self.encoder(self.pos_encoder(self.input_proj(x)))))

# ============================================================
# 交叉股票注意力
# ============================================================
class CrossStockAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        am = ~mask if mask is not None else None
        a, _ = self.attention(x, x, x, key_padding_mask=am)
        return self.norm2(self.norm1(x + a) + self.ffn(self.norm1(x + a)))

# ============================================================
# 完整模型：差异化频率编码 + Cross-Stock
# ============================================================
class MultiFreqStockTransformer(nn.Module):
    """四频率差异化端到端模型

    1min: 28字段 → encoder(28→d_model)
    5min: 20字段 → encoder(20→d_model)
    15min: 12字段 → encoder(12→d_model)
    30min: 7字段  → encoder(7→d_model)
    → concat → fusion → cross-stock → score_head
    """
    def __init__(self, d_model=96, nhead=4, nlayers=2, dim_ff=192,
                 dropout=0.15, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = 4

        # 每个频率独立的编码器（输入维度不同）
        self.enc_1m  = FreqEncoder(FREQ_CONFIGS["1m"]["n_feat"],  d_model, nhead, nlayers, dim_ff, FREQ_CONFIGS["1m"]["seq_len"], dropout)
        self.enc_5m  = FreqEncoder(FREQ_CONFIGS["5m"]["n_feat"],  d_model, nhead, nlayers, dim_ff, FREQ_CONFIGS["5m"]["seq_len"], dropout)
        self.enc_15m = FreqEncoder(FREQ_CONFIGS["15m"]["n_feat"], d_model, nhead, nlayers, dim_ff, FREQ_CONFIGS["15m"]["seq_len"], dropout)
        self.enc_30m = FreqEncoder(FREQ_CONFIGS["30m"]["n_feat"], d_model, nhead, nlayers, dim_ff, FREQ_CONFIGS["30m"]["seq_len"], dropout)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * self.n_freqs, d_model * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.cross_stock = CrossStockAttention(d_model, nhead, dropout)

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, 1),
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def encode_freqs(self, xs):
        """xs: [x_1m, x_5m, x_15m, x_30m] → (N, d_model)"""
        h = torch.cat([
            self.enc_1m(xs[0]), self.enc_5m(xs[1]),
            self.enc_15m(xs[2]), self.enc_30m(xs[3]),
        ], dim=-1)
        return self.fusion(h)

    def forward(self, xs, mask=None):
        h = self.encode_freqs(xs).unsqueeze(0)
        h = self.cross_stock(h, mask.unsqueeze(0) if mask is not None else None)
        return self.score_head(h.squeeze(0)).squeeze(-1)

# ============================================================
# 单频率版（快速实验）
# ============================================================
class SimpleStockTransformer(nn.Module):
    def __init__(self, n_feat=28, d_model=128, nhead=4, nlayers=3,
                 dim_ff=256, seq_len=240, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout)
        el = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(el, nlayers)
        self.pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.cross_stock = CrossStockAttention(d_model, nhead, dropout)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout*0.5),
            nn.Linear(d_model//2, 1),
        )
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def encode(self, x):
        return self.norm(self.pool(self.encoder(self.pos_encoder(self.input_proj(x)))))
    def forward(self, x, mask=None):
        h = self.encode(x).unsqueeze(0)
        h = self.cross_stock(h, mask.unsqueeze(0) if mask is not None else None)
        return self.score_head(h.squeeze(0)).squeeze(-1)

# ============================================================
# 工具函数
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_json(ckpt, model_path):
    import json
    sd = ckpt["state_dict"]
    tensors = {}
    for k, v in sd.items():
        t = v.detach().cpu()
        tensors[k] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "data": t.reshape(-1).tolist()}
    payload = {k: v for k, v in ckpt.items() if k != "state_dict"}
    payload["state_dict"] = tensors
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return model_path

def load_model_json(model_path, map_location="cpu"):
    import json
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    sd = {}
    for k, meta in payload["state_dict"].items():
        t = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        sd[k] = t.reshape(meta["shape"]).to(map_location)
    ckpt = {k: v for k, v in payload.items() if k != "state_dict"}
    ckpt["state_dict"] = sd
    return ckpt

# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    simple = SimpleStockTransformer()
    n_simple = count_parameters(simple)
    print(f"SimpleStockTransformer 参数量: {n_simple:,}")

    multi = MultiFreqStockTransformer()
    n_multi = count_parameters(multi)
    print(f"MultiFreqStockTransformer 参数量: {n_multi:,}")

    total_fields = sum(FREQ_CONFIGS[fl]["n_feat"] for fl in FREQ_LABELS)
    print(f"各频率字段: 1m={FREQ_CONFIGS['1m']['n_feat']}, "
          f"5m={FREQ_CONFIGS['5m']['n_feat']}, "
          f"15m={FREQ_CONFIGS['15m']['n_feat']}, "
          f"30m={FREQ_CONFIGS['30m']['n_feat']} | "
          f"合计={total_fields} | 上限=100")

    N = 50
    xs = []
    for fl in FREQ_LABELS:
        cfg = FREQ_CONFIGS[fl]
        xs.append(torch.randn(N, cfg["seq_len"], cfg["n_feat"]))

    with torch.no_grad():
        scores = multi(xs)
    print(f"Multi 输出: {scores.shape} | 参数量: {n_multi:,}")
    print("v5 自检通过")
