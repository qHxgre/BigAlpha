"""
DisFT-GNN 因子挖掘系统 — 比赛入口
====================================
BigAlpha 2026 AI因子挖掘赛道（AI智能赛道）

合规声明：
- 全部训练在Notebook内完成，不上传预训练权重（规避"私自上传外部数据"违规）
- 随机种子固定，确保运行时审计可复现
- 教师模型使用未来标签（LUPI范式），代码独立隔离，main函数只调用学生推理
- AI技术应用关键环节已标注 [AI应用环节X]

main函数返回三列因子表：date, instrument, factor
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional

# ============================================================
# 模块导入
# ============================================================
from config import CONFIG
from feature_engineering import FeatureExtractor
from graph_construction import GraphBuilder, TimeSeriesSplitter, normalize_adj
from models import TeacherModel, StudentModel
from losses import StudentLossConfig, StudentTotalLoss
from training import (
    TeacherTrainer, StudentTrainer, DistillabilityValidator,
    extract_teacher_representations, set_seed,
)
from factor_output import (
    FactorProcessor, FactorValidator, generate_factor_table,
)


# ============================================================
# [AI应用环节标注汇总]
# 环节1: 自动化特征工程（feature_engineering.py - FeatureExtractor, LearnableFeatureAggregator）
# 环节2: 教师模型训练-未来信息编码（models.py - FutureTrendEncoder）
# 环节3: 多通道双线性融合（models.py - MultiChannelBilinearFusion）
# 环节4: HSIC蒸馏（losses.py - HSICLoss）
# 环节5: IC排序损失（losses.py - ICRankLoss, StudentTotalLoss）
# 环节6: 可蒸馏性验证（training.py - DistillabilityValidator）
# 环节7: 学生模型推理-时空GNN（models.py - StudentModel）
# 环节8: B项感知投影（models.py - StudentModel.factor_projection）
# ============================================================


# ============================================================
# 数据加载（适配BigQuant平台API）
# ============================================================

def load_data(start_date: str, end_date: str) -> Dict:
    """
    从BigQuant平台加载原始数据。

    需要根据实际平台API适配，以下为接口定义。
    BigQuant平台通常使用 dai.DataSource() 或类似API。

    Returns:
        data: {
            'dates': List[str],               # 交易日列表
            'stock_list': List[str],           # 股票代码列表
            'kline_1min': Dict,                # {date: {stock: DataFrame}}
            'orderbook': Dict,                 # {date: {stock: DataFrame}}
            'daily_returns': pd.DataFrame,     # 日频收益率
            'industry_map': Dict,              # {stock: industry}
            'market_cap': Dict,                # {stock: log_market_cap}
            'barra_factors': pd.DataFrame,     # BARRA风格因子（本地预检用）
        }
    """
    # ============================================================
    # 以下为BigQuant平台数据加载代码模板
    # 实际使用时需取消注释并适配平台API
    # ============================================================

    # import dai
    #
    # # 获取中证1000历史成分股
    # stock_list = list(dai.query(
    #     "SELECT instrument FROM cn_stock_index_component "
    #     "WHERE index_code='000852.SH' "
    #     "AND date >= '{}' AND date <= '{}'".format(start_date, end_date)
    # )['instrument'].unique())
    #
    # # 获取交易日历
    # dates = sorted(dai.query(
    #     "SELECT date FROM cn_stock_trading_calendar "
    #     "WHERE date >= '{}' AND date <= '{}' AND is_open=1".format(start_date, end_date)
    # )['date'].dt.strftime('%Y-%m-%d').unique())
    #
    # # 获取1分钟K线数据
    # kline_1min = {}
    # for date in dates:
    #     kline_df = dai.query(
    #         "SELECT * FROM cn_stock_bar_1min "
    #         "WHERE date = '{}' AND instrument IN ({})".format(
    #             date, ','.join(["'{}'".format(s) for s in stock_list])
    #         )
    #     )
    #     kline_1min[date] = {stock: group for stock, group in kline_df.groupby('instrument')}
    #
    # # 获取盘口快照数据
    # # orderbook = ...
    #
    # # 获取日频收益率
    # daily_returns = dai.query(
    #     "SELECT date, instrument, return FROM cn_stock_bar_1d "
    #     "WHERE date >= '{}' AND date <= '{}'".format(start_date, end_date)
    # )
    #
    # # 获取行业分类
    # industry_df = dai.query(
    #     "SELECT instrument, industry FROM cn_stock_industry "
    #     "WHERE date = '{}'".format(end_date)
    # )
    # industry_map = dict(zip(industry_df['instrument'], industry_df['industry']))
    #
    # # 获取市值数据
    # market_cap_df = dai.query(
    #     "SELECT instrument, log(market_cap) as log_cap FROM cn_stock_valuation "
    #     "WHERE date >= '{}' AND date <= '{}'".format(start_date, end_date)
    # )
    # market_cap = dict(zip(market_cap_df['instrument'], market_cap_df['log_cap']))

    # ============================================================
    # 模拟数据（开发测试用，实际提交时替换为平台API）
    # ============================================================
    print(f"[数据加载] 使用模拟数据进行测试 (start={start_date}, end={end_date})")

    np.random.seed(42)
    dates = pd.bdate_range(start_date, end_date).strftime('%Y-%m-%d').tolist()
    stock_list = [f"{i:06d}.SZ" for i in range(1, 101)]  # 100只测试用

    # 模拟日频收益率
    daily_returns = pd.DataFrame({
        'date': np.repeat(dates, len(stock_list)),
        'instrument': stock_list * len(dates),
        'return': np.random.randn(len(dates) * len(stock_list)) * 0.02,
    })

    # 模拟行业和市值
    industries = ['银行', '非银金融', '医药生物', '电子', '计算机', '机械', '化工', '食品饮料']
    industry_map = {s: np.random.choice(industries) for s in stock_list}
    market_cap = {s: np.random.uniform(20, 25) for s in stock_list}

    return {
        'dates': dates,
        'stock_list': stock_list,
        'daily_returns': daily_returns,
        'industry_map': industry_map,
        'market_cap': market_cap,
        # 以下为实际平台提供时使用
        'kline_1min': {},
        'orderbook': {},
        'barra_factors': None,
    }


# ============================================================
# 数据预处理
# ============================================================

def prepare_features_and_graphs(
    data: Dict,
    config=None,
) -> Dict:
    """
    [AI应用环节1] 特征工程 + [Layer 1] 图构建。

    将原始数据转化为模型可消费的特征序列和图序列。

    Returns:
        prepared: {
            'features': {date: tensor(N, L, M)},
            'adjs': {date: tensor(N, N)},
            'future_returns': {date: tensor(N,)},
            'future_labels': {date: tensor(N, T)},
            'stock_list': List[str],
            'dates': List[str],
        }
    """
    config = config or CONFIG
    m_config = config.model

    stock_list = data['stock_list']
    dates = data['dates']
    N = len(stock_list)
    L = m_config.lookback_short
    M = m_config.input_dim

    print(f"\n[预处理] 特征工程 + 图构建 (N={N}, L={L}, M={M})")

    # --- 特征提取 ---
    feature_extractor = FeatureExtractor()

    # 提取每日特征（实际平台使用分钟数据，这里用模拟）
    daily_features = {}
    for date in dates:
        # 实际平台：从kline_1min和orderbook提取
        # 模拟：随机生成特征
        feat = np.random.randn(N, M).astype(np.float32) * 0.1
        # 模拟部分停牌（用0填充而非NaN，实际平台数据需清洗）
        mask = np.random.random(N) < 0.05
        feat[mask] = 0.0
        daily_features[date] = feat

    # --- 图构建 ---
    graph_builder = GraphBuilder(config)
    graph_builder.set_static_info(data['industry_map'], data['market_cap'])

    # 获取日频收益率矩阵（用于相关性邻接矩阵）
    returns_pivot = data['daily_returns'].pivot_table(
        index='date', columns='instrument', values='return'
    ).reindex(columns=stock_list)

    daily_adjs = {}
    for i, date in enumerate(dates):
        # 使用截至前一日的收益计算相关性
        if i > 0:
            past_returns = returns_pivot.iloc[:i].values
            past_returns = np.nan_to_num(past_returns, nan=0.0)
        else:
            past_returns = None

        A_fused, _, _, _ = graph_builder.build_fused_adj(stock_list, past_returns)
        daily_adjs[date] = torch.tensor(A_fused, dtype=torch.float32)

    # --- 构建时序窗口 ---
    splitter = TimeSeriesSplitter(config)
    samples = splitter.create_rolling_windows(dates, lookback=L, future_horizon=config.train.future_horizon)

    # 构建每个样本的时序特征张量
    features_dict = {}
    adjs_dict = {}
    future_returns_dict = {}
    future_labels_dict = {}

    for sample in samples:
        t = sample['t']
        history = sample['history']

        # 构建时序特征 (N, L, M)
        feat_seq = np.zeros((N, L, M), dtype=np.float32)
        for j, h_date in enumerate(history):
            if h_date in daily_features:
                feat_seq[:, j, :] = daily_features[h_date]
        features_dict[t] = torch.tensor(feat_seq)

        # 邻接矩阵
        adjs_dict[t] = daily_adjs.get(t, torch.eye(N))

        # 未来收益率
        future_date = sample['future'][0]
        if future_date in returns_pivot.index:
            future_ret = returns_pivot.loc[future_date, stock_list].values
            future_ret = np.nan_to_num(future_ret, nan=0.0)
        else:
            future_ret = np.zeros(N)
        future_returns_dict[t] = torch.tensor(future_ret, dtype=torch.float32)

        # 未来标签（二值化）
        future_labels_dict[t] = (future_returns_dict[t] > config.train.future_threshold).float().unsqueeze(-1)  # (N, T)

    print(f"[预处理] 完成，共 {len(features_dict)} 个样本")

    return {
        'features': features_dict,
        'adjs': adjs_dict,
        'future_returns': future_returns_dict,
        'future_labels': future_labels_dict,
        'stock_list': stock_list,
        'dates': list(features_dict.keys()),
    }


def split_data(prepared: Dict, config=None) -> Dict:
    """按时间划分训练/验证/测试集"""
    config = config or CONFIG
    splitter = TimeSeriesSplitter(config)

    all_dates = prepared['dates']
    splits = splitter.split_dates(all_dates)

    def collect(date_list):
        features = torch.stack([prepared['features'][d] for d in date_list if d in prepared['features']])
        adjs = torch.stack([prepared['adjs'][d] for d in date_list if d in prepared['adjs']])
        future_returns = torch.stack([prepared['future_returns'][d] for d in date_list if d in prepared['future_returns']])
        future_labels = torch.stack([prepared['future_labels'][d] for d in date_list if d in prepared['future_labels']])
        dates = [d for d in date_list if d in prepared['features']]
        return features, adjs, future_returns, future_labels, dates

    train_data = collect(splits['train'])
    val_data = collect(splits['val'])
    test_data = collect(splits['test'])

    print(f"\n[数据划分] train={len(train_data[4])}, val={len(val_data[4])}, test={len(test_data[4])}")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'stock_list': prepared['stock_list'],
    }


# ============================================================
# 主函数（比赛入口）
# ============================================================

def main():
    """
    [比赛入口] DisFT-GNN因子生成主函数。

    全部训练在Notebook内完成（合规），3小时时间预算：
    - 数据加载+预处理: ~30分钟
    - 教师训练: ~35分钟
    - 可蒸馏性验证: ~5分钟
    - 学生训练: ~30分钟
    - 因子生成: ~25分钟
    合计: ~125分钟，留约35%安全余量。

    Returns:
        factor_df: pd.DataFrame[date, instrument, factor]
    """
    # ============================================================
    # [合规] 固定随机种子，确保审计可复现
    # ============================================================
    set_seed(CONFIG.train.seed)

    start_time = time.time()
    print("=" * 70)
    print("DisFT-GNN 因子挖掘系统")
    print("BigAlpha 2026 AI因子挖掘赛道（AI智能赛道）")
    print("=" * 70)

    # ============================================================
    # Step 1: 数据加载
    # ============================================================
    print("\n[Step 1] 数据加载...")
    data = load_data(CONFIG.data.train_start, CONFIG.data.test_end)

    # ============================================================
    # Step 2: [AI应用环节1] 特征工程 + 图构建
    # ============================================================
    print("\n[Step 2] 特征工程与图构建...")
    prepared = prepare_features_and_graphs(data, CONFIG)
    splits = split_data(prepared, CONFIG)

    train_feat, train_adj, train_ret, train_label, train_dates = splits['train']
    val_feat, val_adj, val_ret, val_label, val_dates = splits['val']
    test_feat, test_adj, test_ret, test_label, test_dates = splits['test']
    stock_list = splits['stock_list']

    device = torch.device('cuda' if torch.cuda.is_available() and CONFIG.train.device == 'cuda' else 'cpu')
    print(f"[设备] 使用: {device}")

    # ============================================================
    # [合规声明] 以下教师模型代码仅用于训练
    # 教师模型使用未来标签(future_labels)是LUPI范式的设计要求
    # 提交的因子仅由学生模型(仅用历史数据)生成
    # 详见AI技术应用说明文档第1.3节
    # ============================================================

    # ============================================================
    # Step 3: [AI应用环节2] 教师模型训练
    # ============================================================
    print("\n[Step 3] 教师模型训练...")
    teacher_trainer = TeacherTrainer(CONFIG)
    teacher_model = teacher_trainer.train(
        train_features=train_feat,
        train_adjs=train_adj,
        train_future_labels=train_label,
        train_future_returns=train_ret,
        val_features=val_feat,
        val_adjs=val_adj,
        val_future_labels=val_label,
        val_future_returns=val_ret,
        device=device,
    )

    # ============================================================
    # Step 4: [AI应用环节6] 可蒸馏性前置验证
    # ============================================================
    print("\n[Step 4] 可蒸馏性验证...")
    validator = DistillabilityValidator(
        input_dim=CONFIG.model.input_dim,
        hidden_dim=32,
        output_dim=CONFIG.model.d_out,
    )
    distill_result = validator.validate(
        teacher_model=teacher_model,
        train_features=train_feat,
        train_adjs=train_adj,
        train_future_labels=train_label,
        val_features=val_feat,
        val_adjs=val_adj,
        val_future_labels=val_label,
        device=device,
    )

    use_distillation = distill_result['decision'] != 'abort'
    if not use_distillation:
        print("[决策] 可蒸馏性R²过低，切换为纯时序GNN方案（无蒸馏）")

    # ============================================================
    # Step 5: 提取教师表征（蒸馏目标）
    # ============================================================
    teacher_train_repr = None
    teacher_val_repr = None
    if use_distillation:
        print("\n[Step 5] 提取教师表征...")
        teacher_train_repr = extract_teacher_representations(
            teacher_model, train_feat, train_adj, train_label, device
        )
        teacher_val_repr = extract_teacher_representations(
            teacher_model, val_feat, val_adj, val_label, device
        )

    # ============================================================
    # Step 6: [AI应用环节5] 学生模型训练（蒸馏+排序损失）
    # ============================================================
    print("\n[Step 6] 学生模型训练...")
    student_trainer = StudentTrainer(CONFIG, use_distillation=use_distillation)
    student_model = student_trainer.train(
        train_features=train_feat,
        train_adjs=train_adj,
        train_future_returns=train_ret,
        val_features=val_feat,
        val_adjs=val_adj,
        val_future_returns=val_ret,
        teacher_model=teacher_model if use_distillation else None,
        teacher_train_repr=teacher_train_repr,
        teacher_val_repr=teacher_val_repr,
        device=device,
    )

    # ============================================================
    # Step 7: [AI应用环节7] 因子生成
    # ============================================================
    print("\n[Step 7] 因子生成...")

    # 生成全部评估区间的因子
    all_dates = train_dates + val_dates + test_dates
    all_features = {}
    all_adjs = {}

    for i, date in enumerate(all_dates):
        if i < len(train_feat):
            all_features[date] = train_feat[i]
            all_adjs[date] = train_adj[i]
        elif i - len(train_dates) < len(val_feat):
            all_features[date] = val_feat[i - len(train_dates)]
            all_adjs[date] = val_adj[i - len(train_dates)]
        else:
            idx = i - len(train_dates) - len(val_dates)
            if idx < len(test_feat):
                all_features[date] = test_feat[idx]
                all_adjs[date] = test_adj[idx]

    factor_df = generate_factor_table(
        student_model=student_model,
        all_features=all_features,
        all_adjs=all_adjs,
        stock_list=stock_list,
        dates=all_dates,
        config=CONFIG,
        device=device,
    )

    # ============================================================
    # Step 8: 合规校验
    # ============================================================
    print("\n[Step 8] 合规校验...")
    validator = FactorValidator()

    # 格式校验
    assert validator.validate_output_format(factor_df), "因子输出格式不合规"

    # 覆盖度校验
    validator.validate_coverage(factor_df, CONFIG.data.max_missing_ratio)

    # 因子质量评估
    returns_df = data['daily_returns'].rename(columns={'return': 'return'})
    metrics = validator.compute_factor_metrics(factor_df, returns_df)
    print(f"\n[因子质量] IC_mean={metrics['ic_mean']:.4f}, IC_IR={metrics['ic_ir']:.4f}, "
          f"Sharpe={metrics['sharpe_ratio']:.4f}")

    # ============================================================
    # 完成
    # ============================================================
    elapsed = (time.time() - start_time) / 60
    print(f"\n{'='*70}")
    print(f"[完成] 总耗时: {elapsed:.1f}分钟")
    print(f"[输出] {len(factor_df)}行, {factor_df['date'].nunique()}个交易日")
    print(f"{'='*70}")

    return factor_df


# ============================================================
# 运行入口
# ============================================================

if __name__ == '__main__':
    factor_df = main()
    print("\n因子表预览:")
    print(factor_df.head(10))
    print(f"\n形状: {factor_df.shape}")
