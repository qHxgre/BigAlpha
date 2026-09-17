import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_sfa_group_ret(df: pd.DataFrame, group_num: int):
    """画分组收益图"""
    df = df.reset_index().sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])

    blues = plt.cm.Blues(np.linspace(0.25, 0.90, group_num))

    plt.figure(figsize=(12, 6))

    # 使用索引作为x轴坐标
    x_indices = df.index

    for i in range(group_num):
        col = str(i)
        if col in df.columns:
            plt.plot(x_indices, df[col].values, color=blues[i], linewidth=1.6, label=f"G{i}")

    if "ls" in df.columns:
        plt.plot(x_indices, df["ls"].values, label='ls', linestyle='--', color='red')

    ax = plt.gca()

    # 智能选择刻度数量（大约10-15个刻度）
    num_points = len(df)
    num_ticks = min(15, max(5, num_points // 10))  # 刻度数量在5-15之间

    # 均匀选择刻度位置
    tick_positions = np.linspace(0, num_points-1, num_ticks, dtype=int)

    # 设置刻度标签为对应的日期
    tick_labels = df.loc[tick_positions, 'date'].dt.strftime('%Y-%m-%d').tolist()

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    plt.title("Group Backtest Returns")
    plt.xlabel("Date")
    plt.ylabel("Net Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 将图像转换为Base64编码
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    c1 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return c1

def plot_avg_backtest(df: pd.DataFrame):
    df = df.reset_index().sort_values('trading_day').reset_index(drop=True)
    df['trading_day'] = pd.to_datetime(df['trading_day'])

    plt.figure(figsize=(14, 8))
    x_indices = df.index

    plt.plot(x_indices, df['portfolio_cumret'].values, color='#FF6B6B', linewidth=2, label='Portfolio')
    plt.plot(x_indices, df['benchmark_cumret'].values, color='#4ECDC4', linewidth=2, label='Benchmark')
    plt.plot(x_indices, df['excess_cumret'].values, color='#FFD166', linewidth=2, linestyle='--', label='Excess Return')

    ax = plt.gca()

    # 智能选择刻度
    num_points = len(df)
    num_ticks = min(12, max(6, num_points // 20))  # 调整刻度密度

    # 确保刻度位置合理
    tick_positions = np.linspace(0, num_points-1, num_ticks, dtype=int)

    # 设置刻度标签
    tick_labels = df.loc[tick_positions, 'trading_day'].dt.strftime('%Y-%m-%d').tolist()

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    # 添加辅助网格
    plt.grid(True, alpha=0.3, linestyle='--')

    # 设置标题和标签
    plt.title("Portfolio Backtest Performance", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Net Value / Return", fontsize=12)

    # 添加图例
    plt.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True)
    plt.tight_layout()

    # 将图像转换为Base64编码
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    c2 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return c2
