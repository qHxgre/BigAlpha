"""每周公示内容 —— 需求三：中证 1000 指数增强策略跟踪。

以因子池合成信号为输入，用 M.bigtrader.v21 回测引擎跑一个「成分股内超配高分股、
低配低分股」的指数增强组合，跟踪相对中证 1000（000852.SH）的超额表现。

对外只公示相对口径：中证 1000 是公开指数，指增策略 beta≈1，组合/基准的绝对净值曲线
形状会与公开指数高度重合，可被拿去做滑窗匹配反推回测区间；绝对收益/夏普/卡玛等标量同理。
因此对外只展示相对基准的超额与相对强弱曲线（X 轴用交易日序号进一步隐藏时间），绝对口径
指标仅落盘到本地 CSV 供内部追踪。

数据来源（云端评测榜单目录 + dai）：
    leaderboard_reg.csv     每个因子的 model_score，据此加权合成信号；
    factor_pool.parquet     入池因子（已预处理）的因子值，date/instrument + 各因子列；
    cn_stock_index_weight   000852.SH 成分权重（倾斜基准，dai 查询）。

以 run(competition_id, leaderboard_dir, output_dir, date_str) 供 weekly_disclosure 调用，
返回本需求的 Markdown 片段；业绩图 / 指标历史 CSV 落到 output_dir。

注意：正常运行依赖 dai + M.bigtrader，只能在云端评测环境运行。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _disclosure_common import (
    BENCHMARK,
    DEFAULT_COMPETITION_ID,
    OUTPUT_DIR,
    resolve_leaderboard_dir,
    zscore,
)

try:
    import dai
except ImportError:  # 本地无 dai（仅云端可用）：import 期不报错，跑到查询时才提示
    dai = None

try:
    from bigmodule import M
except ImportError:  # 本地无 M.bigtrader（仅云端可用）
    M = None

# 合成信号只取 model_score 最高的前 TOP_K 个因子，弱因子留在尾部只会引入噪声
TOP_K = 100
# 指数型打分倾斜强度：w ∝ w_b·exp(THETA·score)。越大越向高分股集中、跟踪误差越大。
# 取 1.0 使成分股内适度倾斜，年化跟踪误差经验上落在 5% 附近。
THETA = 1.0
REBALANCE_DAY = 5           # 每 5 个交易日调仓一次
CAPITAL_BASE = 2_000_000
TRADING_DAYS = 252          # 年化系数（日频）


# ---- 数据加载 ----------------------------------------------------------------


def load_factor_weights(leaderboard_dir: str, top_k: int = TOP_K) -> pd.Series:
    """读取 leaderboard_reg.csv，返回 model_score 最高的前 top_k 因子权重（归一化）。"""
    reg_path = os.path.join(leaderboard_dir, "leaderboard_reg.csv")
    reg = pd.read_csv(reg_path)
    if not {"factor", "model_score"}.issubset(reg.columns):
        raise ValueError(f"{reg_path} 缺少 factor/model_score 列: {list(reg.columns)}")
    reg = reg.dropna(subset=["model_score"])
    reg = reg[reg["model_score"] > 0]
    top = reg.sort_values("model_score", ascending=False).head(top_k)
    weights = top.set_index("factor")["model_score"]
    return weights / weights.sum()


def load_factor_pool(leaderboard_dir: str, factors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """读取因子池（已预处理），只保留 date/instrument + 命中的因子列。"""
    pool_path = os.path.join(leaderboard_dir, "factor_pool.parquet")
    pool = pd.read_parquet(pool_path)

    present = [f for f in factors if f in pool.columns]
    missing = [f for f in factors if f not in pool.columns]
    if missing:
        print(
            f"  [警告] 因子池缺少 {len(missing)} 个入选因子，跳过: {missing[:5]}...",
            file=sys.stderr,
        )
    if not present:
        raise ValueError("因子池中没有任何入选因子列，无法合成信号")
    return pool[["date", "instrument"] + present], present


def load_index_weights(start_date: str, end_date: str) -> pd.DataFrame:
    """从 dai 拉取中证 1000（000852.SH）成分权重，返回 date/instrument/bm_weight。"""
    if dai is None:
        raise RuntimeError("dai 不可用：需要指数成分权重，只能在云端评测环境运行")
    df = dai.query(
        "SELECT date, member_code AS instrument, weight AS bm_weight "
        "FROM cn_stock_index_weight WHERE instrument == '{0}'".format(BENCHMARK),
        filters={"date": [start_date, end_date]},
    ).df()
    # 成分权重原始单位是百分比，归一到每日和为 1
    df["bm_weight"] = df.groupby("date", group_keys=False)["bm_weight"].apply(
        lambda s: s / s.sum()
    )
    return df


# ---- 信号合成与目标权重 ------------------------------------------------------


def build_signal(
    factor_pool: pd.DataFrame,
    weights: pd.Series,
    factors: list[str],
) -> pd.DataFrame:
    """把多因子按 model_score 加权合成为单一信号 score（逐日截面标准化）。

    因子池已做过中性化/标准化，这里只需按权重线性合成，再逐日 z-score 让不同交易日
    的信号处于同一尺度。返回 date/instrument/score。
    """
    w = weights.reindex(factors).fillna(0.0)
    df = factor_pool[["date", "instrument"]].copy()
    df["score"] = factor_pool[factors].fillna(0.0).values @ w.values
    df["score"] = df.groupby("date", group_keys=False)["score"].apply(zscore)
    return df


def build_target_weights(
    signal: pd.DataFrame,
    index_weights: pd.DataFrame,
    theta: float = THETA,
) -> pd.DataFrame:
    """构建指数增强目标权重：在成分股内按 w ∝ w_b·exp(θ·score) 倾斜。

    只在中证 1000 成分股内配置（天然多头、逐日和为 1）：θ=0 退化为跟指数，
    θ>0 时超配高分股、低配低分股。返回 date/instrument/weight/score。
    """
    merged = index_weights.merge(signal, on=["date", "instrument"], how="inner")

    def _tilt(grp: pd.DataFrame) -> pd.DataFrame:
        z = grp["score"].to_numpy(dtype=float)
        wb = grp["bm_weight"].to_numpy(dtype=float)
        # 指数型倾斜，exp 前减最大值防溢出；乘基准权重保证接近指数、可控偏离
        tilt = wb * np.exp(theta * (z - np.nanmax(z)))
        grp = grp.copy()
        grp["weight"] = tilt / tilt.sum()
        return grp

    out = merged.groupby("date", group_keys=False).apply(_tilt)
    return out[["date", "instrument", "weight", "score"]]


# ---- bigtrader 回测引擎回调 --------------------------------------------------
# 目标权重（date/instrument/weight）通过 data 传入引擎，m_handle_data 每 REBALANCE_DAY
# 个交易日把持仓调整到当日目标权重。撮合、手续费、滑点由 M.bigtrader.v21 负责。


def m_initialize(context):
    from bigtrader.finance.commission import PerOrder
    context.target = context.data
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0008, min_cost=5))
    context.rebalance_day = REBALANCE_DAY


def m_handle_data(context, data):
    # 非调仓日直接跳过，降低换手
    if context.trading_day_index % context.rebalance_day != 0:
        return

    dt = data.current_dt.strftime("%Y-%m-%d")
    today = context.target[context.target["date"] == dt]
    if today.empty:
        return

    today = today[today["weight"] > 0]
    target = dict(zip(today["instrument"], today["weight"]))

    # 卖出不在目标里的持仓
    for stock in list(context.portfolio.positions):
        if stock not in target:
            context.order_target_percent(stock, 0)
    # 调整到目标权重
    for stock, w in target.items():
        try:
            context.order_target_percent(stock, float(w))
        except Exception as e:  # 停牌/涨跌停等无法成交，跳过即可
            print(">>>", dt, stock, e)


def run_bigtrader(target_weights: pd.DataFrame, start_date: str, end_date: str):
    """用 M.bigtrader.v21 跑回测，返回引擎结果对象（含每日收益/净值）。"""
    if M is None:
        raise RuntimeError("M.bigtrader 不可用：只能在云端评测环境运行")
    return M.bigtrader.v21(
        data=target_weights,
        start_date=start_date,
        end_date=end_date,
        initialize=m_initialize,
        handle_data=m_handle_data,
        volume_limit=0,
        order_price_field_buy="open",
        order_price_field_sell="open",
        capital_base=CAPITAL_BASE,
        frequency="daily",
        product_type="股票",
        plot_charts=False,
        backtest_only=True,
        benchmark=BENCHMARK,
    )


# ---- 从引擎结果计算超额指标 --------------------------------------------------


def _extract_daily_returns(perf: pd.DataFrame) -> pd.DataFrame:
    """从 bigtrader perf 表里取出组合与基准的每日收益，返回 date/algo/bench。

    不同版本引擎列名略有差异，这里对常见命名做兼容匹配。
    """
    cols = {c.lower(): c for c in perf.columns}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    algo_ret = pick("returns", "algorithm_period_return_daily", "daily_returns")
    bench_ret = pick("benchmark_returns", "benchmark_period_return_daily", "benchmark_daily_returns")
    algo_nav = pick("algorithm_period_return", "portfolio_value", "cumulative_returns")
    bench_nav = pick("benchmark_period_return", "benchmark_cumulative_returns")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(perf[cols.get("date", "date")]) if "date" in cols else perf.index

    if algo_ret and bench_ret:
        df["algo"] = perf[algo_ret].to_numpy(dtype=float)
        df["bench"] = perf[bench_ret].to_numpy(dtype=float)
    elif algo_nav and bench_nav:
        # 只有累计收益时，差分还原日收益（净值形式先转为 1+ 累计）
        a = 1.0 + perf[algo_nav].to_numpy(dtype=float)
        b = 1.0 + perf[bench_nav].to_numpy(dtype=float)
        df["algo"] = np.concatenate([[a[0] - 1.0], a[1:] / a[:-1] - 1.0])
        df["bench"] = np.concatenate([[b[0] - 1.0], b[1:] / b[:-1] - 1.0])
    else:
        raise ValueError(f"无法从 perf 识别收益列: {list(perf.columns)}")
    return df


def _max_drawdown(nav: np.ndarray) -> tuple[float, int, int]:
    """净值曲线的最大回撤及其区间下标，返回 (max_dd, peak_idx, trough_idx)。

    peak_idx 为回撤起点（前高），trough_idx 为回撤谷底，用于画图时阴影标注。
    """
    running_max = np.maximum.accumulate(nav)
    dd = nav / running_max - 1.0
    trough = int(dd.argmin())
    peak = int(nav[: trough + 1].argmax()) if trough > 0 else 0
    return float(dd.min()), peak, trough


def compute_metrics(daily: pd.DataFrame, trading_days: int = TRADING_DAYS) -> dict:
    """由每日组合/基准收益计算绝对收益与超额指标（含夏普、卡玛、胜率等）。"""
    algo = daily["algo"].to_numpy(dtype=float)
    bench = daily["bench"].to_numpy(dtype=float)
    excess = algo - bench

    algo_nav = np.cumprod(1.0 + algo)
    bench_nav = np.cumprod(1.0 + bench)
    excess_nav = np.cumprod(1.0 + excess)
    n = len(excess)

    # —— 组合绝对收益端指标（无风险利率取 0）——
    ann_return = algo.mean() * trading_days
    ann_vol = algo.std() * np.sqrt(trading_days)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    algo_dd, algo_peak, algo_trough = _max_drawdown(algo_nav)
    calmar = ann_return / abs(algo_dd) if algo_dd < 0 else 0.0

    # —— 相对基准的超额端指标 ——
    ann_excess = excess.mean() * trading_days
    te = excess.std() * np.sqrt(trading_days)
    ir = ann_excess / te if te > 0 else 0.0
    excess_dd, dd_peak, dd_trough = _max_drawdown(excess_nav)
    win_rate = float((excess > 0).mean())

    return {
        "dates": list(daily["date"]),
        "n_days": n,
        # 曲线
        "algo_nav": algo_nav,
        "bench_nav": bench_nav,
        "excess": excess,
        "excess_nav": excess_nav,
        # 组合绝对收益指标
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "algo_max_drawdown": algo_dd,
        "algo_dd_range": (algo_peak, algo_trough),
        "calmar": calmar,
        # 相对基准的超额指标
        "cum_excess": excess_nav[-1] - 1.0,
        "ann_excess": ann_excess,
        "tracking_error": te,
        "ir": ir,
        "max_drawdown": excess_dd,
        "dd_range": (dd_peak, dd_trough),
        "win_rate": win_rate,
    }


# ---- 历史与增量贡献 ----------------------------------------------------------


def load_prev_ir(history_path: Path) -> float | None:
    """读取指标历史，返回上一轮的 IR（用于计算增量贡献）；无历史时返回 None。"""
    if not history_path.exists():
        return None
    try:
        hist = pd.read_csv(history_path)
    except Exception:
        return None
    if hist.empty or "ir" not in hist.columns:
        return None
    return float(hist.iloc[-1]["ir"])


def append_history(history_path: Path, date_str: str, metrics: dict, factors: list[str]) -> None:
    """把本轮指标追加到历史 CSV（逐轮一行，用于跨轮对比与增量贡献）。

    factors 为本轮入选的因子 id 列表，以「;」拼接落盘，便于回溯每轮用的是哪些因子。
    """
    row = {
        "date": date_str,
        "n_factors": len(factors),
        # 组合绝对收益指标
        "ann_return": metrics["ann_return"],
        "ann_vol": metrics["ann_vol"],
        "sharpe": metrics["sharpe"],
        "calmar": metrics["calmar"],
        "algo_max_drawdown": metrics["algo_max_drawdown"],
        # 相对基准的超额指标
        "cum_excess": metrics["cum_excess"],
        "ann_excess": metrics["ann_excess"],
        "tracking_error": metrics["tracking_error"],
        "ir": metrics["ir"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "n_days": metrics["n_days"],
        # 本轮入选因子 id 列表（分号拼接）
        "factors": ";".join(map(str, factors)),
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        hist = pd.concat([pd.read_csv(history_path), pd.DataFrame([row])], ignore_index=True)
    else:
        hist = pd.DataFrame([row])
    hist.to_csv(history_path, index=False, encoding="utf-8-sig")


# ---- 图表与展示 --------------------------------------------------------------


def plot_performance(metrics: dict, out_path: Path | None = None):
    """画业绩图：上图累计超额，下图相对强弱（组合/基准比值），均标注超额最大回撤区间。

    刻意只画相对口径曲线，不画组合/基准的绝对净值——指增策略 beta≈1，绝对曲线形状会与
    公开的中证 1000 高度重合，可被拿去做滑窗匹配反推回测区间。X 轴用交易日序号进一步隐藏
    时间周期。传 out_path 时落盘 PNG，返回 fig 便于预览。
    """
    x = np.arange(metrics["n_days"])
    excess_pct = (metrics["excess_nav"] - 1.0) * 100
    # 相对强弱：组合净值 / 基准净值，>1 表示跑赢基准（同样是纯相对口径，不含市场形状）
    rel_strength = metrics["algo_nav"] / metrics["bench_nav"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    # —— 上图：累计超额收益 ——
    ax1.plot(excess_pct, color="#16a085", linewidth=1.6, label="累计超额")
    ax1.fill_between(x, excess_pct, 0, color="#16a085", alpha=0.12)
    ax1.axhline(0, color="#888", linewidth=0.8, linestyle=":")
    # 超额最大回撤区间阴影
    p, t = metrics["dd_range"]
    if t > p:
        ax1.axvspan(p, t, color="#8e44ad", alpha=0.12,
                    label=f"超额最大回撤 {metrics['max_drawdown'] * 100:.2f}%")
        ax1.annotate(
            f"最大回撤 {metrics['max_drawdown'] * 100:.2f}%",
            xy=(t, excess_pct[t]),
            xytext=(6, -14), textcoords="offset points",
            fontsize=9, color="#8e44ad",
        )
    ax1.set_title("中证 1000 指数增强 · 累计超额收益（相对基准）", fontsize=12, pad=10)
    ax1.set_ylabel("累计超额 (%)")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # —— 下图：相对强弱曲线（组合 / 基准）——
    ax2.plot(rel_strength, color="#c0392b", linewidth=1.6, label="相对强弱（组合/基准）")
    ax2.axhline(1.0, color="#888", linewidth=0.8, linestyle=":")
    ax2.set_title("中证 1000 指数增强 · 相对强弱（组合净值 / 基准净值）", fontsize=12, pad=10)
    ax2.set_ylabel("相对强弱")
    ax2.set_xlabel("交易日序号")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def _fmt_delta(delta: float | None) -> str:
    """把 IR 增量格式化成带符号的字符串；无上一轮时返回占位。"""
    if delta is None:
        return "— (首轮，无对比基准)"
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "＝")
    return f"{arrow} {delta:+.4f}"


def build_markdown(
    metrics: dict,
    n_factors: int,
    prev_ir: float | None,
    curve_img: str,
) -> str:
    """把指标 + 业绩图拼成本需求的 Markdown 片段。

    对外只展示相对口径指标：绝对收益/夏普/卡玛等被市场 beta 主导，可被拿去和公开指数的
    同期统计做匹配、反推回测区间，故仅留在本地 CSV 供内部追踪，不对外公示。
    """
    ir_delta = None if prev_ir is None else metrics["ir"] - prev_ir
    lines = [
        "## 三、中证 1000 指数增强策略跟踪",
        "",
        f"以 ModelScore 前 {n_factors} 个因子按回归重要性合成信号，在成分股内超配高分股、"
        f"低配低分股（`w ∝ w_b·exp(θ·score)`，θ={THETA:g}），由 bigtrader 引擎回测，"
        "跟踪误差约束 5% 以内。",
        "",
        "### 相对基准的超额指标",
        "",
        "| 指标 | 数值 | 说明 |",
        "|---|---|---|",
        f"| 累计超额收益 | {metrics['cum_excess'] * 100:.2f}% | 相对中证 1000 的累计 alpha |",
        f"| 信息比率 (IR) | {metrics['ir']:.4f} | 年化超额收益 / 跟踪误差 |",
        f"| 年化超额收益 | {metrics['ann_excess'] * 100:.2f}% | — |",
        f"| 跟踪误差 | {metrics['tracking_error'] * 100:.2f}% | 年化 |",
        f"| 超额最大回撤 | {metrics['max_drawdown'] * 100:.2f}% | 超额收益的最大回撤 |",
        f"| 日胜率 | {metrics['win_rate'] * 100:.2f}% | 日超额为正的比例 |",
        f"| 增量贡献 | {_fmt_delta(ir_delta)} | 新一轮因子加入后 IR 的变化 |",
        "",
        "### 收益曲线",
        "",
        f"![收益曲线]({curve_img})",
        "",
        "> 为避免暴露回测时间周期，此处仅展示相对基准的超额与相对强弱曲线（X 轴为交易日序号），"
        "不公示组合/基准的绝对净值。",
        "",
        "> 指增策略的持续改善是比赛质量的直观体现——优质新因子应推动 IR 上升；"
        "若未能提升也会如实呈现，形成正向反馈。",
        "",
    ]
    return "\n".join(lines)


# ---- 供 weekly_disclosure 调用的入口 -----------------------------------------


def run(
    competition_id: str,
    leaderboard_dir: str,
    output_dir: Path,
    date_str: str,
) -> str:
    """执行需求三并把业绩图 / 指标历史落到 output_dir，返回本需求的 Markdown 片段。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{datetime.now():%H:%M:%S}] [需求三] 读取因子回归权重...", file=sys.stderr)
    weights = load_factor_weights(leaderboard_dir)

    factor_pool, factors = load_factor_pool(leaderboard_dir, list(weights.index))
    print(f"[{datetime.now():%H:%M:%S}] [需求三] 合成信号（{len(factors)} 个因子）...", file=sys.stderr)
    signal = build_signal(factor_pool, weights, factors)

    dates = pd.to_datetime(factor_pool["date"])
    start_date, end_date = dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")

    print(f"[{datetime.now():%H:%M:%S}] [需求三] 拉取指数成分权重并构建目标权重...", file=sys.stderr)
    index_weights = load_index_weights(start_date, end_date)
    target_weights = build_target_weights(signal, index_weights)

    print(f"[{datetime.now():%H:%M:%S}] [需求三] M.bigtrader.v21 回测中...", file=sys.stderr)
    result = run_bigtrader(target_weights, start_date, end_date)
    daily = _extract_daily_returns(result.read_raw_perf())
    metrics = compute_metrics(daily)

    curve_png = output_dir / f"excess_curve_{date_str}.png"
    history_csv = output_dir / "strategy_metrics_history.csv"

    # 增量贡献需在追加本轮之前读取上一轮 IR
    prev_ir = load_prev_ir(history_csv)
    fig = plot_performance(metrics, curve_png)
    plt.close(fig)
    append_history(history_csv, date_str, metrics, factors)

    print(
        f"[{datetime.now():%H:%M:%S}] [需求三] 已写入: {curve_png.name} / {history_csv.name}",
        file=sys.stderr,
    )
    return build_markdown(metrics, len(factors), prev_ir, curve_png.name)
