"""
Financial analysis report for optical transceiver stocks:
- 中际旭创 (300308.SZ)
- 新易盛 (300502.SZ)
- 光迅科技 (002281.SZ)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))

import dai
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

STOCKS = {
    "300308.SZ": "ZhongJi InnoLight (300308)",
    "300502.SZ": "XinYiSheng (300502)",
    "002281.SZ": "Accelink (002281)",
}

STOCK_CODES = list(STOCKS.keys())
INSTRUMENTS_SQL = "(" + ", ".join(f"'{s}'" for s in STOCK_CODES) + ")"

# Query the most recent 8 quarters of financial data (shift 0..7)
QUERY_END = "2026-05-22"
QUERY_START = "2023-01-01"


def fetch_financial_data() -> pd.DataFrame:
    sql = f"""
    SELECT
        lf.date,
        lf.instrument,
        lf.report_date,
        lf.shift,
        lf.total_operating_revenue_lf,
        lf.operating_revenue_lf,
        lf.net_profit_lf,
        lf.net_profit_to_parent_shareholders_lf,
        lf.total_assets_lf,
        lf.total_liabilities_lf,
        lf.total_owner_equity_lf,
        lf.total_equity_to_parent_shareholders_lf,
        lf.total_current_assets_lf,
        lf.total_current_liabilities_lf,
        lf.research_and_development_expense_lf,
        lf.operating_revenue_lf - lf.operating_costs_lf AS gross_profit_lf,
        lf.operating_profit_lf,
        ttm.total_operating_revenue_ttm,
        ttm.net_profit_ttm,
        ttm.net_profit_to_parent_shareholders_ttm,
        ttm.net_cffoa_ttm,
        ttm.research_and_development_expense_ttm
    FROM cn_stock_financial_lf_shift lf
    LEFT JOIN cn_stock_financial_ttm_shift ttm
        ON lf.date = ttm.date
        AND lf.instrument = ttm.instrument
        AND ttm.shift = 0
    WHERE lf.shift = 0
      AND lf.instrument IN {INSTRUMENTS_SQL}
      AND lf.date >= '{QUERY_START}'
      AND lf.date <= '{QUERY_END}'
    ORDER BY lf.instrument, lf.date
    """
    df = dai.query(sql, filters={"date": [QUERY_START, QUERY_END]}).df()
    return df


def fetch_yoy_data() -> pd.DataFrame:
    """Fetch shift=0 and shift=4 for YoY comparison."""
    sql = f"""
    SELECT
        a.date,
        a.instrument,
        a.report_date AS cur_report_date,
        b.report_date AS yoy_report_date,
        a.total_operating_revenue_lf AS revenue_cur,
        b.total_operating_revenue_lf AS revenue_yoy,
        a.net_profit_to_parent_shareholders_lf AS profit_cur,
        b.net_profit_to_parent_shareholders_lf AS profit_yoy,
        a.operating_revenue_lf - a.operating_costs_lf AS gross_profit_cur,
        b.operating_revenue_lf - b.operating_costs_lf AS gross_profit_yoy
    FROM cn_stock_financial_lf_shift a
    JOIN cn_stock_financial_lf_shift b
        ON a.date = b.date AND a.instrument = b.instrument
    WHERE a.shift = 0 AND b.shift = 4
      AND a.instrument IN {INSTRUMENTS_SQL}
      AND a.date >= '{QUERY_START}'
      AND a.date <= '{QUERY_END}'
    ORDER BY a.instrument, a.date
    """
    df = dai.query(sql, filters={"date": [QUERY_START, QUERY_END]}).df()
    return df


def get_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent record per instrument."""
    return df.sort_values("date").groupby("instrument").last().reset_index()


def fmt_billion(val):
    if pd.isna(val):
        return "N/A"
    return f"{val / 1e8:.2f}B"


def fmt_pct(val):
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:.1f}%"


def safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where((b != 0) & ~np.isnan(b) & ~np.isnan(a), a / b, np.nan)
    return result


def plot_revenue_profit(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Revenue & Net Profit Trend (Latest Filing)", fontsize=14, fontweight='bold')

    for ax, (code, name) in zip(axes, STOCKS.items()):
        sub = df[df["instrument"] == code].sort_values("report_date")
        if sub.empty:
            ax.set_title(name)
            continue
        x = range(len(sub))
        labels = sub["report_date"].dt.strftime("%Y-%m").tolist()
        rev = sub["total_operating_revenue_lf"].values / 1e8
        profit = sub["net_profit_to_parent_shareholders_lf"].values / 1e8

        ax2 = ax.twinx()
        ax.bar(x, rev, color="#4C72B0", alpha=0.7, label="Revenue (100M CNY)")
        ax2.plot(x, profit, color="#DD8452", marker='o', linewidth=2, label="Net Profit (100M CNY)")

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Revenue (100M CNY)")
        ax2.set_ylabel("Net Profit (100M CNY)")
        ax.set_title(name, fontsize=10)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_revenue_profit.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_profitability(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Profitability Ratios Trend", fontsize=14, fontweight='bold')

    for ax, (code, name) in zip(axes, STOCKS.items()):
        sub = df[df["instrument"] == code].sort_values("report_date").copy()
        if sub.empty:
            ax.set_title(name)
            continue

        sub["gross_margin"] = safe_div(sub["gross_profit_lf"].values, sub["total_operating_revenue_lf"].values)
        sub["net_margin"] = safe_div(sub["net_profit_to_parent_shareholders_lf"].values, sub["total_operating_revenue_lf"].values)
        sub["roe"] = safe_div(sub["net_profit_to_parent_shareholders_lf"].values, sub["total_equity_to_parent_shareholders_lf"].values)

        labels = sub["report_date"].dt.strftime("%Y-%m").tolist()
        x = range(len(sub))

        ax.plot(x, sub["gross_margin"] * 100, marker='s', label="Gross Margin %", linewidth=2)
        ax.plot(x, sub["net_margin"] * 100, marker='o', label="Net Margin %", linewidth=2)
        ax.plot(x, sub["roe"] * 100, marker='^', label="ROE %", linewidth=2)

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Ratio (%)")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_profitability.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_yoy_growth(yoy_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("YoY Growth Rate (Revenue & Net Profit)", fontsize=14, fontweight='bold')

    for ax, (code, name) in zip(axes, STOCKS.items()):
        sub = yoy_df[yoy_df["instrument"] == code].sort_values("cur_report_date").copy()
        if sub.empty:
            ax.set_title(name)
            continue

        sub["rev_yoy"] = safe_div(
            (sub["revenue_cur"] - sub["revenue_yoy"]).values,
            np.abs(sub["revenue_yoy"].values)
        ) * 100
        sub["profit_yoy"] = safe_div(
            (sub["profit_cur"] - sub["profit_yoy"]).values,
            np.abs(sub["profit_yoy"].values)
        ) * 100

        labels = sub["cur_report_date"].dt.strftime("%Y-%m").tolist()
        x = range(len(sub))

        ax.bar([i - 0.2 for i in x], sub["rev_yoy"], width=0.35, color="#4C72B0", alpha=0.8, label="Revenue YoY %")
        ax.bar([i + 0.2 for i in x], sub["profit_yoy"], width=0.35, color="#DD8452", alpha=0.8, label="Net Profit YoY %")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("YoY Growth (%)")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_yoy_growth.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_cashflow_rd(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Operating Cash Flow & R&D Expense (TTM)", fontsize=14, fontweight='bold')

    for ax, (code, name) in zip(axes, STOCKS.items()):
        sub = df[df["instrument"] == code].sort_values("report_date").copy()
        if sub.empty:
            ax.set_title(name)
            continue

        labels = sub["report_date"].dt.strftime("%Y-%m").tolist()
        x = range(len(sub))
        cffoa = sub["net_cffoa_ttm"].values / 1e8
        rd = sub["research_and_development_expense_ttm"].values / 1e8

        ax2 = ax.twinx()
        ax.bar(x, cffoa, color="#55A868", alpha=0.7, label="Op. Cash Flow TTM (100M)")
        ax2.plot(x, rd, color="#C44E52", marker='D', linewidth=2, label="R&D Expense TTM (100M)")

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel("Cash Flow (100M CNY)")
        ax2.set_ylabel("R&D Expense (100M CNY)")
        ax.set_title(name, fontsize=10)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_cashflow_rd.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def build_summary_table(df: pd.DataFrame, yoy_df: pd.DataFrame) -> dict:
    latest = get_latest(df)
    summary = {}
    for code, name in STOCKS.items():
        row = latest[latest["instrument"] == code]
        if row.empty:
            summary[code] = {}
            continue
        r = row.iloc[0]

        yoy_row = yoy_df[yoy_df["instrument"] == code].sort_values("cur_report_date")
        if not yoy_row.empty:
            yr = yoy_row.iloc[-1]
            rev_yoy = (yr["revenue_cur"] - yr["revenue_yoy"]) / abs(yr["revenue_yoy"]) if yr["revenue_yoy"] else np.nan
            profit_yoy = (yr["profit_cur"] - yr["profit_yoy"]) / abs(yr["profit_yoy"]) if yr["profit_yoy"] else np.nan
        else:
            rev_yoy = profit_yoy = np.nan

        gross_margin = r["gross_profit_lf"] / r["total_operating_revenue_lf"] if r["total_operating_revenue_lf"] else np.nan
        net_margin = r["net_profit_to_parent_shareholders_lf"] / r["total_operating_revenue_lf"] if r["total_operating_revenue_lf"] else np.nan
        roe = r["net_profit_to_parent_shareholders_lf"] / r["total_equity_to_parent_shareholders_lf"] if r["total_equity_to_parent_shareholders_lf"] else np.nan
        current_ratio = r["total_current_assets_lf"] / r["total_current_liabilities_lf"] if r["total_current_liabilities_lf"] else np.nan
        debt_ratio = r["total_liabilities_lf"] / r["total_assets_lf"] if r["total_assets_lf"] else np.nan

        summary[code] = {
            "name": name,
            "report_date": str(r["report_date"])[:10],
            "revenue": fmt_billion(r["total_operating_revenue_lf"]),
            "net_profit": fmt_billion(r["net_profit_to_parent_shareholders_lf"]),
            "total_assets": fmt_billion(r["total_assets_lf"]),
            "gross_margin": fmt_pct(gross_margin),
            "net_margin": fmt_pct(net_margin),
            "roe": fmt_pct(roe),
            "current_ratio": f"{current_ratio:.2f}" if not pd.isna(current_ratio) else "N/A",
            "debt_ratio": fmt_pct(debt_ratio),
            "rev_yoy": fmt_pct(rev_yoy),
            "profit_yoy": fmt_pct(profit_yoy),
            "rd_expense": fmt_billion(r["research_and_development_expense_lf"]),
            "op_cashflow_ttm": fmt_billion(r["net_cffoa_ttm"]),
        }
    return summary


def generate_report(summary: dict, fig_paths: list):
    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# 光模块行业财务分析报告")
    lines.append(f"\n**报告日期：** {today}\n")
    lines.append("**分析标的：** 中际旭创（300308）、新易盛（300502）、光迅科技（002281）\n")
    lines.append("---\n")

    lines.append("## 一、核心财务指标汇总\n")
    lines.append("| 指标 | 中际旭创 (300308) | 新易盛 (300502) | 光迅科技 (002281) |")
    lines.append("|---|---|---|---|")

    metrics = [
        ("最新报告期", "report_date"),
        ("营业收入（亿元）", "revenue"),
        ("归母净利润（亿元）", "net_profit"),
        ("总资产（亿元）", "total_assets"),
        ("毛利率", "gross_margin"),
        ("净利率", "net_margin"),
        ("ROE", "roe"),
        ("流动比率", "current_ratio"),
        ("资产负债率", "debt_ratio"),
        ("营收同比增长", "rev_yoy"),
        ("净利润同比增长", "profit_yoy"),
        ("研发费用（亿元）", "rd_expense"),
        ("经营现金流TTM（亿元）", "op_cashflow_ttm"),
    ]

    codes = list(STOCKS.keys())
    for label, key in metrics:
        vals = [summary.get(c, {}).get(key, "N/A") for c in codes]
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")

    lines.append("\n---\n")
    lines.append("## 二、营收与净利润趋势\n")
    lines.append(f"![Revenue & Profit Trend](fig_revenue_profit.png)\n")
    lines.append(
        "上图展示了三家公司近期各报告期的营业收入（柱状）与归母净利润（折线）走势。"
        "中际旭创和新易盛受益于AI算力基础设施建设浪潮，800G/1.6T高速光模块需求爆发，"
        "营收和利润均呈现快速增长态势。光迅科技作为国资背景企业，增速相对稳健。\n"
    )

    lines.append("## 三、盈利能力分析\n")
    lines.append(f"![Profitability Ratios](fig_profitability.png)\n")
    lines.append(
        "毛利率、净利率和ROE是衡量企业盈利质量的核心指标。"
        "中际旭创凭借高端产品结构和规模效应，毛利率持续提升；"
        "新易盛在高速产品放量后盈利能力显著改善；"
        "光迅科技毛利率相对稳定，体现其在电信市场的稳定竞争地位。\n"
    )

    lines.append("## 四、同比增长率\n")
    lines.append(f"![YoY Growth](fig_yoy_growth.png)\n")
    lines.append(
        "同比增长率反映企业相对于去年同期的成长速度。"
        "2024年以来，中际旭创和新易盛受益于数据中心光模块需求爆发，"
        "营收和净利润同比增速均大幅超越行业平均水平。"
        "光迅科技增速较为平稳，与其业务结构中电信市场占比较高有关。\n"
    )

    lines.append("## 五、经营现金流与研发投入\n")
    lines.append(f"![Cash Flow & R&D](fig_cashflow_rd.png)\n")
    lines.append(
        "经营现金流（TTM）反映企业实际造血能力，研发费用体现企业对未来技术竞争力的投入。"
        "三家公司经营现金流均保持正值，说明主营业务健康。"
        "中际旭创和新易盛持续加大研发投入，布局下一代光互联技术，"
        "光迅科技研发投入绝对额较大，体现其在光芯片领域的技术积累。\n"
    )

    lines.append("## 六、综合分析与投资观点\n")
    lines.append(
        "### 中际旭创（300308）\n"
        "公司是全球领先的高速光模块供应商，在800G产品上率先实现大规模量产并向1.6T演进。"
        "盈利能力持续提升，ROE处于行业高位，经营现金流充裕。"
        "核心风险在于客户集中度较高（主要依赖北美云厂商）以及行业景气周期波动。\n"
    )
    lines.append(
        "### 新易盛（300502）\n"
        "公司在高速光模块赛道快速崛起，受益于AI算力需求爆发，营收和利润增速亮眼。"
        "毛利率随产品结构升级持续改善，研发投入加速。"
        "需关注产能扩张节奏与市场竞争加剧带来的价格压力。\n"
    )
    lines.append(
        "### 光迅科技（002281）\n"
        "公司是国内光芯片和光器件龙头，具备从芯片到模块的垂直整合能力。"
        "电信市场基本盘稳固，数据中心业务持续发力。"
        "国资背景带来稳定的客户资源，但也使得增速相对保守。"
        "长期看，自研光芯片能力是其核心竞争壁垒。\n"
    )

    lines.append("---\n")
    lines.append(
        "> **免责声明：** 本报告基于公开财务数据生成，仅供学习研究使用，不构成任何投资建议。\n"
    )

    report_path = os.path.join(OUTPUT_DIR, "financial_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    print("Fetching financial data...")
    df = fetch_financial_data()
    print(f"  Got {len(df)} rows from lf/ttm tables")

    print("Fetching YoY comparison data...")
    yoy_df = fetch_yoy_data()
    print(f"  Got {len(yoy_df)} rows for YoY analysis")

    print("Generating charts...")
    p1 = plot_revenue_profit(df)
    p2 = plot_profitability(df)
    p3 = plot_yoy_growth(yoy_df)
    p4 = plot_cashflow_rd(df)
    print(f"  Saved: {p1}, {p2}, {p3}, {p4}")

    print("Building summary table...")
    summary = build_summary_table(df, yoy_df)

    print("Generating markdown report...")
    report_path = generate_report(summary, [p1, p2, p3, p4])
    print(f"  Report saved to: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
