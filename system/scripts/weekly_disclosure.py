"""每周公示内容 —— 需求一：前 10 因子画像。

依据最新评估结果，对 ModelScore 排名前 10 的因子公布两类聚合特征：

    - BARRA 风格暴露：各因子在市值(SIZE)、Beta、动量(MOMENTUM)、波动率(RESVOL) 等
      10 个 BARRA 风格上的平均暴露，提示尚未充分挖掘的维度；
    - 行业分布：各因子在申万一级行业上的平均 IC，揭示因子的行业偏好。

公布的是聚合特征，仅以排名（Top1..Top10）匿名标识因子，不涉及具体构造逻辑，
也不披露因子归属，不影响知识产权保护。

数据来源（云端评测榜单目录）：
    leaderboard_reg.csv     ModelScore（因子池回归得分），据此取前 10 因子；
    factor_pool.parquet     入池因子（已预处理）的因子值，date/instrument + 各因子列；
    bigalpha_2026_exposure  BARRA 风格暴露 + 行业哑变量 + 下一期收益 ret（dai 查询）。

用法：
    python weekly_disclosure.py [competition_id]

    competition_id 默认：76ad3f56-ec2b-431a-890e-139a7f4bbcba

输出（files/weekly_disclosure/）：
    factor_style_exposure_<date>.csv    前 10 因子风格暴露表
    factor_industry_ic_<date>.csv       前 10 因子行业 IC 表
    factor_style_exposure_<date>.png    风格暴露热力图
    factor_industry_ic_<date>.png       行业 IC 热力图
    factor_portrait_<date>.md           汇总 Markdown 公示正文（内嵌上述图表）

注意：本脚本只能在云端评测环境运行（依赖 dai 查询风格暴露表）。
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境（云端）后端，import 后立刻设定
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import dai
except ImportError:  # 本地无 dai（仅云端可用）：import 期不报错，跑到查询时才提示
    dai = None

# 让 matplotlib 能渲染中文标签（行业/风格中文名），找不到中文字体时静默退回默认字体
for _cn_font in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "Arial Unicode MS"]:
    if _cn_font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.sans-serif"] = [_cn_font]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

_scripts_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"

# 云端评测榜单目录；不存在时回退到脚本目录下的 files/leaderboard（与 daily_score_digest 一致）
REMOTE_LEADERBOARD_BASE = (
    "/home/aiuser/work/workspace/BigAlpha/system/files"
    "/{competition_id}/leaderboard"
)
LOCAL_LEADERBOARD_FALLBACK = Path(_scripts_dir) / "files" / "leaderboard"
OUTPUT_DIR = Path(_scripts_dir) / "files" / "weekly_disclosure"

TOP_N = 10

# BARRA 风格因子列（市值、Beta、动量、波动率 等 10 个风格维度）
STYLE_FACTORS = [
    "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL",
    "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE",
]

# exposure_df 中除 date/instrument/风格因子/辅助列外，其余列即申万一级行业哑变量。
_NON_INDUSTRY_COLS = set(
    ["date", "instrument", "industry_level1_code", "float_market_cap", "weights", "ret"]
    + STYLE_FACTORS
)

# 行业哑变量列 → 申万一级行业中文名（仅用于公示展示，缺失的列名原样透传）
INDUSTRY_NAME_CN = {
    "AGRIFOREST": "农林牧渔", "MINING": "采掘", "CHEM": "化工",
    "IRONSTEEL": "钢铁", "NONFERMETAL": "有色金属", "ELECTRONICS": "电子",
    "AUTO": "汽车", "HOUSEAPP": "家用电器", "FOODBEVER": "食品饮料",
    "TEXTILE": "纺织服装", "LIGHTINDUS": "轻工制造", "HEALTH": "医药生物",
    "UTILITIES": "公用事业", "TRANSPORTATION": "交通运输", "REALESTATE": "房地产",
    "COMMETRADE": "商业贸易", "LEISERVICE": "休闲服务", "BANK": "银行",
    "NONBANKFINAN": "非银金融", "CONGLOMERATES": "综合", "CONMAT": "建筑材料",
    "BUILDDECO": "建筑装饰", "ELECEQP": "电气设备", "MACHIEQUIP": "机械设备",
    "AERODEF": "国防军工", "COMPUTER": "计算机", "MEDIA": "传媒",
    "TELECOM": "通信", "COAL": "煤炭", "PETRO": "石油石化",
    "ENVP": "环保", "BEAUTY": "美容护理",
}

# 风格因子 → 中文名（公示展示用）
STYLE_NAME_CN = {
    "SIZE": "市值", "BETA": "Beta", "MOMENTUM": "动量", "RESVOL": "残差波动率",
    "SIZENL": "非线性市值", "BTOP": "账面市值比", "LIQUIDTY": "流动性",
    "EARNYILD": "盈利收益", "GROWTH": "成长", "LEVERAGE": "杠杆",
}


# ---- 数据加载 ----------------------------------------------------------------


def resolve_leaderboard_dir(competition_id: str) -> str:
    """定位榜单目录：优先云端路径，缺失时回退到本地 files/leaderboard。"""
    remote = REMOTE_LEADERBOARD_BASE.format(competition_id=competition_id)
    if os.path.isdir(remote):
        return remote
    print(
        f"  [提示] 云端目录不存在，回退到本地目录: {LOCAL_LEADERBOARD_FALLBACK}",
        file=sys.stderr,
    )
    return str(LOCAL_LEADERBOARD_FALLBACK)


def load_top_factors(leaderboard_dir: str, top_n: int = TOP_N) -> list[str]:
    """读取 leaderboard_reg.csv，返回 ModelScore 排名前 top_n 的因子 id。"""
    reg_path = os.path.join(leaderboard_dir, "leaderboard_reg.csv")
    reg = pd.read_csv(reg_path)
    if not {"factor", "model_score"}.issubset(reg.columns):
        raise ValueError(f"{reg_path} 缺少 factor/model_score 列: {list(reg.columns)}")
    reg = reg.dropna(subset=["model_score"])
    return list(reg.sort_values("model_score", ascending=False)["factor"].astype(str).values[:top_n])


def load_factor_data(
    leaderboard_dir: str, submission_ids: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """读取因子池（已预处理）的因子值，只保留 date/instrument + 命中的前 N 因子列。

    factor_pool.parquet 是各队伍入池因子的宽表；前 10 因子理应都在其中，但个别因子
    若因回归产物与因子池不同步而缺列，这里跳过并告警，避免整脚本因 KeyError 中断。
    """
    pool_path = os.path.join(leaderboard_dir, "factor_pool.parquet")
    pool = pd.read_parquet(pool_path)

    present = [sid for sid in submission_ids if sid in pool.columns]
    missing = [sid for sid in submission_ids if sid not in pool.columns]
    if missing:
        print(
            f"  [警告] 因子池缺少 {len(missing)} 个前 {TOP_N} 因子，跳过: {missing}",
            file=sys.stderr,
        )
    if not present:
        raise ValueError("因子池中没有任何前 10 因子列，无法生成画像")

    return pool[["date", "instrument"] + present], present


def load_exposure(factor_data: pd.DataFrame) -> pd.DataFrame:
    """按因子数据的日期区间，从 dai 拉取 BARRA 风格暴露 + 行业哑变量 + ret。"""
    if dai is None:
        raise RuntimeError("dai 不可用：本脚本依赖风格暴露表，只能在云端评测环境运行")
    dates = pd.to_datetime(factor_data["date"])
    sd = dates.min().strftime("%Y-%m-%d")
    ed = dates.max().strftime("%Y-%m-%d")
    return dai.query(
        "SELECT * FROM bigalpha_2026_exposure",
        filters={"date": [sd, ed]},
    ).df()


# ---- 特征计算 ----------------------------------------------------------------


def _zscore(s: pd.Series) -> pd.Series:
    """截面标准化，std 为 0 或非有限时返回全 0。"""
    std = s.std()
    if std == 0 or not np.isfinite(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def compute_style_exposure(
    factor_data: pd.DataFrame,
    exposure_df: pd.DataFrame,
    submission_ids: list[str],
) -> pd.DataFrame:
    """计算每个因子在各 BARRA 风格上的平均暴露。

    逐日将因子值截面标准化后，对（同样标准化的）风格因子做多元回归，取回归系数
    （即剔除风格间共线性后的纯暴露），再对所有交易日求平均。

    返回 DataFrame，index 为 factor，columns 为风格因子。
    """
    merged = factor_data.merge(
        exposure_df[["date", "instrument"] + STYLE_FACTORS],
        on=["date", "instrument"],
        how="inner",
    )

    exposure_records: dict[str, list[np.ndarray]] = {f: [] for f in submission_ids}
    n_style = len(STYLE_FACTORS)

    for _, day in merged.groupby("date"):
        # 风格因子矩阵（含截距），逐日标准化保证同尺度
        X = day[STYLE_FACTORS].apply(_zscore)
        X = X.assign(_const=1.0)
        Xv = X.values
        finite_rows = np.isfinite(Xv).all(axis=1)

        for f in submission_ids:
            y = _zscore(day[f]).values
            mask = np.isfinite(y) & finite_rows
            # 样本数需多于自变量数（风格 + 截距），否则回归无意义
            if mask.sum() < n_style + 1:
                continue
            try:
                coef = np.linalg.pinv(Xv[mask]) @ y[mask]
            except np.linalg.LinAlgError:
                continue
            exposure_records[f].append(coef[:n_style])

    rows = {}
    for f in submission_ids:
        recs = exposure_records[f]
        rows[f] = (
            np.nanmean(np.vstack(recs), axis=0)
            if recs else np.full(n_style, np.nan)
        )
    return pd.DataFrame.from_dict(rows, orient="index", columns=STYLE_FACTORS)


def compute_industry_ic(
    factor_data: pd.DataFrame,
    exposure_df: pd.DataFrame,
    submission_ids: list[str],
) -> pd.DataFrame:
    """计算每个因子在各申万一级行业的平均 IC。

    按 date × 行业分组，计算因子值与下一期收益 ret 的截面 Spearman 相关（Rank IC），
    再对交易日求平均。

    返回 DataFrame，index 为 factor，columns 为行业代码。
    """
    merged = factor_data.merge(
        exposure_df[["date", "instrument", "industry_level1_code", "ret"]],
        on=["date", "instrument"],
        how="inner",
    )

    # 每个 factor 的 {industry: [每日 IC]}
    ic_records: dict[str, dict[object, list[float]]] = {
        f: defaultdict(list) for f in submission_ids
    }

    for (_, industry), grp in merged.groupby(["date", "industry_level1_code"]):
        ret = grp["ret"]
        # 单日单行业样本太少，Spearman 相关不稳定，跳过
        if ret.notna().sum() < 3:
            continue
        for f in submission_ids:
            ic = grp[f].corr(ret, method="spearman")
            if np.isfinite(ic):
                ic_records[f][industry].append(ic)

    industries = sorted(merged["industry_level1_code"].dropna().unique())
    rows = {}
    for f in submission_ids:
        rows[f] = {
            ind: (np.mean(ic_records[f][ind]) if ic_records[f][ind] else np.nan)
            for ind in industries
        }
    return pd.DataFrame.from_dict(rows, orient="index", columns=industries)


# ---- 匿名化与展示 ------------------------------------------------------------


def anonymize_index(df: pd.DataFrame, submission_ids: list[str]) -> pd.DataFrame:
    """把 index 从因子 id 换成匿名排名标签 Top1..TopN（按 ModelScore 顺序）。

    公示只展示聚合特征、不披露因子归属，故对外用排名标识替代真实 submission id。
    """
    label = {sid: f"Top{i}" for i, sid in enumerate(submission_ids, start=1)}
    out = df.reindex(submission_ids)
    out.index = [label[sid] for sid in submission_ids]
    return out


def rename_industry_cols(df: pd.DataFrame) -> pd.DataFrame:
    """行业代码列名换成中文名（未知代码原样保留）。"""
    return df.rename(columns=lambda c: INDUSTRY_NAME_CN.get(c, c))


def rename_style_cols(df: pd.DataFrame) -> pd.DataFrame:
    """风格因子列名换成「中文(代码)」（如 市值(SIZE)），兼顾可读性与可追溯。"""
    return df.rename(columns=lambda c: f"{STYLE_NAME_CN.get(c, c)}({c})")


# ---- 图表 --------------------------------------------------------------------


def plot_heatmap(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """把因子×特征矩阵画成热力图并落盘 PNG。

    行是因子（Top1..TopN），列是风格/行业；以 0 为中心的红蓝发散色阶，
    正暴露/正 IC 偏红、负偏蓝，每格标注数值。
    """
    data = df.astype(float)
    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=(max(6.0, n_cols * 0.7), max(3.0, n_rows * 0.5)))

    # 以 0 为中心对称取色范围，正负暴露对比清晰；全 NaN 时退回 1.0 避免报错
    vmax = np.nanmax(np.abs(data.values)) if np.isfinite(data.values).any() else 1.0
    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(data.index, fontsize=9)
    ax.set_title(title, fontsize=12, pad=10)

    # 逐格标注数值，背景深处用白字保证可读
    for i in range(n_rows):
        for j in range(n_cols):
            val = data.values[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_markdown(
    style_exposure: pd.DataFrame,
    industry_ic: pd.DataFrame,
    competition_id: str,
    date_str: str,
    style_img: str,
    industry_img: str,
) -> str:
    """把两张表 + 对应热力图拼成对外公示的 Markdown 正文。

    style_img / industry_img 是图片相对 Markdown 文件的路径（同目录下的文件名）。
    """
    lines = [
        f"# {date_str} 前 {TOP_N} 因子画像",
        "",
        f"比赛 `{competition_id}` ModelScore 排名前 {TOP_N} 的因子聚合特征。",
        "公布的是聚合特征，以排名匿名标识（Top1 分数最高），不涉及构造逻辑，不披露因子归属。",
        "",
        "## （1）BARRA 风格暴露",
        "",
        "逐日对风格因子做多元回归取纯暴露系数，再对交易日求平均。数值越大表示该风格上正暴露越强。",
        "",
        f"![BARRA 风格暴露热力图]({style_img})",
        "",
        "<details><summary>展开数值表</summary>",
        "",
        style_exposure.round(4).to_markdown(),
        "",
        "</details>",
        "",
        "## （2）行业分布（平均 Rank IC）",
        "",
        "因子值与下一期收益在各申万一级行业内的截面 Spearman 相关（Rank IC），对交易日求平均。",
        "",
        f"![行业 IC 热力图]({industry_img})",
        "",
        "<details><summary>展开数值表</summary>",
        "",
        industry_ic.round(4).to_markdown(),
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def save_outputs(
    style_exposure: pd.DataFrame,
    industry_ic: pd.DataFrame,
    competition_id: str,
    output_dir: Path,
) -> None:
    """把风格暴露表/行业 IC 表（CSV）、热力图（PNG）与汇总 Markdown 落盘到 output_dir。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    style_csv = output_dir / f"factor_style_exposure_{date_str}.csv"
    industry_csv = output_dir / f"factor_industry_ic_{date_str}.csv"
    style_png = output_dir / f"factor_style_exposure_{date_str}.png"
    industry_png = output_dir / f"factor_industry_ic_{date_str}.png"
    md_path = output_dir / f"factor_portrait_{date_str}.md"

    style_exposure.round(6).to_csv(style_csv, encoding="utf-8-sig")
    industry_ic.round(6).to_csv(industry_csv, encoding="utf-8-sig")

    plot_heatmap(style_exposure, f"前 {TOP_N} 因子 BARRA 风格暴露", style_png)
    plot_heatmap(industry_ic, f"前 {TOP_N} 因子行业 Rank IC", industry_png)

    # Markdown 与图片同目录，用相对文件名引用，便于整目录一起分发
    md_path.write_text(
        build_markdown(
            style_exposure, industry_ic, competition_id, date_str,
            style_img=style_png.name, industry_img=industry_png.name,
        ),
        encoding="utf-8",
    )
    print(
        f"[{datetime.now():%H:%M:%S}] 已写入: {style_csv.name} / {industry_csv.name} / "
        f"{style_png.name} / {industry_png.name} / {md_path.name}",
        file=sys.stderr,
    )


# ---- CLI 入口 ----------------------------------------------------------------


def main(argv: list[str]) -> int:
    competition_id = argv[0] if argv else DEFAULT_COMPETITION_ID
    leaderboard_dir = resolve_leaderboard_dir(competition_id)

    print(f"[{datetime.now():%H:%M:%S}] 读取 ModelScore 前 {TOP_N} 因子...", file=sys.stderr)
    submission_ids = load_top_factors(leaderboard_dir)

    factor_data, submission_ids = load_factor_data(leaderboard_dir, submission_ids)
    print(f"[{datetime.now():%H:%M:%S}] 拉取风格暴露表...", file=sys.stderr)
    exposure_df = load_exposure(factor_data)

    style_exposure = compute_style_exposure(factor_data, exposure_df, submission_ids)
    industry_ic = compute_industry_ic(factor_data, exposure_df, submission_ids)

    # 匿名化 + 中文列名，仅用于对外展示
    style_show = rename_style_cols(anonymize_index(style_exposure, submission_ids))
    industry_show = rename_industry_cols(anonymize_index(industry_ic, submission_ids))

    print("=== 前 10 因子 BARRA 风格暴露 ===")
    print(style_show.round(4))
    print("\n=== 前 10 因子行业 IC ===")
    print(industry_show.round(4))

    save_outputs(style_show, industry_show, competition_id, OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

