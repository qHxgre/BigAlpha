"""每周公示内容展示脚本。

从评估系统产物（leaderboard_*.csv / factor_analyze.json）读取最新结果，
按公示模板输出以下三部分：

  (1) 前 10 因子画像（ModelScore 排名前 10）：BARRA 风格暴露、行业分布
  (2) 团队细节得分：权重前 20 因子占比、各因子 ModelScore 百分位排名及趋势、A/B 项分位
  (3) 指数增强策略跟踪：累计超额收益、IR、最大回撤、增量贡献

用法:
    python weekly_disclosure.py <比赛ID> [--leaderboard-dir <路径>] [--team <team_id>]
                                [--out <输出路径>]

默认 leaderboard_dir 为:
    /home/aiuser/work/workspace/BigAlpha/system/competitions/<competition_id>/leaderboard/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# 把父目录加入 sys.path，复用 _client.AlphathonClient
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from _client import AlphathonClient

# ---- 默认路径 ---------------------------------------------------------------

DEFAULT_COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
DEFAULT_LEADERBOARD_BASE = (
    "/home/aiuser/work/workspace/BigAlpha/system/competitions"
    "/{competition_id}/leaderboard"
)

# ---- 小工具 -----------------------------------------------------------------


def _read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [警告] 无法读取 {path}: {e}", file=sys.stderr)
        return None


def _safe(val: Any) -> Any:
    """把 NaN/Inf 转成 None，方便格式化。"""
    if isinstance(val, float) and not math.isfinite(val):
        return None
    return val


def _fmt(val: Any, fmt: str = ".4f", fallback: str = "N/A") -> str:
    v = _safe(val)
    if v is None:
        return fallback
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return fallback


def _pct(val: Any, fallback: str = "N/A") -> str:
    v = _safe(val)
    if v is None:
        return fallback
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return fallback


# ---- 模拟 BARRA 风格暴露（基于现有指标） ------------------------------------

_STYLE_COLS = {
    "ic_mean":    "IC 均值（信息系数均值，反映因子预测能力）",
    "ic_ir":      "IC IR（信息系数稳定性，类动量/质量风格）",
    "sharpe_ratio": "Sharpe（因子多空组合夏普比，波动率相关）",
    "stress_ic_ir": "Stress IC IR（压力测试 ICIR，尾部风险暴露）",
}


def _barra_exposure_table(row: pd.Series) -> str:
    """用现有单因子指标模拟 BARRA 风格暴露表格。"""
    lines = ["| 风格维度 | 指标值 | 说明 |", "|---|---|---|"]
    for col, desc in _STYLE_COLS.items():
        val = _fmt(row.get(col), ".4f")
        lines.append(f"| {col} | {val} | {desc} |")
    return "\n".join(lines)


# ---- 第一部分：前 10 因子画像 -----------------------------------------------


def section_top10_factors(
    leaderboard_dir: str,
    client: AlphathonClient,
    competition_id: str,
) -> str:
    reg_csv = os.path.join(leaderboard_dir, "leaderboard_reg.csv")
    sfa_csv = os.path.join(leaderboard_dir, "leaderboard_sfa.csv")
    summary_csv = os.path.join(leaderboard_dir, "submissions_summary.csv")

    reg = _read_csv(reg_csv)
    sfa = _read_csv(sfa_csv)
    summary = _read_csv(summary_csv)

    if reg is None or "model_score" not in reg.columns:
        return "**（1）前 10 因子画像**\n\n> 暂无回归产物，本期跳过。\n"

    reg = reg.copy()
    reg["model_score"] = pd.to_numeric(reg["model_score"], errors="coerce")
    reg = reg.sort_values("model_score", ascending=False, na_position="last")
    top10 = reg.head(10).reset_index(drop=True)

    # 从 summary 拿单因子分析指标
    if summary is not None and "submission_id" in summary.columns:
        summary = summary.set_index("submission_id")

    lines = ["**（1）前 10 因子画像**（ModelScore 排名前 10）\n"]
    lines.append(f"*数据截至 {datetime.now():%Y-%m-%d %H:%M}*\n")

    for rank, (_, r) in enumerate(top10.iterrows(), start=1):
        factor_id = str(r.get("factor", ""))
        model_score = _fmt(r.get("model_score"), ".4f")
        sel_rate = _pct(r.get("selection_rate"))

        lines.append(f"### 第 {rank} 名  (ModelScore = {model_score}，入选率 {sel_rate})")
        lines.append(f"- factor_id: `{factor_id}`\n")

        # BARRA 风格暴露（从 summary 读）
        lines.append("**BARRA 风格暴露（聚合特征，不含因子构造逻辑）**\n")
        if summary is not None and factor_id in summary.index:
            sub_row = summary.loc[factor_id]
            lines.append(_barra_exposure_table(sub_row))
        else:
            lines.append("| 风格维度 | 指标值 | 说明 |")
            lines.append("|---|---|---|")
            for col, desc in _STYLE_COLS.items():
                lines.append(f"| {col} | N/A | {desc} |")

        lines.append("")
        lines.append("**行业分布（各行业平均 IC，揭示因子行业偏好）**\n")
        lines.append("> 行业 IC 数据暂未收录，如需展示请在因子分析阶段输出 industry_ic.json。\n")
        lines.append("---")

    lines.append("> 公布的是**聚合特征**，不涉及具体构造逻辑，不影响知识产权保护。\n")
    return "\n".join(lines)


# ---- 第二部分：团队细节得分 --------------------------------------------------


def section_team_scores(
    leaderboard_dir: str,
    client: AlphathonClient,
    competition_id: str,
    team_id: str | None,
) -> str:
    reg_csv = os.path.join(leaderboard_dir, "leaderboard_reg.csv")
    final_csv = os.path.join(leaderboard_dir, "leaderboard_final.csv")
    summary_csv = os.path.join(leaderboard_dir, "submissions_summary.csv")

    reg = _read_csv(reg_csv)
    final = _read_csv(final_csv)
    summary = _read_csv(summary_csv)

    lines = ["**（2）团队细节得分**\n"]

    # 权重前 20 因子
    if reg is not None and "model_score" in reg.columns:
        reg = reg.copy()
        reg["model_score"] = pd.to_numeric(reg["model_score"], errors="coerce")
        top20_ids = set(
            reg.sort_values("model_score", ascending=False, na_position="last")
            .head(20)["factor"]
            .astype(str)
        )
    else:
        top20_ids = set()

    if team_id:
        # 拉该队伍的提交
        team_submissions = _get_team_submissions(client, competition_id, team_id, summary)
        team_sub_ids = {str(s) for s in team_submissions}

        overlap = team_sub_ids & top20_ids
        lines.append(f"* 当前权重前 20 的因子中，**本团队占据 {len(overlap)} 个**（不披露其他团队因子内容）。\n")

        # 各因子 ModelScore 百分位排名
        if reg is not None and "model_score" in reg.columns and final is not None:
            reg_copy = reg.copy()
            reg_copy["model_score"] = pd.to_numeric(reg_copy["model_score"], errors="coerce")
            reg_copy["b_pct"] = reg_copy["model_score"].rank(pct=True)
            reg_copy["factor"] = reg_copy["factor"].astype(str)
            team_reg = reg_copy[reg_copy["factor"].isin(team_sub_ids)]

            if not team_reg.empty:
                lines.append("* **本团队各因子 ModelScore 百分位排名及变化趋势**\n")
                lines.append("| 因子 ID | ModelScore | 百分位排名 | 入选率 |")
                lines.append("|---|---|---|---|")
                for _, r in team_reg.sort_values("b_pct", ascending=False).iterrows():
                    fid = str(r["factor"])
                    ms = _fmt(r.get("model_score"), ".4f")
                    pct = _pct(r.get("b_pct"))
                    sel = _pct(r.get("selection_rate")) if "selection_rate" in r.index else "N/A"
                    lines.append(f"| `{fid}` | {ms} | {pct} | {sel} |")
                lines.append("")
            else:
                lines.append("* 本团队暂无因子进入回归因子池。\n")
        else:
            lines.append("* 回归产物暂未生成，ModelScore 百分位排名待更新。\n")

        # A/B 项得分及全场排名分位
        if final is not None and "id" in final.columns:
            final_copy = final.copy()
            for col in ["a_score", "b_score", "final_score"]:
                final_copy[col] = pd.to_numeric(final_copy.get(col), errors="coerce")
            team_final = final_copy[final_copy["id"].astype(str).isin(team_sub_ids)]

            if not team_final.empty:
                total = len(final_copy.dropna(subset=["final_score"]))
                lines.append("* **本团队 A 项、B 项得分及全场排名分位**\n")
                lines.append("| 因子 ID | A 项得分 | B 项得分 | 最终得分 | 全场分位 |")
                lines.append("|---|---|---|---|---|")
                for _, r in team_final.sort_values("final_score", ascending=False, na_position="last").iterrows():
                    fid = str(r.get("id", ""))
                    a = _fmt(r.get("a_score"), ".4f")
                    b = _fmt(r.get("b_score"), ".4f")
                    fs = _fmt(r.get("final_score"), ".4f")
                    rank_val = _safe(r.get("final_score"))
                    if rank_val is not None and total > 0:
                        rank_pct = (final_copy["final_score"] <= rank_val).sum() / total
                        rank_str = f"{rank_pct * 100:.1f}%"
                    else:
                        rank_str = "N/A"
                    lines.append(f"| `{fid}` | {a} | {b} | {fs} | {rank_str} |")
                lines.append("")
            else:
                lines.append("* 本团队暂无最终得分记录。\n")
        else:
            lines.append("* 最终得分榜单暂未生成，A/B 项分位待更新。\n")
    else:
        lines.append("> 未指定 --team 参数，跳过团队细节得分。使用 `--team <team_id>` 查看本团队详情。\n")

    return "\n".join(lines)


def _get_team_submissions(
    client: AlphathonClient,
    competition_id: str,
    team_id: str,
    summary: pd.DataFrame | None,
) -> list[str]:
    """获取指定队伍的提交 ID 列表。

    优先从 summary 按 group 列过滤（group = user_id，见 scoring.group_key）；
    若 summary 不可用则回退到 API 查询。
    """
    if summary is not None and "submission_id" in summary.columns and "group" in summary.columns:
        matched = summary[summary["group"].astype(str) == str(team_id)]
        if not matched.empty:
            return list(matched["submission_id"].astype(str))

    # 回退：直接查该 user_id 的提交
    try:
        subs = client.list_submissions(
            competition_id,
            constraints={"user_id": team_id},
            order_by=["-created_at"],
        )
        return [str(s["id"]) for s in subs]
    except Exception:
        return []


# ---- 第三部分：指数增强策略跟踪 ----------------------------------------------


def _compute_strategy_metrics(reg: pd.DataFrame, final: pd.DataFrame) -> dict:
    """用回归和最终得分数据模拟指增策略关键指标（简化版）。

    真实指增策略需要价格行情数据回测，此处用因子池表现指标近似展示。
    如需精确指标，请在因子池回归阶段同步输出 index_enhancement_metrics.json。
    """
    metrics: dict[str, Any] = {}

    if "model_score" in reg.columns:
        ms = pd.to_numeric(reg["model_score"], errors="coerce").dropna()
        if len(ms) > 0:
            metrics["avg_model_score"] = float(ms.mean())
            metrics["top_model_score"] = float(ms.max())

    if final is not None and "final_score" in final.columns:
        fs = pd.to_numeric(final["final_score"], errors="coerce").dropna()
        if len(fs) > 0:
            metrics["avg_final_score"] = float(fs.mean())

    if "selection_rate" in reg.columns:
        sr = pd.to_numeric(reg["selection_rate"], errors="coerce").dropna()
        if len(sr) > 0:
            metrics["avg_selection_rate"] = float(sr.mean())

    return metrics


def section_index_enhancement(leaderboard_dir: str) -> str:
    reg_csv = os.path.join(leaderboard_dir, "leaderboard_reg.csv")
    final_csv = os.path.join(leaderboard_dir, "leaderboard_final.csv")
    metrics_json = os.path.join(leaderboard_dir, "index_enhancement_metrics.json")

    lines = ["**（3）指数增强策略跟踪**\n"]
    lines.append(
        "平台基于每轮回归权重构建**中证 1000 指数增强策略**：以合成因子值为信号，"
        "在成分股内超配高分股、低配低分股，跟踪误差约束 5% 以内。\n"
    )

    # 优先读预先产出的指标文件
    if os.path.exists(metrics_json):
        try:
            with open(metrics_json, encoding="utf-8") as f:
                m = json.load(f)
            lines.append("| 指标 | 本期值 | 说明 |")
            lines.append("|---|---|---|")
            lines.append(f"| 累计超额收益 | {_fmt(m.get('cum_excess_return'), '.4f')} | 相对中证 1000 的累计 alpha |")
            lines.append(f"| 信息比率（IR） | {_fmt(m.get('ir'), '.4f')} | 年化超额收益 / 跟踪误差 |")
            lines.append(f"| 最大回撤 | {_fmt(m.get('max_drawdown'), '.4f')} | 超额收益的最大回撤 |")
            lines.append(f"| 增量贡献 | {_fmt(m.get('incremental_ir'), '.4f')} | 新一轮因子加入后策略 IR 的变化 |")
            lines.append("")
        except Exception as e:
            lines.append(f"> 读取 index_enhancement_metrics.json 失败：{e}\n")
    else:
        reg = _read_csv(reg_csv)
        final = _read_csv(final_csv)

        if reg is not None:
            m = _compute_strategy_metrics(reg, final)
            lines.append("| 指标 | 本期近似值 | 说明 |")
            lines.append("|---|---|---|")
            lines.append(f"| 平均 ModelScore | {_fmt(m.get('avg_model_score'), '.4f')} | 因子池平均预测能力 |")
            lines.append(f"| 最高 ModelScore | {_fmt(m.get('top_model_score'), '.4f')} | 因子池最优单因子能力 |")
            lines.append(f"| 平均入选率 | {_pct(m.get('avg_selection_rate'))} | 跨窗口平均有效因子占比 |")
            lines.append(f"| 平均最终得分 | {_fmt(m.get('avg_final_score'), '.4f')} | 全体提交 0.3A+0.7B 均值 |")
            lines.append("")
            lines.append(
                "> 精确指增回测指标（累计超额收益 / IR / 最大回撤 / 增量贡献）需行情数据支撑。"
                "如已产出 `leaderboard/index_enhancement_metrics.json`，脚本会优先读取并展示。\n"
            )
        else:
            lines.append("> 回归产物暂未生成，指增策略指标待更新。\n")

    lines.append(
        "> 指增策略的持续改善是比赛质量的直观体现——优质新因子应推动 IR 上升；"
        "若未能提升也会如实呈现，形成正向反馈。\n"
    )
    return "\n".join(lines)


# ---- 主流程 -----------------------------------------------------------------


def generate_disclosure(
    competition_id: str,
    leaderboard_dir: str,
    team_id: str | None,
    client: AlphathonClient,
) -> str:
    now = datetime.now()
    parts = [
        f"# 每轮公示内容\n",
        f"**发布时间：{now:%Y-%m-%d %H:%M}**　比赛 ID：`{competition_id}`\n",
        "每日固定时点，根据最新评估结果公布以下信息，帮助参赛者明确优化方向。\n",
        "---",
        section_top10_factors(leaderboard_dir, client, competition_id),
        "---",
        section_team_scores(leaderboard_dir, client, competition_id, team_id),
        "---",
        section_index_enhancement(leaderboard_dir),
    ]
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="生成每周公示内容（前10因子画像 / 团队细节得分 / 指增策略跟踪）"
    )
    parser.add_argument(
        "competition_id",
        nargs="?",
        default=DEFAULT_COMPETITION_ID,
        help="比赛 ID（默认 AI 因子挖掘赛道）",
    )
    parser.add_argument(
        "--leaderboard-dir",
        default=None,
        help="榜单目录路径（默认 competitions/<id>/leaderboard）",
    )
    parser.add_argument(
        "--team",
        default=None,
        dest="team_id",
        help="查看指定团队（user_id）的细节得分",
    )
    parser.add_argument("--out", default=None, help="结果 Markdown 输出路径")
    parser.add_argument("--base-url", default=None, help="覆盖 ALPHATHON_API_BASE_URL")
    parser.add_argument("--token", default=None, help="覆盖 ALPHATHON_API_TOKEN")
    args = parser.parse_args(argv)

    leaderboard_dir = args.leaderboard_dir or DEFAULT_LEADERBOARD_BASE.format(
        competition_id=args.competition_id
    )

    client = AlphathonClient(base_url=args.base_url, token=args.token)

    content = generate_disclosure(
        competition_id=args.competition_id,
        leaderboard_dir=leaderboard_dir,
        team_id=args.team_id,
        client=client,
    )

    print(content)

    if args.out:
        out_path = args.out
    else:
        out_path = f"weekly_disclosure_{datetime.now():%Y%m%d_%H%M%S}.md"

    Path(out_path).write_text(content, encoding="utf-8")
    print(f"\n结果已写入: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
