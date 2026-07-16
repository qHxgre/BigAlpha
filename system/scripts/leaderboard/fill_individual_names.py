"""补全排行榜个人条目（type=individual）的 team_name。

数据源：
    user_info.csv（system/files/scripts/leaderboard_crawl/user_info.csv）
    列 id 对应榜单里的 user_id，优先取 nickname，nickname 为空则用 username。

用法：
    python fill_individual_names.py [<leaderboard_csv> ...]
    不传参数时默认处理：
        system/files/scripts/leaderboard_crawl/20260715/
        leaderboard_63dd885c-2488-4efd-9c61-9e3a536f172c_20260715_1500.csv
    就地覆盖写回同一文件（保留 utf-8-sig 编码，与 crawl_leaderboad.py 落盘格式一致）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.paths import LEADERBOARD_CRAWL_DIR

USER_INFO_CSV = LEADERBOARD_CRAWL_DIR / "user_info.csv"

DEFAULT_TARGET = (
    LEADERBOARD_CRAWL_DIR
    / "20260715"
    / "leaderboard_523f9302-5b4b-42bd-bce1-f232e7c74316_20260715_1500.csv"
)


def build_user_name_map() -> dict[str, str]:
    """读取 user_info.csv，返回 {id: 姓名}，优先 nickname，为空则用 username。"""
    df = pd.read_csv(USER_INFO_CSV, dtype=str, encoding="utf-8-sig").fillna("")
    names = df["nickname"].where(df["nickname"].str.strip() != "", df["username"])
    return dict(zip(df["id"], names))


def fill_one(csv_path: Path, name_map: dict[str, str]) -> int:
    """把单个榜单 CSV 里 type=individual 行的 team_name 用 name_map 补全，返回补全条数。"""
    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    mask = df["type"] == "individual"
    filled = df.loc[mask, "user_id"].map(name_map)
    n_filled = int(filled.notna().sum())
    df.loc[mask, "team_name"] = filled.where(filled.notna(), df.loc[mask, "team_name"])
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return n_filled


def main(argv: list[str]) -> int:
    targets = [Path(p) for p in argv] or [DEFAULT_TARGET]
    name_map = build_user_name_map()
    for csv_path in targets:
        if not csv_path.exists():
            print(f"  [跳过] 文件不存在: {csv_path}", file=sys.stderr)
            continue
        n = fill_one(csv_path, name_map)
        print(f"  {csv_path.name}: 补全 {n} 条个人条目 team_name", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
