"""汇总每个 submission 的 sfa_status.json，合成一个 DataFrame 供后续筛选。

评测系统会在 system/files/{competition_id}/submissions/{submission_id}/ 下产出一份
sfa_status.json，形如：
    {"submission_id": "...", "status": "lookahead", "finished_at": "2026-07-14T13:56:17"}

本脚本遍历该目录，读取每份 sfa_status.json，合成一张按 submission 一行的表，
落地成 CSV 方便后续按 status（如 lookahead）等条件筛选。

要查哪场比赛，改下面的 COMPETITION_ID 即可，然后直接跑：
    python system/scripts/submissions/collect_sfa_status.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
import pandas as pd

from common.paths import DATA_ROOT, resolve_submissions_dir

# ===== 配置：要汇总哪场比赛的 submission =====================================
COMPETITION_ID = '76ad3f56-ec2b-431a-890e-139a7f4bbcba'
# ===========================================================================


def collect_sfa_status(submissions_dir: Path) -> pd.DataFrame:
    """遍历 submissions 目录下每个子目录的 sfa_status.json，合成一张表。

    每行对应一个 submission，列取自 sfa_status.json 的字段（submission_id/status/
    finished_at 等）。缺文件或解析失败的目录会记一行并标注 read_error，不中断整体。
    """
    rows: list[dict] = []
    for sub_dir in sorted(p for p in submissions_dir.iterdir() if p.is_dir()):
        status_file = sub_dir / "sfa_status.json"
        if not status_file.exists():
            rows.append({"submission_id": sub_dir.name, "read_error": "missing sfa_status.json"})
            continue
        try:
            data = json.loads(status_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            rows.append({"submission_id": sub_dir.name, "read_error": str(e)})
            continue
        # 以目录名兜底 submission_id，其余字段按 JSON 原样铺开
        row = {"submission_id": sub_dir.name}
        row.update(data)
        rows.append(row)

    df = pd.DataFrame(rows)
    # submission_id 放首列，read_error（若有）放末列，其余保持原顺序
    if not df.empty:
        cols = ["submission_id"] + [c for c in df.columns if c not in ("submission_id", "read_error")]
        if "read_error" in df.columns:
            cols.append("read_error")
        df = df[cols]
    return df


def main() -> None:
    submissions_dir = resolve_submissions_dir(COMPETITION_ID)

    if not submissions_dir.is_dir():
        print(f"提交目录不存在: {submissions_dir}", file=sys.stderr)
        sys.exit(1)

    df = collect_sfa_status(submissions_dir)
    print(f"=== 比赛 {COMPETITION_ID}：共 {len(df)} 个 submission ===")

    if "status" in df.columns:
        print("按 status 分布：")
        print(df["status"].value_counts(dropna=False).to_string())

    out_dir = DATA_ROOT / "submissions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sfa_status_{COMPETITION_ID}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已写出: {out_path}")


if __name__ == "__main__":
    main()
