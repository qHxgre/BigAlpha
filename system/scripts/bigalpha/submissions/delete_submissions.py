"""按 submission_id 删除对应的 submission 文件夹。

评测系统会在 system/files/{competition_id}/submissions/{submission_id}/ 下产出每个
submission 的目录。本脚本按下方 SUBMISSION_IDS 列表逐个删除对应目录（整个文件夹）。

要删哪场比赛的哪些 submission，改下面的 COMPETITION_ID 和 SUBMISSION_IDS，然后跑：
    python system/scripts/submissions/delete_submissions.py

默认 DRY_RUN=True 只打印将要删除的目录、不真正删；确认无误后改成 False 再执行。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用

from common.ids import dedup_keep_order
from common.paths import resolve_submissions_dir

# ===== 配置：要删哪场比赛、哪些 submission ===================================
COMPETITION_ID = '76ad3f56-ec2b-431a-890e-139a7f4bbcba'

# True: 只预览、不真正删除；确认无误后改 False 再跑
DRY_RUN = True

SUBMISSION_IDS = [
    '0df4367d-8744-47cf-ac64-dc9d04bb6d9e',
    '20aab63b-3b7b-4abc-a9ee-cc2a6b6d2f22',
    '25895aaa-c689-4317-9d62-0f35e101f59b',
    '2f3bf319-187a-4354-8878-539fe1e6a12d',
    '304a1156-5101-4468-8286-54286a9f8022',
    '39a013dd-958c-4ea8-8fdb-ced36df31a05',
    '3aa708c2-a847-44fb-8e37-af020e74e1b9',
    '452eb5c3-a3e9-4da3-a99d-a8072479272f',
    '51b9ff13-e613-4018-956b-62d2457d82fc',
    '636cf6e4-df21-4623-a42a-3a96985c59f8',
    '762132ef-ab39-417c-aded-4f6512a5c565',
    '81d00539-1c11-4929-ac64-cb5f70f035b9',
    '85b08f64-c8a0-4fe7-b86c-b08ec21fcb7b',
    '9dd0f06c-0cce-469e-8518-a740851ae753',
    'a579450c-a68d-464c-aa0c-3301e95c20d3',
    'b1a8a4bf-b0ed-4960-b268-379edee75702',
    'be8fac5c-b5e3-4c56-8064-49470a56d88a',
    'c2e48b6f-4af6-4895-b681-af12e7d717b4',
    'c3473862-e835-4468-a8c6-6664dd8190bb',
    'c4d4b57d-4f71-4b4f-8f02-c71582c8b5ad',
    'd0a62292-cbf4-4a42-9229-3cf7878d086c',
    'dbf9c596-b64e-44b7-9d7f-8bcd932ef302',
    'e5aed358-2e49-4e72-87cc-9c303df1984f',
    'eaaca541-1621-4a47-810e-44737bb5f845',
]
# ===========================================================================


def main() -> None:
    submissions_dir = resolve_submissions_dir(COMPETITION_ID)
    if not submissions_dir.is_dir():
        print(f"提交目录不存在: {submissions_dir}", file=sys.stderr)
        sys.exit(1)

    ids = dedup_keep_order(SUBMISSION_IDS)
    mode = "预览（DRY_RUN）" if DRY_RUN else "实删"
    print(f"=== 比赛 {COMPETITION_ID}：{mode}，待处理 {len(ids)} 个 submission ===")
    print(f"提交根目录: {submissions_dir}\n")

    deleted = missing = failed = 0
    for sid in ids:
        sub_dir = submissions_dir / sid
        if not sub_dir.is_dir():
            print(f"[缺失] {sid} -> 目录不存在，跳过")
            missing += 1
            continue
        if DRY_RUN:
            print(f"[将删] {sub_dir}")
            deleted += 1
            continue
        try:
            shutil.rmtree(sub_dir)
            print(f"[已删] {sub_dir}")
            deleted += 1
        except OSError as e:
            print(f"[失败] {sid} -> {e}", file=sys.stderr)
            failed += 1

    verb = "将删除" if DRY_RUN else "已删除"
    print(f"\n汇总：{verb} {deleted}，缺失 {missing}，失败 {failed}")
    if DRY_RUN:
        print("当前为预览模式，未真正删除。确认无误后把 DRY_RUN 改为 False 再运行。")


if __name__ == "__main__":
    main()
