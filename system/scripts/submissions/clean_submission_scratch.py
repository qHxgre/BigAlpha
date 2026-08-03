"""清理已存在的 submission 目录下残留的数据缓存目录（历史遗留，非评测产物）。

背景：端到端模型赛道（competition_id 523f9302-5b4b-42bd-bce1-f232e7c74316）的用户代码
调用平台数据接口（dai.query 等）时，会在提交目录下自行产出查询结果缓存目录，命名不固定
但基本都带 cache 字样（如 bigalpha_memmap_cache_P4_raw_trunk_gated_relative_aux/），
单个能到几个 GB；runner.py 的 _cleanup_scratch 之前只清固定名字的 .cache/tmp，扫不到
这类命名各异的缓存，导致历史提交目录里一直攒着。

清理规则放宽为：目录名包含 "cache"（大小写不敏感）且目录总大小超过 SIZE_THRESHOLD_GB
（默认 1GB）才会被清理，避免误删体积很小、可能另有用途的同名目录。只扫每个 submission
子目录根一级，submission 目录下的其它内容（提交原始文件、judge_runner.py、stdout、
各产物文件等）不动。

用法：
    python system/scripts/submissions/clean_submission_scratch.py       # 先看会删哪些目录
    python system/scripts/submissions/clean_submission_scratch.py -y    # 确认后直接删

默认先打印预览、要求确认才真正删；加 -y 跳过确认。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用

from common.paths import resolve_submissions_dir

# ===== 配置：要清哪场比赛 =====================================================
COMPETITION_ID = "523f9302-5b4b-42bd-bce1-f232e7c74316"

# 目录名包含该关键字（大小写不敏感）才算候选；再叠加下面的大小阈值判断是否真的清理。
CACHE_NAME_KEYWORD = "cache"
SIZE_THRESHOLD_GB = 1.0
SIZE_THRESHOLD_BYTES = int(SIZE_THRESHOLD_GB * 1024**3)
# ===========================================================================


def find_scratch_dirs(submissions_dir: Path) -> list[Path]:
    """遍历每个 submission 子目录，收集其根下名字含 CACHE_NAME_KEYWORD 的子目录。"""
    scratch_dirs: list[Path] = []
    for sub_dir in sorted(p for p in submissions_dir.iterdir() if p.is_dir()):
        for entry in sorted(sub_dir.iterdir()):
            if entry.is_dir() and CACHE_NAME_KEYWORD in entry.name.lower():
                scratch_dirs.append(entry)
    return scratch_dirs


def dir_size(path: Path) -> int:
    """粗略统计目录总大小（字节），读不到的文件跳过不计。"""
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-y", "--yes", action="store_true", help="跳过确认，直接删除")
    args = parser.parse_args()

    submissions_dir = resolve_submissions_dir(COMPETITION_ID)
    if not submissions_dir.is_dir():
        print(f"提交目录不存在: {submissions_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"=== 比赛 {COMPETITION_ID}：扫描提交根目录 {submissions_dir} ===")
    print(f"匹配规则：目录名含 '{CACHE_NAME_KEYWORD}'（不区分大小写）且大小超过 {SIZE_THRESHOLD_GB}GB\n")
    candidates = find_scratch_dirs(submissions_dir)
    if not candidates:
        print("未发现任何名字含 cache 的目录，无需清理。")
        return

    scratch_dirs = []
    sizes = []
    skipped = 0
    for d in candidates:
        size = dir_size(d)
        if size > SIZE_THRESHOLD_BYTES:
            scratch_dirs.append(d)
            sizes.append(size)
        else:
            skipped += 1

    if not scratch_dirs:
        print(f"发现 {len(candidates)} 个含 cache 的目录，但均未超过 {SIZE_THRESHOLD_GB}GB，无需清理。")
        return

    print(f"发现 {len(scratch_dirs)} 个超过 {SIZE_THRESHOLD_GB}GB 的残留目录（另有 {skipped} 个因体积过小被跳过）：")
    total = 0
    for d, size in zip(scratch_dirs, sizes):
        total += size
        print(f"  {human_size(size):>10}  {d}")
    print(f"\n合计大小: {human_size(total)}")

    if not args.yes:
        answer = input(f"\n确认删除以上 {len(scratch_dirs)} 个目录吗？[y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消，未删除任何文件。")
            return

    deleted = failed = 0
    for d, size in zip(scratch_dirs, sizes):
        try:
            shutil.rmtree(d)
            print(f"[已删] ({human_size(size)}) {d}")
            deleted += 1
        except OSError as e:
            print(f"[失败] {d} -> {e}", file=sys.stderr)
            failed += 1

    print(f"\n汇总：已删除 {deleted}，失败 {failed}，释放约 {human_size(total)}")


if __name__ == "__main__":
    main()
