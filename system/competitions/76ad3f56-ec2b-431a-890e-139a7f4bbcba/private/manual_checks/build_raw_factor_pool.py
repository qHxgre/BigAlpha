"""从各 submission 的原始因子文件构建 factor_pool_raw.parquet。

默认以正式 ``factor_pool.parquet`` 的因子列为准，确保原始因子池和正式因子池
包含完全相同的 submission，并保持相同的列顺序。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .config import PATHS, CheckPaths


KEY_COLUMNS = ("date", "instrument")
RAW_FACTOR_FILENAME = "raw_factor.parquet"
RAW_FACTOR_POOL_FILENAME = "factor_pool_raw.parquet"


def build_raw_factor_pool(
    paths: CheckPaths = PATHS,
    *,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """构建与正式因子池成员一致的原始因子池。

    Parameters
    ----------
    paths:
        一次私榜运行的路径配置。
    output_path:
        输出文件；默认写入 artifacts/factor_pool_raw.parquet。
    overwrite:
        是否覆盖已经存在的输出文件。
    """
    processed_pool_path = paths.factor_pool_path
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else paths.artifacts_dir / RAW_FACTOR_POOL_FILENAME
    )

    if not processed_pool_path.exists():
        raise FileNotFoundError(f"正式因子池不存在: {processed_pool_path}")
    if target.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在，请传入 overwrite=True: {target}")

    processed_columns = list(pd.read_parquet(processed_pool_path).columns)
    submission_ids = [column for column in processed_columns if column not in KEY_COLUMNS]
    if not submission_ids:
        raise ValueError(f"正式因子池中没有 submission 因子列: {processed_pool_path}")

    print(
        f"[原始因子池] 以正式因子池为准，共 {len(submission_ids)} 个 submission",
        flush=True,
    )
    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for index, submission_id in enumerate(submission_ids, start=1):
        raw_path = paths.run_dir / "submissions" / submission_id / RAW_FACTOR_FILENAME
        if not raw_path.exists():
            errors.append(f"{submission_id}: 文件不存在 ({raw_path})")
            continue

        try:
            raw = pd.read_parquet(raw_path, columns=[*KEY_COLUMNS, "factor"])
        except Exception as exc:
            errors.append(f"{submission_id}: 读取失败 ({exc})")
            continue

        duplicated = raw.duplicated(list(KEY_COLUMNS), keep=False)
        if duplicated.any():
            errors.append(
                f"{submission_id}: date/instrument 存在 {int(duplicated.sum())} 行重复记录"
            )
            continue

        frame = raw.rename(columns={"factor": submission_id}).set_index(list(KEY_COLUMNS))
        frames.append(frame)
        if index == 1 or index % 10 == 0 or index == len(submission_ids):
            print(
                f"[原始因子池] 已读取 {index}/{len(submission_ids)} 个 submission",
                flush=True,
            )

    if errors:
        detail = "\n".join(f"  - {error}" for error in errors)
        raise RuntimeError(
            f"原始因子池构建中止，{len(errors)} 个 submission 无法入池:\n{detail}"
        )

    pool = pd.concat(frames, axis=1, join="outer").reset_index()
    pool = pool[[*KEY_COLUMNS, *submission_ids]]
    pool = pool.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        pool.to_parquet(temporary, index=False)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    print(
        f"[原始因子池] 构建完成: {target}，{len(pool):,} 行，"
        f"{len(submission_ids)} 个因子",
        flush=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="自定义输出 parquet 路径")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件")
    args = parser.parse_args()
    build_raw_factor_pool(output_path=args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
