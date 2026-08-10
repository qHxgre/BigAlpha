"""下载并固化所有 selected_for_private=True 的提交。"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
API_SERVER = os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver"))
for path in (API_SERVER, HERE):
    if path not in sys.path:
        sys.path.append(path)

from judge.api import AlphathonAPI
from judge.paths import FILE_DIR
from fileio import jsonable, write_json

COMPETITION_ID = "523f9302-5b4b-42bd-bce1-f232e7c74316"
DEFAULT_OUTPUT = os.path.join(FILE_DIR, COMPETITION_ID, "private", "prepared")


def submission_files(submission: dict) -> dict:
    files = (submission.get("data") or {}).get("files")
    if files is None:
        files = submission.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"files 字段类型错误: {type(files).__name__}")
    return files


def safe_name(file_id: str, info: dict | None, used: set[str]) -> str:
    raw = (info or {}).get("name") or file_id
    name = os.path.basename(str(raw).replace("\\", "/")) or file_id
    stem, suffix = os.path.splitext(name)
    candidate, index = name, 2
    while candidate.casefold() in used:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-id", default=time.strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    output = os.path.abspath(args.output_root)
    os.makedirs(output, exist_ok=True)
    staging = os.path.join(output, f".preparing-{os.getpid()}-{time.time_ns()}")
    os.makedirs(os.path.join(staging, "submissions"))

    api = AlphathonAPI()
    selected = api.query_submissions(
        competition_id=COMPETITION_ID,
        constraints={"selected_for_private": True},
    )
    if not selected:
        shutil.rmtree(staging, ignore_errors=True)
        print("没有 selected_for_private=True 的 submission", file=sys.stderr)
        return 1

    records, errors = [], []
    for submission in selected:
        sid = str(submission["id"])
        destination = os.path.join(staging, "submissions", sid)
        os.makedirs(destination)
        downloaded, used = [], set()
        try:
            for file_id, info in submission_files(submission).items():
                if info is not None and not isinstance(info, dict):
                    raise RuntimeError(f"文件 {file_id} 元数据类型错误")
                name = safe_name(str(file_id), info, used)
                api.get_submission_file(sid, str(file_id), info, save_to=os.path.join(destination, name))
                downloaded.append({"file_id": str(file_id), "name": name})
            records.append({
                "submission_id": sid,
                "user_id": str(submission.get("user_id")),
                "public_score": submission.get("public_score"),
                "relative_path": f"submissions/{sid}",
                "files": downloaded,
                "submission": jsonable(submission),
            })
        except Exception as exc:
            errors.append({"submission_id": sid, "error": f"{type(exc).__name__}: {exc}"})
            shutil.rmtree(destination, ignore_errors=True)

    if errors:
        write_json(os.path.join(output, "preparation_errors.json"), {"errors": errors})
        shutil.rmtree(staging, ignore_errors=True)
        print(f"{len(errors)} 个 submission 下载失败", file=sys.stderr)
        return 1

    metadata = {
        "competition_id": COMPETITION_ID,
        "batch_id": args.batch_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "summary": {"private_submission_count": len(records)},
        "submissions": records,
    }
    write_json(os.path.join(staging, "metadata.json"), metadata)
    target = os.path.join(output, "submissions")
    if os.path.exists(target):
        shutil.rmtree(target)
    os.replace(os.path.join(staging, "submissions"), target)
    os.replace(os.path.join(staging, "metadata.json"), os.path.join(output, "metadata.json"))
    shutil.rmtree(staging, ignore_errors=True)
    error_path = os.path.join(output, "preparation_errors.json")
    if os.path.exists(error_path):
        os.remove(error_path)
    print(f"prepared submissions: {len(records)}; output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
