"""人工确认后发布私榜批次。"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
API_SERVER = os.path.abspath(os.path.join(HERE, "..", "..", "..", "alphathonapiserver"))
for path in (API_SERVER, HERE):
    if path not in sys.path:
        sys.path.append(path)

from judge.api import AlphathonAPI
from judge.paths import FILE_DIR
from fileio import update_manifest, write_jsonl

COMPETITION_ID = "523f9302-5b4b-42bd-bce1-f232e7c74316"
RUNS_DIR = os.path.join(FILE_DIR, COMPETITION_ID, "private", "runs")
RUN_DIR = "/Users/xiehao/Desktop/workspace/BigAlpha/system/files/private/523f9302-5b4b-42bd-bce1-f232e7c74316/private/runs/20260812_180115"
DRY_RUN = True


def load(path: str):
    with open(path, encoding="utf-8") as reader:
        return json.load(reader)


def find_run() -> str:
    if RUN_DIR:
        return os.path.abspath(RUN_DIR)
    if not os.path.isdir(RUNS_DIR):
        raise RuntimeError(f"批次目录不存在: {RUNS_DIR}")
    candidates = []
    for name in os.listdir(RUNS_DIR):
        path = os.path.join(RUNS_DIR, name)
        manifest_path = os.path.join(path, "manifest.json")
        if os.path.isfile(manifest_path):
            manifest = load(manifest_path)
            if manifest.get("status") == "review_pending" and not manifest.get("published"):
                candidates.append(path)
    if not candidates:
        raise RuntimeError("没有待发布的私榜批次")
    return max(candidates, key=os.path.getmtime)


def main() -> int:
    run = find_run()
    manifest_path = os.path.join(run, "manifest.json")
    manifest = load(manifest_path)
    if manifest.get("competition_id") != COMPETITION_ID or manifest.get("mode") != "private":
        raise RuntimeError("批次不属于本比赛私榜")
    pending = []
    with open(os.path.join(run, "pending_publish.jsonl"), encoding="utf-8") as reader:
        pending = [json.loads(line) for line in reader if line.strip()]
    if not pending:
        raise RuntimeError("待发布列表为空")
    if len(pending) != int(manifest.get("submission_count") or 0):
        raise RuntimeError("待发布记录数量与批次 submission_count 不一致")
    for record in pending:
        print(record["submission_id"], record["payload"].get("private_score"))
    if DRY_RUN:
        return 0
    if input("输入 PUBLISH 确认发布：").strip() != "PUBLISH":
        return 1
    results, failures = [], 0
    api = AlphathonAPI()
    for record in pending:
        result = dict(record)
        try:
            api.update_submission_score(submission_id=record["submission_id"], **record["payload"])
            result["status"] = "success"
        except Exception as exc:
            failures += 1
            result.update(status="failed", error=str(exc))
        results.append(result)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"publish_result_{stamp}.jsonl"
    write_jsonl(os.path.join(run, result_file), results)
    update_manifest(manifest_path, status="publish_failed" if failures else "published",
                    published=not failures, publish_failures=failures,
                    publish_result_file=result_file)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
