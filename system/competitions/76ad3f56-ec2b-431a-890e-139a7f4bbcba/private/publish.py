"""人工核验后发布私榜批次中的分数。"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

# publish.py 作为脚本直接运行；入口形式与 public.py / private.py 保持一致。
paths = [
    "/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver",
    os.path.dirname(os.path.abspath(__file__)),
]
for path in paths:
    if path not in sys.path:
        sys.path.append(path)

from judge.api import AlphathonAPI

COMPETITION_ID = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as reader:
        return json.load(reader)


def write_json_atomic(path: str, value: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as writer:
        json.dump(value, writer, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def load_latest_payloads(path: str) -> dict[str, dict]:
    """同一 submission 多次更新时，按写入顺序合并成最终发布快照。"""
    latest: dict[str, dict] = {}
    with open(path, encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sid = str(record["submission_id"])
            payload = record.get("payload") or {}
            if not payload:
                raise ValueError(f"line {line_no}: payload 为空")
            latest.setdefault(sid, {}).update(payload)
    return latest


class Publisher:
    """发布入口：默认选择最新的待审核批次，并在发布前要求人工确认。"""

    RUNS_DIR = os.path.join(
        "/home/aiuser/work/workspace/BigAlpha/system/files",
        COMPETITION_ID,
        "private",
        "runs",
    )
    # 留空时自动选择最新的 review_pending 批次；也可以填入明确的批次目录。
    RUN_DIR = ""
    # 临时预览时改为 True；跳过人工确认仅用于已由外部流程完成确认的场景。
    DRY_RUN = False
    SKIP_CONFIRMATION = False

    def _find_run_dir(self) -> str:
        if self.RUN_DIR:
            return os.path.abspath(self.RUN_DIR)
        if not os.path.isdir(self.RUNS_DIR):
            raise RuntimeError(f"runs directory does not exist: {self.RUNS_DIR}")

        candidates = []
        for root, dirs, _files in os.walk(self.RUNS_DIR):
            # submissions/artifacts 等内部目录不可能是批次根目录，无需继续递归。
            dirs[:] = [
                name
                for name in dirs
                if name not in {"submissions", "artifacts", "logs", "analyze"}
            ]
            run_dir = root
            manifest_path = os.path.join(run_dir, "manifest.json")
            pending_path = os.path.join(run_dir, "pending_publish.jsonl")
            if not os.path.isfile(manifest_path) or not os.path.isfile(pending_path):
                continue
            try:
                manifest = load_json(manifest_path)
            except (OSError, ValueError):
                continue
            if (
                manifest.get("competition_id") == COMPETITION_ID
                and manifest.get("mode") == "private"
                and manifest.get("status") == "review_pending"
                and not manifest.get("published")
            ):
                candidates.append(run_dir)
        if not candidates:
            raise RuntimeError(f"no review_pending batch found in: {self.RUNS_DIR}")
        return max(candidates, key=os.path.getmtime)

    def run(self) -> int:
        try:
            run_dir = self._find_run_dir()
        except RuntimeError as exc:
            print(exc)
            return 1

        manifest_path = os.path.join(run_dir, "manifest.json")
        pending_path = os.path.join(run_dir, "pending_publish.jsonl")
        if not os.path.isfile(manifest_path) or not os.path.isfile(pending_path):
            print(f"invalid run directory: {run_dir}")
            return 1

        manifest = load_json(manifest_path)
        if manifest.get("competition_id") != COMPETITION_ID or manifest.get("mode") != "private":
            print("manifest does not belong to this private competition")
            return 1
        if manifest.get("status") != "review_pending":
            print(f"batch is not ready for review/publish: status={manifest.get('status')}")
            return 1
        if manifest.get("published"):
            print("batch has already been published; refusing duplicate publication")
            return 1

        payloads = load_latest_payloads(pending_path)
        expected = int(manifest.get("submission_count") or 0)
        print(f"run directory: {run_dir}")
        print(f"batch: {manifest.get('batch_id')}")
        print(f"evaluation range: {manifest.get('date_start')} -> {manifest.get('date_end')}")
        print(f"selected submissions: {expected}; submissions with pending score: {len(payloads)}")
        for sid, payload in sorted(payloads.items()):
            print(f"- {sid}: private_score={payload.get('private_score')}")
        if not payloads:
            print("no pending scores found; refusing publication")
            return 1
        if len(payloads) != expected:
            print("warning: pending score count differs from selected submission count")
        if self.DRY_RUN:
            print("dry-run only; no score was published")
            return 0

        if not self.SKIP_CONFIRMATION:
            answer = input("确认已人工核验该批次？输入 PUBLISH 后更新线上私榜：").strip()
            if answer != "PUBLISH":
                print("publish cancelled; no score was published")
                return 1

        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(run_dir, f"publish_result_{timestamp}.jsonl")
        failures = 0
        api = AlphathonAPI()
        with open(result_path, "w", encoding="utf-8") as result_file:
            for sid, payload in sorted(payloads.items()):
                result = {"submission_id": sid, "payload": payload}
                try:
                    api.update_submission_score(submission_id=sid, **payload)
                    result["status"] = "success"
                except Exception as exc:
                    failures += 1
                    result.update(status="failed", error=str(exc))
                result_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                result_file.flush()

        manifest.update(
            publish_attempted_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            publish_result_file=os.path.basename(result_path),
            publish_failures=failures,
        )
        if failures:
            manifest["status"] = "publish_failed"
            write_json_atomic(manifest_path, manifest)
            print(f"publish completed with {failures} failure(s); see {result_path}")
            return 1

        manifest.update(status="published", published=True)
        write_json_atomic(manifest_path, manifest)
        print(f"publish succeeded: {len(payloads)} submission(s)")
        print(f"result log: {result_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(Publisher().run())
