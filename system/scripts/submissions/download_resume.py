"""从 resume_info.csv 中按学校白名单筛选简历，批量下载 PDF 附件。

resume_info.csv（common.paths.DATA_ROOT/resume/resume_info.csv）每行是一份投递记录，
列为 name/school/attachment，attachment 是一段 JSON，含 wiki 附件 id/name 等字段。
本脚本按 TARGET_SCHOOLS 白名单（含中英文及常见简称的归一化匹配）筛出目标学校的
简历，逐个下载 PDF 到本地目录，并写出一份下载清单 CSV。

下载方式说明（实测得出）：
    /wiki/api/attachments.redirect?id=<id> 会 302 跳转到 CDN 静态地址，但该 CDN 地址
    常年返回 404；直接请求 {server}/wiki/static/upload/{id[:2]}/{id}.pdf 不带 Range
    头同样 404，但带上 `Range: bytes=0-` 请求头就能拿到完整文件（206 Partial
    Content，Content-Range 里的总大小与实际文件一致）。鉴权用 common.auth.load_auth()
    读到的 token，作为 Cookie: bigjwt=<token> 带上。

用法：
    python system/scripts/submissions/download_resume.py       # 先看筛选结果，确认后下载
    python system/scripts/submissions/download_resume.py -y     # 跳过确认，直接下载（用于自动化）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 使 `from common...` 可用
from common.auth import load_auth
from common.paths import DATA_ROOT

try:
    import requests
except ImportError:
    print("请先安装依赖: python3 -m pip install requests", file=sys.stderr)
    sys.exit(1)

# ===== 配置 =================================================================
RESUME_DIR = DATA_ROOT / "resume"
RESUME_CSV = RESUME_DIR / "resume_info.csv"
DOWNLOAD_DIR = RESUME_DIR / "downloads"
MANIFEST_CSV = RESUME_DIR / "downloaded_manifest.csv"

# 每次下载之间的间隔（秒），避免短时间内请求过密。
SLEEP_BETWEEN = 0.3

# 目标学校白名单：仅保留最顶尖一批（国内 C9 头部 + 港三大/CUHK深圳 + 海外 Top，
# 如 MIT/Stanford/Harvard/Oxbridge/Ivy 部分校等），单校份数不做上限，
# 命中总量取决于实际投递分布。
# key 为学校名的各种写法（中英文/简称/大小写变体），value 为归一后的规范名，
# 归一化时会去空格、全角转半角、转小写后再比较，所以这里不用逐一穷举大小写。
TARGET_SCHOOLS: dict[str, str] = {
    # ---- 国内 C9 ----
    "清华": "清华大学", "清华大学": "清华大学",
    "北京大学": "北京大学", "Peking University": "北京大学", "北京大学/stony brook university": "北京大学",
    "复旦大学": "复旦大学",
    "上海交通大学": "上海交通大学",
    "浙江大学": "浙江大学",
    "南京大学": "南京大学",
    "中国科学技术大学": "中国科学技术大学", "university of science and technology of china": "中国科学技术大学",
    # ---- 港三大 + CUHK深圳 ----
    "香港大学": "香港大学", "The University of Hong Kong": "香港大学", "University of Hong Kong": "香港大学", "HKU": "香港大学",
    "香港中文大学": "香港中文大学", "香港中文大學": "香港中文大学",
    "香港科技大学": "香港科技大学",
    # ---- 海外顶尖（美/英/新/瑞士等） ----
    "MIT": "MIT",
    "Stanford University": "Stanford University", "斯坦福大学": "Stanford University",
    "Harvard University": "Harvard University",
    "University of Cambridge": "University of Cambridge", "剑桥大学": "University of Cambridge",
    "University of Oxford": "University of Oxford", "牛津大学": "University of Oxford",
    "Imperial College London": "Imperial College London", "帝国理工": "Imperial College London",
    "帝国理工大学": "Imperial College London", "帝国理工学院": "Imperial College London",
    "ETH Zurich": "ETH Zurich",
    "University of Chicago": "University of Chicago",
    "Princeton University": "Princeton University", "Princeton university": "Princeton University",
    "Cornell University": "Cornell University", "cornell University": "Cornell University",
    "Columbia University": "Columbia University", "哥伦比亚大学": "Columbia University",
    "University of Pennsylvania": "University of Pennsylvania", "宾夕法尼亚大学": "University of Pennsylvania",
    "Caltech": "Caltech", "加州理工学院": "Caltech",
    "Carnegie Mellon University (CMU)": "Carnegie Mellon University",
    "UC Berkeley": "UC Berkeley", "加州大学伯克利分校": "UC Berkeley", "加州伯克利": "UC Berkeley",
}
# ===========================================================================


def normalize_school(name: str) -> str:
    """归一化学校名：全角转半角、去空格/句点/中间点，转小写，方便做别名匹配。"""
    s = unicodedata.normalize("NFKC", (name or "").strip())
    s = s.replace("（", "(").replace("）", ")").replace("，", ",")
    s = s.replace("。", "").replace("·", "").replace(" ", "")
    return s.lower()


_NORMALIZED_LOOKUP = {normalize_school(k): v for k, v in TARGET_SCHOOLS.items()}


def match_school(school: str) -> str | None:
    """学校名是否在白名单里，命中则返回规范名，否则 None。"""
    return _NORMALIZED_LOOKUP.get(normalize_school(school))


def load_candidates(csv_path: Path) -> list[dict]:
    """读 resume_info.csv，按白名单筛出命中的行，附上归一后的学校名和附件信息。"""
    if not csv_path.exists():
        print(f"未找到简历信息文件: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            school = (row.get("school") or "").strip()
            canon = match_school(school)
            if not canon:
                continue
            attachment_raw = row.get("attachment") or ""
            try:
                attachment = json.loads(attachment_raw)
            except json.JSONDecodeError:
                continue
            attachment_id = (attachment.get("id") or "").strip()
            if not attachment_id:
                continue
            rows.append({
                "name": (row.get("name") or "").strip(),
                "school": school,
                "school_canonical": canon,
                "attachment_id": attachment_id,
                "attachment_name": attachment.get("name") or "",
            })
    return rows


def build_pdf_url(server: str, attachment_id: str) -> str:
    """附件下载地址：{server}/wiki/static/upload/<id前2位>/<id>.pdf。"""
    return f"{server}/wiki/static/upload/{attachment_id[:2]}/{attachment_id}.pdf"


def safe_filename(name: str, school: str, attachment_id: str) -> str:
    """生成本地文件名：<姓名>_<学校>_<附件id前8位>.pdf，过滤路径不安全字符。"""
    def clean(s: str) -> str:
        s = s.strip() or "unknown"
        for ch in '/\\:*?"<>|':
            s = s.replace(ch, "_")
        return s
    return f"{clean(name)}_{clean(school)}_{attachment_id[:8]}.pdf"


def download_one(session: requests.Session, server: str, token: str, attachment_id: str, dest: Path) -> tuple[bool, str]:
    """下载单份简历 PDF。必须带 Range 头，否则该接口会返回 404（见文件头说明）。"""
    url = build_pdf_url(server, attachment_id)
    headers = {
        "Cookie": f"bigjwt={token}",
        "Range": "bytes=0-",
        "User-Agent": "Mozilla/5.0",
    }
    try:
        resp = session.get(url, headers=headers, timeout=30)
    except requests.RequestException as e:
        return False, f"请求异常: {e}"
    if resp.status_code not in (200, 206):
        return False, f"HTTP {resp.status_code}"
    if resp.headers.get("Content-Type", "").split(";")[0].strip() != "application/pdf":
        return False, f"非 PDF 响应(Content-Type={resp.headers.get('Content-Type')})"
    dest.write_bytes(resp.content)
    return True, "OK"


def print_school_breakdown(candidates: list[dict]) -> None:
    """按规范学校名统计简历份数，从多到少打印，方便下载前核对筛选结果。"""
    counts = Counter(c["school_canonical"] for c in candidates)
    print(f"命中白名单学校的简历: 共 {len(candidates)} 份，涉及 {len(counts)} 所学校：")
    for school, n in counts.most_common():
        print(f"  {school}: {n} 份")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-y", "--yes", action="store_true", help="跳过确认，直接下载")
    args = parser.parse_args()

    candidates = load_candidates(RESUME_CSV)
    if not candidates:
        print("没有命中任何简历，请检查 TARGET_SCHOOLS 配置。", file=sys.stderr)
        sys.exit(1)

    print_school_breakdown(candidates)

    if not args.yes:
        answer = input(f"\n确认下载以上 {len(candidates)} 份简历吗？[y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消，未下载任何文件。")
            return

    token, server = load_auth()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    manifest_rows: list[dict] = []
    ok_count = fail_count = 0
    for i, c in enumerate(candidates, 1):
        filename = safe_filename(c["name"], c["school_canonical"], c["attachment_id"])
        dest = DOWNLOAD_DIR / filename
        if dest.exists():
            print(f"  [{i}/{len(candidates)}] 已存在，跳过: {filename}")
            manifest_rows.append({**c, "file_path": str(dest), "download_ok": True, "error": ""})
            ok_count += 1
            continue

        ok, msg = download_one(session, server, token, c["attachment_id"], dest)
        manifest_rows.append({**c, "file_path": str(dest) if ok else "", "download_ok": ok, "error": "" if ok else msg})
        if ok:
            ok_count += 1
            print(f"  [{i}/{len(candidates)}] ✓ {filename}")
        else:
            fail_count += 1
            print(f"  [{i}/{len(candidates)}] ✗ {c['name']} ({c['school']}) —— {msg}", file=sys.stderr)
        if SLEEP_BETWEEN:
            time.sleep(SLEEP_BETWEEN)

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["name", "school", "school_canonical", "attachment_id", "attachment_name", "file_path", "download_ok", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\n=== 完成：成功 {ok_count} 份，失败 {fail_count} 份 ===")
    print(f"下载目录: {DOWNLOAD_DIR}")
    print(f"清单文件: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
