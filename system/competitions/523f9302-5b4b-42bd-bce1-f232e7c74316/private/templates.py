"""端到端模型私榜 runner 模板。"""
from __future__ import annotations

_TEMPLATE = '''
__USER_CODE__

def judge_runner_main():
    import json
    score_data = main(__DATASETS__, "__DATE_START__", "__DATE_END__")
    if "score" in score_data.columns:
        score_data = score_data.rename(columns={"score": "factor"})
    from bigmodule import M
    result = M.bigalpha_eval._latest(
        factor_data=score_data,
        start_date="__DATE_START__",
        end_date="__DATE_END__",
        show=True,
    )
    result["raw_factor"].to_parquet("raw_score.parquet")
    result["process_factor"].to_parquet("process_score.parquet")
    with open("score_analyze.json", "w", encoding="utf-8") as writer:
        json.dump(result["factor_analyze"], writer, ensure_ascii=False, default=str)
'''


def build_runner(user_code: str, datasets: dict[str, str], start: str, end: str) -> str:
    return (_TEMPLATE.replace("__USER_CODE__", user_code)
            .replace("__DATASETS__", repr(datasets))
            .replace("__DATE_START__", start)
            .replace("__DATE_END__", end))
