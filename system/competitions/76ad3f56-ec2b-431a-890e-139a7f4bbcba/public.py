import os
import sys
paths = ['/home/aiuser/work/workspace/BigAlpha/system/alphathonapiserver']
for path in paths:
    if path not in sys.path:
        sys.path.append(path)
from judge.judgebase import JudgeBase


# bigalpha_factorminer 源码目录：M.factorlens._latest 即该 bigmodule，
# 这里复用它的 DataProcess 预处理逻辑，把处理后的因子数据落盘。
# 注意：competitions 目录名与 eval 目录名同为本场比赛的目录 id（区别于下方的 competition_id）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR_ID = os.path.basename(_HERE)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_MINER_SRC = os.path.join(_REPO_ROOT, "eval", _DIR_ID, "src", "bigalpha_factorminer")


JUDGE_RUNNER_CODE = '''
__USER_CODE__

def judge_runner_main():
    import pandas as pd

    data = main("cpt_jyc_2025_stock_csi1000_bar1m_test", "2025-01-01", "2025-07-31 23:59:59")

    from bigmodule import M
    result = M.factorlens._latest(data=data, m_cached=False)

    with open("output.data", "w") as writer:
        writer.write(result._result.id)
'''.replace("__MINER_SRC__", _MINER_SRC)


class Judge(JudgeBase):
    competition_id = "76ad3f56-ec2b-431a-890e-139a7f4bbcba"
    mode = "public"
    JUDGE_RUNNER_CODE = JUDGE_RUNNER_CODE

    def compute_score(self, df):
        df["score"] = (
            df["rank_ic"].rank(pct=True) * 0.4
            + df["rank_ir"].rank(pct=True) * 0.3
            + df["sharp_ratio"].rank(pct=True) * 0.2
            + df["turnover"].rank(pct=True, ascending=False) * 0.1
        )
        return df


if __name__ == "__main__":
    Judge().run()
    """aaa"""
