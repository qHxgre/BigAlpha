"""Frozen 2024 validation entry point for the raw dual-view Transformer."""
from pathlib import Path

from bigmodule import M

import e2e_dual_view_transformer as dual
import validate_full_universe_streaming_2024 as streaming


HERE = Path(__file__).resolve().parent
MODEL_PATH = str(HERE / "e2e_dual_view_streaming.json")
SCORES_PATH = HERE / "e2e_dual_view_streaming_scores_2024.parquet"


if __name__ == "__main__":
    streaming.model = dual
    streaming.MODEL_PATH = MODEL_PATH
    print("trainable_parameters", dual.model_parameter_count(), flush=True)
    streaming.train_streaming(dual.TRAIN_TABLE)
    scores = dual.predict(
        {"bar5m": dual.TRAIN_TABLE},
        "2024-01-01 00:00:00",
        "2024-12-31 23:59:59",
        MODEL_PATH,
    )
    scores.to_parquet(SCORES_PATH, index=False)
    coverage = scores.groupby("date", observed=True)["instrument"].size()
    print(
        "score_contract",
        len(scores),
        scores["date"].nunique(),
        int(coverage.min()),
        int(coverage.max()),
        flush=True,
    )
    M.bigalpha_eval._latest(factor_data=scores, show=False)
