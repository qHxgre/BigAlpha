# BigQuant V38: Tail-Locked Body Ranker

V38 turns the successful score-level ablation into one trainable model:

- a V37-compatible anchor learns tail geometry for 6 epochs;
- a small residual body head learns only the anchor-defined middle 80%;
- bottom/top 10% membership is determined only from the anchor prediction;
- final scores retain standardized anchor tail values and place body ranks
  strictly inside the two tail boundaries (`value_interval`).

V38 does not load V24 or V37 weights. The final checkpoint is one model trained
from scratch and public inference runs one raw-data encoder.

## Fixed Schedule

| Stage | Epochs | Trainable modules | Target/loss |
|---|---:|---|---|
| Anchor | 1-6 | raw encoder, GRU, dynamic factor prior/posterior | 0.65 Huber + 0.35 daily correlation, plus V37 teacher/regularizers |
| Body | 7-10 | body MLP only | middle-80% linear rank: 0.60 Huber + 0.40 daily correlation |

The anchor really stops after epoch 6, but its learning-rate scheduler retains
V37's original 14-epoch horizon. This reproduces the learning-rate path used by
the V37 epoch-6 checkpoint instead of prematurely decaying the LR.

Stage B first performs one frozen anchor pass and writes about 0.5 GiB of FP16
stock embeddings. Four body epochs reuse this cache. It is deleted in a
`finally` block after training or failure.

## AI Studio: Development Training

Upload `v38_submit.py`, then run:

```python
import importlib
import sys

sys.path.insert(0, "/home/aiuser/work/v38")
import v38_submit as v38
importlib.reload(v38)

datasources = {
    "bar5m": "bigalpha_2026_stock_bar5m",
    "bar1m": "bigalpha_2026_stock_bar1m",
}

development_path = v38.train_development_and_save(
    datasources,
    model_path="/home/aiuser/work/v38/v38_development_2019_2023.json",
    cache_dir="/home/aiuser/work/v24_final_cache",
    checkpoint_dir="/home/aiuser/work/v38/checkpoints",
    force_cache=False,
)
print(development_path)
print(v38.checkpoint_file_info(development_path))
```

The expected checkpoints are:

```text
v38_epoch_06_anchor.json
v38_epoch_07_body.json
v38_epoch_08_body.json
v38_epoch_09_body.json
v38_epoch_10_body.json
```

## Shared-Data Checkpoint Evaluation

`v38_checkpoint_eval.py` is local-only and should not be uploaded in the public
submission. It verifies that all checkpoints share the exact frozen anchor,
queries every month once, runs the raw encoder once, then evaluates all body
heads on the shared stock states.

```python
import importlib
import sys

sys.path.insert(0, "/home/aiuser/work/v38")
import v38_checkpoint_eval as v38_eval
importlib.reload(v38_eval)

paths = v38_eval.checkpoint_paths(
    "/home/aiuser/work/v38/checkpoints"
)
scores, timing = v38_eval.predict_checkpoints_shared_data(
    datasources,
    paths,
    score_dir="/home/aiuser/work/v38/checkpoint_scores",
)
print(timing)
```

Evaluate the generated scores with the official module:

```python
from bigmodule import M
import pandas as pd

rows = []
for name, score_data in scores.items():
    result = M.bigalpha_eval._latest(
        factor_data=score_data,
        start_date="2024-01-01",
        end_date="2024-12-31",
        show=False,
    )
    metrics = result["factor_analyze"]
    rows.append({
        "checkpoint": name,
        "ic_mean": metrics["ic_mean"],
        "ic_ir": metrics["ic_ir"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "stress_ic_ir": metrics["stress_ic_ir"],
    })

pd.DataFrame(rows).sort_values("ic_ir", ascending=False)
```

The epoch-6 checkpoint is evaluated with V38's final `value_interval` output
rule and a zero body correction. It is therefore the correct architectural
control, not the original raw V37 score.

## Full 2019-2024 Training

After selecting the fixed V38 schedule, run:

```python
final_path = v38.train_and_save(
    datasources,
    model_path="/home/aiuser/work/v38/v38_tail_locked_body_model.json",
    cache_dir="/home/aiuser/work/v24_final_cache",
    checkpoint_dir=None,
    force_cache=False,
)
print(final_path)
print(v38.checkpoint_file_info(final_path))
```

## Local Full-2024 Inference

```python
score_data = v38.main_with_model_path(
    datasources,
    start_date="2024-01-01 00:00:00",
    end_date="2024-12-31 23:59:59",
    model_path="/home/aiuser/work/v38/v38_development_2019_2023.json",
)

assert list(score_data.columns) == ["date", "instrument", "score"]
assert not score_data.duplicated(["date", "instrument"]).any()
assert score_data["score"].notna().all()
print(score_data.shape)
```

## Submission Files

Upload exactly these six files:

```text
v38_submission.ipynb
v38_submit.py
v38_tail_locked_body_model.json
v38_config.json
README_v38.md
requirements.txt
```

Do not upload development checkpoints or `v38_checkpoint_eval.py`. Multiple
V38 model JSON files make automatic model resolution intentionally fail rather
than silently selecting the wrong weight.

## Efficiency Checks

```python
print(v38.validate_model_invariants())
print(v38.benchmark_model_step())
print(v38.benchmark_inference_step())
```

Public inference has the same monthly query and unique stock-day encoding path
as V37. The additional work is a 192→256→128→1 MLP and one sort of roughly
1,000 values per day; no second raw-data model and no stock-by-stock attention
matrix are used.
