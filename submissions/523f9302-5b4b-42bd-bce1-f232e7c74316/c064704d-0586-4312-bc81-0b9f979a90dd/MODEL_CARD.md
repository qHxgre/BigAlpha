# Model card: strict 2019-2023 holdout ensemble

This is the conservative submission variant for the wording that defines
2019-2023 as public training data and 2024 as public validation data.

It uses the same two-branch, 3,052,866-parameter end-to-end architecture and
the same 26 raw fields as the primary model. Both branches were trained from
scratch on 2019-01-01 through 2023-12-31. Scores are fused 50/50 after
same-day cross-sectional standardization.

On 241 adjacent-day pairs in the untouched 2024 holdout, local diagnostics
before organizer BARRA residualization were:

| Metric | Value |
|---|---:|
| Pearson IC mean | 0.138198 |
| Pearson ICIR | 1.256354 |
| Rank IC mean | 0.086641 |
| Rank ICIR | 0.678749 |
| Ten-group long-short Sharpe | 9.220408 |
| Worst quarterly Pearson IC | 0.116180 |
| Worst monthly Pearson IC | 0.080658 |

Only per-field logarithms, training-set standardization, and missing-value
filling are used. There are no engineered factors, external datasets, or
external pretrained weights. Exact branch hyperparameters and seeds are in
`model_config.json`; `train_model.py` provides the private retraining path.
