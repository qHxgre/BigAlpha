# BigAlpha2026 AI-track reproducibility note — H09

## Factor identity

- Candidate: `H09`
- Frozen source: `factors/stage4_h09_range_residual_after_volatility5.py`
- Source SHA-256: `9cfd19ca3a2c83bffa4fd772ab68e99e170033b3ee6be5dc3c27566bf287d4d1`

## Formula and frozen direction

- Formula: compute `(max(high) - min(low)) / mean(valid pre_close)` by date and instrument; within each date apply average-tie rank z-scores to that daily range and `volatility_5`; regress the range rank-z on an intercept and volatility rank-z. The frozen research signal is the residual.
- Frozen research direction: negative; lower residual values are favorable. The competition evaluator assumes larger submitted values are better, so the submission adapter returns the negative residual. This is a deterministic orientation conversion of the preregistered direction, not a formula, parameter, or post-result direction change.
- Freeze point: source commit `b826520ddaf1b901852237bd67a97ee40cf65e29`, committed `2026-07-23T13:23:46+08:00`.

## Data, date range, and universe

- Replaceable minute source: `datasources["bar1m"]` fields `date,instrument,high,low,pre_close`; fixed auxiliary sources: `bigalpha_2026_factorlib` fields `date,instrument,volatility_5` and `bigalpha_2026_instruments` fields `date,instrument`.
- Public-training interval: `2019-01-01` through `2024-12-31`.
- Universe: historical CSI 1000 membership, joined on the same date and instrument before aggregation and residualization.
- Hidden/private evaluation data used in development or validation: no.

## Preprocessing and normalization

- Parse timestamps deterministically, normalize to trading dates, cast instruments to strings, and coerce price and volatility inputs to finite numeric values.
- Filter minute bars and factor-library controls to same-day historical CSI 1000 membership. Use finite strictly-positive pre-close values in the daily denominator.
- Apply average-tie cross-sectional ranks and population-standard-deviation (`ddof=0`) z-scores. Require at least two jointly finite rows and non-degenerate volatility breadth; the fixed numerical degeneracy threshold is `1e-12`.
- Submitted values are the sign-oriented daily residuals (`-raw_residual`) required to express the frozen negative direction under the platform's higher-is-better convention. No local winsorization, neutralization beyond the frozen single-control residual, tuning, or post-result sign selection is performed. Official evaluation applies winsorization, standardization, and neutralization.

## Leakage controls

The residual is fit independently inside each date using only same-day daily range, same-day `volatility_5`, and same-day historical-universe membership. It uses no return label, future control, future universe membership, cross-date fitted coefficient, hidden/private data, or look-ahead transformation. The evaluator-provided date bounds constrain every official query.

## Runtime environment and dependencies

The verified public-training batch used BigQuant K2 with 4 CPU cores and 16 GiB memory, Python 3, pandas 2.2.1, NumPy 1.26.4, and the platform-provided `dai` module verified importable on 2026-08-01. The `dai` distribution version and Python micro-version were not exposed by the sanitized preflight and are therefore not invented. Execution is CPU-only; ordering, ranks, regression, and output sorting are deterministic.

The submission code cell intentionally defines only `main(datasources, start_date, end_date)`. Every import, fixed auxiliary table name, numerical threshold, output-column constant, and helper is defined inside `main`, so the notebook remains executable when the official backend extracts that function into an otherwise empty namespace and ignores all code outside it. The replaceable minute-table identity is read only from `datasources["bar1m"]`.

## Determinism and random seeds

The implementation contains no random operation and requires no seed. SQL queries, historical-membership joins, daily groups, average-tie ranks, closed-form one-control least squares, and final sorting all use fixed keys and deterministic operations.

## Reproduction steps

1. Open the single notebook in the BigQuant environment with access to the three official public tables.
2. Submit the sole code cell, whose only top-level statement is the self-contained `main` definition; no earlier cell or global notebook state is required.
3. Let the evaluator extract and call `main({"bar1m": "evaluation_bar1m"}, start_date, end_date)` with its evaluator-provided table identity and inclusive bounds.
4. Confirm the returned DataFrame contains exactly `date,instrument,factor`, has unique daily keys, and satisfies the competition missingness rule.
5. Use the platform's official evaluation path for scoring; keep local public-training estimates labeled as estimates.

## Runtime measurement

- Shared-batch public-training estimate: `3329.8891655680104` seconds for the complete ten-candidate 2019–2024 K2 campaign.
- Scope: this is a conservative shared-batch estimate and is not notebook-specific.
- Exact H09 cold-notebook runtime: `395.06168700999115` seconds on K2 4C/16 GiB.
- Exact H09 peak RSS: `5980737536` bytes.
- Required maximum: strictly less than 10,800 seconds.

## Known limitations

The public-training result is reviewer-ready evidence, not an official submission score. Official global percentile transforms cannot be reproduced exactly from a local batch. H09 also requires sufficient same-day cross-sectional breadth after joining minute range, `volatility_5`, and historical membership; constant controls yield no finite residual rows. No hidden/private evaluation data has been inspected, and this pack has not been submitted.
