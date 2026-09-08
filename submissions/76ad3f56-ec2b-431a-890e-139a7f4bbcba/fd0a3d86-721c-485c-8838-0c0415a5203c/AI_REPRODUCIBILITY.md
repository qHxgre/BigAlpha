# BigAlpha2026 AI-track reproducibility note - H06

## Factor identity

- Candidate: `H06`
- Frozen source: `factors/stage4_h06_minute_depth_delta_ofi_proxy.py`
- Source SHA-256: `8640334bb9e0ec233d5b96b200a4e6ba61ee517adeb85ccd0d091df9b0a67307`

## Formula and frozen direction

- Formula: for each instrument-day, sum the valid adjacent-minute changes in `bid_volume1 - ask_volume1`; a transition is valid only when both current and previous bid/ask depths are finite and non-negative.
- Direction: positive.
- Freeze point: source commit `5cac7aad1a050a2e7218d9b868f5874695b7cdd4`, committed `2026-08-01T21:50:21+08:00`.

## Data, date range, and universe

- Replaceable minute source: `datasources["bar1m"]` fields `date,instrument,bid_volume1,ask_volume1`; fixed auxiliary source: `bigalpha_2026_instruments` fields `date,instrument`.
- Public-training interval: `2019-01-01` through `2024-12-31`.
- Universe: historical CSI 1000 membership, joined on the same date and instrument before aggregation.
- Hidden/private evaluation data used in development or validation: no.

## Preprocessing and normalization

- Parse timestamps deterministically, reject duplicate minute keys, normalize timestamps to trading dates, cast instruments to strings, and coerce depth fields to numeric values.
- Keep only finite, non-negative bid/ask depths. Compute adjacent changes only within the same instrument and trading day, require at least one valid transition, then sum those changes.
- Submitted values are raw daily factor values. No local winsorization, standardization, neutralization, tuning, or sign selection is performed.
- Official platform evaluation applies winsorization, standardization, and neutralization.

## Leakage controls

Each factor row uses only same-day minute depth and same-day historical-universe membership. It does not use labels, future observations, current constituents projected backward, hidden/private data, or any cross-date fit. Date bounds are supplied by the evaluator and applied to every official query.

## Runtime environment and dependencies

The verified public-training batch used BigQuant K2 with 4 CPU cores and 16 GiB memory, Python 3, pandas 2.2.1, NumPy 1.26.4, and the platform-provided `dai` module verified importable on 2026-08-01. Execution is CPU-only and deterministic.

The notebook has exactly one top-level executable definition: `main(datasources, start_date, end_date)`. All imports, fixed auxiliary table names, output-column constants, and helper functions are defined inside `main`, so execution does not depend on notebook globals or on any setup cell. The replaceable minute-table identity is read only from `datasources["bar1m"]`.

## Determinism and random seeds

The implementation contains no random operation and requires no seed. Queries, grouping, transitions, and final output use fixed keys and stable ordering.

## Reproduction steps

1. Open the single notebook in the BigQuant environment with access to the two official public tables.
2. Let the official backend extract and execute only `main(datasources, start_date, end_date)`; no other notebook code is required.
3. Call `main({"bar1m": "evaluation_bar1m"}, start_date, end_date)` with the evaluator-provided table identity and inclusive bounds.
4. Confirm the returned DataFrame contains exactly `date,instrument,factor`, has unique daily keys, and satisfies the competition missingness rule.
5. Use the platform's official evaluation path for scoring; do not substitute the local public-training estimate for an official score.

## Runtime measurement

- Exact H06 cold-notebook public-training runtime: `729.8163201190182` seconds on K2 4C/16 GiB.
- Exact H06 peak RSS: `5946437632` bytes.
- Aggregate-only benchmark receipt: `.long-task/night-20260801-top10-fullrange-001/artifacts/phase-3/h09-h06-submission-benchmark-20260802/benchmark_summary.json`.
- Required maximum: strictly less than 10,800 seconds.

## Known limitations

The public-training result is reviewer-ready evidence, not an official submission score. Coverage depends on valid minute depth and historical-universe availability. No hidden/private evaluation data has been inspected, and this pack has not been submitted.
