# BigAlpha E2E strict-cutoff submission

Use this package if the organizer confirms that the explicit 2019-2023
training / 2024 validation wording remains controlling. The included public
weights never train on 2024; that year is used only for holdout selection.

The notebook exposes `main(datasources, start_date, end_date)` and returns only
`date`, `instrument`, and `score`. The package includes one notebook, inference
and training code, dependency declarations, exact hyperparameters, seed 42,
and text-only Base64 JSON weights.

For private retraining the platform can call `train_model.main` with injected
datasources and a training interval. No downloaded dataset or credential is
included in this archive.
