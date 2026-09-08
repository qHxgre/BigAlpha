"""Inference entry for the selected raw dual-view Transformer."""
from pathlib import Path

from e2e_dual_view_transformer import predict as _predict


MODEL_PATH = str(
    Path(__file__).resolve().with_name("e2e_dual_view_streaming_final.json")
)


def predict(datasources, start_date, end_date):
    return _predict(datasources, start_date, end_date, MODEL_PATH)
