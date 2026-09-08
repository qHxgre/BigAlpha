"""Final 2019-2024 full-universe retrain of the selected dual-view model."""
from pathlib import Path

import e2e_dual_view_transformer as dual
import validate_full_universe_streaming_2024 as streaming


HERE = Path(__file__).resolve().parent
FINAL_END = "2024-12-31 23:59:59"
streaming.model = dual
streaming.TRAIN_END = FINAL_END
streaming.MODEL_PATH = str(HERE / "e2e_dual_view_streaming_final.json")
dual.TRAIN_END = FINAL_END


if __name__ == "__main__":
    print("final_train_end", streaming.TRAIN_END, flush=True)
    print("trainable_parameters", dual.model_parameter_count(), flush=True)
    streaming.train_streaming(dual.TRAIN_TABLE)
