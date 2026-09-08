import gc
import contextlib
import hashlib
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import base64

import dai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import structlog

    logger = structlog.get_logger()
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "alstm_teacher_plan_lstm"
MODEL_VERSION = 6
# Keep the public checkpoint filename unchanged so only the two code files need
# to know that the architecture is ALSTM. Copy the chosen local checkpoint to
# this name before uploading the submission directory.
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_HERE, "lstm_best.json"))

BATCH = int(os.environ.get("BATCH", 512))
BARS_PER_DAY = 240
SEQ_DAYS = 1
SEQ_LEN = BARS_PER_DAY * SEQ_DAYS

# Keep each pandas frame and window tensor bounded. They remain conservative by
# default, but high-memory platform sessions can increase them before import.
INSTRUMENT_CHUNK_SIZE = int(os.environ.get("INSTRUMENT_CHUNK_SIZE", 100))
DATE_CHUNK_MONTHS = int(os.environ.get("DATE_CHUNK_MONTHS", 3))
if INSTRUMENT_CHUNK_SIZE < 1 or DATE_CHUNK_MONTHS < 1 or BATCH < 1:
    raise ValueError(
        "INSTRUMENT_CHUNK_SIZE, DATE_CHUNK_MONTHS, and BATCH must be positive."
    )

# stage_submission_checkpoint.py rewrites this marked line together with
# lstm_best.json, keeping the public checkpoint and private retraining aligned.
# The corrected MSE+IC experiment selected epoch 10 on the 2024 validation
# year. Public inference loads whichever checkpoint is supplied and does not
# enforce this number; it is used only for private/full-data retraining.
SELECTED_EPOCH = 10  # STAGED_SELECTED_EPOCH
EPOCHS = int(os.environ.get("MAX_SELECTION_EPOCHS", SELECTED_EPOCH))
LR = float(os.environ.get("LR", 3e-4))
SEED = int(os.environ.get("SEED", 42))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 0))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 0))
LOSS_NAME = "mse_ic"
RANK_LOSS_WEIGHT = 0.1

# Platform-side model selection.  The last N calendar months exposed by the
# training datasource are held out, every epoch is evaluated with the official
# BigAlpha evaluator, and the four competition metrics are percentile-ranked
# across our own epoch candidates.  The winning epoch count is then used for a
# clean from-scratch refit on all available training data.
AUTO_SELECT = os.environ.get("AUTO_SELECT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
VALIDATION_MONTHS = int(os.environ.get("VALIDATION_MONTHS", 12))
VALIDATION_EMBARGO_DAYS = int(os.environ.get("VALIDATION_EMBARGO_DAYS", 45))
MIN_SELECTION_EPOCH = int(os.environ.get("MIN_SELECTION_EPOCH", 1))
OFFICIAL_EVAL_REQUIRED = os.environ.get("OFFICIAL_EVAL_REQUIRED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
SELECTION_HISTORY_PATH = os.environ.get(
    "SELECTION_HISTORY_PATH",
    os.path.join(_HERE, "alstm_selection_history.json"),
)
COMPETITION_METRICS = (
    "ic_mean",
    "ic_ir",
    "sharpe_ratio",
    "stress_ic_ir",
)
ENABLE_RUNTIME_CACHE = os.environ.get("ENABLE_RUNTIME_CACHE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
RUNTIME_CACHE_VERSION = 3
RUNTIME_CACHE_ROOT = os.path.realpath(
    os.environ.get("RUNTIME_CACHE_ROOT", tempfile.gettempdir())
)
RUNTIME_CACHE_BACKEND = os.environ.get("RUNTIME_CACHE_BACKEND", "auto").strip().lower()
if RUNTIME_CACHE_BACKEND not in {"auto", "memory", "disk"}:
    raise ValueError("RUNTIME_CACHE_BACKEND must be auto, memory, or disk.")
_MEMORY_RUNTIME_CACHE = {}

FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "deal_number",
    "bid_price1",
    "ask_price1",
    "bid_volume1",
    "ask_volume1",
    "bid_price2",
    "ask_price2",
    "bid_volume2",
    "ask_volume2",
    "bid_price3",
    "ask_price3",
    "bid_volume3",
    "ask_volume3",
    "bid_num_orders1",
    "ask_num_orders1",
    "bid_num_orders2",
    "ask_num_orders2",
    "bid_num_orders3",
    "ask_num_orders3",
]

MODEL_CFG = dict(
    n_feat=len(FEATURE_COLS),
    seq_len=SEQ_LEN,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    rnn_type="LSTM",
    attention_hidden_size=64,
)


class QuantALSTMLSTM(nn.Module):
    """Qlib-style attentive RNN with its recurrent cell fixed to LSTM."""

    def __init__(
        self,
        n_feat,
        seq_len=SEQ_LEN,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        rnn_type="LSTM",
        attention_hidden_size=64,
    ):
        super().__init__()
        if str(rnn_type).upper() != "LSTM":
            raise ValueError("This ALSTM submission requires rnn_type='LSTM'.")
        if n_feat < 1 or seq_len < 1:
            raise ValueError("n_feat and seq_len must be positive.")
        if hidden_size < 2 or num_layers < 1 or attention_hidden_size < 1:
            raise ValueError("Invalid ALSTM width/depth configuration.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.n_feat = int(n_feat)
        self.seq_len = int(seq_len)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.rnn_type = "LSTM"
        self.attention_hidden_size = int(attention_hidden_size)

        self.input_net = nn.Sequential(
            nn.Linear(self.n_feat, self.hidden_size),
            nn.Tanh(),
        )
        self.rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.attention_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.attention_hidden_size),
            nn.Dropout(self.dropout),
            nn.Tanh(),
            nn.Linear(self.attention_hidden_size, 1, bias=False),
            nn.Softmax(dim=1),
        )
        self.output_layer = nn.Linear(self.hidden_size * 2, 1)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, F], received {tuple(x.shape)}.")
        if x.shape[1] != self.seq_len or x.shape[2] != self.n_feat:
            raise ValueError(
                f"Expected [B, {self.seq_len}, {self.n_feat}], "
                f"received {tuple(x.shape)}."
            )

        recurrent_output, _ = self.rnn(self.input_net(x))
        attention_weights = self.attention_net(recurrent_output)
        attention_context = torch.sum(
            recurrent_output * attention_weights,
            dim=1,
        )
        final_state = recurrent_output[:, -1, :]
        fused_state = torch.cat((final_state, attention_context), dim=1)
        # Keep [B, 1] to match ArrayDataset's labels and prevent broadcasting.
        return self.output_layer(fused_state)


def _log_info(message, **kwargs):
    try:
        logger.info(message, **kwargs)
    except TypeError:
        logger.info("%s %s", message, kwargs)


class _TeeText(io.StringIO):
    """Capture evaluator logs while preserving normal notebook output."""

    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, value):
        self.stream.write(value)
        return super().write(value)

    def flush(self):
        self.stream.flush()
        return super().flush()


_METRIC_ALIASES = {
    "ic_mean": {"icmean", "meanic"},
    "ic_ir": {"icir", "icinformationratio"},
    "sharpe_ratio": {"sharperatio", "sharpe", "longshortsharpe", "sr"},
    "stress_ic_ir": {"stressicir", "stressir", "stress"},
}
_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _normalized_metric_key(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _metrics_from_text(value):
    metrics = {}
    text_value = str(value)
    patterns = {
        "stress_ic_ir": ("stress_ic_ir", "stress_icir", "stress_ir"),
        "sharpe_ratio": ("sharpe_ratio", "long_short_sharpe", "sharpe"),
        "ic_mean": ("ic_mean", "mean_ic"),
        "ic_ir": ("ic_ir", "icir"),
    }
    for metric, aliases in patterns.items():
        for alias in aliases:
            match = re.search(
                rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])\s*[=:]\s*({_NUMBER_PATTERN})",
                text_value,
                flags=re.IGNORECASE,
            )
            if match:
                number = _finite_float(match.group(1))
                if number is not None:
                    metrics[metric] = number
                    break
    return metrics


def _extract_official_metrics(outputs, captured_text=""):
    """Extract four metrics across known BigModule Outputs representations."""
    found = _metrics_from_text(captured_text)
    seen = set()

    def visit(value, depth=0):
        if depth > 5 or value is None or len(found) == len(COMPETITION_METRICS):
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)

        if isinstance(value, str):
            if len(value) <= 2_000_000:
                found.update({k: v for k, v in _metrics_from_text(value).items() if k not in found})
            return
        if isinstance(value, pd.DataFrame):
            if not value.empty:
                for record in value.head(20).to_dict("records"):
                    visit(record, depth + 1)
            return
        if isinstance(value, pd.Series):
            visit(value.to_dict(), depth + 1)
            return
        if isinstance(value, dict):
            items = list(value.items())[:200]
        elif isinstance(value, (list, tuple)):
            for item in value[:200]:
                visit(item, depth + 1)
            return
        else:
            items = []
            for metric, aliases in _METRIC_ALIASES.items():
                if metric in found:
                    continue
                for attribute in (metric, *aliases):
                    try:
                        candidate = getattr(value, attribute)
                    except Exception:
                        continue
                    number = _finite_float(candidate)
                    if number is not None:
                        found[metric] = number
                        break
            try:
                raw_attributes = vars(value)
            except (TypeError, ValueError):
                raw_attributes = None
            if isinstance(raw_attributes, dict):
                items = list(raw_attributes.items())[:200]

        for key, item in items:
            normalized = _normalized_metric_key(key)
            matched_metric = next(
                (
                    metric
                    for metric, aliases in _METRIC_ALIASES.items()
                    if normalized in aliases or normalized == _normalized_metric_key(metric)
                ),
                None,
            )
            if matched_metric is not None:
                number = _finite_float(item)
                if number is not None:
                    found[matched_metric] = number
            visit(item, depth + 1)

    visit(outputs)
    missing = [metric for metric in COMPETITION_METRICS if metric not in found]
    if missing:
        raise RuntimeError(
            "Official evaluator ran, but its Outputs object did not expose all four metrics. "
            f"Missing={missing}. Run one platform probe with type(result), vars(result), and "
            "dir(result), then update _extract_official_metrics for that platform version."
        )
    return {metric: float(found[metric]) for metric in COMPETITION_METRICS}


def evaluate_with_official_metrics(score_data):
    """Run BigAlpha's official evaluator and return scalar competition metrics."""
    if score_data.empty:
        raise ValueError("Cannot evaluate an empty score frame.")
    try:
        from bigmodule import M
    except ImportError as exc:
        raise RuntimeError(
            "AUTO_SELECT requires BigQuant's bigmodule package. Run this training script "
            "inside the official platform, or set AUTO_SELECT=0 for fixed-epoch training."
        ) from exc

    capture = _TeeText(sys.stdout)
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        outputs = M.bigalpha_eval._latest(factor_data=score_data, show=False)
    metrics = _extract_official_metrics(outputs, capture.getvalue())
    _log_info("official validation metrics", **metrics)
    return metrics


def _selection_date_ranges(training_start, training_end):
    if VALIDATION_MONTHS < 1:
        raise ValueError("VALIDATION_MONTHS must be positive.")
    if VALIDATION_EMBARGO_DAYS < 0:
        raise ValueError("VALIDATION_EMBARGO_DAYS cannot be negative.")

    full_start = pd.to_datetime(training_start)
    full_end = pd.to_datetime(training_end)
    validation_cutoff = (
        full_end - pd.Timedelta(days=VALIDATION_EMBARGO_DAYS)
    ).normalize()
    cutoff_month_end = validation_cutoff.to_period("M").end_time.normalize()
    validation_end = (
        validation_cutoff
        if validation_cutoff == cutoff_month_end
        else (validation_cutoff.to_period("M") - 1).end_time.normalize()
    )
    validation_start = (
        validation_end.to_period("M") - (VALIDATION_MONTHS - 1)
    ).start_time
    selection_train_end = validation_start - pd.Timedelta(seconds=1)
    if selection_train_end <= full_start or validation_end <= validation_start:
        raise ValueError(
            "Training range is too short for automatic selection: "
            f"full={training_start}..{training_end}, validation_months={VALIDATION_MONTHS}, "
            f"embargo_days={VALIDATION_EMBARGO_DAYS}."
        )
    return (
        full_start.strftime("%Y-%m-%d %H:%M:%S"),
        selection_train_end.strftime("%Y-%m-%d %H:%M:%S"),
        validation_start.strftime("%Y-%m-%d %H:%M:%S"),
        validation_end.strftime("%Y-%m-%d 23:59:59"),
    )


def _rank_epoch_candidates(rows):
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        raise RuntimeError("No epoch candidates were evaluated.")
    for metric in COMPETITION_METRICS:
        if metric not in frame or not np.isfinite(frame[metric]).all():
            raise RuntimeError(f"Candidate metric {metric!r} is missing or non-finite.")
        frame[f"{metric}_pct"] = frame[metric].rank(method="average", pct=True)
    percentile_columns = [f"{metric}_pct" for metric in COMPETITION_METRICS]
    frame["selection_score"] = frame[percentile_columns].mean(axis=1)
    frame["worst_component_pct"] = frame[percentile_columns].min(axis=1)
    return frame.sort_values(
        ["selection_score", "worst_component_pct", "epoch"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _write_selection_history(frame, selected_epoch, metadata):
    payload = {
        "selection_method": "equal_weight_candidate_percentile_rank",
        "competition_metric_weights": {metric: 0.25 for metric in COMPETITION_METRICS},
        "selected_epoch": int(selected_epoch),
        **metadata,
        "candidates": json.loads(frame.to_json(orient="records")),
    }
    directory = os.path.dirname(SELECTION_HISTORY_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_path = SELECTION_HISTORY_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, SELECTION_HISTORY_PATH)


def _stats_fingerprint(stats):
    digest = hashlib.sha256()
    for values in stats:
        array = np.asarray(values, dtype=np.float32)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _runtime_cache_key(kind, **payload):
    normalized = {
        "version": RUNTIME_CACHE_VERSION,
        "kind": kind,
        **payload,
    }
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _memory_limit_bytes():
    """Best-effort effective RAM limit, respecting Linux cgroup limits."""
    candidates = []
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        candidates.append(page_size * page_count)
    except (AttributeError, OSError, ValueError):
        pass
    for path in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            with open(path, "r", encoding="ascii") as file:
                raw = file.read().strip()
            if raw and raw != "max":
                value = int(raw)
                # Some cgroup-v1 hosts expose an effectively-unlimited sentinel.
                if 0 < value < (1 << 60):
                    candidates.append(value)
        except (OSError, ValueError):
            pass
    return min(candidates) if candidates else 0


def _active_cache_backend():
    if RUNTIME_CACHE_BACKEND != "auto":
        return RUNTIME_CACHE_BACKEND
    # A full six-year float16 cache is roughly 25--30 GiB for this universe.
    # Keep ample headroom for pandas and PyTorch; smaller workers use disk.
    return "memory" if _memory_limit_bytes() >= 96 * (1024**3) else "disk"


def _is_memory_cache_path(path):
    return isinstance(path, str) and path.startswith("memory://")


def _make_runtime_cache(phase):
    if not ENABLE_RUNTIME_CACHE:
        return None
    backend = _active_cache_backend()
    if backend == "memory":
        token = hashlib.sha256(
            f"{phase}:{time.time_ns()}:{random.random()}".encode("ascii")
        ).hexdigest()[:16]
        path = f"memory://bigalpha_alstm_{phase}_{token}"
        _log_info(
            "runtime cache created",
            phase=phase,
            path=path,
            backend=backend,
            memory_limit_gib=round(_memory_limit_bytes() / (1024**3), 1),
        )
        return path
    os.makedirs(RUNTIME_CACHE_ROOT, exist_ok=True)
    path = tempfile.mkdtemp(
        prefix=f"bigalpha_alstm_{phase}_",
        dir=RUNTIME_CACHE_ROOT,
    )
    _log_info("runtime cache created", phase=phase, path=path, backend=backend)
    return path


def _remove_runtime_cache(path, phase):
    if not path:
        return
    if _is_memory_cache_path(path):
        keys = [key for key in _MEMORY_RUNTIME_CACHE if key.startswith(path)]
        for key in keys:
            del _MEMORY_RUNTIME_CACHE[key]
        gc.collect()
        _log_info(
            "runtime cache removed",
            phase=phase,
            path=path,
            backend="memory",
            entries=len(keys),
        )
        return
    if not os.path.exists(path):
        return
    temp_root = RUNTIME_CACHE_ROOT
    resolved = os.path.realpath(path)
    expected_prefix = f"bigalpha_alstm_{phase}_"
    if (
        os.path.commonpath([temp_root, resolved]) != temp_root
        or not os.path.basename(resolved).startswith(expected_prefix)
    ):
        raise RuntimeError(f"Refusing to remove unexpected runtime cache path: {path}")
    gc.collect()
    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        # Windows keeps active memmaps locked; the BigQuant worker is Linux,
        # but cleanup must never turn an otherwise successful model into a
        # failed run. The isolated worker will reclaim its temp directory.
        _log_info(
            "runtime cache cleanup deferred",
            phase=phase,
            path=resolved,
            error=str(exc),
        )
        return
    _log_info("runtime cache removed", phase=phase, path=resolved)


def _save_npy_atomic(path, values):
    temp_path = path + ".tmp"
    with open(temp_path, "wb") as file:
        np.save(file, values, allow_pickle=False)
    os.replace(temp_path, path)


def _write_json_atomic(path, payload):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
    os.replace(temp_path, path)


def _progress_due(done, total, every=PROGRESS_EVERY):
    return done == 1 or done == total or (every > 0 and done % every == 0)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(ckpt, model_path=MODEL_PATH):
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    tensors = {}
    for key, value in ckpt["state_dict"].items():
        tensor = value.detach().cpu().contiguous()
        tensors[key] = {
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
            "data_b64": base64.b64encode(tensor.numpy().tobytes()).decode("ascii"),
        }

    payload = {key: value for key, value in ckpt.items() if key != "state_dict"}
    payload["state_dict"] = tensors
    tmp_path = model_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, model_path)


def load_model(model_path=MODEL_PATH, map_location="cpu"):
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state_dict = {}
    for key, meta in payload["state_dict"].items():
        if "data_b64" in meta:
            array = np.frombuffer(base64.b64decode(meta["data_b64"]), dtype=np.dtype(meta["dtype"])).copy()
            tensor = torch.from_numpy(array)
        else:
            tensor = torch.tensor(meta["data"], dtype=getattr(torch, meta["dtype"]))
        state_dict[key] = tensor.reshape(meta["shape"]).to(map_location)

    ckpt = {key: value for key, value in payload.items() if key != "state_dict"}
    ckpt["state_dict"] = state_dict
    return ckpt


def pool(sd, ed):
    df = dai.query(
        "SELECT DISTINCT instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    return df["instrument"].astype(str).tolist()


def _query_bar_data(table, sd, ed, instruments, feature_cols):
    columns = ["date", "instrument"] + list(dict.fromkeys(["close"] + feature_cols))
    sql = f"SELECT {', '.join(columns)} FROM {table} ORDER BY instrument, date"
    t0 = time.time()
    _log_info(
        "query bar data",
        table=table,
        start=sd,
        end=ed,
        instruments=len(instruments),
        features=len(feature_cols),
    )
    df = dai.query(
        sql,
        filters={"date": [sd, ed], "instrument": instruments},
    ).df()
    _log_info("query done", rows=len(df), seconds=round(time.time() - t0, 2))
    return df


def _resolve_bar1m_table(datasources):
    if datasources is None:
        return "bigalpha_2026_stock_bar1m"
    if "bar1m" not in datasources:
        raise KeyError("This LSTM model requires datasources['bar1m'] because it was trained on 1-minute bars.")
    return datasources["bar1m"]


def _query_training_bounds(table):
    """Use every row exposed by the platform's training datasource."""
    frame = dai.query(
        f"SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM {table}",
        # BigQuant requires an explicit opt-in whenever no partition filter is
        # supplied, including aggregate-only MIN/MAX discovery queries.
        full_db_scan=True,
    ).df()
    if frame.empty or pd.isna(frame.loc[0, "min_date"]) or pd.isna(frame.loc[0, "max_date"]):
        raise RuntimeError(f"Could not determine training date bounds for {table}.")
    start = pd.to_datetime(frame.loc[0, "min_date"]).strftime("%Y-%m-%d 00:00:00")
    end = pd.to_datetime(frame.loc[0, "max_date"]).strftime("%Y-%m-%d 23:59:59")
    _log_info("resolved full training range", table=table, start=start, end=end)
    return start, end


def _iter_instrument_chunks(instruments, chunk_size=INSTRUMENT_CHUNK_SIZE):
    instruments = list(instruments)
    for start in range(0, len(instruments), chunk_size):
        yield instruments[start : start + chunk_size]


def _iter_date_chunks(sd, ed):
    """Yield configured multi-month ranges clipped to [sd, ed]."""
    start_ts, end_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    period = start_ts.to_period("M")
    while period.start_time <= end_ts:
        next_period = period + DATE_CHUNK_MONTHS
        block_start = max(start_ts, period.start_time)
        block_end = min(end_ts, next_period.start_time - pd.Timedelta(seconds=1))
        yield block_start.strftime("%Y-%m-%d %H:%M:%S"), block_end.strftime("%Y-%m-%d %H:%M:%S")
        period = next_period


def _prepare_frame(df, feature_cols):
    if len(df) == 0:
        return df

    # The query result is owned by this function. Mutating it avoids a full
    # deep copy of a potentially million-row 1-minute block.
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    price_like = [
        col
        for col in feature_cols
        if col in {"open", "high", "low", "close", "amount"}
        or col.startswith("bid_price")
        or col.startswith("ask_price")
    ]
    for col in price_like:
        df.loc[df[col] == -1, col] = np.nan

    scale_like = [
        col
        for col in feature_cols
        if "volume" in col or col in {"amount", "deal_number", "num_trades"} or "num_orders" in col
    ]
    for col in scale_like:
        df[col] = np.log1p(df[col].clip(lower=0))

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[feature_cols] = df.groupby("instrument", sort=False)[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def build_dataset(
    table,
    sd,
    ed,
    mode,
    instruments,
    stats=None,
    feature_cols=None,
    normalize=True,
    seq_len=None,
    standardize_labels=True,
    query_end=None,
    output_dtype=np.float32,
):
    t0 = time.time()
    feature_cols = list(feature_cols or FEATURE_COLS)
    seq_len = SEQ_LEN if seq_len is None else int(seq_len)
    output_dtype = np.dtype(output_dtype)
    if output_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError(f"output_dtype must be float16 or float32, got {output_dtype}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    start_ts, end_ts = pd.to_datetime(sd), pd.to_datetime(ed)

    # A short calendar buffer covers weekends and exchange holidays.  The
    # requested output is still clipped to [sd, ed], so adjacent blocks do not
    # create duplicate samples.
    buffer_days = max(7, int(np.ceil(seq_len / BARS_PER_DAY)) * 3)
    buf = (start_ts - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d 00:00:00")
    df = _query_bar_data(table, buf, query_end or ed, instruments, feature_cols)
    if len(df) == 0:
        return np.empty((0, seq_len, len(feature_cols)), dtype=output_dtype), None, pd.DataFrame(), stats

    df = _prepare_frame(df, feature_cols)
    wins, ys, keys = [], [], []
    n_instruments = int(df["instrument"].nunique())
    _log_info(
        "window build start",
        mode=mode,
        rows=len(df),
        instruments=n_instruments,
        bars_per_day=BARS_PER_DAY,
        seq_len=seq_len,
    )

    for group_idx, (ins, sub) in enumerate(df.groupby("instrument", sort=False), start=1):
        if len(sub) < seq_len:
            if _progress_due(group_idx, n_instruments):
                _log_info(
                    "window build progress",
                    mode=mode,
                    instruments_done=group_idx,
                    instruments_total=n_instruments,
                    samples=len(wins),
                )
            continue

        feats = sub[feature_cols].to_numpy(np.float32)
        day = sub["date"].dt.normalize().to_numpy()
        close_pos = np.flatnonzero(np.append(day[1:] != day[:-1], True))
        close_px = sub["close"].to_numpy(np.float64)[close_pos]
        dates = day[close_pos]

        for k, pos in enumerate(close_pos):
            d = pd.Timestamp(dates[k])
            if pos + 1 < seq_len or d < start_ts or d > end_ts:
                continue

            label = np.float32(0.0)
            if k + 1 < len(close_pos) and close_px[k] > 0:
                ret = close_px[k + 1] / close_px[k] - 1.0
                if np.isfinite(ret):
                    label = np.float32(ret)
                elif mode == "train":
                    continue
            elif mode == "train":
                continue

            wins.append(feats[pos - seq_len + 1 : pos + 1])
            ys.append(label)
            keys.append((d, ins))

        if _progress_due(group_idx, n_instruments):
            _log_info(
                "window build progress",
                mode=mode,
                instruments_done=group_idx,
                instruments_total=n_instruments,
                samples=len(wins),
            )

    if not wins:
        return np.empty((0, seq_len, len(feature_cols)), dtype=output_dtype), None, pd.DataFrame(), stats

    normalized_during_stack = False
    if output_dtype == np.dtype(np.float16) and (not normalize or stats is not None):
        # Training retains the block as float16. Filling the destination one
        # window at a time avoids ever materializing the same block as a full
        # float32 tensor first (about 590 MiB for 25,735 x 240 x 25).
        X = np.empty(
            (len(wins), seq_len, len(feature_cols)),
            dtype=np.float16,
        )
        if normalize and stats is not None:
            mean = np.asarray(stats[0], dtype=np.float32)
            std = np.asarray(stats[1], dtype=np.float32)
            for window_idx, window in enumerate(wins):
                X[window_idx] = (window - mean) / std
            normalized_during_stack = True
        else:
            for window_idx, window in enumerate(wins):
                X[window_idx] = window
    else:
        # np.stack already preserves the float32 dtype of every feature window.
        # copy=False avoids allocating another full window tensor here.
        X = np.stack(wins).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.float32)
    idx_df = pd.DataFrame(keys, columns=["date", "instrument"])

    if mode == "train" and standardize_labels:
        y_series = pd.Series(y)
        y_z = y_series.groupby(idx_df["date"]).transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-8)
        )
        keep = np.isfinite(y_z.to_numpy())
        X = X[keep]
        y = y_z.to_numpy(np.float32)[keep]
        idx_df = idx_df.loc[keep].reset_index(drop=True)
        if len(X) == 0:
            return np.empty((0, seq_len, len(feature_cols)), dtype=output_dtype), None, pd.DataFrame(), stats

    if normalize and mode == "train" and stats is None:
        flat = X.reshape(-1, len(feature_cols))
        stats = (flat.mean(axis=0).astype(np.float32), (flat.std(axis=0) + 1e-6).astype(np.float32))

    if normalize and stats is not None and not normalized_during_stack:
        mean, std = stats
        # Work in place so normalization does not briefly retain X, X-mean,
        # and the normalized result at the same time.
        np.subtract(X, np.asarray(mean, dtype=np.float32), out=X)
        np.divide(X, np.asarray(std, dtype=np.float32), out=X)

    if X.dtype != output_dtype:
        X = X.astype(output_dtype, copy=False)

    _log_info(
        "dataset built",
        mode=mode,
        shape=X.shape,
        dtype=str(X.dtype),
        seconds=round(time.time() - t0, 2),
    )
    return X, y, idx_df, stats


def _inference_block_cache_path(
    cache_dir,
    table,
    sd,
    ed,
    instruments,
    stats,
    feature_cols,
    seq_len,
):
    if cache_dir is None:
        return None
    key = _runtime_cache_key(
        "inference_block",
        table=str(table),
        start=str(sd),
        end=str(ed),
        instruments=[str(value) for value in instruments],
        feature_cols=list(feature_cols),
        seq_len=int(seq_len),
        stats=_stats_fingerprint(stats),
    )
    return os.path.join(cache_dir, f"infer_{key}")


def _save_inference_block_cache(path, X, idx_df):
    if path is None:
        return
    if _is_memory_cache_path(path):
        _MEMORY_RUNTIME_CACHE[path] = (
            np.asarray(X, dtype=np.float16),
            pd.to_datetime(idx_df["date"]).to_numpy(dtype="datetime64[ns]"),
            idx_df["instrument"].astype(str).to_numpy(dtype=str),
        )
        return
    os.makedirs(path, exist_ok=True)
    _save_npy_atomic(
        os.path.join(path, "features.npy"),
        np.asarray(X, dtype=np.float16),
    )
    _save_npy_atomic(
        os.path.join(path, "dates.npy"),
        pd.to_datetime(idx_df["date"]).to_numpy(dtype="datetime64[ns]"),
    )
    _save_npy_atomic(
        os.path.join(path, "instruments.npy"),
        idx_df["instrument"].astype(str).to_numpy(dtype=str),
    )
    _write_json_atomic(
        os.path.join(path, "metadata.json"),
        {"samples": int(len(idx_df))},
    )


def _load_inference_block_cache(path):
    if path is None:
        return None
    if _is_memory_cache_path(path):
        payload = _MEMORY_RUNTIME_CACHE.get(path)
        if payload is None:
            return None
        X, dates, instruments = payload
        idx_df = pd.DataFrame({"date": dates, "instrument": instruments})
        _log_info("inference cache hit", path=path, samples=len(X))
        return X, None, idx_df
    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        X = np.load(
            os.path.join(path, "features.npy"), mmap_mode="r", allow_pickle=False
        )
        dates = np.load(
            os.path.join(path, "dates.npy"), mmap_mode="r", allow_pickle=False
        )
        instruments = np.load(
            os.path.join(path, "instruments.npy"), mmap_mode="r", allow_pickle=False
        )
        samples = int(metadata["samples"])
        if len(X) != samples or len(dates) != samples or len(instruments) != samples:
            raise ValueError("cached inference block length mismatch")
        idx_df = pd.DataFrame({"date": dates, "instrument": instruments})
    except Exception as exc:
        _log_info("inference cache invalid; rebuilding", path=path, error=str(exc))
        return None
    _log_info("inference cache hit", path=path, samples=len(X))
    return X, None, idx_df


def iter_dataset_blocks(
    table,
    sd,
    ed,
    mode,
    instruments,
    stats,
    feature_cols=None,
    seq_len=None,
    cache_dir=None,
):
    """Yield bounded month/instrument tensors instead of one market-wide tensor."""
    for block_sd, block_ed in _iter_date_chunks(sd, ed):
        for instrument_chunk in _iter_instrument_chunks(instruments):
            active_feature_cols = list(feature_cols or FEATURE_COLS)
            active_seq_len = SEQ_LEN if seq_len is None else int(seq_len)
            cache_path = _inference_block_cache_path(
                cache_dir,
                table,
                block_sd,
                block_ed,
                instrument_chunk,
                stats,
                active_feature_cols,
                active_seq_len,
            )
            cached = _load_inference_block_cache(cache_path)
            if cached is not None:
                X, y, idx_df = cached
                yield X, y, idx_df
                del X, y, idx_df
                gc.collect()
                continue
            X, y, idx_df, _ = build_dataset(
                table,
                block_sd,
                block_ed,
                mode,
                instrument_chunk,
                stats,
                feature_cols,
                normalize=True,
                seq_len=seq_len,
                output_dtype=np.float16,
            )
            if len(X):
                _save_inference_block_cache(cache_path, X, idx_df)
                if cache_path is not None:
                    _log_info(
                        "inference cache stored",
                        path=cache_path,
                        samples=len(X),
                    )
                    cached = _load_inference_block_cache(cache_path)
                    if cached is None:
                        raise RuntimeError(
                            f"Could not reload inference cache after writing: {cache_path}"
                        )
                    del X, y, idx_df
                    X, y, idx_df = cached
                yield X, y, idx_df
                # The generator otherwise retains the previous block while the
                # next one is being built, temporarily doubling peak memory.
                del X, y, idx_df
                gc.collect()
            else:
                del X, y, idx_df
                gc.collect()


class ChunkedFeatureArray:
    """Index several feature arrays as one without concatenating/copying them."""

    def __init__(self, parts, valid_indices=None):
        self.parts = list(parts)
        self.ends = np.cumsum([len(part) for part in self.parts], dtype=np.int64)
        self.total_size = int(self.ends[-1]) if len(self.ends) else 0
        self.valid_indices = (
            None if valid_indices is None else np.asarray(valid_indices, dtype=np.int64)
        )

    def __len__(self):
        return self.total_size if self.valid_indices is None else len(self.valid_indices)

    def __getitem__(self, idx):
        global_idx = int(idx)
        if self.valid_indices is not None:
            global_idx = int(self.valid_indices[global_idx])
        part_idx = int(np.searchsorted(self.ends, global_idx, side="right"))
        part_start = 0 if part_idx == 0 else int(self.ends[part_idx - 1])
        return self.parts[part_idx][global_idx - part_start]


class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Training blocks are retained as float16 on the host to cap RAM, then
        # promoted per sample before entering the model.
        sample = self.X[idx]
        if not sample.flags.writeable:
            sample = np.array(sample, copy=True)
        return torch.from_numpy(sample).float(), torch.tensor(
            [self.y[idx]], dtype=torch.float32
        )


def _training_block_cache_path(
    cache_dir,
    table,
    sd,
    ed,
    instruments,
    stats,
    training_end,
):
    if cache_dir is None:
        return None
    key = _runtime_cache_key(
        "training_block",
        table=str(table),
        start=str(sd),
        end=str(ed),
        training_end=str(training_end),
        instruments=[str(value) for value in instruments],
        feature_cols=list(FEATURE_COLS),
        seq_len=int(SEQ_LEN),
        stats=_stats_fingerprint(stats),
    )
    return os.path.join(cache_dir, f"train_{key}")


def _save_training_block_cache(path, X, y):
    if path is None:
        return
    if _is_memory_cache_path(path):
        _MEMORY_RUNTIME_CACHE[path] = (X, np.asarray(y, dtype=np.float32))
        return
    os.makedirs(path, exist_ok=True)
    parts = X.parts if isinstance(X, ChunkedFeatureArray) else [np.asarray(X)]
    part_files = []
    for index, part in enumerate(parts):
        filename = f"features_{index:03d}.npy"
        _save_npy_atomic(os.path.join(path, filename), np.asarray(part, dtype=np.float16))
        part_files.append(filename)
    _save_npy_atomic(os.path.join(path, "labels.npy"), np.asarray(y, dtype=np.float32))

    valid_file = None
    valid_indices = getattr(X, "valid_indices", None)
    if valid_indices is not None:
        valid_file = "valid_indices.npy"
        _save_npy_atomic(
            os.path.join(path, valid_file),
            np.asarray(valid_indices, dtype=np.int64),
        )
    _write_json_atomic(
        os.path.join(path, "metadata.json"),
        {
            "part_files": part_files,
            "valid_indices_file": valid_file,
            "samples": int(len(y)),
        },
    )


def _load_training_block_cache(path):
    if path is None:
        return None
    if _is_memory_cache_path(path):
        payload = _MEMORY_RUNTIME_CACHE.get(path)
        if payload is None:
            return None
        X, y = payload
        _log_info("training cache hit", path=path, samples=len(y))
        return X, y, pd.DataFrame(index=pd.RangeIndex(len(y)))
    metadata_path = os.path.join(path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        parts = [
            np.load(os.path.join(path, filename), mmap_mode="r", allow_pickle=False)
            for filename in metadata["part_files"]
        ]
        valid_file = metadata.get("valid_indices_file")
        valid_indices = (
            None
            if valid_file is None
            else np.load(
                os.path.join(path, valid_file), mmap_mode="r", allow_pickle=False
            )
        )
        y = np.load(
            os.path.join(path, "labels.npy"), mmap_mode="r", allow_pickle=False
        )
        X = ChunkedFeatureArray(parts, valid_indices=valid_indices)
        if len(X) != len(y) or len(y) != int(metadata["samples"]):
            raise ValueError("cached feature/label length mismatch")
    except Exception as exc:
        _log_info("training cache invalid; rebuilding", path=path, error=str(exc))
        return None
    _log_info("training cache hit", path=path, samples=len(y))
    return X, y, pd.DataFrame(index=pd.RangeIndex(len(y)))


def _get_training_block(
    table,
    sd,
    ed,
    instruments,
    stats,
    training_end,
    cache_dir=None,
):
    cache_path = _training_block_cache_path(
        cache_dir,
        table,
        sd,
        ed,
        instruments,
        stats,
        training_end,
    )
    cached = _load_training_block_cache(cache_path)
    if cached is not None:
        return cached
    X, y, idx_df = _collect_training_block(
        table,
        sd,
        ed,
        instruments,
        stats,
        training_end,
    )
    if len(X):
        _save_training_block_cache(cache_path, X, y)
        _log_info("training cache stored", path=cache_path, samples=len(y))
    return X, y, idx_df


def negative_batch_corr(pred, target, eps=1e-8):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.numel() <= 1:
        return pred.new_tensor(0.0)

    pred = pred - pred.mean()
    target = target - target.mean()
    pred_std = pred.pow(2).mean().sqrt()
    target_std = target.pow(2).mean().sqrt()
    corr = (pred * target).mean() / (pred_std * target_std + eps)
    return -corr


def mixed_rank_mse_loss(pred, target, mse_loss_fn):
    # Explicit flattening prevents PyTorch from silently broadcasting [B]
    # predictions against [B, 1] labels. The model already returns [B, 1], but
    # retaining this guard makes private retraining robust to future changes.
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    if pred.shape != target.shape:
        raise RuntimeError(f"Prediction/target mismatch: {pred.shape} vs {target.shape}")
    mse = mse_loss_fn(pred, target)
    if RANK_LOSS_WEIGHT <= 0:
        return mse
    return mse + RANK_LOSS_WEIGHT * negative_batch_corr(pred, target)


def _train_one_block(model, opt, loss_fn, X, y, device):
    dataset = ArrayDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    model.train()
    total_loss, batches = 0.0, 0
    total_batches = len(loader)
    t0 = time.time()
    for batch_idx, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = torch.clamp(yb.to(device, non_blocking=True), min=-3.0, max=3.0)
        opt.zero_grad(set_to_none=True)
        loss = mixed_rank_mse_loss(model(xb), yb, loss_fn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        batches += 1
        if _progress_due(batch_idx, total_batches):
            _log_info(
                "train batch progress",
                batch=batch_idx,
                batches=total_batches,
                loss=round(total_loss / max(batches, 1), 6),
                seconds=round(time.time() - t0, 2),
            )
    return total_loss, batches


def _collect_training_block(table, sd, ed, instruments, stats, training_end):
    """Build one bounded date block while preserving cross-sectional labels."""
    X_parts, y_parts, idx_parts = [], [], []
    # A month-end sample needs the following trading day's close for its label.
    # Never look beyond the configured training horizon.
    label_query_end = min(
        pd.to_datetime(ed) + pd.Timedelta(days=14),
        pd.to_datetime(training_end),
    ).strftime("%Y-%m-%d %H:%M:%S")
    for instrument_chunk in _iter_instrument_chunks(instruments):
        X, y, idx_df, _ = build_dataset(
            table,
            sd,
            ed,
            "train",
            instrument_chunk,
            stats,
            FEATURE_COLS,
            normalize=True,
            standardize_labels=False,
            query_end=label_query_end,
            output_dtype=np.float16,
        )
        if len(X) == 0:
            continue
        # This mirrors the reusable local cache and halves the largest retained
        # host allocation. ArrayDataset converts each sample back to float32.
        X_parts.append(X)
        y_parts.append(y)
        idx_parts.append(idx_df)
        del X, y, idx_df
        gc.collect()

    if not X_parts:
        return (
            np.empty((0, SEQ_LEN, len(FEATURE_COLS)), dtype=np.float16),
            None,
            pd.DataFrame(columns=["date", "instrument"]),
        )

    y_raw = np.concatenate(y_parts).astype(np.float32, copy=False)
    idx_df = pd.concat(idx_parts, ignore_index=True)
    del y_parts, idx_parts

    # Do this after all instrument chunks have been joined. Standardizing each
    # chunk separately would leak the arbitrary chunk boundaries into labels.
    y_series = pd.Series(y_raw)
    y_z = y_series.groupby(idx_df["date"]).transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-8)
    ).to_numpy(np.float32)
    keep = np.isfinite(y_z)
    X = ChunkedFeatureArray(
        X_parts,
        valid_indices=None if keep.all() else np.flatnonzero(keep),
    )
    if not keep.all():
        idx_df = idx_df.loc[keep].reset_index(drop=True)
    return X, y_z[keep], idx_df


def _compute_stats(table, instruments, training_start, training_end):
    """Compute global raw-bar statistics without materializing any windows."""
    sum_x, sum_sq_x, n_rows = None, None, 0
    sum_x = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    sum_sq_x = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    stats_sd = training_start
    stats_ed = training_end

    for block_sd, block_ed in _iter_date_chunks(stats_sd, stats_ed):
        start_ts, end_ts = pd.to_datetime(block_sd), pd.to_datetime(block_ed)
        # Include a little history only for forward filling, then exclude it
        # from the aggregates so every source row is counted once.
        query_sd = (start_ts - pd.Timedelta(days=7)).strftime("%Y-%m-%d 00:00:00")
        for instrument_chunk in _iter_instrument_chunks(instruments):
            df = _query_bar_data(table, query_sd, block_ed, instrument_chunk, FEATURE_COLS)
            if len(df) == 0:
                continue
            df = _prepare_frame(df, FEATURE_COLS)
            target = df.loc[
                (df["date"] >= start_ts) & (df["date"] <= end_ts),
                FEATURE_COLS,
            ]
            # Row slices keep temporary float64 work arrays well below the size
            # of the pandas block itself.
            for row_start in range(0, len(target), 250_000):
                values = target.iloc[row_start : row_start + 250_000].to_numpy(
                    dtype=np.float64,
                    copy=True,
                )
                sum_x += values.sum(axis=0)
                sum_sq_x += np.einsum("ij,ij->j", values, values, optimize=True)
                n_rows += len(values)
                del values
            del target, df
            gc.collect()
        _log_info("stats block done", start=block_sd, end=block_ed, rows=n_rows)

    if n_rows == 0:
        raise RuntimeError("No rows found for global normalization stats.")
    mean = sum_x / n_rows
    var = np.maximum(sum_sq_x / n_rows - mean**2, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def predict_scores_streaming(
    model,
    table,
    sd,
    ed,
    instruments,
    stats,
    feature_cols,
    device,
    batch_size=BATCH,
    seq_len=SEQ_LEN,
    cache_dir=None,
):
    """Predict bounded blocks and retain only the small date/instrument/score frames."""
    model.eval()
    result_parts = []
    total_samples = 0
    block_number = 0
    date_block_count = sum(1 for _ in _iter_date_chunks(sd, ed))
    instrument_block_count = int(np.ceil(len(instruments) / INSTRUMENT_CHUNK_SIZE))
    _log_info(
        "streaming prediction plan",
        date_blocks=date_block_count,
        instrument_blocks=instrument_block_count,
        total_blocks=date_block_count * instrument_block_count,
        months_per_block=DATE_CHUNK_MONTHS,
        instruments_per_block=INSTRUMENT_CHUNK_SIZE,
    )
    with torch.no_grad():
        for X, _, idx_df in iter_dataset_blocks(
            table,
            sd,
            ed,
            "infer",
            instruments,
            stats,
            feature_cols,
            seq_len,
            cache_dir=cache_dir,
        ):
            block_number += 1
            block_predictions = []
            start = 0
            active_batch_size = min(int(batch_size), len(X))
            while start < len(X):
                xb = None
                batch_values = None
                stop = min(start + active_batch_size, len(X))
                try:
                    batch_values = np.asarray(
                        X[start:stop], dtype=np.float32
                    )
                    if not batch_values.flags.writeable:
                        batch_values = batch_values.copy()
                    xb = torch.from_numpy(batch_values).to(
                        device, non_blocking=True
                    )
                    block_predictions.append(model(xb).detach().cpu().numpy().reshape(-1))
                    start = stop
                except (RuntimeError, MemoryError) as exc:
                    message = str(exc).lower()
                    is_oom = isinstance(exc, MemoryError) or any(
                        marker in message
                        for marker in ("out of memory", "defaultcpuallocator", "can't allocate")
                    )
                    if not is_oom or active_batch_size <= 32:
                        raise
                    active_batch_size = max(32, active_batch_size // 2)
                    _log_info(
                        "prediction batch reduced after oom",
                        block=block_number,
                        batch_size=active_batch_size,
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                finally:
                    if xb is not None:
                        del xb
                    if batch_values is not None:
                        del batch_values
            scores = np.concatenate(block_predictions).astype(np.float32, copy=False)
            part = idx_df[["date", "instrument"]].copy()
            part["score"] = scores
            result_parts.append(part)
            total_samples += len(part)
            _log_info(
                "prediction block done",
                block=block_number,
                block_samples=len(part),
                total_samples=total_samples,
            )
            del X, idx_df, block_predictions, scores, part
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not result_parts:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    return pd.concat(result_parts, ignore_index=True)


def _align_scores_to_official_pool(scores, sd, ed):
    """Apply the same daily CSI 1000 alignment used by the inference notebook."""
    if scores.empty:
        return scores
    stock_pool = dai.query(
        "SELECT date, instrument FROM bigalpha_2026_instruments",
        filters={"date": [sd, ed]},
    ).df()
    stock_pool["date"] = pd.to_datetime(stock_pool["date"]).dt.normalize()
    stock_pool["instrument"] = stock_pool["instrument"].astype(str)
    scores = scores.copy()
    scores["date"] = pd.to_datetime(scores["date"]).dt.normalize()
    scores["instrument"] = scores["instrument"].astype(str)
    return (
        pd.merge(scores, stock_pool, on=["date", "instrument"], how="inner")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score"])
        .drop_duplicates(["date", "instrument"])[["date", "instrument", "score"]]
        .reset_index(drop=True)
    )


def _train_epochs(
    model,
    optimizer,
    loss_fn,
    device,
    table,
    instruments,
    stats,
    training_start,
    training_end,
    epochs,
    phase,
    after_epoch=None,
    cache_dir=None,
):
    training_blocks = list(_iter_date_chunks(training_start, training_end))
    instrument_block_count = int(np.ceil(len(instruments) / INSTRUMENT_CHUNK_SIZE))
    _log_info(
        "training block plan",
        phase=phase,
        date_blocks_per_epoch=len(training_blocks),
        instrument_blocks=instrument_block_count,
        data_builds_per_epoch=len(training_blocks) * instrument_block_count,
        epochs=epochs,
        months_per_block=DATE_CHUNK_MONTHS,
        instruments_per_block=INSTRUMENT_CHUNK_SIZE,
    )
    for epoch in range(int(epochs)):
        epoch_number = epoch + 1
        epoch_started = time.time()
        random.shuffle(training_blocks)
        epoch_loss, epoch_batches = 0.0, 0
        _log_info(
            "epoch start",
            phase=phase,
            epoch=epoch_number,
            learning_rate=LR,
            blocks=len(training_blocks),
        )
        for block_sd, block_ed in training_blocks:
            _log_info(
                "train block start",
                phase=phase,
                epoch=epoch_number,
                start=block_sd,
                end=block_ed,
            )
            X, y, idx_df = _get_training_block(
                table,
                block_sd,
                block_ed,
                instruments,
                stats,
                training_end,
                cache_dir=cache_dir,
            )
            if len(X) == 0:
                _log_info(
                    "train block skipped",
                    phase=phase,
                    epoch=epoch_number,
                    start=block_sd,
                    end=block_ed,
                    reason="empty dataset",
                )
                del X, y, idx_df
                gc.collect()
                continue
            loss_sum, batches = _train_one_block(
                model, optimizer, loss_fn, X, y, device
            )
            epoch_loss += loss_sum
            epoch_batches += batches
            _log_info(
                "train block done",
                phase=phase,
                epoch=epoch_number,
                start=block_sd,
                end=block_ed,
                samples=len(X),
                batches=batches,
                loss=round(loss_sum / max(batches, 1), 6),
            )
            del X, y, idx_df
            gc.collect()
        mean_loss = epoch_loss / max(epoch_batches, 1)
        _log_info(
            "epoch done",
            phase=phase,
            epoch=epoch_number,
            loss=mean_loss,
            seconds=round(time.time() - epoch_started, 2),
        )
        if after_epoch is not None:
            after_epoch(epoch_number, model, mean_loss)


def _train_and_save_repeated_scan(datasources=None, model_path=MODEL_PATH):
    """Legacy repeated-scan trainer retained only as a reference fallback."""
    train_and_save_started = time.time()
    if AUTO_SELECT and EPOCHS < 1:
        raise ValueError("MAX_SELECTION_EPOCHS must be positive.")
    if AUTO_SELECT and (
        MIN_SELECTION_EPOCH < 1 or MIN_SELECTION_EPOCH > EPOCHS
    ):
        raise ValueError(
            "MIN_SELECTION_EPOCH must be between 1 and MAX_SELECTION_EPOCHS."
        )
    set_seed(SEED)
    table = _resolve_bar1m_table(datasources)
    training_start, training_end = _query_training_bounds(table)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.MSELoss()

    selected_epoch = int(SELECTED_EPOCH)
    selection_metadata = None
    if AUTO_SELECT:
        selection_cache_dir = _make_runtime_cache("selection")
        selection_train_cache = (
            None
            if selection_cache_dir is None
            else os.path.join(selection_cache_dir, "train")
        )
        selection_validation_cache = (
            None
            if selection_cache_dir is None
            else os.path.join(selection_cache_dir, "validation")
        )
        (
            selection_train_start,
            selection_train_end,
            validation_start,
            validation_end,
        ) = _selection_date_ranges(training_start, training_end)
        selection_instruments = pool(selection_train_start, selection_train_end)
        validation_instruments = pool(validation_start, validation_end)
        _log_info(
            "automatic selection split",
            train_start=selection_train_start,
            train_end=selection_train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            embargo_days=VALIDATION_EMBARGO_DAYS,
        )
        selection_stats = _compute_stats(
            table,
            selection_instruments,
            selection_train_start,
            selection_train_end,
        )
        selection_model = QuantALSTMLSTM(**MODEL_CFG).to(device)
        selection_optimizer = torch.optim.Adam(
            selection_model.parameters(), lr=LR, weight_decay=1e-4
        )
        candidate_rows = []
        evaluator_failed = []

        def evaluate_epoch(epoch_number, active_model, mean_loss):
            if epoch_number < MIN_SELECTION_EPOCH or evaluator_failed:
                return
            validation_started = time.time()
            _log_info("validation epoch start", epoch=epoch_number)
            scores = predict_scores_streaming(
                active_model,
                table,
                validation_start,
                validation_end,
                validation_instruments,
                selection_stats,
                FEATURE_COLS,
                device,
                batch_size=BATCH,
                seq_len=SEQ_LEN,
                cache_dir=selection_validation_cache,
            )
            scores = _align_scores_to_official_pool(
                scores, validation_start, validation_end
            )
            try:
                metrics = evaluate_with_official_metrics(scores)
            except Exception as exc:
                if OFFICIAL_EVAL_REQUIRED:
                    raise
                evaluator_failed.append(str(exc))
                _log_info(
                    "official evaluator unavailable; fixed epoch fallback enabled",
                    error=str(exc),
                    fallback_epoch=SELECTED_EPOCH,
                )
                return
            candidate_rows.append(
                {"epoch": int(epoch_number), "train_loss": float(mean_loss), **metrics}
            )
            _log_info(
                "validation epoch done",
                epoch=epoch_number,
                seconds=round(time.time() - validation_started, 2),
                **metrics,
            )
            del scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _train_epochs(
            selection_model,
            selection_optimizer,
            loss_fn,
            device,
            table,
            selection_instruments,
            selection_stats,
            selection_train_start,
            selection_train_end,
            EPOCHS,
            "selection",
            after_epoch=evaluate_epoch,
            cache_dir=selection_train_cache,
        )
        if candidate_rows:
            ranked_candidates = _rank_epoch_candidates(candidate_rows)
            selected_epoch = int(ranked_candidates.loc[0, "epoch"])
            selection_metadata = {
                "selection_train_start": selection_train_start,
                "selection_train_end": selection_train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "validation_months": VALIDATION_MONTHS,
                "validation_embargo_days": VALIDATION_EMBARGO_DAYS,
                "official_evaluator": "M.bigalpha_eval._latest",
            }
            _write_selection_history(
                ranked_candidates, selected_epoch, selection_metadata
            )
            _log_info(
                "automatic epoch selected",
                selected_epoch=selected_epoch,
                selection_score=float(ranked_candidates.loc[0, "selection_score"]),
                history_path=SELECTION_HISTORY_PATH,
            )
        elif OFFICIAL_EVAL_REQUIRED:
            raise RuntimeError("Automatic selection produced no valid epoch candidates.")
        del selection_model, selection_optimizer, selection_stats
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _remove_runtime_cache(selection_cache_dir, "selection")

    # The selection model intentionally excludes the validation period.  Start
    # over and refit the chosen epoch count on every row exposed to training.
    set_seed(SEED)
    instruments = pool(training_start, training_end)
    stats = _compute_stats(table, instruments, training_start, training_end)
    model = QuantALSTMLSTM(**MODEL_CFG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    full_refit_cache_dir = _make_runtime_cache("full_refit")
    _train_epochs(
        model,
        optimizer,
        loss_fn,
        device,
        table,
        instruments,
        stats,
        training_start,
        training_end,
        selected_epoch,
        "full_refit",
        cache_dir=full_refit_cache_dir,
    )
    _remove_runtime_cache(full_refit_cache_dir, "full_refit")

    save_model(
        {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "architecture": "qlib_style_alstm",
            "rnn_type": "LSTM",
            "state_dict": model.state_dict(),
            "model_cfg": MODEL_CFG,
            "feature_cols": FEATURE_COLS,
            "mean": stats[0].tolist(),
            "std": stats[1].tolist(),
            "seq_len": SEQ_LEN,
            "seq_days": SEQ_DAYS,
            "bars_per_day": BARS_PER_DAY,
            "bar_frequency": "1m",
            "required_datasource": "bar1m",
            "instrument_chunk_size": INSTRUMENT_CHUNK_SIZE,
            "date_chunk_months": DATE_CHUNK_MONTHS,
            "runtime_cache_enabled": ENABLE_RUNTIME_CACHE,
            "runtime_cache_version": RUNTIME_CACHE_VERSION,
            "target": "next_day_close_to_close",
            "loss_name": LOSS_NAME,
            "rank_loss_weight": RANK_LOSS_WEIGHT,
            "learning_rate": LR,
            "seed": SEED,
            "selected_epoch": selected_epoch,
            "configured_max_selection_epoch": EPOCHS,
            "auto_selected": bool(AUTO_SELECT and selection_metadata is not None),
            "selection_metadata": selection_metadata,
            "training_start": training_start,
            "training_end": training_end,
            "uses_all_available_training_data": True,
            "internal_validation": bool(AUTO_SELECT and selection_metadata is not None),
            "final_refit_uses_internal_validation_rows": True,
        },
        model_path,
    )
    _log_info(
        "model saved",
        path=model_path,
        total_seconds=round(time.time() - train_and_save_started, 2),
    )
    return model_path

# ---------------------------------------------------------------------------
# Teacher-plan online trainer
#
# Adapted from main_train_alstm_teacher_plan.py. Unlike the legacy platform
# path above, raw float16 windows are built exactly once and reused for
# selection and official validation. The validated winning checkpoint is saved
# directly; validation rows are never folded back into training.
# ---------------------------------------------------------------------------
from dataclasses import dataclass

platform = sys.modules[__name__]

# Re-export the inference interface expected by lstm_predict.ipynb.
MODEL_PATH = platform.MODEL_PATH
BATCH = int(os.environ.get("ONLINE_BATCH", platform.BATCH))
QuantALSTMLSTM = platform.QuantALSTMLSTM
pool = platform.pool
load_model = platform.load_model
predict_scores_streaming = platform.predict_scores_streaming
_resolve_bar1m_table = platform._resolve_bar1m_table

FEATURE_COLS = list(platform.FEATURE_COLS)
MODEL_CFG = dict(platform.MODEL_CFG)
SEQ_LEN = platform.SEQ_LEN
SEED = platform.SEED
LR = platform.LR
RANK_LOSS_WEIGHT = platform.RANK_LOSS_WEIGHT
MAX_SELECTION_EPOCHS = int(
    os.environ.get("ONLINE_MAX_SELECTION_EPOCHS", 15)
)
MIN_SELECTION_EPOCH = int(
    os.environ.get("ONLINE_MIN_SELECTION_EPOCH", platform.MIN_SELECTION_EPOCH)
)
INSTRUMENT_CHUNK_SIZE = int(os.environ.get("ONLINE_INSTRUMENT_CHUNK_SIZE", 200))
DATE_CHUNK_MONTHS = int(os.environ.get("ONLINE_DATE_CHUNK_MONTHS", 6))
STATS_BATCH_SAMPLES = int(os.environ.get("ONLINE_STATS_BATCH_SAMPLES", 512))
NUM_WORKERS = int(os.environ.get("ONLINE_NUM_WORKERS", 0))
MODEL_VERSION = 8

if min(
    BATCH,
    MAX_SELECTION_EPOCHS,
    MIN_SELECTION_EPOCH,
    INSTRUMENT_CHUNK_SIZE,
    DATE_CHUNK_MONTHS,
    STATS_BATCH_SAMPLES,
) < 1:
    raise ValueError("All ONLINE_* size and epoch settings must be positive.")
if MIN_SELECTION_EPOCH > MAX_SELECTION_EPOCHS:
    raise ValueError("ONLINE_MIN_SELECTION_EPOCH cannot exceed selection epochs.")


@dataclass
class RawWindowBlock:
    start: str
    end: str
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    instruments: np.ndarray

    @property
    def nbytes(self):
        return int(
            self.X.nbytes
            + self.y.nbytes
            + self.dates.nbytes
            + self.instruments.nbytes
        )


class RawWindowDataset(Dataset):
    """Serve cached float16 windows; normalization happens once per GPU batch."""

    def __init__(self, block, indices):
        self.X = block.X
        self.y = block.y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        # The DataLoader collates float16 host samples. Casting and
        # normalization on the GPU avoid millions of small NumPy operations.
        return (
            torch.from_numpy(self.X[source_index]),
            torch.tensor(self.y[source_index], dtype=torch.float32),
        )


def _log(message, **kwargs):
    platform._log_info(message, **kwargs)


def _iter_date_blocks(sd, ed):
    start_ts, end_ts = pd.to_datetime(sd), pd.to_datetime(ed)
    cursor = start_ts.to_period("M")
    while cursor.start_time <= end_ts:
        next_cursor = cursor + DATE_CHUNK_MONTHS
        block_start = max(start_ts, cursor.start_time)
        block_end = min(end_ts, next_cursor.start_time - pd.Timedelta(seconds=1))
        yield (
            block_start.strftime("%Y-%m-%d %H:%M:%S"),
            block_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        cursor = next_cursor


def _iter_instrument_blocks(instruments):
    instruments = list(instruments)
    for start in range(0, len(instruments), INSTRUMENT_CHUNK_SIZE):
        yield instruments[start : start + INSTRUMENT_CHUNK_SIZE]


def _range_indices(block, sd, ed):
    start = np.datetime64(pd.to_datetime(sd).to_datetime64())
    end = np.datetime64(pd.to_datetime(ed).to_datetime64())
    return np.flatnonzero((block.dates >= start) & (block.dates <= end))


def _build_one_raw_block(table, sd, ed, instruments, training_end):
    started = time.time()
    feature_parts, label_parts, index_parts = [], [], []
    query_end = min(
        pd.to_datetime(ed) + pd.Timedelta(days=14),
        pd.to_datetime(training_end),
    ).strftime("%Y-%m-%d %H:%M:%S")

    chunks = list(_iter_instrument_blocks(instruments))
    for chunk_number, instrument_chunk in enumerate(chunks, start=1):
        X, y, idx_df, _ = platform.build_dataset(
            table,
            sd,
            ed,
            "train",
            instrument_chunk,
            stats=None,
            feature_cols=FEATURE_COLS,
            normalize=False,
            seq_len=SEQ_LEN,
            standardize_labels=False,
            query_end=query_end,
            output_dtype=np.float16,
        )
        if len(X):
            feature_parts.append(X)
            label_parts.append(y)
            index_parts.append(idx_df)
        _log(
            "online raw cache chunk done",
            start=sd,
            end=ed,
            chunk=chunk_number,
            chunks=len(chunks),
            samples=0 if y is None else len(y),
        )
        del X, y, idx_df
        gc.collect()

    if not feature_parts:
        return None

    # This is the server script's reusable yearly float16 cache, built online.
    X = np.concatenate(feature_parts, axis=0).astype(np.float16, copy=False)
    y_raw = np.concatenate(label_parts).astype(np.float32, copy=False)
    idx_df = pd.concat(index_parts, ignore_index=True)
    del feature_parts, label_parts, index_parts
    gc.collect()

    # Cross-sectional target standardization must happen after every instrument
    # chunk for the date has been joined.
    y_z = (
        pd.Series(y_raw)
        .groupby(pd.to_datetime(idx_df["date"]).dt.normalize())
        .transform(lambda values: (values - values.mean()) / (values.std() + 1e-8))
        .to_numpy(np.float32)
    )
    keep = np.isfinite(y_z)
    if not keep.all():
        X = X[keep]
        y_z = y_z[keep]
        idx_df = idx_df.loc[keep].reset_index(drop=True)

    block = RawWindowBlock(
        start=sd,
        end=ed,
        X=X,
        y=y_z,
        dates=pd.to_datetime(idx_df["date"]).to_numpy(dtype="datetime64[ns]"),
        instruments=idx_df["instrument"].astype(str).to_numpy(dtype=str),
    )
    _log(
        "online raw cache block ready",
        start=sd,
        end=ed,
        samples=len(block.X),
        cache_gib=round(block.nbytes / (1024**3), 3),
        seconds=round(time.time() - started, 2),
    )
    return block


def _build_raw_cache(table, training_start, training_end, instruments):
    started = time.time()
    blocks = []
    date_blocks = list(_iter_date_blocks(training_start, training_end))
    _log(
        "online one-pass cache build start",
        date_blocks=len(date_blocks),
        months_per_block=DATE_CHUNK_MONTHS,
        instruments=len(instruments),
        instruments_per_query=INSTRUMENT_CHUNK_SIZE,
        memory_limit_gib=round(platform._memory_limit_bytes() / (1024**3), 1),
    )
    for block_number, (sd, ed) in enumerate(date_blocks, start=1):
        block = _build_one_raw_block(
            table, sd, ed, instruments, training_end
        )
        if block is not None:
            blocks.append(block)
        _log(
            "online cache build progress",
            block=block_number,
            blocks=len(date_blocks),
            retained_gib=round(sum(item.nbytes for item in blocks) / (1024**3), 3),
        )
    if not blocks:
        raise RuntimeError("The platform datasource produced no trainable windows.")
    _log(
        "online one-pass cache ready",
        blocks=len(blocks),
        samples=sum(len(block.X) for block in blocks),
        retained_gib=round(sum(block.nbytes for block in blocks) / (1024**3), 3),
        seconds=round(time.time() - started, 2),
    )
    return blocks


def _compute_cached_stats(blocks, sd, ed):
    started = time.time()
    total = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    total_sq = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    observations = 0
    samples = 0
    for block in blocks:
        indices = _range_indices(block, sd, ed)
        samples += len(indices)
        for start in range(0, len(indices), STATS_BATCH_SAMPLES):
            chosen = indices[start : start + STATS_BATCH_SAMPLES]
            values = np.asarray(block.X[chosen], dtype=np.float64)
            total += values.sum(axis=(0, 1))
            total_sq += np.einsum("ijk,ijk->k", values, values, optimize=True)
            observations += int(values.shape[0] * values.shape[1])
            del values
    if observations == 0:
        raise RuntimeError(f"No cached samples available for stats range {sd}..{ed}.")
    mean = total / observations
    variance = np.maximum(total_sq / observations - mean**2, 1e-8)
    stats = mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
    _log(
        "online cached stats ready",
        start=sd,
        end=ed,
        samples=samples,
        seconds=round(time.time() - started, 2),
    )
    return stats


def _normalization_tensors(stats, device):
    mean = torch.as_tensor(stats[0], dtype=torch.float32, device=device).view(1, 1, -1)
    std = torch.as_tensor(stats[1], dtype=torch.float32, device=device).view(1, 1, -1)
    return mean, std


def _train_cached_epochs(
    model,
    optimizer,
    blocks,
    stats,
    sd,
    ed,
    epochs,
    device,
    phase,
    after_epoch=None,
):
    loss_fn = nn.MSELoss()
    mean_tensor, std_tensor = _normalization_tensors(stats, device)
    active_blocks = [(block, _range_indices(block, sd, ed)) for block in blocks]
    active_blocks = [(block, indices) for block, indices in active_blocks if len(indices)]
    if not active_blocks:
        raise RuntimeError(f"No cached training samples for {sd}..{ed}.")

    for epoch in range(1, int(epochs) + 1):
        started = time.time()
        random.shuffle(active_blocks)
        total_loss, total_batches, total_samples = 0.0, 0, 0
        model.train()
        _log("online epoch start", phase=phase, epoch=epoch, blocks=len(active_blocks))
        for block_number, (block, indices) in enumerate(active_blocks, start=1):
            dataset = RawWindowDataset(block, indices)
            loader = DataLoader(
                dataset,
                batch_size=BATCH,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
            block_loss, block_batches = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device, dtype=torch.float32, non_blocking=True)
                xb.sub_(mean_tensor).div_(std_tensor)
                yb = torch.clamp(
                    yb.to(device, dtype=torch.float32, non_blocking=True),
                    min=-3.0,
                    max=3.0,
                )
                optimizer.zero_grad(set_to_none=True)
                loss = platform.mixed_rank_mse_loss(model(xb), yb, loss_fn)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                block_loss += float(loss.item())
                block_batches += 1
            total_loss += block_loss
            total_batches += block_batches
            total_samples += len(dataset)
            _log(
                "online train block done",
                phase=phase,
                epoch=epoch,
                block=block_number,
                blocks=len(active_blocks),
                samples=len(dataset),
                loss=round(block_loss / max(block_batches, 1), 6),
            )
            del loader, dataset
            gc.collect()
        mean_loss = total_loss / max(total_batches, 1)
        _log(
            "online epoch done",
            phase=phase,
            epoch=epoch,
            samples=total_samples,
            batches=total_batches,
            loss=round(mean_loss, 6),
            seconds=round(time.time() - started, 2),
        )
        if after_epoch is not None:
            after_epoch(epoch, model, mean_loss)


def _predict_cached(model, blocks, stats, sd, ed, device):
    mean_tensor, std_tensor = _normalization_tensors(stats, device)
    frames = []
    model.eval()
    with torch.no_grad():
        for block in blocks:
            indices = _range_indices(block, sd, ed)
            if not len(indices):
                continue
            dataset = RawWindowDataset(block, indices)
            loader = DataLoader(
                dataset,
                batch_size=BATCH,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
            predictions = []
            for xb, _ in loader:
                xb = xb.to(device, dtype=torch.float32, non_blocking=True)
                xb.sub_(mean_tensor).div_(std_tensor)
                predictions.append(model(xb).detach().cpu().numpy().reshape(-1))
            frame = pd.DataFrame(
                {
                    "date": block.dates[indices],
                    "instrument": block.instruments[indices],
                    "score": np.concatenate(predictions).astype(np.float32, copy=False),
                }
            )
            frames.append(frame)
            del loader, dataset, predictions
    if not frames:
        return pd.DataFrame(columns=["date", "instrument", "score"])
    return pd.concat(frames, ignore_index=True)


def _write_online_history(ranked_candidates, selected_epoch, metadata):
    path = os.path.join(os.path.dirname(MODEL_PATH), "alstm_online_selection_history.json")
    payload = {
        "selection_method": "official_four_metric_equal_weight_percentile",
        "selected_epoch": int(selected_epoch),
        **metadata,
        "candidates": json.loads(ranked_candidates.to_json(orient="records")),
    }
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)
    return path


def _last_complete_calendar_year_split(training_start, training_end):
    """Train before, and validate on, the datasource's last complete year."""
    full_start = pd.to_datetime(training_start)
    full_end = pd.to_datetime(training_end)
    validation_year = int(full_end.year)
    expected_year_end = pd.Timestamp(validation_year, 12, 31, 23, 59, 59)
    if full_end < expected_year_end:
        validation_year -= 1
    validation_start = pd.Timestamp(validation_year, 1, 1)
    validation_end = min(
        full_end,
        pd.Timestamp(validation_year, 12, 31, 23, 59, 59),
    )
    selection_train_end = validation_start - pd.Timedelta(seconds=1)
    if validation_year < full_start.year or selection_train_end <= full_start:
        raise ValueError(
            "At least one complete training year before the validation year is required: "
            f"full={training_start}..{training_end}."
        )
    return (
        full_start.strftime("%Y-%m-%d %H:%M:%S"),
        selection_train_end.strftime("%Y-%m-%d %H:%M:%S"),
        validation_start.strftime("%Y-%m-%d %H:%M:%S"),
        validation_end.strftime("%Y-%m-%d %H:%M:%S"),
    )


def train_and_save(datasources=None, model_path=MODEL_PATH):
    """Train on prior years and save the best last-year validated checkpoint."""
    started = time.time()
    platform.set_seed(SEED)
    table = platform._resolve_bar1m_table(datasources)
    training_start, training_end = platform._query_training_bounds(table)
    instruments = platform.pool(training_start, training_end)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(
        "teacher-plan online training start",
        table=table,
        start=training_start,
        end=training_end,
        instruments=len(instruments),
        features=len(FEATURE_COLS),
        device=str(device),
    )

    # The expensive DAI/window stage happens exactly once.
    blocks = _build_raw_cache(table, training_start, training_end, instruments)
    (
        selection_train_start,
        selection_train_end,
        validation_start,
        validation_end,
    ) = _last_complete_calendar_year_split(training_start, training_end)
    _log(
        "online calendar-year selection split",
        train_start=selection_train_start,
        train_end=selection_train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        refit=False,
    )
    selection_stats = _compute_cached_stats(
        blocks, selection_train_start, selection_train_end
    )

    selection_model = QuantALSTMLSTM(**MODEL_CFG).to(device)
    selection_optimizer = torch.optim.Adam(
        selection_model.parameters(), lr=LR, weight_decay=1e-4
    )
    candidate_rows = []
    candidate_states = {}

    def evaluate_epoch(epoch, active_model, train_loss):
        if epoch < MIN_SELECTION_EPOCH:
            return
        validation_started = time.time()
        scores = _predict_cached(
            active_model,
            blocks,
            selection_stats,
            validation_start,
            validation_end,
            device,
        )
        scores = platform._align_scores_to_official_pool(
            scores, validation_start, validation_end
        )
        metrics = platform.evaluate_with_official_metrics(scores)
        candidate_rows.append(
            {"epoch": int(epoch), "train_loss": float(train_loss), **metrics}
        )
        # The final equal-weight percentile winner is known only after every
        # candidate has been evaluated, so retain each small model state in RAM.
        candidate_states[int(epoch)] = {
            key: value.detach().cpu().clone()
            for key, value in active_model.state_dict().items()
        }
        _log(
            "online official validation done",
            epoch=epoch,
            seconds=round(time.time() - validation_started, 2),
            **metrics,
        )
        del scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _train_cached_epochs(
        selection_model,
        selection_optimizer,
        blocks,
        selection_stats,
        selection_train_start,
        selection_train_end,
        MAX_SELECTION_EPOCHS,
        device,
        "selection",
        after_epoch=evaluate_epoch,
    )
    if not candidate_rows:
        raise RuntimeError("Official automatic selection produced no candidates.")
    ranked_candidates = platform._rank_epoch_candidates(candidate_rows)
    selected_epoch = int(ranked_candidates.loc[0, "epoch"])
    selection_model.load_state_dict(candidate_states[selected_epoch])
    selection_metadata = {
        "datasource_start": training_start,
        "datasource_end": training_end,
        "selection_train_start": selection_train_start,
        "selection_train_end": selection_train_end,
        "validation_start": validation_start,
        "validation_end": validation_end,
        "validation_scheme": "last_complete_calendar_year",
        "validation_months": 12,
        "validation_embargo_days": 0,
        "full_refit": False,
    }
    history_path = _write_online_history(
        ranked_candidates, selected_epoch, selection_metadata
    )
    _log(
        "online best epoch selected",
        selected_epoch=selected_epoch,
        score=float(ranked_candidates.loc[0, "selection_score"]),
        history_path=history_path,
    )
    platform.save_model(
        {
            "model_name": "alstm_teacher_plan_validated",
            "model_version": MODEL_VERSION,
            "architecture": "qlib_style_alstm",
            "rnn_type": "LSTM",
            "state_dict": selection_model.state_dict(),
            "model_cfg": MODEL_CFG,
            "feature_cols": FEATURE_COLS,
            "mean": selection_stats[0].tolist(),
            "std": selection_stats[1].tolist(),
            "seq_len": SEQ_LEN,
            "seq_days": platform.SEQ_DAYS,
            "bars_per_day": platform.BARS_PER_DAY,
            "bar_frequency": "1m",
            "loss_name": "mse_ic",
            "rank_loss_weight": RANK_LOSS_WEIGHT,
            "learning_rate": LR,
            "selected_epoch": selected_epoch,
            "epoch": selected_epoch,
            "configured_max_selection_epoch": MAX_SELECTION_EPOCHS,
            "seed": SEED,
            "training_start": selection_train_start,
            "training_end": selection_train_end,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "uses_all_available_training_data": False,
            "full_refit": False,
            "saved_checkpoint_was_officially_validated": True,
            "online_cache_strategy": "one_pass_raw_float16_memory",
            "online_date_chunk_months": DATE_CHUNK_MONTHS,
            "online_instrument_chunk_size": INSTRUMENT_CHUNK_SIZE,
            "selection": selection_metadata,
        },
        model_path,
    )
    _log(
        "validated teacher-plan model saved",
        model_path=model_path,
        selected_epoch=selected_epoch,
        validation_start=validation_start,
        validation_end=validation_end,
        total_seconds=round(time.time() - started, 2),
    )
    return model_path


if __name__ == "__main__":
    train_and_save({"bar1m": "bigalpha_2026_stock_bar1m"})
