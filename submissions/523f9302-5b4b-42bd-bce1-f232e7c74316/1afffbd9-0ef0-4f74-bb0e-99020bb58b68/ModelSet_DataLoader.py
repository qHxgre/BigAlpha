import os
import queue
import sys
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Support_BaseData import get_batch_codes, get_standard_details, DEFAULT_MODEL_JSON
from Support_FeatureCal import get_batch_feature
from Support_LabelCal import get_batch_label
from Support_MemProfile import MemMode, resolve_profile
from Support_RollStore import RollingBaseStore

BatchData = Tuple[dict, dict]
_SENTINEL = object()

def _resolve_details_path(details_path: str) -> str:
    """相对路径锚定到上传根（本文件目录）；绝对路径且存在则原样返回。"""
    if os.path.isabs(details_path):
        if os.path.exists(details_path):
            return details_path
        # 旧绝对路径失效：退回同级终稿
        return os.path.join(_ROOT, DEFAULT_MODEL_JSON)  # "transformer_model.json"
    return os.path.join(_ROOT, details_path)


class PrefetchDataLoader:
    """
    后台双线程：Base 灌内存 ∥ Batch 组装。

    use_rolling=True 时必须按交易日升序；shuffle 将被忽略。
    """

    def __init__(
        self,
        date_list: List[str],
        seq_len: int = 240,
        details_path: str = DEFAULT_MODEL_JSON,
        mem_mode: MemMode = "low",
        prefetch_size: Optional[int] = None,
        shuffle: bool = False,
        use_rolling: bool = True,
        need_label: bool = True
    ):
        self.date_list = list(date_list)
        self.seq_len = seq_len
        self.details_path = _resolve_details_path(details_path)
        self.use_rolling = use_rolling
        self.profile = resolve_profile(mem_mode)
        self.need_label = need_label

        size = self.profile.prefetch_size if prefetch_size is None else int(prefetch_size)
        if size < 1:
            raise ValueError(
                "prefetch_size 必须 >= 1（有界预取）。"
                "同步 benchmark 请自行循环调用 _fetch_one，不要传 0。"
            )
        self.prefetch_size = size

        if use_rolling and shuffle:
            print("[PrefetchDataLoader] use_rolling=True，已忽略 shuffle=True")
            shuffle = False
        self.shuffle = shuffle

        self._scale_details = get_standard_details(details_path)
        horizon_end = max(self.date_list) if self.date_list else None
        self._store = (
            RollingBaseStore(
                seq_len=seq_len,
                scale_details=self._scale_details,
                profile=self.profile,
                horizon_end=horizon_end,
            )
            if use_rolling
            else None
        )

        self._codes_cache: Dict[str, Tuple[List[str], Dict[str, int]]] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=self.prefetch_size)
        self._stop_event = threading.Event()
        self._base_thread: Optional[threading.Thread] = None
        self._batch_thread: Optional[threading.Thread] = None
        self._fetch_times: List[float] = []
        self._wait_times: List[float] = []
        self._base_times: List[float] = []

    def _get_codes(self, date: str) -> Tuple[List[str], Dict[str, int]]:
        hit = self._codes_cache.get(date)
        if hit is not None:
            return hit
        stock_list, code_to_idx = get_batch_codes(date)
        self._codes_cache[date] = (stock_list, code_to_idx)
        return stock_list, code_to_idx

    def _warmup_codes(self, dates: List[str]) -> None:
        for d in dates:
            self._get_codes(d)

    def _fetch_one(self, date: str) -> BatchData:
        stock_list, code_to_idx = self._get_codes(date)
        feature = get_batch_feature(
            date=date,
            seq_len=self.seq_len,
            stock_list=stock_list,
            code_to_idx=code_to_idx,
            details_path=self.details_path,
            scale_details=self._scale_details,
            store=self._store,
        )
        
        if not self.need_label:
            return feature, None
        
        label = get_batch_label(
            date=date,
            stock_list=stock_list,
            code_to_idx=code_to_idx,
        )
        
        return feature, label

    # ------------------------------------------------------------------
    # Thread 1: DAI → memory
    # ------------------------------------------------------------------

    def _base_worker(self, date_list: List[str]) -> None:
        assert self._store is not None
        try:
            for date in date_list:
                if self._stop_event.is_set():
                    break
                if not self._store.wait_can_prefetch(date):
                    break
                t0 = time.perf_counter()
                self._store.prefetch_for(date)
                self._base_times.append(time.perf_counter() - t0)
        except Exception as exc:
            self._store.set_error(exc)
            try:
                self._queue.put(exc)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Thread 2: memory → batch queue
    # ------------------------------------------------------------------

    def _batch_worker(self, date_list: List[str]) -> None:
        for date in date_list:
            if self._stop_event.is_set():
                break
            if self._store is not None:
                self._store.set_assemble_date(date)
            t0 = time.perf_counter()
            try:
                data = self._fetch_one(date)
                if self._store is not None:
                    self._store.advance_assemble(date)
                self._fetch_times.append(time.perf_counter() - t0)
                self._queue.put(data)
            except Exception as exc:
                if self._store is not None:
                    self._store.set_error(exc)
                self._queue.put(exc)
                break
        self._queue.put(_SENTINEL)

    def __iter__(self) -> Iterator[BatchData]:
        self.stop()
        self._stop_event.clear()
        self._fetch_times.clear()
        self._wait_times.clear()
        self._base_times.clear()

        order = list(self.date_list)
        if self.shuffle:
            rng = np.random.default_rng(int(time.time()))
            rng.shuffle(order)
        elif self.use_rolling:
            order = sorted(order)

        if self._store is not None:
            self._store.reset()
            self._warmup_codes(order)

        self._queue = queue.Queue(maxsize=self.prefetch_size)

        threads: List[threading.Thread] = []
        if self._store is not None:
            self._base_thread = threading.Thread(
                target=self._base_worker,
                args=(order,),
                daemon=True,
                name="BasePrefetch",
            )
            self._base_thread.start()
            threads.append(self._base_thread)

        self._batch_thread = threading.Thread(
            target=self._batch_worker,
            args=(order,),
            daemon=True,
            name="BatchAssemble",
        )
        self._batch_thread.start()
        threads.append(self._batch_thread)

        try:
            while True:
                t_wait = time.perf_counter()
                item = self._queue.get()
                self._wait_times.append(time.perf_counter() - t_wait)

                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            self.stop()

    def __len__(self) -> int:
        return len(self.date_list)

    def stop(self) -> None:
        self._stop_event.set()
        if self._store is not None:
            self._store.stop()

        if self._queue is not None:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        for th in (self._base_thread, self._batch_thread):
            if th is not None and th.is_alive():
                th.join(timeout=60)
        self._base_thread = None
        self._batch_thread = None
