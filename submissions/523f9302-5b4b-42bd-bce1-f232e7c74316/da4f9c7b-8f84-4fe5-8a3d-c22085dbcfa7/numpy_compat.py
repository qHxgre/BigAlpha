"""Compatibility shims for old parquet readers under NumPy 2.x."""

from __future__ import annotations

import numpy as np


def patch_numpy_for_pyarrow() -> None:
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_

    if not hasattr(np, "sctypes"):
        int_types = [np.int8, np.int16, np.int32, np.int64]
        uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]
        float_types = [np.float16, np.float32, np.float64]
        complex_types = [np.complex64, np.complex128]
        for name in ("int_", "longlong"):
            typ = getattr(np, name, None)
            if typ is not None and typ not in int_types:
                int_types.append(typ)
        for name in ("uint", "ulonglong"):
            typ = getattr(np, name, None)
            if typ is not None and typ not in uint_types:
                uint_types.append(typ)
        for name in ("float128",):
            typ = getattr(np, name, None)
            if typ is not None and typ not in float_types:
                float_types.append(typ)
        for name in ("complex256",):
            typ = getattr(np, name, None)
            if typ is not None and typ not in complex_types:
                complex_types.append(typ)
        np.sctypes = {
            "int": int_types,
            "uint": uint_types,
            "float": float_types,
            "complex": complex_types,
            "others": [np.bool_, np.object_, np.str_, np.bytes_],
        }


patch_numpy_for_pyarrow()
