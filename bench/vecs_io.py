from __future__ import annotations

import numpy as np


def _read_records(path: str, max_count: int, dtype: type) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"{path} is empty")

    dimension = int(raw[0])
    if dimension <= 0:
        raise ValueError(f"{path} declares a non-positive dimension {dimension}")

    stride = dimension + 1
    if raw.size % stride != 0:
        raise ValueError(f"{path} size is not a multiple of its record length")

    headers = raw.reshape(-1, stride)[:, 0]
    if not bool(np.all(headers == dimension)):
        raise ValueError(f"{path} contains records of differing dimension")

    values = raw.view(dtype).reshape(-1, stride)[:, 1:]
    if max_count:
        values = values[:max_count]
    return np.ascontiguousarray(values)


def read_fvecs(path: str, max_count: int = 0) -> np.ndarray:
    return _read_records(path, max_count, np.float32)


def read_ivecs(path: str, max_count: int = 0) -> np.ndarray:
    return _read_records(path, max_count, np.int32)


def write_fvecs(path: str, values: np.ndarray) -> None:
    matrix = np.ascontiguousarray(values, dtype=np.float32)
    count, dimension = matrix.shape
    records = np.empty((count, dimension + 1), dtype=np.int32)
    records[:, 0] = dimension
    records[:, 1:] = matrix.view(np.int32)
    records.tofile(path)


def write_ivecs(path: str, values: np.ndarray) -> None:
    matrix = np.ascontiguousarray(values, dtype=np.int32)
    count, dimension = matrix.shape
    records = np.empty((count, dimension + 1), dtype=np.int32)
    records[:, 0] = dimension
    records[:, 1:] = matrix
    records.tofile(path)
