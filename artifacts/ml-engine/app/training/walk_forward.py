"""Expanding-window walk-forward splitter.

Given a DataFrame already sorted by `timestamp_ms` ascending, yields
(train_idx, test_idx) numpy arrays where every test index is strictly
after every train index. No shuffling, no future leakage. Test windows are
contiguous and non-overlapping; the train set EXPANDS with each fold so
the final fold trains on all-but-the-last test window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class WalkForwardConfig:
    n_folds: int = 5
    min_train_size: int = 100
    test_size: int | None = None  # if None, derived from total / (n_folds + 1)


def walk_forward_splits(
    df: pd.DataFrame, config: WalkForwardConfig
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if "timestamp_ms" not in df.columns:
        raise ValueError("DataFrame must have a timestamp_ms column")
    n = len(df)
    if n == 0:
        return
    # Sanity check that the caller already sorted by time.
    ts = df["timestamp_ms"].to_numpy()
    if not np.all(ts[:-1] <= ts[1:]):
        raise ValueError("DataFrame must be sorted by timestamp_ms ascending")

    n_folds = max(1, config.n_folds)
    test_size = config.test_size or max(1, (n - config.min_train_size) // n_folds)
    if test_size <= 0:
        return

    train_end = config.min_train_size
    if train_end >= n:
        return

    for _ in range(n_folds):
        test_start = train_end
        test_end = min(test_start + test_size, n)
        if test_end <= test_start:
            break
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx
        train_end = test_end
        if train_end >= n:
            break
