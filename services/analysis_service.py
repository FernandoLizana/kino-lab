from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from config import Config


def draws_matrix(draws: Iterable) -> np.ndarray:
    rows = []
    for d in draws:
        if hasattr(d, "numbers"):
            rows.append(d.numbers())
        else:
            rows.append(list(d["numbers"]))
    if not rows:
        return np.empty((0, Config.DRAW_SIZE), dtype=np.int16)
    return np.asarray(rows, dtype=np.int16)


def frequency(matrix: np.ndarray) -> dict[int, int]:
    counts = {n: 0 for n in range(1, Config.POOL_SIZE + 1)}
    if matrix.size:
        flat = matrix.ravel()
        for n, c in zip(*np.unique(flat, return_counts=True)):
            counts[int(n)] = int(c)
    return counts


def gaps(matrix: np.ndarray) -> dict[int, int]:
    """Atraso: sorteos desde la última aparición (0 = salió en el último)."""
    result = {n: len(matrix) for n in range(1, Config.POOL_SIZE + 1)}
    if matrix.size == 0:
        return result
    for offset, row in enumerate(matrix[::-1]):
        for n in row:
            n = int(n)
            if result[n] == len(matrix):
                result[n] = offset
    return result


def pair_counts(matrix: np.ndarray, top: int = 30) -> list[dict]:
    counter: Counter[tuple[int, int]] = Counter()
    for row in matrix:
        for a, b in combinations(sorted(int(x) for x in row), 2):
            counter[(a, b)] += 1
    return [
        {"pair": list(pair), "count": count}
        for pair, count in counter.most_common(top)
    ]


def triple_counts(matrix: np.ndarray, top: int = 20) -> list[dict]:
    counter: Counter[tuple[int, int, int]] = Counter()
    for row in matrix:
        for triple in combinations(sorted(int(x) for x in row), 3):
            counter[triple] += 1
    return [
        {"triple": list(triple), "count": count}
        for triple, count in counter.most_common(top)
    ]


def parity_distribution(matrix: np.ndarray) -> dict[str, int]:
    dist: Counter[str] = Counter()
    for row in matrix:
        even = int(np.sum(row % 2 == 0))
        odd = Config.DRAW_SIZE - even
        dist[f"{even}P-{odd}I"] += 1
    return dict(dist)


def sum_stats(matrix: np.ndarray) -> dict:
    if matrix.size == 0:
        return {"mean": 0, "min": 0, "max": 0, "std": 0}
    sums = matrix.sum(axis=1)
    return {
        "mean": float(sums.mean()),
        "min": int(sums.min()),
        "max": int(sums.max()),
        "std": float(sums.std()),
    }


def consecutive_stats(matrix: np.ndarray) -> dict:
    values = []
    for row in matrix:
        sorted_row = np.sort(row)
        values.append(int(np.sum(np.diff(sorted_row) == 1)))
    if not values:
        return {"mean": 0, "max": 0, "distribution": {}}
    dist = Counter(values)
    return {
        "mean": float(np.mean(values)),
        "max": int(np.max(values)),
        "distribution": {str(k): int(v) for k, v in sorted(dist.items())},
    }


def build_analysis(draws) -> dict:
    matrix = draws_matrix(draws)
    freq = frequency(matrix)
    gap = gaps(matrix)
    return {
        "draw_count": int(matrix.shape[0]),
        "frequency": freq,
        "gaps": gap,
        "pairs": pair_counts(matrix),
        "triples": triple_counts(matrix),
        "parity": parity_distribution(matrix),
        "sum": sum_stats(matrix),
        "consecutive": consecutive_stats(matrix),
        "expected_frequency": float(matrix.shape[0] * Config.DRAW_SIZE / Config.POOL_SIZE)
        if matrix.shape[0]
        else 0.0,
    }


def features_table(matrix: np.ndarray) -> pd.DataFrame:
    """Tabla auxiliar de features por sorteo."""
    rows = []
    for row in matrix:
        sorted_row = np.sort(row)
        even = int(np.sum(sorted_row % 2 == 0))
        consec = int(np.sum(np.diff(sorted_row) == 1))
        low = int(np.sum(sorted_row <= 8))
        mid = int(np.sum((sorted_row >= 9) & (sorted_row <= 17)))
        high = int(np.sum(sorted_row >= 18))
        rows.append(
            {
                "sum": int(sorted_row.sum()),
                "even": even,
                "odd": Config.DRAW_SIZE - even,
                "consecutive": consec,
                "low": low,
                "mid": mid,
                "high": high,
            }
        )
    return pd.DataFrame(rows)
