from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import COMBINATION_COUNT, DRAW_SIZE, POOL_SIZE


@lru_cache(maxsize=None)
def binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


# Tabla estática C[n][k] para n<=25, k<=14
_BINOM = np.zeros((POOL_SIZE + 1, DRAW_SIZE + 1), dtype=np.int64)
for _n in range(POOL_SIZE + 1):
    for _k in range(min(_n, DRAW_SIZE) + 1):
        _BINOM[_n, _k] = math.comb(_n, _k)


def combination_rank(numbers: np.ndarray | list[int]) -> int:
    """Índice denso 0 .. C(n,k)-1 para una combinación ordenada 1-based."""
    combo = np.asarray(numbers, dtype=np.int64)
    if combo.size != DRAW_SIZE:
        raise ValueError(f"Se esperaban {DRAW_SIZE} números")
    if np.any(combo[1:] <= combo[:-1]):
        combo = np.sort(combo)

    rank = 0
    prev = 0
    for i in range(DRAW_SIZE):
        x = int(combo[i]) - 1
        k_remaining = DRAW_SIZE - i
        for j in range(prev, x):
            rank += int(_BINOM[POOL_SIZE - 1 - j, k_remaining - 1])
        prev = x + 1
    return int(rank)


def combination_unrank(rank: int) -> np.ndarray:
    """Recupera la combinación 1-based ordenada desde un índice denso."""
    if rank < 0 or rank >= COMBINATION_COUNT:
        raise ValueError(f"Rank fuera de rango: {rank}")

    out = np.empty(DRAW_SIZE, dtype=np.int64)
    x = 0
    remaining = int(rank)
    for i in range(DRAW_SIZE):
        k_remaining = DRAW_SIZE - i
        while True:
            count = int(_BINOM[POOL_SIZE - 1 - x, k_remaining - 1])
            if remaining < count:
                out[i] = x + 1
                x += 1
                break
            remaining -= count
            x += 1
    return out


def ranks_from_combinations(combos: np.ndarray) -> np.ndarray:
    """Ranking para un lote (N, 14) ya ordenado 1-based."""
    if combos.ndim != 2 or combos.shape[1] != DRAW_SIZE:
        raise ValueError(f"Se esperaba shape (N, {DRAW_SIZE})")

    n_rows = combos.shape[0]
    ranks = np.empty(n_rows, dtype=np.int64)
    binom = _BINOM
    pool = POOL_SIZE
    draw = DRAW_SIZE

    # Bucle local (más rápido que llamar combination_rank por fila)
    for r in range(n_rows):
        rank = 0
        prev = 0
        row = combos[r]
        for i in range(draw):
            x = int(row[i]) - 1
            k_remaining = draw - i
            for j in range(prev, x):
                rank += int(binom[pool - 1 - j, k_remaining - 1])
            prev = x + 1
        ranks[r] = rank
    return ranks


def masks_from_combinations(combos: np.ndarray) -> np.ndarray:
    """Codifica cada combinación como bitmask uint32 (bits 0..24)."""
    masks = np.zeros(combos.shape[0], dtype=np.uint32)
    for j in range(DRAW_SIZE):
        masks |= (np.uint32(1) << (combos[:, j].astype(np.uint32) - np.uint32(1)))
    return masks
