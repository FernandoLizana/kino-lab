from __future__ import annotations

from itertools import combinations
from typing import Iterator

import numpy as np

from config import Config


def combo_to_mask(numbers: np.ndarray | list[int]) -> np.uint32:
    mask = np.uint32(0)
    for n in numbers:
        mask |= np.uint32(1) << np.uint32(int(n) - 1)
    return mask


def mask_to_combo(mask: int | np.uint32) -> list[int]:
    value = int(mask)
    return [i + 1 for i in range(Config.POOL_SIZE) if value & (1 << i)]


def overlap_count(mask_a: int, mask_b: int) -> int:
    return int(bin(int(mask_a) & int(mask_b)).count("1"))


def iter_all_combinations() -> Iterator[tuple[int, ...]]:
    return combinations(range(1, Config.POOL_SIZE + 1), Config.DRAW_SIZE)


def sample_combinations(
    n: int,
    seed: int | None = 42,
    chunk_size: int | None = None,
) -> Iterator[np.ndarray]:
    """Genera combinaciones aleatorias únicas por lotes (sin materializar todas)."""
    chunk_size = chunk_size or Config.DEFAULT_CHUNK_SIZE
    rng = np.random.default_rng(seed)
    remaining = max(0, int(n))
    seen: set[int] = set()

    while remaining > 0:
        batch_n = min(chunk_size, remaining * 2, 100_000)
        noise = rng.random((batch_n, Config.POOL_SIZE))
        idx = np.argpartition(noise, Config.DRAW_SIZE - 1, axis=1)[:, : Config.DRAW_SIZE]
        combos = np.sort(idx + 1, axis=1).astype(np.int16)

        keep = []
        for row in combos:
            mask = int(combo_to_mask(row))
            if mask in seen:
                continue
            seen.add(mask)
            keep.append(row)
            if len(keep) >= remaining:
                break
        if not keep:
            # Extremely unlikely for modest n vs C(25,14)
            break
        batch = np.asarray(keep, dtype=np.int16)
        yield batch
        remaining -= len(batch)


def features_for_combo(combo: np.ndarray | list[int]) -> dict:
    arr = np.asarray(combo, dtype=np.int16)
    sorted_arr = np.sort(arr)
    even = int(np.sum(sorted_arr % 2 == 0))
    consecutive = int(np.sum(np.diff(sorted_arr) == 1))
    low = int(np.sum(sorted_arr <= 8))
    mid = int(np.sum((sorted_arr >= 9) & (sorted_arr <= 17)))
    high = int(np.sum(sorted_arr >= 18))
    return {
        "numbers": [int(x) for x in sorted_arr],
        "even_count": even,
        "odd_count": Config.DRAW_SIZE - even,
        "total_sum": int(sorted_arr.sum()),
        "consecutive": consecutive,
        "low": low,
        "mid": mid,
        "high": high,
        "mask": int(combo_to_mask(sorted_arr)),
    }
