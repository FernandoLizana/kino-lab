"""Monte Carlo en streaming: no materializa las simulaciones en disco.

Usa un acumulador denso de C(25,14) ≈ 4.46M contadores (~36 MB con uint64).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import COMBINATION_COUNT, DRAW_SIZE, POOL_SIZE
from .combo import combination_unrank, ranks_from_combinations
from .dataset import DEFAULT_OUTPUT, frequency_table, load_canonical


def _sample_batch_uniform(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """(batch, 14) números 1-based ordenados, sin reemplazo, uniforme."""
    noise = rng.random((batch_size, POOL_SIZE))
    idx = np.argpartition(noise, DRAW_SIZE - 1, axis=1)[:, :DRAW_SIZE]
    return np.sort(idx + 1, axis=1)


def _sample_batch_weighted(
    batch_size: int,
    rng: np.random.Generator,
    probabilities: np.ndarray,
) -> np.ndarray:
    """Muestreo ponderado sin reemplazo vía Gumbel-top-k."""
    logits = np.log(np.clip(probabilities, 1e-18, None))
    gumbel = rng.gumbel(size=(batch_size, POOL_SIZE))
    scores = logits + gumbel
    idx = np.argpartition(scores, -DRAW_SIZE, axis=1)[:, -DRAW_SIZE:]
    return np.sort(idx + 1, axis=1)


def run_streaming_simulation(
    *,
    n_simulations: int = 1_000_000,
    batch_size: int = 50_000,
    top: int = 100,
    seed: int | None = 42,
    weighted: bool = False,
    dataset_path: Path = DEFAULT_OUTPUT,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    if n_simulations <= 0:
        raise ValueError("n_simulations debe ser > 0")
    if batch_size <= 0:
        raise ValueError("batch_size debe ser > 0")

    rng = np.random.default_rng(seed)
    counts = np.zeros(COMBINATION_COUNT, dtype=np.uint64)

    probabilities = None
    if weighted:
        frame = load_canonical(dataset_path)
        freq = frequency_table(frame).reindex(range(1, POOL_SIZE + 1), fill_value=0)
        probabilities = freq.to_numpy(dtype=np.float64)
        total = probabilities.sum()
        if total <= 0:
            raise RuntimeError("No hay frecuencias históricas para ponderar")
        probabilities = probabilities / total

    remaining = n_simulations
    done = 0
    while remaining > 0:
        current = min(batch_size, remaining)
        if probabilities is None:
            batch = _sample_batch_uniform(current, rng)
        else:
            batch = _sample_batch_weighted(current, rng, probabilities)
        ranks = ranks_from_combinations(batch)
        counts += np.bincount(ranks, minlength=COMBINATION_COUNT).astype(np.uint64)
        done += current
        remaining -= current
        print(
            f"simuladas {done:,}/{n_simulations:,} | "
            f"contadores en RAM ~ {counts.nbytes / 1e6:.1f} MB"
        )

    top_idx = np.argpartition(counts, -top)[-top:]
    top_idx = top_idx[np.argsort(counts[top_idx])[::-1]]

    rows = []
    for rank in top_idx:
        combo = combination_unrank(int(rank))
        rows.append(
            {
                **{f"n{i}": int(v) for i, v in enumerate(combo, start=1)},
                "count": int(counts[rank]),
                "rank_id": int(rank),
            }
        )
    result = pd.DataFrame(rows)

    out_dir = output_dir or (Path(__file__).resolve().parent.parent / "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "weighted" if weighted else "uniform"
    out_path = out_dir / f"top{top}_{mode}_{n_simulations}.csv"
    result.to_csv(out_path, index=False)
    print(f"Top {top} guardado en {out_path}")
    return result
