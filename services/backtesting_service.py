from __future__ import annotations

from typing import Callable

import numpy as np

from config import Config
from services.analysis_service import draws_matrix
from services.scoring_service import normalize_weights, score_sample


def hits_against(combo: list[int], actual: list[int]) -> int:
    return len(set(combo) & set(actual))


def run_walk_forward(
    draws,
    *,
    sample_size: int = 3000,
    top_k: int = 5,
    seed: int = 42,
    weights: dict | None = None,
    min_history: int = 50,
    max_steps: int | None = 30,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """
    Backtesting walk-forward sin fuga de información:
    para el sorteo i solo usa draws[:i].
    """
    weights = normalize_weights(weights)
    matrix = draws_matrix(draws)
    n = matrix.shape[0]
    if n < min_history + 1:
        raise ValueError(
            f"Se necesitan al menos {min_history + 1} sorteos para backtesting"
        )

    start = min_history
    end = n - 1
    if max_steps is not None:
        end = min(end, start + max_steps - 1)

    model_hits: list[int] = []
    random_hits: list[int] = []
    details: list[dict] = []
    rng = np.random.default_rng(seed)

    total_steps = max(1, end - start + 1)
    for step, i in enumerate(range(start, end + 1)):
        if should_cancel and should_cancel():
            break

        history = matrix[:i]
        actual = [int(x) for x in matrix[i]]

        scored = score_sample(
            history,
            sample_size=sample_size,
            seed=seed + i,
            weights=weights,
            top_k=top_k,
        )
        best_hits = [hits_against(item["numbers"], actual) for item in scored]
        best = max(best_hits) if best_hits else 0
        avg_top = float(np.mean(best_hits)) if best_hits else 0.0

        # Baseline aleatorio con el mismo top_k
        rand_best = 0
        for _ in range(top_k):
            pick = np.sort(rng.choice(np.arange(1, Config.POOL_SIZE + 1), Config.DRAW_SIZE, replace=False))
            rand_best = max(rand_best, hits_against([int(x) for x in pick], actual))

        model_hits.append(int(best))
        random_hits.append(int(rand_best))
        details.append(
            {
                "index": i,
                "actual": actual,
                "model_best_hits": int(best),
                "model_avg_top_hits": avg_top,
                "random_best_hits": int(rand_best),
                "top_combo": scored[0]["numbers"] if scored else [],
            }
        )
        if progress_cb:
            progress_cb(
                100.0 * (step + 1) / total_steps,
                f"Paso {step + 1}/{total_steps} · aciertos modelo={best} azar={rand_best}",
            )

    def dist(values: list[int]) -> dict[str, int]:
        out = {str(k): 0 for k in range(0, Config.DRAW_SIZE + 1)}
        for v in values:
            out[str(v)] = out.get(str(v), 0) + 1
        return out

    metrics = {
        "steps": len(model_hits),
        "sample_size": sample_size,
        "top_k": top_k,
        "seed": seed,
        "min_history": min_history,
        "model_mean_hits": float(np.mean(model_hits)) if model_hits else 0.0,
        "model_max_hits": int(np.max(model_hits)) if model_hits else 0,
        "random_mean_hits": float(np.mean(random_hits)) if random_hits else 0.0,
        "random_max_hits": int(np.max(random_hits)) if random_hits else 0,
        "model_distribution": dist(model_hits),
        "random_distribution": dist(random_hits),
        "advantage_vs_random": (
            float(np.mean(model_hits) - np.mean(random_hits)) if model_hits else 0.0
        ),
        "details_tail": details[-20:],
    }
    return metrics
