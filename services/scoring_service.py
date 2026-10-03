from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np

from config import Config
from services.analysis_service import frequency, gaps, pair_counts, triple_counts
from services.combination_service import features_for_combo, sample_combinations


def normalize_weights(weights: dict[str, float] | None) -> dict[str, float]:
    base = dict(Config.DEFAULT_WEIGHTS)
    if weights:
        for key, value in weights.items():
            if key in base:
                try:
                    base[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return base


def _build_context(matrix: np.ndarray) -> dict:
    freq = frequency(matrix)
    gap = gaps(matrix)
    expected = (
        matrix.shape[0] * Config.DRAW_SIZE / Config.POOL_SIZE if matrix.shape[0] else 1.0
    )
    pair_map = {
        tuple(item["pair"]): item["count"] for item in pair_counts(matrix, top=200)
    }
    triple_map = {
        tuple(item["triple"]): item["count"] for item in triple_counts(matrix, top=100)
    }
    sums = matrix.sum(axis=1) if matrix.size else np.array([182.0])
    sum_mean = float(sums.mean()) if sums.size else 182.0
    sum_std = float(sums.std()) if sums.size > 1 else 25.0
    return {
        "freq": freq,
        "gap": gap,
        "expected": max(expected, 1e-9),
        "pair_map": pair_map,
        "triple_map": triple_map,
        "sum_mean": sum_mean,
        "sum_std": max(sum_std, 1e-9),
        "max_gap": max(gap.values()) if gap else 1,
    }


def score_combo(combo: list[int] | np.ndarray, ctx: dict, weights: dict[str, float]) -> dict:
    feats = features_for_combo(combo)
    nums = feats["numbers"]

    # Frecuencia: favorece números históricamente más frecuentes (normalizado)
    freq_score = sum(ctx["freq"].get(n, 0) / ctx["expected"] for n in nums) / Config.DRAW_SIZE

    # Atraso: favorece números más atrasados
    max_gap = max(ctx["max_gap"], 1)
    gap_score = sum(ctx["gap"].get(n, 0) / max_gap for n in nums) / Config.DRAW_SIZE

    # Paridad: ideal cerca de 7-7
    parity_score = 1.0 - abs(feats["even_count"] - feats["odd_count"]) / Config.DRAW_SIZE

    # Suma: cerca de la media histórica
    z = abs(feats["total_sum"] - ctx["sum_mean"]) / ctx["sum_std"]
    sum_score = float(np.exp(-0.5 * z * z))

    # Bandas bajas/medias/altas: ideal ~4-5-5 o similar
    ideal = Config.DRAW_SIZE / 3
    bands_score = 1.0 - (
        abs(feats["low"] - ideal)
        + abs(feats["mid"] - ideal)
        + abs(feats["high"] - ideal)
    ) / (2 * Config.DRAW_SIZE)

    # Consecutivos: penaliza extremos (0 o muchos)
    consec = feats["consecutive"]
    consecutive_score = 1.0 - abs(consec - 2) / Config.DRAW_SIZE

    # Parejas históricas
    pair_hits = 0.0
    pair_denom = 0.0
    top_pair_max = max(ctx["pair_map"].values()) if ctx["pair_map"] else 1
    for a, b in combinations(nums, 2):
        pair_denom += 1
        pair_hits += ctx["pair_map"].get((a, b), 0) / top_pair_max
    pairs_score = pair_hits / max(pair_denom, 1)

    # Tríos históricos
    triple_hits = 0.0
    triple_denom = 0.0
    top_triple_max = max(ctx["triple_map"].values()) if ctx["triple_map"] else 1
    for triple in combinations(nums, 3):
        triple_denom += 1
        triple_hits += ctx["triple_map"].get(triple, 0) / top_triple_max
    triples_score = triple_hits / max(triple_denom, 1)

    components = {
        "frequency": float(freq_score),
        "gap": float(gap_score),
        "parity": float(parity_score),
        "sum_balance": float(sum_score),
        "bands": float(max(0.0, bands_score)),
        "consecutive": float(max(0.0, consecutive_score)),
        "pairs": float(pairs_score),
        "triples": float(triples_score),
    }

    total_w = sum(max(0.0, weights[k]) for k in components) or 1.0
    score = sum(max(0.0, weights[k]) * components[k] for k in components) / total_w

    return {
        **feats,
        "score": float(score),
        "components": components,
        "numbers_csv": "-".join(f"{n:02d}" for n in nums),
    }


def score_sample(
    matrix: np.ndarray,
    *,
    sample_size: int,
    seed: int | None,
    weights: dict[str, float] | None = None,
    top_k: int = 100,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict]:
    weights = normalize_weights(weights)
    ctx = _build_context(matrix)
    best: list[dict] = []

    processed = 0
    for batch in sample_combinations(sample_size, seed=seed):
        if should_cancel and should_cancel():
            break
        for row in batch:
            scored = score_combo(row, ctx, weights)
            best.append(scored)
            processed += 1
        best.sort(key=lambda x: x["score"], reverse=True)
        best = best[: max(top_k * 3, top_k)]
        if progress_cb:
            progress_cb(min(99.0, 100.0 * processed / max(sample_size, 1)), f"{processed:,} combinaciones")

    best = sorted(best, key=lambda x: x["score"], reverse=True)[:top_k]
    for i, item in enumerate(best, start=1):
        item["rank"] = i
    if progress_cb:
        progress_cb(100.0, f"Listo: top {len(best)}")
    return best
