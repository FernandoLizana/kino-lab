from __future__ import annotations

import random
from collections import Counter
from datetime import date

from .combinations import generate_many, hits_against, sample_uniform
from .profiles import GameProfile


def frequency_ticket(profile: GameProfile, past_draws: list[list[int]], seed: int) -> list[int]:
    """Elige los r números más frecuentes del pasado permitido. Empates por semilla."""
    counts: Counter[int] = Counter()
    for draw in past_draws:
        counts.update(draw)
    rng = random.Random(seed)
    ranked = sorted(range(1, profile.n + 1), key=lambda n: (-counts[n], rng.random()))
    return sorted(ranked[: profile.r])


def evaluate_walk_forward(
    profile: GameProfile,
    draws: list[dict],
    *,
    strategy: str,
    tickets_per_draw: int,
    seed: int,
    min_train: int = 30,
    cutoff: date | None = None,
    holdout_from: date | None = None,
    fixed_tickets: list[list[int]] | None = None,
) -> dict:
    """
    En cada paso solo se usa información estrictamente anterior.
    cutoff: no entrenar ni evaluar antes de esa fecha (opcional).
    holdout_from: período final que no se usa para elegir la estrategia.
    """
    ordered = sorted(draws, key=lambda d: d["draw_date"])
    if holdout_from:
        train_pool = [d for d in ordered if d["draw_date"] < holdout_from]
        eval_pool = [d for d in ordered if d["draw_date"] >= holdout_from]
    else:
        train_pool = []
        eval_pool = ordered

    rng = random.Random(seed)
    rows = []
    hist: list[list[int]] = [d["numbers"] for d in train_pool]
    usable = [d for d in eval_pool if cutoff is None or d["draw_date"] >= cutoff]

    for i, draw in enumerate(usable):
        past = hist + [d["numbers"] for d in usable[:i]]
        if len(past) < min_train and strategy == "frequency":
            continue
        tickets: list[list[int]] = []
        if strategy == "uniform":
            tickets = [
                sample_uniform(profile, random.Random(seed + 1000 * i + t))
                for t in range(tickets_per_draw)
            ]
        elif strategy == "fixed":
            tickets = list(fixed_tickets or [])
            if not tickets:
                tickets = [sample_uniform(profile, rng)]
        elif strategy == "frequency":
            base = frequency_ticket(profile, past, seed)
            tickets = [base]
            extra_n = tickets_per_draw - 1
            if extra_n > 0:
                extra = generate_many(profile, extra_n, seed + i, include=base[:2])
                tickets.extend(extra["tickets"])
            tickets = tickets[:tickets_per_draw]
        else:
            raise ValueError(f"Estrategia no soportada: {strategy}")
        scores = [hits_against(draw["numbers"], t) for t in tickets]
        rows.append(
            {
                "date": draw["draw_date"].isoformat(),
                "hits": scores,
                "best": max(scores) if scores else 0,
                "tickets": tickets,
            }
        )

    bests = [r["best"] for r in rows]
    return {
        "kind": "resultado_experimental",
        "strategy": strategy,
        "seed": seed,
        "tickets_per_draw": tickets_per_draw,
        "evaluated": len(rows),
        "min_train": min_train,
        "cutoff": cutoff.isoformat() if cutoff else None,
        "holdout_from": holdout_from.isoformat() if holdout_from else None,
        "mean_best_hits": sum(bests) / len(bests) if bests else 0.0,
        "hit_histogram": {str(h): bests.count(h) for h in sorted(set(bests))},
        "rows": rows[-40:],
        "all_best": bests,
        "label": (
            "retrospectiva_exploratoria"
            if strategy == "fixed"
            else "walk_forward_sin_fuga"
        ),
        "warning": (
            "Comparar estrategias exige las mismas fechas, las mismas reglas "
            "y la misma cantidad de boletos. Un promedio más alto aquí no es "
            "una ventaja demostrada sobre el azar."
        ),
    }


def compare_strategies(
    profile: GameProfile,
    draws: list[dict],
    strategies: list[str],
    **kwargs,
) -> dict:
    results = {name: evaluate_walk_forward(profile, draws, strategy=name, **kwargs) for name in strategies}
    return {
        "kind": "resultado_experimental",
        "reference": "uniform",
        "results": results,
        "same_draws": True,
        "same_ticket_count": True,
    }
