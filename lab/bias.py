from __future__ import annotations

import random

from .combinations import sample_uniform
from .probability import build_probability_table
from .profiles import GameProfile


def weighted_draw_without_replacement(weights: list[float], k: int, rng: random.Random) -> list[int]:
    """
    Muestreo ponderado sin reposición (algoritmo de Efraimidis-Spirakis):
    clave = u^(1/w). Los pesos NO son probabilidades marginales de inclusión.
    """
    keyed = []
    for i, w in enumerate(weights, start=1):
        ww = max(float(w), 1e-12)
        u = rng.random()
        keyed.append((u ** (1.0 / ww), i))
    keyed.sort(reverse=True)
    return sorted(item[1] for item in keyed[:k])


def generate_synthetic(
    profile: GameProfile,
    n_draws: int,
    seed: int,
    *,
    biased: bool = False,
    intensity: float = 0.0,
) -> dict:
    rng = random.Random(seed)
    weights = [1.0] * profile.n
    if biased:
        intensity = max(0.0, min(3.0, intensity))
        for i in range(profile.n):
            weights[i] = 1.0 + intensity * ((i + 1) / profile.n)
    draws = []
    for _ in range(n_draws):
        if biased:
            nums = weighted_draw_without_replacement(weights, profile.k, rng)
        else:
            nums = sample_uniform(profile, rng)
        draws.append(nums)
    table = build_probability_table(profile)
    counts = [0] * (profile.n + 1)
    for draw in draws:
        for n in draw:
            counts[n] += 1
    expected = n_draws * profile.k / profile.n
    return {
        "kind": "datos_sinteticos",
        "synthetic": True,
        "biased": biased,
        "intensity": intensity if biased else 0.0,
        "n_draws": n_draws,
        "seed": seed,
        "rule": (
            "uniforme sin reposición"
            if not biased
            else "Efraimidis-Spirakis con peso 1 + intensidad·(número/N)"
        ),
        "weights": weights,
        "note": "Los pesos no equivalen a P(el número sale).",
        "counts": counts[1:],
        "expected_if_uniform": expected,
        "draws": draws,
        "control": "uniforme" if not biased else "sesgado",
        "theoretical_hits_mean": table.expected_hits,
        "warning": (
            "Una muestra chica inventa patrones. Aumentar n suele disolver "
            "falsos indicios. Esto no audita un sorteo real."
        ),
    }
