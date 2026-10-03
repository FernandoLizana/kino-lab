from __future__ import annotations

import random
from dataclasses import dataclass

from .probability import comb
from .profiles import GameProfile


class ComboError(ValueError):
    pass


def validate_combo(profile: GameProfile, numbers: list[int]) -> list[int]:
    cleaned = [int(n) for n in numbers]
    if profile.replacement:
        if len(cleaned) != profile.r:
            raise ComboError(f"Se esperaban {profile.r} números.")
    else:
        if len(cleaned) != profile.r:
            raise ComboError(f"Se esperaban {profile.r} números distintos.")
        if len(set(cleaned)) != profile.r:
            raise ComboError("Este juego no permite repetidos dentro del boleto.")
    if any(n < 1 or n > profile.n for n in cleaned):
        raise ComboError(f"Números fuera de rango 1–{profile.n}.")
    return sorted(cleaned) if not profile.ordered else cleaned


def remaining_universe(profile: GameProfile, include: list[int], exclude: list[int]) -> list[int]:
    include_s = set(include)
    exclude_s = set(exclude)
    if include_s & exclude_s:
        raise ComboError("Un número no puede estar incluido y excluido a la vez.")
    if any(n < 1 or n > profile.n for n in include_s | exclude_s):
        raise ComboError(f"Filtros fuera de rango 1–{profile.n}.")
    if len(include_s) > profile.r:
        raise ComboError("Hay más números fijos que el tamaño del boleto.")
    pool = [n for n in range(1, profile.n + 1) if n not in exclude_s and n not in include_s]
    need = profile.r - len(include_s)
    if need > len(pool):
        raise ComboError(
            f"Tras los filtros quedan {len(pool)} números y se necesitan {need}."
        )
    return pool


def allowed_count(profile: GameProfile, include: list[int], exclude: list[int]) -> int:
    pool = remaining_universe(profile, include, exclude)
    need = profile.r - len(set(include))
    return comb(len(pool), need)


def _unrank(pool: list[int], k: int, rank: int) -> list[int]:
    chosen: list[int] = []
    n = len(pool)
    remaining = rank
    for i in range(k):
        for idx, value in enumerate(pool):
            left = n - idx - 1
            need = k - i - 1
            count = comb(left, need)
            if remaining < count:
                chosen.append(value)
                pool = pool[idx + 1 :]
                n = len(pool)
                break
            remaining -= count
        else:
            raise ComboError("Rank fuera del conjunto permitido.")
    return chosen


def sample_uniform(
    profile: GameProfile,
    rng: random.Random,
    include: list[int] | None = None,
    exclude: list[int] | None = None,
    even_count: int | None = None,
) -> list[int]:
    include = list(include or [])
    exclude = list(exclude or [])
    total = allowed_count(profile, include, exclude)
    if total == 0:
        raise ComboError("No queda ninguna combinación posible con esos filtros.")
    if even_count is None:
        rank = rng.randrange(total)
        extra = _unrank(remaining_universe(profile, include, exclude), profile.r - len(set(include)), rank)
        return validate_combo(profile, include + extra)

    # Paridad: enumeración acotada por rejection con tope, no bucle infinito.
    max_tries = min(10_000, max(200, total * 4))
    for _ in range(max_tries):
        rank = rng.randrange(total)
        extra = _unrank(remaining_universe(profile, include, exclude), profile.r - len(set(include)), rank)
        combo = validate_combo(profile, include + extra)
        if sum(1 for n in combo if n % 2 == 0) == even_count:
            return combo
    raise ComboError(
        "Las restricciones (incluida la paridad) son incompatibles o demasiado "
        "estrechas. Afloja los filtros."
    )


def generate_many(
    profile: GameProfile,
    count: int,
    seed: int,
    include: list[int] | None = None,
    exclude: list[int] | None = None,
    even_count: int | None = None,
    allow_duplicate_tickets: bool = False,
) -> dict:
    if count < 1 or count > 200:
        raise ComboError("Pide entre 1 y 200 combinaciones.")
    total = allowed_count(profile, include or [], exclude or [])
    if not allow_duplicate_tickets and count > total:
        raise ComboError(
            f"Solo existen {total} combinaciones permitidas; pediste {count} distintas."
        )
    rng = random.Random(seed)
    tickets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    limit = max(500, count * 50)
    while len(tickets) < count:
        attempts += 1
        if attempts > limit:
            raise ComboError("No se pudieron completar las combinaciones con esas restricciones.")
        combo = sample_uniform(profile, rng, include, exclude, even_count)
        key = tuple(combo)
        if not allow_duplicate_tickets and key in seen:
            continue
        seen.add(key)
        tickets.append(combo)
    return {
        "kind": "generacion",
        "profile": profile.slug,
        "seed": seed,
        "sampling": "uniforme_sobre_conjunto_filtrado",
        "allowed": total,
        "filters": {
            "include": sorted(set(include or [])),
            "exclude": sorted(set(exclude or [])),
            "even_count": even_count,
            "note": "Los filtros son preferencias de selección; no cambian la probabilidad de un boleto válido.",
        },
        "tickets": tickets,
    }


def hits_against(draw: list[int], ticket: list[int]) -> int:
    return len(set(draw) & set(ticket))


@dataclass
class SavedCombo:
    numbers: list[int]
    label: str
    favorite: bool = False
