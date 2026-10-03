from __future__ import annotations

import math
import random
import time
from typing import Any, Callable

from .combinations import hits_against, sample_uniform
from .probability import build_probability_table
from .profiles import GameProfile

QUICK_N = 10_000
DETAILED_N = 100_000
MAX_N = 1_000_000
BATCH = 1_000


def _draw_uniform(profile: GameProfile, rng: random.Random) -> list[int]:
    return sample_uniform(profile, rng)


def run_monte_carlo(
    profile: GameProfile,
    n_sim: int,
    seed: int,
    tickets: list[list[int]],
    *,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    should_pause: Callable[[], bool] | None = None,
    start_done: int = 0,
    start_hits: dict[int, int] | None = None,
    start_rng_state: Any = None,
) -> dict:
    if n_sim < 1 or n_sim > MAX_N:
        raise ValueError(f"Simulaciones entre 1 y {MAX_N} (modo detallado: {DETAILED_N}).")
    if not tickets:
        raise ValueError("Necesitas al menos un boleto.")
    rng = random.Random(seed)
    if start_rng_state is not None:
        rng.setstate(start_rng_state)
    table = build_probability_table(profile)
    hits = dict(start_hits or {j: 0 for j in range(table.support[0], table.support[1] + 1)})
    started = time.monotonic()
    done = start_done
    ticket_hits = [0] * len(tickets)

    while done < n_sim:
        if should_cancel and should_cancel():
            break
        if should_pause and should_pause():
            return _pack(
                profile, n_sim, seed, tickets, table, hits, ticket_hits, done, started,
                status="paused", rng=rng,
            )
        batch = min(BATCH, n_sim - done)
        for _ in range(batch):
            draw = _draw_uniform(profile, rng)
            best = 0
            for i, ticket in enumerate(tickets):
                h = hits_against(draw, ticket)
                ticket_hits[i] += h
                if h > best:
                    best = h
            hits[best] = hits.get(best, 0) + 1
        done += batch
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed else 0
        remaining = (n_sim - done) / rate if rate else None
        if progress_cb:
            progress_cb(
                100.0 * done / n_sim,
                f"{done}/{n_sim} sorteos · {elapsed:.1f}s"
                + (f" · ~{remaining:.0f}s restantes" if remaining else ""),
            )

    status = "cancelled" if (should_cancel and should_cancel() and done < n_sim) else "done"
    return _pack(profile, n_sim, seed, tickets, table, hits, ticket_hits, done, started, status, rng)


def _pack(profile, n_sim, seed, tickets, table, hits, ticket_hits, done, started, status, rng):
    elapsed = time.monotonic() - started
    observed = []
    for j in range(table.support[0], table.support[1] + 1):
        count = hits.get(j, 0)
        p_obs = count / done if done else 0.0
        p_th = table.pmf[j]
        se = math.sqrt(p_th * (1 - p_th) / done) if done else 0.0
        observed.append(
            {
                "hits": j,
                "count": count,
                "observed": p_obs,
                "theoretical": p_th,
                "se": se,
                "note": (
                    "0 casos observados; la probabilidad teórica no es cero."
                    if count == 0 and p_th > 0
                    else ""
                ),
            }
        )
    return {
        "kind": "simulacion",
        "status": status,
        "profile": profile.slug,
        "n_requested": n_sim,
        "n_completed": done,
        "seed": seed,
        "tickets": tickets,
        "elapsed_s": elapsed,
        "hits": observed,
        "mean_hits_per_ticket": [h / done if done else 0 for h in ticket_hits],
        "checkpoint": {
            "done": done,
            "hits": hits,
            "rng_state": rng.getstate(),
        },
        "limits": {"quick": QUICK_N, "detailed": DETAILED_N, "max": MAX_N},
        "method": (
            "Intervalo aproximado ±1 SE binomial usando la probabilidad teórica. "
            "Una simulación corta puede no ver eventos raros."
        ),
    }
