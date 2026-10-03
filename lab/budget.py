from __future__ import annotations

import random

from .combinations import hits_against, sample_uniform
from .profiles import GameProfile


def run_budget(
    profile: GameProfile,
    *,
    cost: float,
    tickets: int,
    draws: int,
    seed: int,
    prizes: dict[int, float],
    currency: str = "CLP",
) -> dict:
    """
    Premios = supuestos del usuario, no una tabla oficial.
    Gasto = cost * tickets * draws. Balance = premios - gasto.
    """
    if cost < 0 or tickets < 1 or draws < 1 or draws > 5_000 or tickets > 50:
        raise ValueError("Límites: 1–50 boletos, 1–5000 sorteos, costo ≥ 0.")
    rng = random.Random(seed)
    book = [sample_uniform(profile, rng) for _ in range(tickets)]
    paths = []
    net = 0.0
    spend_step = cost * tickets
    wins = 0.0
    end_balances = []
    losses = 0
    for _ in range(1):  # una trayectoria detallada + resumen
        pass
    # Varias ejecuciones para la distribución final
    n_runs = 40
    for run in range(n_runs):
        r = random.Random(seed + 17 * run)
        bal = 0.0
        spent = 0.0
        won = 0.0
        series = []
        for d in range(draws):
            spent += spend_step
            draw = sample_uniform(profile, r)
            prize = 0.0
            for ticket in book:
                h = hits_against(draw, ticket)
                prize += float(prizes.get(h, 0.0))
            won += prize
            bal = won - spent
            if run == 0:
                series.append({"draw": d + 1, "spent": spent, "won": won, "net": bal})
        end_balances.append(bal)
        if bal < 0:
            losses += 1
        if run == 0:
            paths = series
            net = bal
            wins = won
    return {
        "kind": "supuesto_hipotetico",
        "currency": currency,
        "cost": cost,
        "tickets": tickets,
        "draws": draws,
        "seed": seed,
        "prizes": prizes,
        "spend": cost * tickets * draws,
        "won_first_run": wins,
        "net_first_run": net,
        "path": paths,
        "end_balances": end_balances,
        "p_loss_est": losses / n_runs,
        "note": (
            "Gasto, premios y balance cuadran por construcción. "
            "Los premios son parámetros tuyos, no una tabla oficial. "
            "No hay transacciones reales."
        ),
        "identity": {
            "historical": [],
            "user_params": ["cost", "tickets", "draws", "prizes", "currency"],
            "hypothetical": ["path", "p_loss_est"],
        },
    }
