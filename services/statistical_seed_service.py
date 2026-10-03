from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Callable

from services.analysis_service import draws_matrix, frequency, gaps
from services.scoring_service import normalize_weights, score_sample


def build_statistical_seed(
    draws,
    *,
    experiment_date: date | None = None,
    recent_count: int = 20,
) -> dict:
    """Build a deterministic SHA-256 seed from the available history.

    Repeating this function with the same ordered history and experiment date
    produces the same digest and numeric RNG seed.
    """
    ordered = sorted(draws, key=lambda draw: draw.draw_date)
    if not ordered:
        raise ValueError("No hay sorteos cargados para generar la semilla")

    matrix = draws_matrix(ordered)
    as_of_date = ordered[-1].draw_date.isoformat()
    run_date = (experiment_date or date.today()).isoformat()
    recent = [
        {
            "date": draw.draw_date.isoformat(),
            "draw_number": draw.draw_number,
            "numbers": [int(number) for number in draw.numbers()],
        }
        for draw in ordered[-recent_count:]
    ]
    payload = {
        "schema": "kino-statistical-seed-v1",
        "experiment_date": run_date,
        "history_as_of": as_of_date,
        "draw_count": len(ordered),
        "frequencies": frequency(matrix),
        "gaps": gaps(matrix),
        "recent_draws": recent,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    # SQLite INTEGER is signed 64-bit; keep the deterministic RNG seed in range.
    numeric_seed = int(digest[:16], 16) & ((1 << 63) - 1)
    return {
        "sha256": digest,
        "numeric_seed": numeric_seed,
        "payload": payload,
    }


def generate_seed_candidates(
    draws,
    *,
    experiment_date: date | None = None,
    sample_size: int = 20_000,
    weights: dict[str, float] | None = None,
    top_k: int = 20,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[dict, list[dict]]:
    seed = build_statistical_seed(draws, experiment_date=experiment_date)
    matrix = draws_matrix(draws)
    results = score_sample(
        matrix,
        sample_size=sample_size,
        seed=seed["numeric_seed"],
        weights=normalize_weights(weights),
        top_k=top_k,
        progress_cb=progress_cb,
        should_cancel=should_cancel,
    )
    return seed, results
