from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from models import LotteryAnalysisSnapshot, LotteryDraw, LotteryGame, db

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover
    scipy_stats = None


MIN_DAY_MONTH_SUPPORT = 8


def _chisquare_uniform(counts: list[int]) -> dict[str, Any]:
    """Chi-cuadrado descriptivo vs uniforme. No implica predicción."""
    observed = np.asarray(counts, dtype=float)
    n = float(observed.sum())
    k = len(observed)
    if k < 2 or n <= 0:
        return {
            "statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "note": "datos insuficientes",
        }
    expected = np.full(k, n / k)
    # Avoid zero expected
    if np.any(expected <= 0):
        return {
            "statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "note": "esperados nulos",
        }
    if scipy_stats is not None:
        stat, p = scipy_stats.chisquare(observed, expected)
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "degrees_of_freedom": int(k - 1),
            "engine": "scipy",
            "note": (
                "Prueba descriptiva vs distribución uniforme histórica; "
                "no cambia probabilidades futuras de un sorteo justo."
            ),
        }
    # Manual chi-square + survival function approximation via regularized gamma
    statistic = float(np.sum((observed - expected) ** 2 / expected))
    df = k - 1
    # Incomplete gamma upper for chi2 survival: P(X>x) = gammaincc(df/2, x/2)
    try:
        from math import gamma

        # Series approximation for upper incomplete ratio (simple)
        # Prefer scipy; fallback p_value None if too rough
        p_value = None
        # Wilson-Hilferty approx for large df
        if df >= 1:
            z = ((statistic / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(
                2 / (9 * df)
            )
            # one-sided normal approx for p (rough)
            p_value = 0.5 * math.erfc(z / math.sqrt(2))
            p_value = float(max(0.0, min(1.0, p_value)))
        return {
            "statistic": statistic,
            "p_value": p_value,
            "degrees_of_freedom": df,
            "engine": "approx",
            "note": (
                "Aproximación sin scipy; solo descriptiva. "
                "No implica ventaja predictiva."
            ),
        }
    except Exception:  # pragma: no cover
        return {
            "statistic": statistic,
            "p_value": None,
            "degrees_of_freedom": df,
            "engine": "manual",
            "note": "p-value no disponible sin scipy",
        }


def _frequency(matrix: np.ndarray, pool_min: int, pool_max: int) -> dict[str, int]:
    counts = {str(n): 0 for n in range(pool_min, pool_max + 1)}
    if matrix.size:
        flat = matrix.ravel()
        for n, c in zip(*np.unique(flat, return_counts=True)):
            key = str(int(n))
            if key in counts:
                counts[key] = int(c)
    return counts


def _gaps(matrix: np.ndarray, pool_min: int, pool_max: int) -> dict[str, int]:
    result = {str(n): len(matrix) for n in range(pool_min, pool_max + 1)}
    if matrix.size == 0:
        return result
    for offset, row in enumerate(matrix[::-1]):
        for n in row:
            key = str(int(n))
            if key in result and result[key] == len(matrix):
                result[key] = offset
    return result


def _pair_counts(matrix: np.ndarray, top: int = 25) -> list[dict]:
    counter: Counter[tuple[int, int]] = Counter()
    for row in matrix:
        for a, b in combinations(sorted(int(x) for x in row), 2):
            counter[(a, b)] += 1
    return [
        {"pair": list(pair), "count": count}
        for pair, count in counter.most_common(top)
    ]


def _triple_counts(matrix: np.ndarray, top: int = 15) -> list[dict]:
    counter: Counter[tuple[int, int, int]] = Counter()
    for row in matrix:
        vals = sorted(int(x) for x in row)
        if len(vals) < 3:
            continue
        for triple in combinations(vals, 3):
            counter[triple] += 1
    return [
        {"triple": list(triple), "count": count}
        for triple, count in counter.most_common(top)
    ]


def _parity(matrix: np.ndarray) -> dict[str, int]:
    dist: Counter[str] = Counter()
    draw_size = matrix.shape[1] if matrix.size else 0
    for row in matrix:
        even = int(np.sum(np.asarray(row) % 2 == 0))
        odd = draw_size - even
        dist[f"{even}P-{odd}I"] += 1
    return dict(dist)


def _sum_stats(matrix: np.ndarray) -> dict[str, float]:
    if matrix.size == 0:
        return {"mean": 0, "min": 0, "max": 0, "std": 0}
    sums = matrix.sum(axis=1).astype(float)
    return {
        "mean": float(sums.mean()),
        "min": float(sums.min()),
        "max": float(sums.max()),
        "std": float(sums.std()),
    }


def _consecutive_stats(matrix: np.ndarray) -> dict[str, Any]:
    runs: list[int] = []
    for row in matrix:
        nums = sorted(int(x) for x in row)
        run = 0
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1] + 1:
                run += 1
        runs.append(run)
    if not runs:
        return {"mean": 0, "distribution": {}}
    dist = Counter(runs)
    return {
        "mean": float(np.mean(runs)),
        "distribution": {str(k): int(v) for k, v in sorted(dist.items())},
    }


def _overlap_prev(matrix: np.ndarray) -> dict[str, Any]:
    if len(matrix) < 2:
        return {"mean": 0, "distribution": {}, "sample": 0}
    overlaps: list[int] = []
    for i in range(1, len(matrix)):
        overlaps.append(len(set(matrix[i]) & set(matrix[i - 1])))
    dist = Counter(overlaps)
    return {
        "mean": float(np.mean(overlaps)),
        "distribution": {str(k): int(v) for k, v in sorted(dist.items())},
        "sample": len(overlaps),
    }


def _calendar_patterns(
    draws: list[LotteryDraw],
) -> dict[str, Any]:
    by_weekday: Counter[str] = Counter()
    by_month: Counter[str] = Counter()
    weekday_names = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    for d in draws:
        by_weekday[weekday_names[d.draw_date.weekday()]] += 1
        by_month[f"{d.draw_date.month:02d}"] += 1

    def _filter(counter: Counter[str]) -> dict[str, int]:
        return {k: v for k, v in counter.items() if v >= MIN_DAY_MONTH_SUPPORT}

    return {
        "by_weekday": dict(by_weekday),
        "by_month": dict(by_month),
        "by_weekday_min_support": _filter(by_weekday),
        "by_month_min_support": _filter(by_month),
        "min_support": MIN_DAY_MONTH_SUPPORT,
        "note": (
            "Solo descriptivo; requiere soporte mínimo "
            f"({MIN_DAY_MONTH_SUPPORT}) para resaltar patrones."
        ),
    }


def _dataset_quality(draws: list[LotteryDraw]) -> dict[str, Any]:
    if not draws:
        return {
            "draw_count": 0,
            "date_from": None,
            "date_to": None,
            "duplicate_dates": 0,
            "missing_dates_hint": None,
        }
    dates = [d.draw_date for d in draws]
    date_counts = Counter(dates)
    duplicates = sum(1 for c in date_counts.values() if c > 1)
    date_from = min(dates)
    date_to = max(dates)
    # Hint only: span vs count (not claiming official calendar gaps)
    span_days = (date_to - date_from).days + 1
    return {
        "draw_count": len(draws),
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "duplicate_dates": duplicates,
        "span_days": span_days,
        "density": round(len(draws) / span_days, 4) if span_days else None,
        "note": (
            "Los 'faltantes' reales dependen del calendario oficial del juego; "
            "aquí solo se reporta densidad del dataset local."
        ),
    }


def build_lottery_analysis(
    game: LotteryGame,
    draws: Iterable[LotteryDraw],
) -> dict[str, Any]:
    rules = game.rules()
    draw_list = list(draws)
    draw_list.sort(key=lambda d: (d.draw_date, d.draw_number or 0))

    main_spec = rules.get("main") or {}
    main_min = int(main_spec.get("min", 1))
    main_max = int(main_spec.get("max", 1))
    main_rows = [d.main() for d in draw_list]
    main_matrix = (
        np.asarray(main_rows, dtype=np.int16)
        if main_rows
        else np.empty((0, int(main_spec.get("count", 0))), dtype=np.int16)
    )

    bonus_specs = rules.get("bonus") or []
    bonus_rows = [d.bonus() for d in draw_list]
    bonus_analyses: list[dict] = []
    offset = 0
    for spec in bonus_specs:
        count = int(spec["count"])
        bmin = int(spec["min"])
        bmax = int(spec["max"])
        chunk_rows = [row[offset : offset + count] for row in bonus_rows]
        offset += count
        bmatrix = (
            np.asarray(chunk_rows, dtype=np.int16)
            if chunk_rows and chunk_rows[0]
            else np.empty((0, count), dtype=np.int16)
        )
        freq = _frequency(bmatrix, bmin, bmax)
        bonus_analyses.append(
            {
                "name": spec.get("name") or "Bonus",
                "count": count,
                "min": bmin,
                "max": bmax,
                "frequency": freq,
                "gaps": _gaps(bmatrix, bmin, bmax),
                "chi_square": _chisquare_uniform(
                    [freq[str(n)] for n in range(bmin, bmax + 1)]
                ),
            }
        )

    main_freq = _frequency(main_matrix, main_min, main_max)
    analysis = {
        "game": {
            "slug": game.slug,
            "name": game.name,
            "country": game.country,
            "rules_summary": (
                f"{main_spec.get('count')} de {main_min}–{main_max}"
            ),
        },
        "disclaimer": (
            "Patrones históricos descriptivos. No cambian las probabilidades "
            "futuras si el mecanismo de sorteo es justo e independiente."
        ),
        "quality": _dataset_quality(draw_list),
        "main": {
            "name": main_spec.get("name") or "Main",
            "frequency": main_freq,
            "gaps": _gaps(main_matrix, main_min, main_max),
            "pairs": _pair_counts(main_matrix),
            "triples": _triple_counts(main_matrix),
            "parity": _parity(main_matrix),
            "sums": _sum_stats(main_matrix),
            "consecutive": _consecutive_stats(main_matrix),
            "overlap_previous": _overlap_prev(main_matrix),
            "chi_square": _chisquare_uniform(
                [main_freq[str(n)] for n in range(main_min, main_max + 1)]
            ),
        },
        "bonus": bonus_analyses,
        "calendar": _calendar_patterns(draw_list),
    }
    return analysis


def save_analysis_snapshot(game: LotteryGame, analysis: dict) -> LotteryAnalysisSnapshot:
    quality = analysis.get("quality") or {}
    date_from = quality.get("date_from")
    date_to = quality.get("date_to")
    snap = LotteryAnalysisSnapshot(
        game_id=game.id,
        draw_count=int(quality.get("draw_count") or 0),
        date_from=date.fromisoformat(date_from) if date_from else None,
        date_to=date.fromisoformat(date_to) if date_to else None,
        summary_json=json.dumps(
            {
                "chi_square_main": (analysis.get("main") or {}).get("chi_square"),
                "sums": (analysis.get("main") or {}).get("sums"),
                "overlap_previous": (analysis.get("main") or {}).get(
                    "overlap_previous"
                ),
            },
            default=str,
        ),
    )
    db.session.add(snap)
    db.session.commit()
    return snap


def compare_games(games_with_draws: list[tuple[LotteryGame, list[LotteryDraw]]]) -> dict:
    """Comparación básica entre juegos con datasets locales."""
    rows = []
    for game, draws in games_with_draws:
        rules = game.rules()
        main = rules.get("main") or {}
        analysis = build_lottery_analysis(game, draws) if draws else None
        rows.append(
            {
                "slug": game.slug,
                "name": game.name,
                "country": game.country,
                "main_matrix": (
                    f"{main.get('count')} / {main.get('min')}-{main.get('max')}"
                ),
                "bonus": [
                    f"{b.get('count')} {b.get('name')} ({b.get('min')}-{b.get('max')})"
                    for b in (rules.get("bonus") or [])
                ],
                "draw_count": len(draws),
                "date_from": analysis["quality"]["date_from"] if analysis else None,
                "date_to": analysis["quality"]["date_to"] if analysis else None,
                "sum_mean": (
                    analysis["main"]["sums"]["mean"] if analysis else None
                ),
                "overlap_mean": (
                    analysis["main"]["overlap_previous"]["mean"]
                    if analysis
                    else None
                ),
                "chi_p": (
                    (analysis["main"]["chi_square"] or {}).get("p_value")
                    if analysis
                    else None
                ),
                "has_data": bool(draws),
            }
        )
    return {
        "disclaimer": (
            "Comparación descriptiva de datasets locales. "
            "No implica que un juego sea 'mejor' para ganar."
        ),
        "games": rows,
    }
