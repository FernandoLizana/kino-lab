"""¿Las combinaciones del Kino se agrupan por color más que el azar?

Cada sorteo se pinta con services.color_service. Se compara contra combinaciones
aleatorias del mismo tamaño (14 o 15) y contra un nulo que baraja el orden
de los sorteos (para ver si el color persiste de un día al siguiente).

Uso:
    python -m research.color_pattern
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from research.tarot_synchronicity import load_longest_history  # noqa: E402
from services.color_service import PIGMENT_NAMES, combo_color  # noqa: E402

REPORT_PATH = BASE_DIR / "research" / "color_report.json"
POOL = 25


def _rows_for_era(hits: np.ndarray, sizes: np.ndarray, draw_size: int) -> list[list[int]]:
    rows = []
    for i, size in enumerate(sizes):
        if int(size) != draw_size:
            continue
        nums = [j + 1 for j in range(POOL) if hits[i, j] > 0]
        rows.append(nums)
    return rows


def _random_combos(draw_size: int, n: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(out) < n:
        pick = rng.choice(np.arange(1, POOL + 1), size=draw_size, replace=False)
        key = tuple(sorted(int(x) for x in pick))
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def _chi2(observed: np.ndarray, expected: np.ndarray) -> float:
    mask = expected > 1e-9
    return float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))


def _pvalue_chi2(stat: float, null: np.ndarray) -> float:
    return float((np.sum(null >= stat) + 1) / (len(null) + 1))


def test_pigment_dominance(rows: list[list[int]], random_rows: list[list[int]]) -> dict:
    obs = Counter(combo_color(r)["dominant_pigment"] for r in rows)
    exp_c = Counter(combo_color(r)["dominant_pigment"] for r in random_rows)
    names = list(PIGMENT_NAMES)
    observed = np.array([obs.get(n, 0) for n in names], dtype=float)
    expected = np.array([exp_c.get(n, 0) for n in names], dtype=float)
    expected = expected * (observed.sum() / max(expected.sum(), 1.0))
    stat = _chi2(observed, expected)
    # Nulo: recuentos multinomiales con las probs del azar.
    probs = expected / expected.sum()
    rng = np.random.default_rng(21)
    null = np.empty(400)
    for i in range(400):
        sim = rng.multinomial(int(observed.sum()), probs)
        null[i] = _chi2(sim.astype(float), expected)
    return {
        "observed": {n: int(obs.get(n, 0)) for n in names},
        "expected": {n: round(float(expected[i]), 1) for i, n in enumerate(names)},
        "chi2": round(stat, 3),
        "p_value": round(_pvalue_chi2(stat, null), 4),
    }


def test_hue_bins(rows: list[list[int]], random_rows: list[list[int]]) -> dict:
    obs = np.bincount([combo_color(r)["hue_bin_12"] for r in rows], minlength=12)
    exp_raw = np.bincount([combo_color(r)["hue_bin_12"] for r in random_rows], minlength=12)
    expected = exp_raw.astype(float) * (obs.sum() / max(exp_raw.sum(), 1.0))
    stat = _chi2(obs.astype(float), expected)
    rng = np.random.default_rng(22)
    probs = expected / expected.sum()
    null = np.empty(400)
    for i in range(400):
        sim = rng.multinomial(int(obs.sum()), probs)
        null[i] = _chi2(sim.astype(float), expected)
    return {
        "observed": [int(x) for x in obs],
        "expected": [round(float(x), 1) for x in expected],
        "chi2": round(stat, 3),
        "p_value": round(_pvalue_chi2(stat, null), 4),
    }


def test_saturation(rows: list[list[int]], random_rows: list[list[int]]) -> dict:
    obs = np.array([combo_color(r)["saturation"] for r in rows])
    rnd = np.array([combo_color(r)["saturation"] for r in random_rows])
    # ¿Las combinaciones reales están más "puras" (números juntos en el aro) que el azar?
    delta = float(obs.mean() - rnd.mean())
    rng = np.random.default_rng(23)
    pooled = np.concatenate([obs, rnd[: len(obs)]])
    null = np.empty(400)
    for i in range(400):
        rng.shuffle(pooled)
        null[i] = float(pooled[: len(obs)].mean() - pooled[len(obs) : 2 * len(obs)].mean())
    p = float((np.sum(np.abs(null) >= abs(delta)) + 1) / (400 + 1))
    return {
        "observed_mean": round(float(obs.mean()), 4),
        "random_mean": round(float(rnd.mean()), 4),
        "delta": round(delta, 4),
        "p_value": round(p, 4),
    }


def test_color_persistence(rows: list[list[int]]) -> dict:
    """¿El tono de un sorteo se parece al del siguiente más que si se baraja el orden?"""
    hues = np.array([combo_color(r)["hue_deg"] for r in rows])
    rad = np.deg2rad(hues)

    def _mean_abs_step(values: np.ndarray) -> float:
        d = np.abs((values[1:] - values[:-1] + 180.0) % 360.0 - 180.0)
        return float(d.mean())

    observed = _mean_abs_step(hues)
    rng = np.random.default_rng(24)
    null = np.empty(400)
    for i in range(400):
        null[i] = _mean_abs_step(rng.permutation(hues))
    p = float((np.sum(null <= observed) + 1) / (400 + 1))
    # Correlación circular lag-1 (sin/cos).
    s, c = np.sin(rad), np.cos(rad)
    corr = float(
        (
            np.corrcoef(s[:-1], s[1:])[0, 1]
            + np.corrcoef(c[:-1], c[1:])[0, 1]
        )
        / 2.0
    )
    return {
        "mean_hue_step_deg": round(observed, 2),
        "null_mean_step_deg": round(float(null.mean()), 2),
        "p_value_closer_than_chance": round(p, 4),
        "circular_lag1_corr": round(corr, 4),
    }


def test_number_frequency_by_pigment(hits: np.ndarray, sizes: np.ndarray, draw_size: int) -> dict:
    """Atajo: si un pigmento 'gana', ¿es solo que esos 5 números salen más?"""
    mask = sizes == draw_size
    totals = hits[mask].sum(axis=0)
    per_pigment = []
    for i, name in enumerate(PIGMENT_NAMES):
        sl = slice(i * 5, (i + 1) * 5)
        per_pigment.append(
            {
                "pigment": name,
                "numbers": f"{i * 5 + 1}-{i * 5 + 5}",
                "appearances": int(totals[sl].sum()),
            }
        )
    counts = np.array([p["appearances"] for p in per_pigment], dtype=float)
    expected = np.full(5, counts.sum() / 5.0)
    stat = _chi2(counts, expected)
    rng = np.random.default_rng(25)
    null = np.empty(400)
    for i in range(400):
        sim = rng.multinomial(int(counts.sum()), np.full(5, 0.2))
        null[i] = _chi2(sim.astype(float), expected)
    return {
        "pigments": per_pigment,
        "chi2": round(stat, 3),
        "p_value": round(_pvalue_chi2(stat, null), 4),
    }


def _print_block(title: str, payload: dict) -> None:
    print(f"\n{title}")
    if "observed" in payload and isinstance(payload["observed"], dict):
        for name, count in payload["observed"].items():
            exp = payload["expected"][name]
            print(f"  {name:<10} {count:5d}  (esperado {exp:6.1f})")
    print(f"  chi2={payload.get('chi2', '—')}  p={payload.get('p_value', payload.get('p_value_closer_than_chance'))}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random", type=int, default=20_000)
    args = parser.parse_args()

    dates, hits, sizes = load_longest_history()
    report: dict = {
        "n_draws": len(dates),
        "date_from": dates[0].isoformat(),
        "date_to": dates[-1].isoformat(),
        "mapping": (
            "tono = promedio circular de 25 arcos de 14.4°; "
            "pigmentos = 1-5 rojo, 6-10 ámbar, 11-15 verde, 16-20 azul, 21-25 violeta"
        ),
        "eras": {},
    }

    print(f"sorteos={len(dates)}  {dates[0]} → {dates[-1]}")
    print(report["mapping"])

    examples = []
    for era_size in (14, 15):
        rows = _rows_for_era(hits, sizes, era_size)
        if len(rows) < 50:
            continue
        print(f"\n======== era {era_size}/25  n={len(rows)} ========", flush=True)
        random_rows = _random_combos(era_size, args.random, seed=era_size * 17)
        dominance = test_pigment_dominance(rows, random_rows)
        hues = test_hue_bins(rows, random_rows)
        sat = test_saturation(rows, random_rows)
        persist = test_color_persistence(rows)
        freq = test_number_frequency_by_pigment(hits, sizes, era_size)

        _print_block("Pigmento dominante de la combinación", {**dominance, "p_value": dominance["p_value"]})
        _print_block("Tono en 12 casillas (cada 30°)", hues)
        print("\nSaturación (¿números más juntos en el aro que el azar?)")
        print(
            f"  real {sat['observed_mean']}  azar {sat['random_mean']}  "
            f"Δ {sat['delta']}  p={sat['p_value']}"
        )
        print("\n¿El color de hoy se parece al de ayer?")
        print(
            f"  paso medio {persist['mean_hue_step_deg']}°  "
            f"nulo {persist['null_mean_step_deg']}°  "
            f"p={persist['p_value_closer_than_chance']}  "
            f"corr={persist['circular_lag1_corr']}"
        )
        _print_block("Frecuencia cruda por pigmento (5 números cada uno)", {**freq, "observed": {p['pigment']: p['appearances'] for p in freq['pigments']}, "expected": {p['pigment']: round(sum(x['appearances'] for x in freq['pigments'])/5, 1) for p in freq['pigments']}})

        report["eras"][str(era_size)] = {
            "n": len(rows),
            "dominance": dominance,
            "hue_bins": hues,
            "saturation": sat,
            "persistence": persist,
            "frequency": freq,
        }
        examples.append(combo_color(rows[-1]))

    last = combo_color(
        [j + 1 for j in range(POOL) if hits[-1, j] > 0]
    )
    report["last_combo_color"] = last
    report["conclusion"] = _conclude(report["eras"])
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nÚltimo sorteo → {last['hex']}  tono {last['hue_deg']}°  "
          f"dominante {last['dominant_pigment']}")
    print(report["conclusion"])
    print(f"reporte: {REPORT_PATH}")


def _conclude(eras: dict) -> str:
    ps = []
    for era, block in eras.items():
        ps.extend(
            [
                block["dominance"]["p_value"],
                block["hue_bins"]["p_value"],
                block["saturation"]["p_value"],
                block["persistence"]["p_value_closer_than_chance"],
                block["frequency"]["p_value"],
            ]
        )
    if not ps:
        return "Sin datos."
    # Bonferroni tosco sobre las pruebas reportadas.
    n = len(ps)
    min_p = min(ps)
    if min_p * n > 0.05:
        return (
            "No hay patrón de color: ni el pigmento dominante, ni el tono, ni la "
            "pureza, ni el color de un sorteo al siguiente se apartan del azar "
            f"(p mínima {min_p:.3f}; con {n} pruebas el umbral es {0.05/n:.3f}). "
            "La combinación sí tiene un color, pero es solo una etiqueta del "
            "conjunto de números, no una señal."
        )
    return (
        f"Alguna prueba quedó bajo {0.05/n:.3f} (p mínima {min_p:.3f}). "
        "Revisar si es frecuencia cruda de números (bandas 1-5 vs 21-25) "
        "y no un color de la combinación."
    )


if __name__ == "__main__":
    main()
