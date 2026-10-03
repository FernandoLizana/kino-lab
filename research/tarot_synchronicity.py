"""Ejercicio: sincronicidad tarot–cielo sobre el histórico más largo posible.

Une historico.csv (desde 1990, incluyendo la era de 15 números) con el
dataset moderno. Traduce cada sorteo a tarot vía efemérides (22:30 Chile).

Dos capas:

  rápida    — Sol, Luna, fase, decano lunar (cambia en días).
  invisible — Júpiter, Saturno, Urano, Neptuno, Plutón, nodo. Solo se ve
              en décadas: Saturno tarda 29 años en dar la vuelta.

El nulo baraja las cartas DENTRO de cada era (15/25 vs 14/25) para no
confundir el cambio de reglamento con una señal.

Uso:
    python -m research.tarot_synchronicity
    python -m research.tarot_synchronicity --permutations 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config  # noqa: E402
from services.astrology_service import compute_astro_features  # noqa: E402
from services.tarot_service import (  # noqa: E402
    ALL_TAROT_BIN_KEYS,
    INVISIBLE_BIN_KEYS,
    TAROT_BIN_KEYS,
    as_tarot_view,
    measure_anchor_synchronicity,
    resolution_table,
)

POOL = Config.POOL_SIZE
REPORT_PATH = BASE_DIR / "research" / "tarot_report.json"
HISTORICO_PATH = BASE_DIR / "historico.csv"
MODERN_CSV = BASE_DIR / "data" / "kino_draws.csv"


def _parse_historico_date(raw: str) -> date:
    raw = raw.strip()
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return datetime.strptime(raw, "%d-%m-%Y").date()


def load_historico_all_eras(path: Path) -> dict[date, list[int]]:
    """Lee historico.csv aceptando 14 o 15 números (era antigua y moderna)."""
    rows: dict[date, list[int]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) < 15:
            continue
        numbers = [int(x) for x in parts[1:]]
        if len(numbers) not in (14, 15):
            continue
        if len(set(numbers)) != len(numbers):
            continue
        if min(numbers) < 1 or max(numbers) > POOL:
            continue
        rows[_parse_historico_date(parts[0])] = sorted(numbers)
    return rows


def load_modern_csv(path: Path) -> dict[date, list[int]]:
    import csv as csv_module

    rows: dict[date, list[int]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv_module.DictReader(handle):
            draw_date = date.fromisoformat(record["draw_date"].strip())
            numbers = [int(record[f"n{i}"]) for i in range(1, 15)]
            if len(set(numbers)) != 14:
                continue
            rows[draw_date] = sorted(numbers)
    return rows


def load_longest_history() -> tuple[list[date], np.ndarray, np.ndarray]:
    """Une 1990→hoy. Si una fecha está en ambos, gana el CSV moderno."""
    merged = load_historico_all_eras(HISTORICO_PATH)
    merged.update(load_modern_csv(MODERN_CSV))
    dates = sorted(merged)
    hits = np.zeros((len(dates), POOL), dtype=np.float64)
    sizes = np.empty(len(dates), dtype=np.int16)
    for i, draw_date in enumerate(dates):
        numbers = merged[draw_date]
        sizes[i] = len(numbers)
        for number in numbers:
            hits[i, number - 1] = 1.0
    return dates, hits, sizes


def _max_abs_z(
    labels: np.ndarray,
    hits: np.ndarray,
    sizes: np.ndarray,
    *,
    min_support: int,
) -> float:
    """|z| máximo carta×número, con tasa base de la era de cada sorteo."""
    best = 0.0
    expected = sizes.astype(np.float64) / POOL
    uniq = np.unique(labels)
    for value in uniq:
        mask = labels == value
        n = int(mask.sum())
        if n < min_support:
            continue
        p0 = float(expected[mask].mean())
        sd = float(np.sqrt(max(p0 * (1.0 - p0), 1e-9)))
        rates = hits[mask].mean(axis=0)
        z = float(np.max(np.abs(rates - p0) * np.sqrt(n) / sd))
        if z > best:
            best = z
    return best


def era_aware_pattern_null(
    label_columns: dict[str, list[str]],
    hits: np.ndarray,
    sizes: np.ndarray,
    *,
    keys: tuple[str, ...],
    permutations: int,
    min_support: int,
    seed: int,
) -> dict:
    """Nulo familiar: baraja etiquetas dentro de cada era (14 vs 15)."""
    coded = []
    for key in keys:
        values, inverse = np.unique(np.asarray(label_columns[key]), return_inverse=True)
        if len(values) <= 1:
            continue
        coded.append(inverse.astype(np.int64))
    if not coded:
        return {"available": False, "reason": "sin bins"}

    eras = (sizes == 15).astype(np.int8)
    observed = 0.0
    for column in coded:
        observed = max(
            observed, _max_abs_z(column, hits, sizes, min_support=min_support)
        )

    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    era_idx = {0: np.flatnonzero(eras == 0), 1: np.flatnonzero(eras == 1)}
    for i in range(permutations):
        best = 0.0
        for column in coded:
            shuffled = column.copy()
            for idx in era_idx.values():
                if idx.size:
                    shuffled[idx] = rng.permutation(shuffled[idx])
            best = max(
                best, _max_abs_z(shuffled, hits, sizes, min_support=min_support)
            )
        null[i] = best

    p_value = float((np.sum(null >= observed) + 1) / (permutations + 1))
    return {
        "available": True,
        "permutations": permutations,
        "min_support": min_support,
        "observed_max_abs_z": round(observed, 4),
        "critical_abs_z": round(float(np.quantile(null, 0.95)), 4),
        "null_max_abs_z_mean": round(float(null.mean()), 4),
        "family_wise_p_value": round(p_value, 4),
        "keys": list(keys),
    }


def layer_synchronicity(
    hits: np.ndarray,
    sizes: np.ndarray,
    label_columns: dict[str, list[str]],
    keys: tuple[str, ...],
    *,
    permutations: int,
) -> dict[str, dict]:
    """Jaccard solo entre sorteos de la misma era (14 con 14, 15 con 15)."""
    out: dict[str, dict] = {}
    for era_size in (14, 15):
        idx = np.flatnonzero(sizes == era_size)
        if idx.size < 30:
            continue
        era_hits = hits[idx]
        for key in keys:
            labels = [label_columns[key][int(i)] for i in idx]
            out[f"{key}|{era_size}"] = measure_anchor_synchronicity(
                era_hits, labels, permutations=permutations
            )
    return out


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument(
        "--min-support",
        type=int,
        default=400,
        help="Soporte mínimo por valor. 400 (no 600) porque la era antigua es semanal.",
    )
    args = parser.parse_args()

    t0 = time.time()
    dates, hits, sizes = load_longest_history()
    n15 = int((sizes == 15).sum())
    n14 = int((sizes == 14).sum())
    print(
        f"sorteos={len(dates)}  {dates[0]} → {dates[-1]}  "
        f"(era 15 números: {n15} · era 14: {n14})",
        flush=True,
    )

    print("efemérides + tarot (capa rápida y capa invisible)...", flush=True)
    views = [as_tarot_view(compute_astro_features(d)) for d in dates]
    bin_columns = {
        key: [str(v["pattern_bins"][key]) for v in views] for key in ALL_TAROT_BIN_KEYS
    }
    resolution = resolution_table(bin_columns, len(dates))

    print("\nRESOLUCIÓN con el histórico largo")
    print(f"  {'variable':<22}{'valores':>8}{'mediana n':>12}{'usable≥400':>12}{'salto mín':>10}")
    for row in resolution:
        usable = row["median_draws_per_value"] >= args.min_support
        jump = row["min_detectable_rate_jump_pp"]
        jump_s = f"{jump:.1f}pp" if jump is not None else "—"
        print(
            f"  {row['variable']:<22}{row['distinct_values']:>8}"
            f"{row['median_draws_per_value']:>12.0f}"
            f"{('sí' if usable else 'no'):>12}{jump_s:>10}"
        )

    print("\nNULO FAMILIAR — capa rápida (Sol/Luna/fase)", flush=True)
    null_fast = era_aware_pattern_null(
        bin_columns,
        hits,
        sizes,
        keys=TAROT_BIN_KEYS,
        permutations=args.permutations,
        min_support=args.min_support,
        seed=1990,
    )
    print(
        f"  observado |z|={null_fast.get('observed_max_abs_z')}  "
        f"crítico={null_fast.get('critical_abs_z')}  "
        f"p_fam={null_fast.get('family_wise_p_value')}"
    )

    print("\nNULO FAMILIAR — capa invisible (Saturno/Júpiter/nodo/exteriores)", flush=True)
    null_slow = era_aware_pattern_null(
        bin_columns,
        hits,
        sizes,
        keys=INVISIBLE_BIN_KEYS,
        permutations=args.permutations,
        min_support=args.min_support,
        seed=1991,
    )
    print(
        f"  observado |z|={null_slow.get('observed_max_abs_z')}  "
        f"crítico={null_slow.get('critical_abs_z')}  "
        f"p_fam={null_slow.get('family_wise_p_value')}"
    )

    print("\nSincronicidad intra-carta por era", flush=True)
    sync = layer_synchronicity(
        hits, sizes, bin_columns, TAROT_BIN_KEYS + INVISIBLE_BIN_KEYS,
        permutations=args.permutations,
    )
    interesting = []
    for name, row in sorted(sync.items()):
        mark = "SEÑAL" if row["significant_at_05"] else "nulo"
        if row["p_value"] <= 0.15:
            interesting.append((name, row, mark))
        if row["significant_at_05"] or row["p_value"] <= 0.10:
            print(
                f"  {name:<28} J={row['observed_mean_jaccard']:.4f} "
                f"(nulo {row['null_mean_jaccard']:.4f}) p={row['p_value']:.3f}  {mark}"
            )
    if not interesting:
        print("  ninguna carta agrupó sorteos más parecidos que el azar (p≤0.15)")

    first_spread = views[0]["tarot"]
    last_spread = views[-1]["tarot"]
    first_slow = views[0]["pattern_bins"]
    last_slow = views[-1]["pattern_bins"]
    print(f"\nCielo del primer sorteo ({dates[0]})")
    print(f"  energía {first_spread['energy']['name']} · ancla {first_spread['pattern']['name']}")
    print(
        f"  invisible: Saturno {first_slow['saturn_major']} / "
        f"Júpiter {first_slow['jupiter_major']} / nodo {first_slow['node_major']}"
    )
    print(f"Cielo del último sorteo ({dates[-1]})")
    print(f"  energía {last_spread['energy']['name']} · ancla {last_spread['pattern']['name']}")
    print(
        f"  invisible: Saturno {last_slow['saturn_major']} / "
        f"Júpiter {last_slow['jupiter_major']} / nodo {last_slow['node_major']}"
    )

    conclusion = _conclude(null_fast, null_slow, resolution, args.min_support)
    report = {
        "n_draws": len(dates),
        "n_era_15": n15,
        "n_era_14": n14,
        "date_from": dates[0].isoformat(),
        "date_to": dates[-1].isoformat(),
        "ephemeris": "Swiss Ephemeris Moshier · 22:30 America/Santiago",
        "correspondence": "Golden Dawn / Book T + arcanos de planetas lentos",
        "resolution": resolution,
        "null_fast": null_fast,
        "null_invisible": null_slow,
        "synchronicity": sync,
        "first_spread": first_spread,
        "last_spread": last_spread,
        "runtime_seconds": round(time.time() - t0, 1),
        "conclusion": conclusion,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{conclusion}")
    print(f"reporte: {REPORT_PATH}  ({report['runtime_seconds']}s)")


def _conclude(fast: dict, slow: dict, resolution: list[dict], min_support: int) -> str:
    usable = [
        r["variable"]
        for r in resolution
        if r["median_draws_per_value"] >= min_support
    ]
    pf = float(fast.get("family_wise_p_value") or 1.0)
    ps = float(slow.get("family_wise_p_value") or 1.0)
    if pf > 0.05 and ps > 0.05:
        return (
            "Ni la capa rápida ni la invisible superan el nulo familiar "
            f"(p_rápida={pf:.3f}, p_invisible={ps:.3f}). "
            "Con efemérides desde 1990 el cielo lento ya es medible "
            f"({', '.join(usable) or 'ninguna variable con soporte'}), "
            "y no aparece un patrón carta→número distinto del azar."
        )
    which = []
    if pf <= 0.05:
        which.append(f"rápida p={pf:.3f}")
    if ps <= 0.05:
        which.append(f"invisible p={ps:.3f}")
    return "Posible señal en " + " y ".join(which) + ". Revisar otro corte temporal."


if __name__ == "__main__":
    main()
