"""Auditoría empírica de variables y métodos de evaluación del laboratorio Kino.

Responde tres preguntas con datos, no con opinión:

1. ¿Qué variables (astrológicas, lunares, estadísticas, calendario) tienen
   asociación con la aparición de cada número más allá del ruido?
2. ¿Qué estrategia de scoring generaliza mejor fuera de muestra?
3. ¿Conviene un score compuesto o una sola variable?

Método: walk-forward estricto (cada predicción usa solo el pasado) + nulos por
permutación (Westfall-Young con estadístico máximo) para controlar el hecho de
que estamos probando decenas de estrategias a la vez.

Incluye un control de potencia con señal plantada: si el arnés no detecta una
dependencia inyectada artificialmente, sus resultados nulos no valen nada.

Uso:
    python -m research.variable_audit                # suite completa
    python -m research.variable_audit --permutations 500 --min-history 800
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import Config  # noqa: E402
from services.astrology_service import (  # noqa: E402
    compute_astro_features,
    group_feature_channels,
)

POOL = Config.POOL_SIZE
DRAW = Config.DRAW_SIZE
BASE_RATE = DRAW / POOL
N_HITS = DRAW
N_MISSES = POOL - DRAW
CACHE_PATH = BASE_DIR / "research" / "_feature_cache.npz"
CACHE_VERSION = 2
REPORT_PATH = BASE_DIR / "research" / "audit_report.json"


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #


@dataclass
class Dataset:
    dates: list[date]
    hits: np.ndarray  # (T, 25) binaria
    numeric: np.ndarray  # (T, D) features numéricas
    numeric_keys: list[str]
    bins: np.ndarray  # (T, B) categóricas codificadas como enteros
    bin_keys: list[str]
    bin_cardinality: list[int]
    moon_longitude: np.ndarray  # (T,) grados

    @property
    def n_draws(self) -> int:
        return len(self.dates)


def load_draws(csv_path: Path) -> tuple[list[date], np.ndarray]:
    """Lee el CSV canónico y devuelve fechas ordenadas + matriz de aciertos."""
    import csv as csv_module

    rows: list[tuple[date, list[int]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv_module.DictReader(handle):
            draw_date = date.fromisoformat(record["draw_date"].strip())
            numbers = [int(record[f"n{i}"]) for i in range(1, DRAW + 1)]
            if len(set(numbers)) != DRAW:
                continue
            rows.append((draw_date, numbers))

    rows.sort(key=lambda item: item[0])
    dates = [item[0] for item in rows]
    hits = np.zeros((len(rows), POOL), dtype=np.float32)
    for idx, (_, numbers) in enumerate(rows):
        for number in numbers:
            hits[idx, number - 1] = 1.0
    return dates, hits


def build_dataset(csv_path: Path, *, use_cache: bool = True) -> Dataset:
    dates, hits = load_draws(csv_path)

    if use_cache and CACHE_PATH.exists():
        cached = np.load(CACHE_PATH, allow_pickle=True)
        if len(cached["dates"]) == len(dates) and int(cached.get("version", 0)) == CACHE_VERSION:
            return Dataset(
                dates=dates,
                hits=hits,
                numeric=cached["numeric"],
                numeric_keys=list(cached["numeric_keys"]),
                bins=cached["bins"],
                bin_keys=list(cached["bin_keys"]),
                bin_cardinality=list(cached["bin_cardinality"]),
                moon_longitude=cached["moon_longitude"],
            )

    payloads = [compute_astro_features(d) for d in dates]

    numeric_keys = sorted(payloads[0]["model_features"].keys())
    numeric = np.asarray(
        [[float(p["model_features"][k]) for k in numeric_keys] for p in payloads],
        dtype=np.float32,
    )

    bin_keys = sorted(payloads[0]["pattern_bins"].keys())
    vocabularies: list[dict[str, int]] = []
    coded: list[list[int]] = []
    for key_idx, key in enumerate(bin_keys):
        vocab: dict[str, int] = {}
        column: list[int] = []
        for payload in payloads:
            value = str(payload["pattern_bins"].get(key, "unknown"))
            column.append(vocab.setdefault(value, len(vocab)))
        vocabularies.append(vocab)
        coded.append(column)

    # Variables de calendario: control honesto no astrológico.
    weekday_vocab: dict[str, int] = {}
    weekday_column = [
        weekday_vocab.setdefault(str(d.weekday()), len(weekday_vocab)) for d in dates
    ]
    bin_keys.append("calendar_weekday")
    vocabularies.append(weekday_vocab)
    coded.append(weekday_column)

    month_vocab: dict[str, int] = {}
    month_column = [
        month_vocab.setdefault(str(d.month), len(month_vocab)) for d in dates
    ]
    bin_keys.append("calendar_month")
    vocabularies.append(month_vocab)
    coded.append(month_column)

    moon_longitude = np.asarray(
        [float(p["bodies"]["moon"]["longitude_deg"]) for p in payloads],
        dtype=np.float64,
    )

    # Posición lunar discretizada: el canal que usa el servicio lunar en producción.
    sector_column = (moon_longitude % 360.0 / 360.0 * POOL).astype(int)
    bin_keys.append("lunar_sector25")
    vocabularies.append({str(v): v for v in range(POOL)})
    coded.append(list(sector_column))

    sign_column = (moon_longitude % 360.0 / 30.0).astype(int)
    bin_keys.append("lunar_zodiac_sign")
    vocabularies.append({str(v): v for v in range(12)})
    coded.append(list(sign_column))

    bins = np.asarray(coded, dtype=np.int16).T
    bin_cardinality = [len(v) for v in vocabularies]

    np.savez_compressed(
        CACHE_PATH,
        version=CACHE_VERSION,
        dates=np.asarray([d.isoformat() for d in dates]),
        numeric=numeric,
        numeric_keys=np.asarray(numeric_keys),
        bins=bins,
        bin_keys=np.asarray(bin_keys),
        bin_cardinality=np.asarray(bin_cardinality),
        moon_longitude=moon_longitude,
    )

    return Dataset(
        dates=dates,
        hits=hits,
        numeric=numeric,
        numeric_keys=numeric_keys,
        bins=bins,
        bin_keys=bin_keys,
        bin_cardinality=bin_cardinality,
        moon_longitude=moon_longitude,
    )


# --------------------------------------------------------------------------- #
# Prueba 1: asociación marginal variable ↔ número (dentro de muestra)
# --------------------------------------------------------------------------- #


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    m = p_values.size
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = ranked <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if passing.any():
        cutoff = np.max(np.nonzero(passing)[0])
        rejected[order[: cutoff + 1]] = True
    return rejected


def _correlation_matrix(features: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """Correlación punto-biserial entre cada feature y cada número."""
    f = features.astype(np.float64)
    y = hits.astype(np.float64)
    f = f - f.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    f_norm = np.linalg.norm(f, axis=0, keepdims=True)
    y_norm = np.linalg.norm(y, axis=0, keepdims=True)
    f_norm[f_norm < 1e-12] = np.inf
    y_norm[y_norm < 1e-12] = np.inf
    return (f / f_norm).T @ (y / y_norm)


def _corr_p_values(corr: np.ndarray, n: int) -> np.ndarray:
    from scipy import stats

    r = np.clip(corr, -0.999999, 0.999999)
    t_stat = r * np.sqrt((n - 2) / (1.0 - r**2))
    return 2.0 * stats.t.sf(np.abs(t_stat), df=n - 2)


def test_marginal_associations(
    data: Dataset, *, permutations: int, rng: np.random.Generator
) -> dict:
    """¿Alguna variable numérica se asocia con algún número más de lo esperado?"""
    corr = _correlation_matrix(data.numeric, data.hits)
    p_values = _corr_p_values(corr, data.n_draws)
    rejected = _benjamini_hochberg(p_values.ravel(), alpha=0.05)
    observed_discoveries = int(rejected.sum())
    observed_max_abs_r = float(np.abs(corr).max())

    null_discoveries = np.zeros(permutations, dtype=np.int32)
    null_max_r = np.zeros(permutations, dtype=np.float64)
    for i in range(permutations):
        shuffled = data.hits[rng.permutation(data.n_draws)]
        null_corr = _correlation_matrix(data.numeric, shuffled)
        null_p = _corr_p_values(null_corr, data.n_draws)
        null_discoveries[i] = int(_benjamini_hochberg(null_p.ravel(), alpha=0.05).sum())
        null_max_r[i] = float(np.abs(null_corr).max())

    flat_idx = np.argsort(p_values.ravel())[:10]
    strongest = []
    for idx in flat_idx:
        f_idx, n_idx = np.unravel_index(idx, corr.shape)
        strongest.append(
            {
                "feature": data.numeric_keys[int(f_idx)],
                "number": int(n_idx) + 1,
                "r": round(float(corr[f_idx, n_idx]), 5),
                "p_value": float(p_values[f_idx, n_idx]),
            }
        )

    return {
        "tests_run": int(corr.size),
        "discoveries_fdr05": observed_discoveries,
        "null_discoveries_mean": float(null_discoveries.mean()),
        "null_discoveries_p95": float(np.percentile(null_discoveries, 95)),
        "max_abs_correlation": round(observed_max_abs_r, 5),
        "null_max_abs_correlation_mean": round(float(null_max_r.mean()), 5),
        "max_correlation_p_value": float(
            (np.sum(null_max_r >= observed_max_abs_r) + 1) / (permutations + 1)
        ),
        "strongest_pairs": strongest,
    }


def test_categorical_associations(
    data: Dataset, *, permutations: int, rng: np.random.Generator
) -> dict:
    """Chi-cuadrado bin ↔ número, con nulo por permutación sobre el máximo."""
    from scipy import stats

    def max_chi2(hits: np.ndarray) -> tuple[float, tuple[int, int] | None]:
        best = 0.0
        best_pair: tuple[int, int] | None = None
        for b_idx in range(data.bins.shape[1]):
            column = data.bins[:, b_idx]
            cardinality = int(column.max()) + 1
            if cardinality < 2:
                continue
            one_hot = np.zeros((data.n_draws, cardinality), dtype=np.float64)
            one_hot[np.arange(data.n_draws), column] = 1.0
            counts_per_value = one_hot.sum(axis=0)
            hits_per_value = one_hot.T @ hits  # (V, 25)
            expected = np.outer(counts_per_value, hits.sum(axis=0)) / data.n_draws
            misses = counts_per_value[:, None] - hits_per_value
            expected_miss = counts_per_value[:, None] - expected
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = np.where(expected > 5, (hits_per_value - expected) ** 2 / expected, 0.0)
                chi2 += np.where(
                    expected_miss > 5, (misses - expected_miss) ** 2 / expected_miss, 0.0
                )
            per_number = chi2.sum(axis=0)
            local_best = float(per_number.max())
            if local_best > best:
                best = local_best
                best_pair = (b_idx, int(per_number.argmax()))
        return best, best_pair

    observed, pair = max_chi2(data.hits)
    null = np.zeros(permutations, dtype=np.float64)
    for i in range(permutations):
        null[i], _ = max_chi2(data.hits[rng.permutation(data.n_draws)])

    detail = None
    if pair is not None:
        b_idx, n_idx = pair
        cardinality = int(data.bins[:, b_idx].max()) + 1
        detail = {
            "bin_variable": data.bin_keys[b_idx],
            "number": n_idx + 1,
            "chi2": round(observed, 3),
            "df": cardinality - 1,
            "nominal_p": float(stats.chi2.sf(observed, cardinality - 1)),
        }

    return {
        "strongest_bin_association": detail,
        "permutation_p_value": float((np.sum(null >= observed) + 1) / (permutations + 1)),
        "null_max_chi2_mean": round(float(null.mean()), 3),
        "null_max_chi2_p95": round(float(np.percentile(null, 95)), 3),
    }


# --------------------------------------------------------------------------- #
# Prueba 2: estrategias walk-forward
# --------------------------------------------------------------------------- #


@dataclass
class Strategy:
    name: str
    family: str
    scores: np.ndarray = field(repr=False)  # (n_steps, 25)


def _circular_distance(a: np.ndarray, b: float) -> np.ndarray:
    diff = np.abs((a - b) % 360.0)
    return np.minimum(diff, 360.0 - diff)


def _weighted_history_scores(
    weights_matrix: np.ndarray, hits: np.ndarray, start: int
) -> np.ndarray:
    """Para cada paso t≥start: promedio de aciertos pasados ponderado por weights."""
    n_steps = hits.shape[0] - start
    out = np.zeros((n_steps, POOL), dtype=np.float32)
    for offset in range(n_steps):
        t = start + offset
        w = weights_matrix[t, :t]
        total = w.sum()
        if total <= 1e-12:
            out[offset] = BASE_RATE
        else:
            out[offset] = (w @ hits[:t]) / total
    return out


def _similarity_strategy(
    features: np.ndarray, hits: np.ndarray, start: int
) -> np.ndarray:
    """Pondera sorteos pasados por similitud coseno del cielo (igual que producción)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    unit = (features / norms).astype(np.float32)
    gram = unit @ unit.T  # (T, T) similitud coseno
    weights = np.clip(0.5 * (gram + 1.0), 0.0, None) ** 2
    return _weighted_history_scores(weights, hits, start)


def _moon_kernel_strategy(
    longitudes: np.ndarray, hits: np.ndarray, start: int, bandwidth: float
) -> np.ndarray:
    diff = np.abs(longitudes[:, None] - longitudes[None, :]) % 360.0
    dist = np.minimum(diff, 360.0 - diff)
    weights = np.exp(-0.5 * (dist / max(bandwidth, 1e-3)) ** 2).astype(np.float32)
    return _weighted_history_scores(weights, hits, start)


def _bin_association_strategy(
    column: np.ndarray, hits: np.ndarray, start: int, *, min_support: int = 3
) -> np.ndarray:
    """Score = tasa histórica de cada número condicionada al bin actual."""
    cardinality = int(column.max()) + 1
    counts = np.zeros((cardinality, POOL), dtype=np.float64)
    totals = np.zeros(cardinality, dtype=np.float64)
    for t in range(start):
        counts[column[t]] += hits[t]
        totals[column[t]] += 1.0

    n_steps = hits.shape[0] - start
    out = np.zeros((n_steps, POOL), dtype=np.float32)
    for offset in range(n_steps):
        t = start + offset
        value = column[t]
        if totals[value] >= min_support:
            out[offset] = counts[value] / totals[value]
        else:
            out[offset] = BASE_RATE
        counts[value] += hits[t]
        totals[value] += 1.0
    return out


def _frequency_strategy(hits: np.ndarray, start: int, *, window: int | None) -> np.ndarray:
    cumulative = np.cumsum(hits, axis=0)
    n_steps = hits.shape[0] - start
    out = np.zeros((n_steps, POOL), dtype=np.float32)
    for offset in range(n_steps):
        t = start + offset
        if window is None:
            out[offset] = cumulative[t - 1] / t
        else:
            lo = max(0, t - window)
            span = t - lo
            out[offset] = (cumulative[t - 1] - (cumulative[lo - 1] if lo > 0 else 0)) / span
    return out


def _gap_strategy(hits: np.ndarray, start: int) -> np.ndarray:
    n_steps = hits.shape[0] - start
    out = np.zeros((n_steps, POOL), dtype=np.float32)
    last_seen = np.full(POOL, -1, dtype=np.int64)
    for t in range(start):
        last_seen[hits[t] > 0] = t
    for offset in range(n_steps):
        t = start + offset
        out[offset] = t - last_seen
        last_seen[hits[t] > 0] = t
    return out


def _ewma_strategy(hits: np.ndarray, start: int, halflife: float) -> np.ndarray:
    decay = 0.5 ** (1.0 / halflife)
    n_steps = hits.shape[0] - start
    out = np.zeros((n_steps, POOL), dtype=np.float32)
    state = np.zeros(POOL, dtype=np.float64)
    for t in range(start):
        state = decay * state + (1 - decay) * hits[t]
    for offset in range(n_steps):
        t = start + offset
        out[offset] = state
        state = decay * state + (1 - decay) * hits[t]
    return out


def build_strategies(data: Dataset, start: int) -> list[Strategy]:
    strategies: list[Strategy] = []
    hits = data.hits

    # Control: ruido puro. Cualquier estrategia debe superarlo para ser interesante.
    rng = np.random.default_rng(20260905)
    strategies.append(
        Strategy(
            "control_random",
            "control",
            rng.random((hits.shape[0] - start, POOL)).astype(np.float32),
        )
    )
    strategies.append(
        Strategy(
            "control_number_index",
            "control",
            np.tile(np.arange(POOL, dtype=np.float32), (hits.shape[0] - start, 1)),
        )
    )

    # Familia estadística clásica.
    strategies.append(
        Strategy("stat_frequency_all", "statistical", _frequency_strategy(hits, start, window=None))
    )
    for window in (25, 50, 100, 250, 500):
        strategies.append(
            Strategy(
                f"stat_frequency_w{window}",
                "statistical",
                _frequency_strategy(hits, start, window=window),
            )
        )
    strategies.append(Strategy("stat_gap_cold", "statistical", _gap_strategy(hits, start)))
    strategies.append(
        Strategy("stat_gap_hot", "statistical", -_gap_strategy(hits, start))
    )
    for halflife in (10.0, 30.0, 80.0, 200.0):
        strategies.append(
            Strategy(
                f"stat_ewma_h{int(halflife)}",
                "statistical",
                _ewma_strategy(hits, start, halflife),
            )
        )

    # Familia astro: vector completo y cada canal por separado.
    strategies.append(
        Strategy(
            "astro_similarity_all",
            "astro_similarity",
            _similarity_strategy(data.numeric, hits, start),
        )
    )
    channels = group_feature_channels(data.numeric_keys)
    key_index = {k: i for i, k in enumerate(data.numeric_keys)}
    for channel_name, channel_keys in sorted(channels.items()):
        cols = [key_index[k] for k in channel_keys if k in key_index]
        if not cols:
            continue
        strategies.append(
            Strategy(
                f"astro_channel_{channel_name}",
                "astro_channel",
                _similarity_strategy(data.numeric[:, cols], hits, start),
            )
        )

    # Familia lunar posicional.
    for bandwidth in (4.0, 8.0, 12.0, 24.0, 45.0):
        strategies.append(
            Strategy(
                f"lunar_kernel_bw{int(bandwidth)}",
                "lunar_position",
                _moon_kernel_strategy(data.moon_longitude, hits, start, bandwidth),
            )
        )
    # Familia de bins categóricos: uno por variable (incluye sector lunar y calendario).
    for b_idx, key in enumerate(data.bin_keys):
        column = data.bins[:, b_idx].astype(np.int64)
        if int(column.max()) + 1 < 2:
            continue
        if key.startswith("calendar_"):
            family = "calendar"
        elif key.startswith("lunar_sector") or key.startswith("lunar_zodiac"):
            family = "lunar_position"
        else:
            family = "astro_bin"
        strategies.append(
            Strategy(f"bin_{key}", family, _bin_association_strategy(column, hits, start))
        )

    return strategies


# --------------------------------------------------------------------------- #
# Métricas y nulos por permutación
# --------------------------------------------------------------------------- #


def _rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Rangos promedio por fila, base 1, repartiendo empates.

    Sin promediar empates, el desempate arbitrario del argsort inflaría el AUC
    de cualquier estrategia que devuelva scores constantes.
    """
    n_rows = scores.shape[0]
    order = np.argsort(scores, axis=1, kind="stable")
    ordered = np.take_along_axis(scores, order, axis=1)

    is_start = np.ones_like(ordered, dtype=bool)
    is_start[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    positions = np.tile(np.arange(POOL, dtype=np.float64), (n_rows, 1))

    group_start = np.maximum.accumulate(np.where(is_start, positions, -1.0), axis=1)
    is_end = np.ones_like(ordered, dtype=bool)
    is_end[:, :-1] = is_start[:, 1:]
    group_end = np.minimum.accumulate(
        np.where(is_end, positions, float(POOL))[:, ::-1], axis=1
    )[:, ::-1]

    averaged = (group_start + group_end) / 2.0 + 1.0
    ranks = np.empty_like(averaged)
    np.put_along_axis(ranks, order, averaged, axis=1)
    return ranks


def _top_mask(scores: np.ndarray) -> np.ndarray:
    """Máscara de los 14 números mejor puntuados por paso."""
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :DRAW]
    mask = np.zeros_like(scores, dtype=np.float64)
    np.put_along_axis(mask, idx, 1.0, axis=1)
    return mask


def evaluate_strategies(
    strategies: list[Strategy],
    outcomes: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    n_steps = outcomes.shape[0]
    auc_offset = N_HITS * (N_HITS + 1) / 2.0
    auc_scale = float(N_HITS * N_MISSES)

    perms = np.stack([rng.permutation(n_steps) for _ in range(permutations)])
    observed_aucs: list[float] = []
    null_auc_matrix = np.zeros((len(strategies), permutations), dtype=np.float64)
    rows: list[dict] = []

    for s_idx, strategy in enumerate(strategies):
        ranks = _rank_matrix(strategy.scores.astype(np.float64))
        top = _top_mask(strategy.scores.astype(np.float64))

        pair_auc = ranks @ outcomes.T  # (n_steps, n_steps) todas las combinaciones
        diag = np.diagonal(pair_auc)
        auc = float(((diag - auc_offset) / auc_scale).mean())
        hits14 = float((top * outcomes).sum(axis=1).mean())

        idx = np.arange(n_steps)[None, :]
        null_means = pair_auc[idx, perms].mean(axis=1)
        null_aucs = (null_means - auc_offset) / auc_scale
        null_auc_matrix[s_idx] = null_aucs

        observed_aucs.append(auc)
        rows.append(
            {
                "strategy": strategy.name,
                "family": strategy.family,
                "auc": round(auc, 5),
                "auc_minus_half": round(auc - 0.5, 5),
                "top14_hits": round(hits14, 4),
                "top14_hits_vs_expected": round(hits14 - DRAW * BASE_RATE, 4),
                # La media del nulo suele estar sobre 0.5: refleja la ventaja
                # estática de los números más frecuentes del período, que no es
                # capacidad predictiva. Por eso el contraste correcto es contra
                # este nulo y no contra 0.5.
                "null_auc_mean": round(float(null_aucs.mean()), 5),
                "auc_minus_null_mean": round(auc - float(null_aucs.mean()), 5),
                "null_auc_sd": round(float(null_aucs.std()), 5),
                # Tolerancia: una estrategia constante produce un nulo degenerado
                # y sin holgura numérica saldría "significativa" por redondeo.
                "p_value_raw": float(
                    (np.sum(null_aucs >= auc - 1e-12) + 1) / (permutations + 1)
                ),
            }
        )

    # Westfall-Young: nulo del máximo sobre TODAS las estrategias a la vez.
    null_max = null_auc_matrix.max(axis=0)
    for row, auc in zip(rows, observed_aucs):
        row["p_value_family_wise"] = float(
            (np.sum(null_max >= auc - 1e-12) + 1) / (permutations + 1)
        )

    best_idx = int(np.argmax(observed_aucs))
    summary = {
        "n_steps": n_steps,
        "n_strategies": len(strategies),
        "permutations": permutations,
        "best_strategy": strategies[best_idx].name,
        "best_auc": round(observed_aucs[best_idx], 5),
        "null_max_auc_mean": round(float(null_max.mean()), 5),
        "null_max_auc_p95": round(float(np.percentile(null_max, 95)), 5),
        "best_p_value_family_wise": rows[best_idx]["p_value_family_wise"],
        "strategies_significant_fwer05": int(
            sum(1 for r in rows if r["p_value_family_wise"] < 0.05)
        ),
    }
    return rows, summary


# --------------------------------------------------------------------------- #
# Prueba 3: maldición del ganador (elegir en una mitad, medir en la otra)
# --------------------------------------------------------------------------- #


def test_selection_stability(
    strategies: list[Strategy], outcomes: np.ndarray, *, n_splits: int
) -> dict:
    n_steps = outcomes.shape[0]
    cut = n_steps // 2
    auc_offset = N_HITS * (N_HITS + 1) / 2.0
    auc_scale = float(N_HITS * N_MISSES)

    per_step = np.zeros((len(strategies), n_steps), dtype=np.float64)
    for s_idx, strategy in enumerate(strategies):
        ranks = _rank_matrix(strategy.scores.astype(np.float64))
        per_step[s_idx] = ((ranks * outcomes).sum(axis=1) - auc_offset) / auc_scale

    selection = per_step[:, :cut].mean(axis=1)
    holdout = per_step[:, cut:].mean(axis=1)
    winner = int(np.argmax(selection))

    correlation = float(np.corrcoef(selection, holdout)[0, 1])
    rank_correlation = float(
        np.corrcoef(
            np.argsort(np.argsort(-selection)), np.argsort(np.argsort(-holdout))
        )[0, 1]
    )

    # Repetición con particiones aleatorias para no depender de un solo corte.
    rng = np.random.default_rng(7)
    shrinkages: list[float] = []
    winner_beats_chance = 0
    winner_names: list[str] = []
    for _ in range(n_splits):
        mask = rng.permutation(n_steps)
        a, b = mask[:cut], mask[cut:]
        sel = per_step[:, a].mean(axis=1)
        hold = per_step[:, b].mean(axis=1)
        champion = int(np.argmax(sel))
        shrinkages.append(float(sel[champion] - hold[champion]))
        winner_beats_chance += int(hold[champion] > 0.5)
        winner_names.append(strategies[champion].name)

    unique_winners = sorted(set(winner_names))
    return {
        "selection_winner": strategies[winner].name,
        "winner_auc_in_selection": round(float(selection[winner]), 5),
        "winner_auc_in_holdout": round(float(holdout[winner]), 5),
        "winner_shrinkage": round(float(selection[winner] - holdout[winner]), 5),
        "selection_vs_holdout_correlation": round(correlation, 4),
        "selection_vs_holdout_rank_correlation": round(rank_correlation, 4),
        "mean_shrinkage_random_splits": round(float(np.mean(shrinkages)), 5),
        "random_splits": n_splits,
        "winner_beat_chance_in_holdout": winner_beats_chance,
        "winner_beat_chance_rate": round(winner_beats_chance / n_splits, 3),
        "distinct_winners_across_splits": len(unique_winners),
    }


# --------------------------------------------------------------------------- #
# Prueba 4: score compuesto vs variable única
# --------------------------------------------------------------------------- #


def test_statistical_resolution(data: Dataset) -> dict:
    """¿Qué tamaño de efecto podría siquiera detectarse con estos datos?

    Antes de preguntar "¿qué variable importa?" hay que saber si el histórico
    alcanza para responder. Una variable con 25 valores reparte 2.456 sorteos en
    celdas de ~98 casos: su resolución es pésima por construcción.
    """
    from scipy import stats

    p = BASE_RATE
    z_power = stats.norm.isf(0.20)  # 80% de potencia
    rows: list[dict] = []

    for b_idx, key in enumerate(data.bin_keys):
        column = data.bins[:, b_idx].astype(np.int64)
        cardinality = int(column.max()) + 1
        counts = np.bincount(column, minlength=cardinality)
        counts = counts[counts > 0]
        if counts.size < 2:
            continue
        median_n = float(np.median(counts))
        n_tests = counts.size * POOL
        z_alpha = stats.norm.isf(0.05 / (2 * n_tests))  # Bonferroni sobre las celdas
        mde = (z_alpha + z_power) * float(np.sqrt(p * (1 - p) / median_n))
        rows.append(
            {
                "variable": key,
                "distinct_values": int(counts.size),
                "median_draws_per_value": round(median_n, 1),
                "min_draws_per_value": int(counts.min()),
                "cells_tested": int(n_tests),
                "min_detectable_lift_pp": round(100 * mde, 2),
                "equivalent_forcing_strength": round(min(1.0, mde / (1 - p)), 3),
            }
        )

    rows.sort(key=lambda r: r["min_detectable_lift_pp"])
    return {
        "note": (
            "min_detectable_lift_pp = cuánto tendría que subir la tasa de aparición "
            "de un número (desde 56%) para detectarse con 80% de potencia tras "
            "corregir por el número de celdas probadas."
        ),
        "base_rate_pct": round(100 * p, 2),
        "variables": rows,
    }


def _zscore_rows(scores: np.ndarray) -> np.ndarray:
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std[std < 1e-12] = 1.0
    return (scores - mean) / std


def test_composite_vs_single(
    strategies: list[Strategy], outcomes: np.ndarray, *, repeats: int = 100
) -> dict:
    """Compara formas de evaluar, repitiendo la partición para no leer ruido.

    Una sola partición da un ganador distinto cada vez; lo que importa es la
    media y la dispersión entre reparticiones.
    """
    n_steps = outcomes.shape[0]
    cut = n_steps // 2
    auc_offset = N_HITS * (N_HITS + 1) / 2.0
    auc_scale = float(N_HITS * N_MISSES)

    usable = [s for s in strategies if s.family != "control"]
    stacked = np.stack([_zscore_rows(s.scores.astype(np.float64)) for s in usable])

    def per_step_auc(scores: np.ndarray) -> np.ndarray:
        ranks = _rank_matrix(scores)
        return ((ranks * outcomes).sum(axis=1) - auc_offset) / auc_scale

    component_auc = np.stack([per_step_auc(stacked[i]) for i in range(len(usable))])
    equal_weight_auc = per_step_auc(stacked.mean(axis=0))

    rng = np.random.default_rng(2024)
    methods: dict[str, list[float]] = {
        "composite_equal_weight": [],
        "composite_precision_weighted": [],
        "best_single_by_selection": [],
        "top5_blend_by_selection": [],
        "worst_single_by_selection": [],
    }
    champions: list[str] = []

    for _ in range(repeats):
        order = rng.permutation(n_steps)
        sel, hold = order[:cut], order[cut:]
        selection_auc = component_auc[:, sel].mean(axis=1)

        champion = int(np.argmax(selection_auc))
        loser = int(np.argmin(selection_auc))
        top5 = np.argsort(-selection_auc)[:5]
        champions.append(usable[champion].name)

        precision = np.clip(selection_auc - 0.5, 0.0, None)
        weights = (
            precision / precision.sum()
            if precision.sum() > 0
            else np.full(len(usable), 1.0 / len(usable))
        )
        weighted = per_step_auc(np.tensordot(weights, stacked, axes=(0, 0)))
        blended = per_step_auc(stacked[top5].mean(axis=0))

        methods["composite_equal_weight"].append(float(equal_weight_auc[hold].mean()))
        methods["composite_precision_weighted"].append(float(weighted[hold].mean()))
        methods["best_single_by_selection"].append(
            float(component_auc[champion, hold].mean())
        )
        methods["top5_blend_by_selection"].append(float(blended[hold].mean()))
        methods["worst_single_by_selection"].append(
            float(component_auc[loser, hold].mean())
        )

    summary = {
        name: {
            "holdout_auc_mean": round(float(np.mean(values)), 5),
            "holdout_auc_sd": round(float(np.std(values)), 5),
            "beats_chance_rate": round(float(np.mean(np.array(values) > 0.5)), 3),
        }
        for name, values in methods.items()
    }

    distinct = sorted(set(champions))
    best = max(summary.items(), key=lambda kv: kv[1]["holdout_auc_mean"])
    spread = max(v["holdout_auc_mean"] for v in summary.values()) - min(
        v["holdout_auc_mean"] for v in summary.values()
    )
    typical_sd = float(np.mean([v["holdout_auc_sd"] for v in summary.values()]))

    return {
        "repeats": repeats,
        "n_components": len(usable),
        "methods": summary,
        "best_method": best[0],
        "spread_between_methods": round(spread, 5),
        "typical_sd_within_method": round(typical_sd, 5),
        "difference_is_within_noise": bool(spread < typical_sd),
        "distinct_champions_across_repeats": len(distinct),
        "most_frequent_champion": max(set(champions), key=champions.count),
    }


# --------------------------------------------------------------------------- #
# Prueba 5: control de potencia con señal plantada
# --------------------------------------------------------------------------- #


def _cell_chi2(column: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """Chi-cuadrado 2x2 por celda (valor_del_bin, número).

    Es el estadístico que replica la búsqueda de patrones de la app: "cuando la
    variable vale v, ¿el número n aparece más de lo normal?".
    """
    n_rows = hits.shape[0]
    cardinality = int(column.max()) + 1
    one_hot = np.zeros((n_rows, cardinality), dtype=np.float64)
    one_hot[np.arange(n_rows), column] = 1.0
    totals = one_hot.sum(axis=0)  # (V,)
    observed = one_hot.T @ hits  # (V, 25)
    global_rate = hits.mean(axis=0)  # (25,)
    expected_hit = np.outer(totals, global_rate)
    expected_miss = np.outer(totals, 1.0 - global_rate)
    observed_miss = totals[:, None] - observed
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected_hit > 5, (observed - expected_hit) ** 2 / expected_hit, 0.0)
        chi2 += np.where(
            expected_miss > 5, (observed_miss - expected_miss) ** 2 / expected_miss, 0.0
        )
    return chi2


def _pattern_search_p_value(
    column: np.ndarray,
    hits: np.ndarray,
    *,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[int, int], float]:
    """Busca la celda más fuerte y la contrasta con un nulo por permutación."""
    chi2 = _cell_chi2(column, hits)
    observed = float(chi2.max())
    flat = int(np.argmax(chi2))
    cell = (flat // chi2.shape[1], flat % chi2.shape[1])
    null = np.empty(permutations)
    for i in range(permutations):
        null[i] = _cell_chi2(column, hits[rng.permutation(hits.shape[0])]).max()
    p_value = float((np.sum(null >= observed) + 1) / (permutations + 1))
    return observed, cell, p_value


def test_power_with_planted_signal(
    data: Dataset, start: int, *, strengths: tuple[float, ...], permutations: int
) -> dict:
    """Inyecta una dependencia real luna→número y mide si el arnés la detecta.

    Sin este control, un resultado nulo es indistinguible de un arnés roto.
    Incluye un detector 'oráculo' que conoce la regla verdadera: marca el techo
    de AUC alcanzable, que para un efecto sobre un solo número es muy bajo.
    """
    rng = np.random.default_rng(31337)
    sector = (data.moon_longitude % 360.0 / 360.0 * POOL).astype(np.int64)
    results: list[dict] = []

    for strength in strengths:
        synthetic = np.zeros_like(data.hits)
        for t in range(data.n_draws):
            pool = rng.permutation(POOL)
            favoured = int(sector[t])
            if rng.random() < strength:
                chosen = np.array(
                    [favoured] + [int(n) for n in pool if n != favoured][: DRAW - 1]
                )
            else:
                chosen = pool[:DRAW]
            synthetic[t, chosen] = 1.0

        oracle = np.zeros((data.n_draws - start, POOL), dtype=np.float32)
        oracle[np.arange(data.n_draws - start), sector[start:]] = 1.0

        probes = [
            Strategy("oracle_knows_the_rule", "oracle", oracle),
            Strategy(
                "learned_lunar_sector25",
                "lunar_position",
                _bin_association_strategy(sector, synthetic, start),
            ),
            Strategy(
                "learned_lunar_kernel_bw12",
                "lunar_position",
                _moon_kernel_strategy(data.moon_longitude, synthetic, start, 12.0),
            ),
            Strategy(
                "astro_similarity_all",
                "astro_similarity",
                _similarity_strategy(data.numeric, synthetic, start),
            ),
        ]
        rows, _ = evaluate_strategies(
            probes,
            synthetic[start:].astype(np.float64),
            permutations=permutations,
            rng=np.random.default_rng(99),
        )

        chi2, cell, cell_p = _pattern_search_p_value(
            sector,
            synthetic,
            permutations=max(50, permutations // 4),
            rng=np.random.default_rng(4242),
        )
        recovered = bool(cell[0] == cell[1])

        results.append(
            {
                "planted_strength": strength,
                "extra_appearance_probability": round(strength * (1 - BASE_RATE), 4),
                "detectors": {
                    r["strategy"]: {
                        "auc": r["auc"],
                        "p_value": r["p_value_raw"],
                        "significant": r["p_value_raw"] < 0.05,
                    }
                    for r in rows
                },
                "pattern_search": {
                    "max_chi2": round(chi2, 2),
                    "top_cell_bin_value": int(cell[0]),
                    "top_cell_number": int(cell[1]) + 1,
                    "recovered_planted_rule": recovered,
                    "permutation_p_value": cell_p,
                    "detected": bool(cell_p < 0.05),
                },
                "auc_detected_by_learned_sector": bool(
                    rows[1]["p_value_raw"] < 0.05
                ),
            }
        )

    detectable = [
        r["planted_strength"]
        for r in results
        if r["pattern_search"]["detected"] and r["pattern_search"]["recovered_planted_rule"]
    ]
    return {
        "runs": results,
        "minimum_detectable_strength": min(detectable) if detectable else None,
        "note": (
            "El AUC tiene techo bajo para efectos concentrados en un número: "
            "mira el oráculo para ver el máximo alcanzable a cada intensidad."
        ),
    }


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def main() -> None:
    # La consola de Windows usa cp1252 y rompe con acentos/flechas.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(BASE_DIR / "data" / "kino_draws.csv"))
    parser.add_argument("--min-history", type=int, default=400)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--marginal-permutations", type=int, default=200)
    parser.add_argument("--chi2-permutations", type=int, default=100)
    parser.add_argument("--power-permutations", type=int, default=300)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    print("[1/6] Cargando sorteos y calculando efemérides...", flush=True)
    data = build_dataset(Path(args.csv), use_cache=not args.no_cache)
    print(
        f"      {data.n_draws} sorteos | {len(data.numeric_keys)} variables numéricas"
        f" | {len(data.bin_keys)} variables categóricas"
        f" | {data.dates[0]} → {data.dates[-1]}",
        flush=True,
    )

    rng = np.random.default_rng(2026)

    print("[2/6] Asociaciones marginales variable↔número (con nulo permutado)...", flush=True)
    marginal = test_marginal_associations(
        data, permutations=args.marginal_permutations, rng=rng
    )
    print(
        f"      {marginal['discoveries_fdr05']} descubrimientos FDR5% de"
        f" {marginal['tests_run']} pruebas"
        f" (nulo esperaba {marginal['null_discoveries_mean']:.1f})",
        flush=True,
    )

    print("[3/6] Asociaciones categóricas (chi-cuadrado máximo permutado)...", flush=True)
    categorical = test_categorical_associations(
        data, permutations=args.chi2_permutations, rng=rng
    )
    print(
        f"      p permutado del bin más fuerte = {categorical['permutation_p_value']:.4f}",
        flush=True,
    )

    start = args.min_history
    print(f"[4/6] Construyendo estrategias walk-forward (historia mínima {start})...", flush=True)
    strategies = build_strategies(data, start)
    outcomes = data.hits[start:].astype(np.float64)
    print(
        f"      {len(strategies)} estrategias × {outcomes.shape[0]} pasos"
        f" = {len(strategies) * outcomes.shape[0]:,} predicciones fuera de muestra",
        flush=True,
    )

    print("[5/6] Evaluando con nulo por permutación (Westfall-Young)...", flush=True)
    rows, summary = evaluate_strategies(
        strategies, outcomes, permutations=args.permutations, rng=rng
    )
    stability = test_selection_stability(strategies, outcomes, n_splits=200)
    composite = test_composite_vs_single(strategies, outcomes, repeats=100)
    resolution = test_statistical_resolution(data)

    print("[6/6] Control de potencia con señal plantada...", flush=True)
    power = test_power_with_planted_signal(
        data,
        start,
        strengths=(0.01, 0.02, 0.05, 0.10, 0.25),
        permutations=args.power_permutations,
    )

    rows_sorted = sorted(rows, key=lambda r: -r["auc"])
    report = {
        "dataset": {
            "draws": data.n_draws,
            "first_draw": data.dates[0].isoformat(),
            "last_draw": data.dates[-1].isoformat(),
            "numeric_features": len(data.numeric_keys),
            "categorical_features": len(data.bin_keys),
            "pool_size": POOL,
            "draw_size": DRAW,
            "base_rate": BASE_RATE,
        },
        "statistical_resolution": resolution,
        "marginal_associations": marginal,
        "categorical_associations": categorical,
        "walk_forward_summary": summary,
        "walk_forward_strategies": rows_sorted,
        "selection_stability": stability,
        "composite_vs_single": composite,
        "power_check": power,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print("RANKING FUERA DE MUESTRA (AUC 0.5 = azar)")
    print("=" * 78)
    print(f"{'estrategia':<42}{'familia':<20}{'AUC':>8}{'p FWER':>8}")
    for row in rows_sorted[:15]:
        print(
            f"{row['strategy']:<42}{row['family']:<20}"
            f"{row['auc']:>8.4f}{row['p_value_family_wise']:>8.3f}"
        )
    print("...")
    for row in rows_sorted[-5:]:
        print(
            f"{row['strategy']:<42}{row['family']:<20}"
            f"{row['auc']:>8.4f}{row['p_value_family_wise']:>8.3f}"
        )

    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    print(f"Mejor AUC observado          : {summary['best_auc']:.4f} ({summary['best_strategy']})")
    print(f"Máximo esperado por azar (p95): {summary['null_max_auc_p95']:.4f}")
    print(f"p familiar del mejor          : {summary['best_p_value_family_wise']:.4f}")
    print(f"Estrategias significativas    : {summary['strategies_significant_fwer05']}")
    print(f"Correlación selección↔holdout : {stability['selection_vs_holdout_correlation']:.3f}")
    print(f"Encogimiento del ganador      : {stability['winner_shrinkage']:.4f}")
    print(f"Ganadores distintos por corte : {stability['distinct_winners_across_splits']}")

    print("\n" + "=" * 78)
    print("FORMA DE EVALUAR (AUC medio en holdout, 100 reparticiones)")
    print("=" * 78)
    for name, stats_row in sorted(
        composite["methods"].items(), key=lambda kv: -kv[1]["holdout_auc_mean"]
    ):
        print(
            f"  {name:<34}{stats_row['holdout_auc_mean']:.4f}"
            f"  ± {stats_row['holdout_auc_sd']:.4f}"
        )
    print(
        f"  Diferencia entre métodos {composite['spread_between_methods']:.4f}"
        f" vs ruido típico {composite['typical_sd_within_method']:.4f}"
        f" → {'indistinguible' if composite['difference_is_within_noise'] else 'distinguible'}"
    )

    print("\n" + "=" * 78)
    print("RESOLUCIÓN: efecto mínimo detectable por variable (tasa base 56%)")
    print("=" * 78)
    print(f"  {'variable':<32}{'valores':>8}{'n/celda':>10}{'lift mín.':>11}")
    for row in resolution["variables"][:6]:
        print(
            f"  {row['variable']:<32}{row['distinct_values']:>8}"
            f"{row['median_draws_per_value']:>10.0f}{row['min_detectable_lift_pp']:>10.1f}pp"
        )
    print("  ...")
    for row in resolution["variables"][-4:]:
        print(
            f"  {row['variable']:<32}{row['distinct_values']:>8}"
            f"{row['median_draws_per_value']:>10.0f}{row['min_detectable_lift_pp']:>10.1f}pp"
        )
    print("\nControl de potencia (señal luna→número inyectada a propósito):")
    print(f"  {'señal':>7}{'AUC oráculo':>13}{'AUC aprendido':>15}{'patrón':>10}{'regla ok':>10}")
    for entry in power["runs"]:
        det = entry["detectors"]
        found = "sí" if entry["pattern_search"]["detected"] else "no"
        ok = "sí" if entry["pattern_search"]["recovered_planted_rule"] else "no"
        print(
            f"  {entry['planted_strength']:>6.0%}"
            f"{det['oracle_knows_the_rule']['auc']:>13.4f}"
            f"{det['learned_lunar_sector25']['auc']:>15.4f}"
            f"{found:>10}{ok:>10}"
        )
    print(f"  Señal mínima detectable: {power['minimum_detectable_strength']}")
    print(f"\nReporte completo: {REPORT_PATH}")
    print(f"Tiempo total: {report['runtime_seconds']}s")


if __name__ == "__main__":
    main()
