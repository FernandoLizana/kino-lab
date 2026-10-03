"""Tarot anclado al cielo del sorteo.

La astrología da el clima energético del día. El tarot traduce ese cielo a un
patrón simbólico discreto (cartas). Esa traducción es el ancla: dos sorteos con
la misma carta-patrón se consideran "el mismo día arquetípico".

Correspondencias: Golden Dawn / Book T (decano → menor 2–10; signo → arcano
mayor; planeta → arcano mayor). No hay baraja aleatoria: la tirada de un
instante es una función del cielo, reproducible.

Esto NO afirma que las cartas causen el Kino. El módulo existe para medir si
hay sincronicidad estadística (asociación carta↔números, o semejanza entre
sorteos que compartieron la misma ancla) más allá del nulo por permutación.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np

from config import Config
from services.astrology_service import (
    AUDIT_SUMMARY,
    MIN_DRAWS_PER_BIN_VALUE,
    SIGN_NAMES,
    TZ_CHILE,
    _empty_assoc,
    _score_from_assoc,
    _update_assoc,
    compute_astro_features,
    discover_patterns_walk_forward,
    estimate_pattern_null,
    format_draw_label,
    next_draw_datetime,
    prior_draws_before_target,
)
from services.lunar_service import GEOMETRIC_TIEBREAK_WEIGHT, _geometric_tiebreak
from services.combination_service import features_for_combo, sample_combinations

# --------------------------------------------------------------------------- #
# Correspondencias Golden Dawn
# --------------------------------------------------------------------------- #

PLANET_MAJORS: dict[str, tuple[int, str]] = {
    "mercury": (1, "El Mago"),
    "moon": (2, "La Sacerdotisa"),
    "venus": (3, "La Emperatriz"),
    "jupiter": (10, "La Rueda de la Fortuna"),
    "neptune": (12, "El Colgado"),
    "mars": (16, "La Torre"),
    "sun": (19, "El Sol"),
    "pluto": (20, "El Juicio"),
    "saturn": (21, "El Mundo"),
    "uranus": (0, "El Loco"),
}

# Índice 0=Aries … 11=Piscis.
SIGN_MAJORS: tuple[tuple[int, str], ...] = (
    (4, "El Emperador"),
    (5, "El Hierofante"),
    (6, "Los Enamorados"),
    (7, "El Carro"),
    (8, "La Fuerza"),
    (9, "El Ermitaño"),
    (11, "La Justicia"),
    (13, "La Muerte"),
    (14, "La Templanza"),
    (15, "El Diablo"),
    (17, "La Estrella"),
    (18, "La Luna"),
)

# Elemento del signo → palo. Aries/Leo/Sagitario = Bastos, etc.
SUIT_BY_ELEMENT = ("bastos", "oros", "espadas", "copas")
SUIT_LABEL = {
    "bastos": "Bastos",
    "oros": "Oros",
    "espadas": "Espadas",
    "copas": "Copas",
}

# Orden caldeo de los decanos, empezando por Marte en el I de Aries.
CHALDEAN_RULERS = ("mars", "sun", "venus", "mercury", "moon", "saturn", "jupiter")

# Fase lunar (8 octiles, misma partición que astrology_service) → arcano.
PHASE_MAJORS: tuple[tuple[int, str], ...] = (
    (0, "El Loco"),
    (2, "La Sacerdotisa"),
    (7, "El Carro"),
    (10, "La Rueda de la Fortuna"),
    (19, "El Sol"),
    (11, "La Justicia"),
    (9, "El Ermitaño"),
    (13, "La Muerte"),
)

# Capa rápida: cambia en días o semanas. Es el ancla cotidiana.
TAROT_BIN_KEYS = (
    "energy_suit",
    "pattern_suit",
    "energy_major",
    "phase_major",
    "anchor_pattern",
    "anchor_pip",
    "suit_pair",
)

# Capa lenta / "invisible": planetas que tardan años en cambiar de signo.
# Solo se puede mirar con el histórico largo (1990→hoy).
INVISIBLE_BIN_KEYS = (
    "jupiter_major",
    "jupiter_suit",
    "saturn_major",
    "saturn_suit",
    "uranus_major",
    "uranus_suit",
    "neptune_major",
    "neptune_suit",
    "pluto_major",
    "pluto_suit",
    "node_major",
    "node_suit",
    "decan_ruler",
    "slow_climate",
)

ALL_TAROT_BIN_KEYS = TAROT_BIN_KEYS + INVISIBLE_BIN_KEYS


def _major(number: int, name: str, *, tradition: str) -> dict:
    return {
        "id": f"major-{number}",
        "name": name,
        "kind": "major",
        "number": number,
        "suit": None,
        "tradition": tradition,
    }


def _minor(pip: int, suit: str, *, tradition: str) -> dict:
    return {
        "id": f"{suit}-{pip}",
        "name": f"{pip} de {SUIT_LABEL[suit]}",
        "kind": "minor",
        "number": pip,
        "suit": suit,
        "tradition": tradition,
    }


def sign_major(sign_index: int) -> dict:
    number, name = SIGN_MAJORS[int(sign_index) % 12]
    sign = SIGN_NAMES[int(sign_index) % 12]
    return _major(number, name, tradition=f"Golden Dawn · signo {sign}")


def planet_major(planet: str) -> dict:
    number, name = PLANET_MAJORS[planet]
    return _major(number, name, tradition=f"Golden Dawn · planeta {planet}")


def phase_major(phase_bin: int) -> dict:
    number, name = PHASE_MAJORS[int(phase_bin) % 8]
    return _major(number, name, tradition=f"ciclo lunar · octil {int(phase_bin) % 8}")


def decan_from_longitude(longitude_deg: float) -> dict:
    """Menor 2–10 del decano Golden Dawn (Book T) para una longitud eclíptica."""
    lon = float(longitude_deg) % 360.0
    sign_index = int(lon // 30.0) % 12
    degree_in_sign = lon % 30.0
    decan = min(int(degree_in_sign // 10.0), 2)
    pip = 2 + (sign_index % 3) * 3 + decan
    suit = SUIT_BY_ELEMENT[sign_index % 4]
    absolute = sign_index * 3 + decan
    ruler = CHALDEAN_RULERS[absolute % 7]
    sign = SIGN_NAMES[sign_index]
    ordinal = ("I", "II", "III")[decan]
    card = _minor(
        pip,
        suit,
        tradition=f"Book T · {ordinal} decanato de {sign} (regente {ruler})",
    )
    card["decan"] = decan
    card["sign_index"] = sign_index
    card["sign_name"] = sign
    card["ruler"] = ruler
    card["absolute_decan"] = absolute
    return card


def build_day_spread(astro: dict) -> dict:
    """Tirada de 3 cartas determinada por el cielo del instante.

    - Energía: arcano del signo solar (clima astrológico del día).
    - Patrón / ancla: menor del decano lunar (cambia ~cada 2.5 días).
    - Clima: arcano de la fase lunar (ciclo emocional del cielo).
    """
    bodies = astro["bodies"]
    sun = bodies["sun"]
    moon = bodies["moon"]
    phase_bin = int(astro.get("moon_phase_bin", 0))

    energy = sign_major(sun["sign_index"])
    energy["role"] = "energía"
    energy["source"] = f"Sol en {sun['sign_name']} {sun['degree_in_sign']:.2f}°"

    pattern = decan_from_longitude(moon["longitude_deg"])
    pattern["role"] = "patrón"
    pattern["source"] = (
        f"Luna en {moon['sign_name']} {moon['degree_in_sign']:.2f}° "
        f"(decano {pattern['decan'] + 1})"
    )

    climate = phase_major(phase_bin)
    climate["role"] = "clima"
    climate["source"] = f"fase {astro.get('moon_phase_name', '—')}"

    sun_decan = decan_from_longitude(sun["longitude_deg"])
    moon_sign = sign_major(moon["sign_index"])

    return {
        "energy": energy,
        "pattern": pattern,
        "climate": climate,
        "sun_decan": sun_decan,
        "moon_sign_major": moon_sign,
        "anchor_id": pattern["id"],
        "anchor_name": pattern["name"],
        "signature": f"{energy['id']}|{pattern['id']}|{climate['id']}",
    }


def extract_tarot_bins(astro: dict, spread: dict | None = None) -> dict[str, str]:
    """Bins de la tirada del día + capa lenta (planetas exteriores y nodo)."""
    spread = spread or build_day_spread(astro)
    energy = spread["energy"]
    pattern = spread["pattern"]
    climate = spread["climate"]
    bodies = astro["bodies"]

    def _body_major_suit(name: str) -> tuple[str, str]:
        sign_index = int(bodies[name]["sign_index"])
        return sign_major(sign_index)["name"], SUIT_BY_ELEMENT[sign_index % 4]

    jup_m, jup_s = _body_major_suit("jupiter")
    sat_m, sat_s = _body_major_suit("saturn")
    ura_m, ura_s = _body_major_suit("uranus")
    nep_m, nep_s = _body_major_suit("neptune")
    plu_m, plu_s = _body_major_suit("pluto")

    node = (astro.get("lunar") or {}).get("true_node") or {}
    node_sign = int(node.get("sign_index", 0))
    node_major_name = sign_major(node_sign)["name"]
    node_suit = SUIT_BY_ELEMENT[node_sign % 4]

    return {
        "energy_suit": SUIT_BY_ELEMENT[int(bodies["sun"]["sign_index"]) % 4],
        "pattern_suit": str(pattern["suit"]),
        "energy_major": energy["name"],
        "phase_major": climate["name"],
        "anchor_pattern": pattern["name"],
        "anchor_pip": str(pattern["number"]),
        "suit_pair": (
            f"{SUIT_BY_ELEMENT[int(bodies['sun']['sign_index']) % 4]}"
            f"+{pattern['suit']}"
        ),
        "jupiter_major": jup_m,
        "jupiter_suit": jup_s,
        "saturn_major": sat_m,
        "saturn_suit": sat_s,
        "uranus_major": ura_m,
        "uranus_suit": ura_s,
        "neptune_major": nep_m,
        "neptune_suit": nep_s,
        "pluto_major": plu_m,
        "pluto_suit": plu_s,
        "node_major": node_major_name,
        "node_suit": node_suit,
        "decan_ruler": str(pattern.get("ruler") or "unknown"),
        "slow_climate": f"{sat_s}+{jup_s}+{node_suit}",
    }


def extract_tarot_model_features(astro: dict, spread: dict | None = None) -> dict[str, float]:
    """Vector numérico pequeño: solo lo que la tirada usa (Sol, Luna, fase)."""
    spread = spread or build_day_spread(astro)
    mf = astro.get("model_features") or {}
    pattern = spread["pattern"]
    return {
        "sun.longitude_sin": float(mf.get("sun.longitude_sin", 0.0)),
        "sun.longitude_cos": float(mf.get("sun.longitude_cos", 0.0)),
        "moon.longitude_sin": float(mf.get("moon.longitude_sin", 0.0)),
        "moon.longitude_cos": float(mf.get("moon.longitude_cos", 0.0)),
        "lunar.phase_sin": float(mf.get("lunar.phase_sin", 0.0)),
        "lunar.phase_cos": float(mf.get("lunar.phase_cos", 0.0)),
        "anchor.decan_norm": float(pattern.get("absolute_decan", 0)) / 35.0,
        "anchor.pip_norm": (float(pattern["number"]) - 2.0) / 8.0,
    }


def as_tarot_view(astro: dict) -> dict:
    spread = build_day_spread(astro)
    bins = extract_tarot_bins(astro, spread)
    return {
        **astro,
        "tarot": spread,
        "model_features": extract_tarot_model_features(astro, spread),
        "pattern_bins": bins,
        "categorical": bins,
        "mode": "tarot",
    }


def resolution_table(bin_columns: dict[str, list[str]], n_draws: int) -> list[dict]:
    """Cuántos sorteos caen en cada valor, y si alcanza para afirmar algo."""
    rows: list[dict] = []
    for key, values in bin_columns.items():
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        sizes = list(counts.values()) or [0]
        median = float(np.median(sizes))
        rows.append(
            {
                "variable": key,
                "distinct_values": len(counts),
                "median_draws_per_value": round(median, 1),
                "min_draws_per_value": int(min(sizes)),
                "max_draws_per_value": int(max(sizes)),
                "usable": median >= MIN_DRAWS_PER_BIN_VALUE,
                "min_detectable_rate_jump_pp": (
                    round(196.0 / max(median, 1.0) ** 0.5, 1)
                    if median > 0
                    else None
                ),
            }
        )
    rows.sort(key=lambda r: r["median_draws_per_value"], reverse=True)
    return rows


def measure_anchor_synchronicity(
    hits: np.ndarray,
    labels: list[str],
    *,
    permutations: int = 200,
    seed: int = 2026,
) -> dict:
    """¿Los sorteos con la misma carta-ancla se parecen más entre sí?

    Mide el Jaccard medio de los conjuntos de 14 números dentro de cada carta,
    y lo compara con barajar las cartas entre fechas. Si no hay sincronicidad,
    el Jaccard intra-carta es el de dos sorteos al azar (~0.39).
    """
    labels_arr = np.asarray(labels)
    uniq, inverse = np.unique(labels_arr, return_inverse=True)

    def _mean_jaccard(group_ids: np.ndarray, rng_pairs: np.random.Generator) -> float:
        # Pares aleatorios por carta, ponderados por tamaño. Un clique de 13
        # sorteos fijos (versión anterior) inventaba "señales" en planetas lentos.
        weights = []
        scores = []
        for idx in range(int(group_ids.max()) + 1):
            members = np.flatnonzero(group_ids == idx)
            if members.size < 2:
                continue
            n_pairs = min(1500, members.size * (members.size - 1) // 2)
            a = rng_pairs.integers(0, members.size, size=n_pairs)
            b = rng_pairs.integers(0, members.size, size=n_pairs)
            keep = a != b
            if not keep.any():
                continue
            left = hits[members[a[keep]]]
            right = hits[members[b[keep]]]
            inter = (left * right).sum(axis=1)
            union = left.sum(axis=1) + right.sum(axis=1) - inter
            valid = union > 0
            if not valid.any():
                continue
            scores.append(float((inter[valid] / union[valid]).mean()))
            weights.append(float(members.size))
        if not scores:
            return 0.0
        w = np.asarray(weights)
        return float(np.dot(w, np.asarray(scores)) / w.sum())

    rng = np.random.default_rng(seed)
    observed = _mean_jaccard(inverse, rng)
    null = np.empty(permutations, dtype=np.float64)
    for i in range(permutations):
        shuffled = rng.permutation(inverse)
        null[i] = _mean_jaccard(shuffled, rng)

    p_value = float((np.sum(null >= observed) + 1) / (permutations + 1))
    return {
        "anchor_values": int(len(uniq)),
        "observed_mean_jaccard": round(observed, 4),
        "null_mean_jaccard": round(float(null.mean()), 4),
        "null_p95_jaccard": round(float(np.quantile(null, 0.95)), 4),
        "p_value": round(p_value, 4),
        "chance_jaccard_two_random_draws": round(
            (Config.DRAW_SIZE * Config.DRAW_SIZE / Config.POOL_SIZE)
            / (
                2 * Config.DRAW_SIZE
                - Config.DRAW_SIZE * Config.DRAW_SIZE / Config.POOL_SIZE
            ),
            4,
        ),
        "permutations": permutations,
        "significant_at_05": p_value <= 0.05,
    }


def build_tarot_number_scores(
    draws,
    features_list: list[dict],
    target_features: dict,
    *,
    return_meta: bool = False,
):
    """Score solo por asociación de las cartas (bins) con soporte suficiente."""
    if not draws:
        scores = [0.5] * Config.POOL_SIZE
        meta = {"mode": "empty"}
        return (scores, meta) if return_meta else scores

    assoc = _empty_assoc()
    for draw, feats in zip(draws, features_list):
        _update_assoc(assoc, feats, [int(x) for x in draw.numbers()])
    scores = _score_from_assoc(assoc, target_features)

    effective_support = min(MIN_DRAWS_PER_BIN_VALUE, max(3, len(draws) // 4))
    usable = sum(
        1
        for key in assoc["feature_draw_counts"]
        for count in assoc["feature_draw_counts"][key].values()
        if count >= effective_support
    )
    total = sum(len(values) for values in assoc["feature_draw_counts"].values())
    spread = target_features.get("tarot") or {}
    meta = {
        "mode": "tarot_anchor_bins",
        "channel_weights": {"tarot_bins": 1.0},
        "bin_values_usable": usable,
        "bin_values_total": total,
        "min_draws_per_bin_value": MIN_DRAWS_PER_BIN_VALUE,
        "anchor_card": spread.get("anchor_name"),
        "spread_signature": spread.get("signature"),
        "score_formula": (
            "tarot_score[n] = asociación histórica de las cartas del día "
            "(solo valores con soporte suficiente)"
        ),
    }
    return (scores, meta) if return_meta else scores


def score_sample_tarot(
    number_scores: list[float],
    *,
    sample_size: int,
    seed: int | None,
    top_k: int = 20,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict]:
    by_number = {i + 1: float(number_scores[i]) for i in range(Config.POOL_SIZE)}
    best: list[dict] = []
    processed = 0
    for batch in sample_combinations(sample_size, seed=seed):
        if should_cancel and should_cancel():
            break
        for row in batch:
            feats = features_for_combo(row)
            nums = feats["numbers"]
            tarot_score = sum(by_number[n] for n in nums) / Config.DRAW_SIZE
            tie = _geometric_tiebreak(feats)
            total = float(tarot_score + GEOMETRIC_TIEBREAK_WEIGHT * tie)
            best.append(
                {
                    **feats,
                    "tarot_score": float(tarot_score),
                    "score": total,
                    "components": {
                        "tarot": float(tarot_score),
                        "geometric_tiebreak": float(tie),
                    },
                    "numbers_csv": "-".join(f"{n:02d}" for n in nums),
                }
            )
            processed += 1
        best.sort(key=lambda x: x["score"], reverse=True)
        best = best[: max(top_k * 3, top_k)]
        if progress_cb:
            progress_cb(
                min(99.0, 100.0 * processed / max(sample_size, 1)),
                f"{processed:,} combinaciones tarot",
            )
    best = sorted(best, key=lambda x: x["score"], reverse=True)[:top_k]
    for i, item in enumerate(best, start=1):
        item["rank"] = i
    return best


def run_tarot_experiment(
    draws,
    *,
    now: datetime | None = None,
    target_dt: datetime | None = None,
    history_sync: dict | None = None,
    sample_size: int = 20_000,
    seed: int = 42,
    top_k: int = 20,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[dict, list[dict]]:
    target = target_dt or next_draw_datetime(now)
    if target.tzinfo is None:
        target = target.replace(tzinfo=TZ_CHILE)
    else:
        target = target.astimezone(TZ_CHILE)

    ordered = prior_draws_before_target(draws, target)
    if len(ordered) < 20:
        raise ValueError(
            "Se necesitan al menos 20 sorteos históricos anteriores al próximo sorteo"
        )

    if progress_cb:
        progress_cb(2.0, "Traduciendo el cielo a tarot")

    features_list: list[dict] = []
    total = len(ordered)
    for i, draw in enumerate(ordered):
        if should_cancel and should_cancel():
            break
        features_list.append(as_tarot_view(compute_astro_features(draw.draw_date)))
        if progress_cb and i % 50 == 0:
            progress_cb(2.0 + 38.0 * i / max(total, 1), f"Tarot {i + 1}/{total}")

    if should_cancel and should_cancel():
        return {"cancelled": True}, []

    if progress_cb:
        progress_cb(42.0, "Nulo tarot por permutación")
    pattern_null = estimate_pattern_null(ordered, features_list)
    if progress_cb:
        progress_cb(50.0, "Patrones tarot walk-forward")
    patterns = discover_patterns_walk_forward(
        ordered, features_list, permutation_null=pattern_null
    )
    pattern_null_public = {k: v for k, v in pattern_null.items() if k != "null_samples"}

    target_astro = compute_astro_features(target)
    target_features = as_tarot_view(target_astro)
    number_scores, score_meta = build_tarot_number_scores(
        ordered, features_list, target_features, return_meta=True
    )

    hits = np.zeros((len(ordered), Config.POOL_SIZE), dtype=np.float64)
    for i, draw in enumerate(ordered):
        for number in draw.numbers():
            hits[i, int(number) - 1] = 1.0
    anchors = [str(f["pattern_bins"]["anchor_pattern"]) for f in features_list]
    if progress_cb:
        progress_cb(54.0, "Midiendo sincronicidad de la ancla")
    synchronicity = measure_anchor_synchronicity(hits, anchors, permutations=200)

    bin_columns = {
        key: [str(f["pattern_bins"][key]) for f in features_list] for key in TAROT_BIN_KEYS
    }
    resolution = resolution_table(bin_columns, len(ordered))

    if progress_cb:
        progress_cb(58.0, "Muestreando combinaciones")

    def combo_progress(pct: float, message: str = "") -> None:
        if progress_cb:
            progress_cb(58.0 + 0.42 * pct, message)

    results = score_sample_tarot(
        number_scores,
        sample_size=sample_size,
        seed=seed,
        top_k=top_k,
        progress_cb=combo_progress,
        should_cancel=should_cancel,
    )

    ranked_numbers = sorted(
        (
            {"number": n, "tarot_score": round(number_scores[n - 1], 4)}
            for n in range(1, Config.POOL_SIZE + 1)
        ),
        key=lambda x: x["tarot_score"],
        reverse=True,
    )

    spread = target_features["tarot"]
    sync = history_sync or {}
    metrics = {
        "mode": "tarot",
        "disclaimer": (
            "Ejercicio de sincronicidad tarot–astrología. La tirada no se baraja: "
            "es una traducción Golden Dawn del cielo del sorteo. El ancla es el "
            "decano lunar. Un mayor tarot_score no implica mayor probabilidad real; "
            "si el sorteo es uniforme, las 4.457.400 combinaciones son equiprobables. "
            "Encontrar una carta 'del día' no demuestra que el tarot influya en el Kino."
        ),
        "audit_note": AUDIT_SUMMARY,
        "pattern_method": (
            "ancla = menor del decano lunar; energía = arcano del signo solar; "
            "clima = arcano de la fase; walk-forward + nulo familiar por permutación"
        ),
        "pattern_null": pattern_null_public,
        "synchronicity": synchronicity,
        "resolution": resolution,
        "score_formula": score_meta.get("score_formula"),
        "adaptive_weights": score_meta,
        "target_spread": {
            "energy": spread["energy"],
            "pattern": spread["pattern"],
            "climate": spread["climate"],
            "signature": spread["signature"],
        },
        "correspondence": "Golden Dawn / Book T",
        "assumptions": {
            "draw_time_local": "22:30",
            "timezone": "America/Santiago",
            "location": target_astro["location"]["name"],
            "lat": target_astro["location"]["lat"],
            "lon": target_astro["location"]["lon"],
            "ephemeris": "Swiss Ephemeris Moshier",
            "anchor": "decano lunar → menor 2–10",
            "energy": "signo solar → arcano mayor",
            "climate": "fase lunar (8 octiles) → arcano mayor",
            "min_draws_per_bin_value": MIN_DRAWS_PER_BIN_VALUE,
            "random_shuffle": False,
        },
        "target_draw_datetime": target.isoformat(),
        "target_draw_date": target.date().isoformat(),
        "target_draw_time": "22:30",
        "timezone": "America/Santiago",
        "target_draw_label": format_draw_label(target),
        "history_as_of": ordered[-1].draw_date.isoformat(),
        "remote_latest_date": sync.get("remote_latest_date"),
        "remote_latest_draw_number": sync.get("remote_latest_draw_number"),
        "history_freshness": sync.get("history_freshness", "already_current"),
        "history_sync_message": sync.get("message"),
        "draw_count": len(ordered),
        "sample_size": sample_size,
        "seed": seed,
        "number_scores": ranked_numbers,
        "patterns": patterns,
        "top_k": len(results),
        "candidate_count": sample_size,
        "target_sky": {
            "local_datetime": target_astro["local_datetime"],
            "moon_phase": target_astro["moon_phase_name"],
            "moon_sign": target_astro["bodies"]["moon"]["sign_name"],
            "sun_sign": target_astro["bodies"]["sun"]["sign_name"],
        },
    }
    if progress_cb:
        progress_cb(100.0, "Ejercicio tarot–astro listo")
    return metrics, results
