from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Callable

import numpy as np

from config import Config
from services.astrology_service import (
    AUDIT_SUMMARY,
    MIN_DRAWS_PER_BIN_VALUE,
    SYNODIC_MONTH_DAYS,
    TZ_CHILE,
    circular_feature_distance,
    compute_astro_features,
    discover_patterns_walk_forward,
    estimate_pattern_null,
    format_draw_label,
    next_draw_datetime,
    prior_draws_before_target,
)
from services.combination_service import features_for_combo, sample_combinations

# Features lunares permitidas (subset explícito; sin planetas ni separaciones).
LUNAR_MODEL_KEYS = (
    "moon.longitude_sin",
    "moon.longitude_cos",
    "moon.latitude_norm",
    "moon.declination_norm",
    "moon.speed_lon_norm",
    "moon.distance_norm",
    "lunar.phase_sin",
    "lunar.phase_cos",
    "lunar.illumination_norm",
    "lunar.age_norm",
    "lunar.apsis_index",
    "lunar.waxing",
    "lunar.near_perigee",
    "lunar.near_apogee",
    "lunar.true_node_sin",
    "lunar.true_node_cos",
    "lunar.mean_node_sin",
    "lunar.mean_node_cos",
    "lunar.hours_since_previous_new_moon_norm",
    "lunar.hours_until_next_new_moon_norm",
    "lunar.hours_to_nearest_new_moon_signed_norm",
    "lunar.hours_since_previous_full_moon_norm",
    "lunar.hours_until_next_full_moon_norm",
    "lunar.hours_to_nearest_full_moon_signed_norm",
)

LUNAR_PATTERN_KEYS = (
    "moon_motion",
    "moon_speed_bin",
    "lunar_illumination_bin",
    "lunar_phase_dir",
    "lunar_phase_octile",
    "lunar_apsis",
    "moon_declination_bin",
    "lunar_true_node_quad",
    "lunar_mean_node_quad",
    "lunar_nearest_lunation_bin",
)

# Peso de desempate geométrico neutral (NO es score estadístico histórico).
GEOMETRIC_TIEBREAK_WEIGHT = 0.02
MOON_KERNEL_BANDWIDTH_DEG = 12.0


def _hours_norm(value: float | None) -> float:
    return float(value or 0.0) / 360.0


def extract_lunar_model_features(astro_features: dict) -> dict[str, float]:
    """Vector lunar-only a partir del payload cacheado de astrology_service."""
    mf = astro_features.get("model_features") or {}
    lunar = astro_features.get("lunar") or {}
    lunations = lunar.get("lunations") or {}
    out: dict[str, float] = {}

    for key in (
        "moon.longitude_sin",
        "moon.longitude_cos",
        "moon.latitude_norm",
        "moon.declination_norm",
        "moon.speed_lon_norm",
        "moon.distance_norm",
        "lunar.phase_sin",
        "lunar.phase_cos",
        "lunar.illumination_norm",
        "lunar.age_norm",
        "lunar.apsis_index",
        "lunar.waxing",
        "lunar.near_perigee",
        "lunar.near_apogee",
        "lunar.true_node_sin",
        "lunar.true_node_cos",
        "lunar.mean_node_sin",
        "lunar.mean_node_cos",
    ):
        if key in mf:
            out[key] = float(mf[key])

    # Horas firmadas desde/hasta (además del nearest ya presente en mf).
    out["lunar.hours_since_previous_new_moon_norm"] = _hours_norm(
        lunations.get("hours_since_previous_new_moon")
    )
    out["lunar.hours_until_next_new_moon_norm"] = _hours_norm(
        lunations.get("hours_until_next_new_moon")
    )
    out["lunar.hours_to_nearest_new_moon_signed_norm"] = float(
        mf.get(
            "lunar.hours_to_nearest_new_moon_signed_norm",
            _hours_norm(lunations.get("hours_to_nearest_new_moon_signed")),
        )
    )
    out["lunar.hours_since_previous_full_moon_norm"] = _hours_norm(
        lunations.get("hours_since_previous_full_moon")
    )
    out["lunar.hours_until_next_full_moon_norm"] = _hours_norm(
        lunations.get("hours_until_next_full_moon")
    )
    out["lunar.hours_to_nearest_full_moon_signed_norm"] = float(
        mf.get(
            "lunar.hours_to_nearest_full_moon_signed_norm",
            _hours_norm(lunations.get("hours_to_nearest_full_moon_signed")),
        )
    )
    return out


def extract_lunar_pattern_bins(astro_features: dict) -> dict[str, str]:
    bins = astro_features.get("pattern_bins") or astro_features.get("categorical") or {}
    return {k: bins[k] for k in LUNAR_PATTERN_KEYS if k in bins}


def assert_lunar_feature_keys_clean(model_features: dict[str, float]) -> None:
    """Garantiza que no hay planetas no-lunares ni separaciones."""
    forbidden_planets = (
        "sun.",
        "mercury.",
        "venus.",
        "mars.",
        "jupiter.",
        "saturn.",
        "uranus.",
        "neptune.",
        "pluto.",
    )
    for key in model_features:
        for prefix in forbidden_planets:
            if key.startswith(prefix):
                raise AssertionError(f"Feature no lunar: {key}")
        if "sep_" in key or "separation" in key:
            raise AssertionError(f"Separación Luna-planeta prohibida: {key}")


def _as_lunar_view(astro_features: dict) -> dict:
    lunar_mf = extract_lunar_model_features(astro_features)
    assert_lunar_feature_keys_clean(lunar_mf)
    lunar_bins = extract_lunar_pattern_bins(astro_features)
    return {
        **astro_features,
        "model_features": lunar_mf,
        "pattern_bins": lunar_bins,
        "categorical": lunar_bins,
        "mode": "lunar",
    }


def _vectorize(features: dict) -> np.ndarray:
    mf = features.get("model_features") or {}
    keys = sorted(mf.keys())
    return np.asarray([float(mf[k]) for k in keys], dtype=float)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _feature_items(features: dict) -> list[tuple[str, str]]:
    bins = features.get("pattern_bins") or {}
    return list(bins.items())


def _empty_assoc() -> dict:
    return {
        "feature_draw_counts": defaultdict(lambda: defaultdict(int)),
        "feature_number_hits": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        "number_hits": defaultdict(int),
        "draw_count": 0,
    }


def _update_assoc(assoc: dict, features: dict, numbers: list[int]) -> None:
    assoc["draw_count"] += 1
    for number in numbers:
        assoc["number_hits"][int(number)] += 1
    for key, value in _feature_items(features):
        assoc["feature_draw_counts"][key][value] += 1
        for number in numbers:
            assoc["feature_number_hits"][key][value][int(number)] += 1


def _logistic_center(raw: np.ndarray) -> list[float]:
    if raw.size == 0:
        return [0.5] * Config.POOL_SIZE
    centered = raw - float(raw.mean())
    scale = max(float(np.std(centered)), 1e-6)
    norm = 1.0 / (1.0 + np.exp(-centered / scale))
    return [float(x) for x in norm]


def _moon_longitude_deg(features: dict) -> float:
    bodies = features.get("bodies") or {}
    moon = bodies.get("moon") or {}
    if "longitude_deg" in moon:
        return float(moon["longitude_deg"]) % 360.0
    # Fallback sin/cos si el payload lunar-only no trae bodies.
    mf = features.get("model_features") or {}
    s = float(mf.get("moon.longitude_sin", 0.0))
    c = float(mf.get("moon.longitude_cos", 1.0))
    return float((np.degrees(np.arctan2(s, c)) + 360.0) % 360.0)


def _lon_to_sector(longitude_deg: float) -> int:
    """Mapea 0–360° a sectores 1–25 (14.4° cada uno)."""
    return int((longitude_deg % 360.0) / 360.0 * Config.POOL_SIZE) % Config.POOL_SIZE + 1


def _score_from_moon_longitude_kernel(
    draws,
    features_list: list[dict],
    target_features: dict,
    *,
    bandwidth_deg: float = MOON_KERNEL_BANDWIDTH_DEG,
) -> list[float]:
    """Correlación exacta: kernel gaussiano circular sobre la longitud lunar."""
    target_lon = _moon_longitude_deg(target_features)
    scores = np.zeros(Config.POOL_SIZE + 1, dtype=float)
    weight_sum = 0.0
    bw = max(float(bandwidth_deg), 1e-3)
    for draw, feats in zip(draws, features_list):
        lon = _moon_longitude_deg(feats)
        dist = circular_feature_distance(lon, target_lon)
        w = float(np.exp(-0.5 * (dist / bw) ** 2))
        if w <= 1e-12:
            continue
        weight_sum += w
        for number in draw.numbers():
            scores[int(number)] += w
    if weight_sum > 0:
        scores[1:] /= weight_sum
    baseline = Config.DRAW_SIZE / Config.POOL_SIZE
    lifts = scores[1:] / max(baseline, 1e-9)
    return _logistic_center(lifts)


def _score_from_moon_sector(
    draws,
    features_list: list[dict],
    target_features: dict,
) -> list[float]:
    """Frecuencia por sector zodiacal 1–25 + vecinos circulares."""
    # Histograma: sector -> conteos por número
    sector_hits = defaultdict(lambda: np.zeros(Config.POOL_SIZE + 1, dtype=float))
    sector_draws = defaultdict(float)
    for draw, feats in zip(draws, features_list):
        sector = _lon_to_sector(_moon_longitude_deg(feats))
        sector_draws[sector] += 1.0
        for number in draw.numbers():
            sector_hits[sector][int(number)] += 1.0

    target_sector = _lon_to_sector(_moon_longitude_deg(target_features))
    scores = np.zeros(Config.POOL_SIZE + 1, dtype=float)
    weight_sum = 0.0
    for offset in range(-2, 3):
        sector = ((target_sector - 1 + offset) % Config.POOL_SIZE) + 1
        n = sector_draws.get(sector, 0.0)
        if n < 1:
            continue
        # Vecinos más cercanos pesan más.
        w = float(np.exp(-0.5 * (offset / 1.25) ** 2))
        weight_sum += w
        scores += w * (sector_hits[sector] / n)
    if weight_sum > 0:
        scores /= weight_sum
    baseline = Config.DRAW_SIZE / Config.POOL_SIZE
    lifts = scores[1:] / max(baseline, 1e-9)
    return _logistic_center(lifts)


def _score_from_similarity(draws, features_list: list[dict], target_features: dict) -> list[float]:
    target_vec = _vectorize(target_features)
    scores = np.zeros(Config.POOL_SIZE + 1, dtype=float)
    weight_sum = 0.0
    for draw, feats in zip(draws, features_list):
        sim = _cosine_sim(_vectorize(feats), target_vec)
        w = max(0.0, 0.5 * (sim + 1.0)) ** 2
        if w <= 1e-12:
            continue
        weight_sum += w
        for number in draw.numbers():
            scores[int(number)] += w
    if weight_sum > 0:
        scores[1:] /= weight_sum
    baseline = Config.DRAW_SIZE / Config.POOL_SIZE
    lifts = scores[1:] / max(baseline, 1e-9)
    return _logistic_center(lifts)


def _score_from_assoc(assoc: dict, target_features: dict) -> list[float]:
    scores = np.zeros(Config.POOL_SIZE + 1, dtype=float)
    draw_count = max(assoc["draw_count"], 1)
    base_rate = Config.DRAW_SIZE / Config.POOL_SIZE
    for key, value in _feature_items(target_features):
        n_feat = assoc["feature_draw_counts"][key].get(value, 0)
        if n_feat < 3:
            continue
        for number in range(1, Config.POOL_SIZE + 1):
            hits = assoc["feature_number_hits"][key][value].get(number, 0)
            rate = hits / n_feat
            lift = rate / base_rate
            emp = assoc["number_hits"].get(number, 0) / draw_count
            emp = max(emp, 1e-6)
            relative = rate / emp
            scores[number] += 0.5 * (lift - 1.0) + 0.5 * (relative - 1.0)
    return _logistic_center(scores[1:])


def _score_hit_advantage(scores: list[float], actual_numbers: list[int]) -> float:
    if not scores or not actual_numbers:
        return 0.0
    arr = np.asarray(scores, dtype=float)
    hit = float(np.mean([arr[n - 1] for n in actual_numbers]))
    return hit - float(arr.mean())


def build_lunar_number_scores(
    draws,
    features_list: list[dict],
    target_features: dict,
    *,
    return_meta: bool = False,
):
    """
    Score lunar acoplado a la posición exacta de la Luna.

    Cuatro canales con el MISMO peso (25% cada uno):
      - kernel circular sobre la longitud eclíptica (correlación exacta)
      - sectores 1–25 derivados de esa longitud
      - similitud del resto del vector lunar
      - bins discretos lunares con soporte suficiente

    Sobre por qué los pesos son iguales y no adaptativos: la auditoría
    (research/variable_audit.py) midió la "precisión histórica" de cada canal y
    encontró que no persiste — la correlación entre rendir bien en un período y
    en el siguiente es -0.29. Ponderar con esa medida era ajustar ruido.

    Sobre por qué se conservan el kernel y el sector pese a su baja resolución
    estadística: son decisiones de DISEÑO del generador, no afirmaciones
    estadísticas. Hacen que la salida dependa de forma exacta y reproducible de
    dónde está la Luna, que es el comportamiento buscado. Lo que no se hace es
    presentarlos como evidencia de que la Luna influye en el sorteo.
    """
    if not draws:
        scores = [0.5] * Config.POOL_SIZE
        meta = {"mode": "empty"}
        return (scores, meta) if return_meta else scores

    assoc = _empty_assoc()
    for draw, feats in zip(draws, features_list):
        _update_assoc(assoc, feats, [int(x) for x in draw.numbers()])

    channels = {
        "moon_longitude_kernel": _score_from_moon_longitude_kernel(
            draws, features_list, target_features
        ),
        "moon_sector": _score_from_moon_sector(draws, features_list, target_features),
        "lunar_vector_similarity": _score_from_similarity(
            draws, features_list, target_features
        ),
        "lunar_bins": _score_from_assoc(assoc, target_features),
    }

    weight = 1.0 / len(channels)
    blended = np.zeros(Config.POOL_SIZE, dtype=float)
    for scores in channels.values():
        blended += weight * np.asarray(scores, dtype=float)

    effective_support = min(MIN_DRAWS_PER_BIN_VALUE, max(3, len(draws) // 4))
    usable_bin_values = sum(
        1
        for key in assoc["feature_draw_counts"]
        for count in assoc["feature_draw_counts"][key].values()
        if count >= effective_support
    )
    total_bin_values = sum(
        len(values) for values in assoc["feature_draw_counts"].values()
    )

    meta = {
        "mode": "equal_channel_weight",
        "channel_weights": {name: round(weight, 6) for name in sorted(channels)},
        "discarded_channels": [],
        "moon_longitude_deg": round(_moon_longitude_deg(target_features), 4),
        "moon_sector": _lon_to_sector(_moon_longitude_deg(target_features)),
        "kernel_bandwidth_deg": MOON_KERNEL_BANDWIDTH_DEG,
        "bin_values_usable": usable_bin_values,
        "bin_values_total": total_bin_values,
        "min_draws_per_bin_value": MIN_DRAWS_PER_BIN_VALUE,
        "score_formula": (
            "lunar_score[n] = promedio simple de 4 canales "
            "(kernel de longitud lunar, sector 1–25, vector lunar, bins con soporte)"
        ),
        "weighting_rationale": (
            "Pesos iguales: la precisión histórica por canal no persiste entre "
            "períodos (correlación -0.29 en la auditoría), así que usarla para "
            "ponderar ajustaba ruido."
        ),
    }
    scores = [float(x) for x in blended]
    return (scores, meta) if return_meta else scores


def _geometric_tiebreak(feats: dict) -> float:
    """Desempate geométrico neutral (paridad/suma/consecutivos); no usa historial."""
    parity = 1.0 - abs(feats["even_count"] - feats["odd_count"]) / Config.DRAW_SIZE
    # Suma ideal aproximada de 14 números ~ 182
    sum_score = float(np.exp(-0.5 * ((feats["total_sum"] - 182) / 25.0) ** 2))
    consec = 1.0 - abs(feats["consecutive"] - 2) / Config.DRAW_SIZE
    return float(max(0.0, (parity + sum_score + consec) / 3.0))


def score_sample_lunar(
    number_lunar_scores: list[float],
    *,
    sample_size: int,
    seed: int | None,
    top_k: int = 20,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict]:
    """
    Puntúa combinaciones SOLO con scores lunares:
      lunar_score_combo = mean(lunar_score[n] for n in combo)
      score = lunar_score_combo + 0.02 * geometric_tiebreak
    No usa score_combo / modelo estadístico histórico.
    """
    by_number = {i + 1: float(number_lunar_scores[i]) for i in range(Config.POOL_SIZE)}
    best: list[dict] = []
    processed = 0

    for batch in sample_combinations(sample_size, seed=seed):
        if should_cancel and should_cancel():
            break
        for row in batch:
            feats = features_for_combo(row)
            nums = feats["numbers"]
            lunar_score = sum(by_number[n] for n in nums) / Config.DRAW_SIZE
            tie = _geometric_tiebreak(feats)
            total = float(lunar_score + GEOMETRIC_TIEBREAK_WEIGHT * tie)
            best.append(
                {
                    **feats,
                    "lunar_score": float(lunar_score),
                    "score": total,
                    "components": {
                        "lunar": float(lunar_score),
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
                f"{processed:,} combinaciones lunares",
            )

    best = sorted(best, key=lambda x: x["score"], reverse=True)[:top_k]
    for i, item in enumerate(best, start=1):
        item["rank"] = i
    if progress_cb:
        progress_cb(100.0, f"Listo: top {len(best)}")
    return best


def discover_lunar_patterns(draws, features_list: list[dict], **kwargs) -> list[dict]:
    """Walk-forward solo sobre bins lunares, con nulo familiar por permutación."""
    kwargs.setdefault(
        "permutation_null", estimate_pattern_null(draws, features_list)
    )
    return discover_patterns_walk_forward(draws, features_list, **kwargs)


def run_lunar_experiment(
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
        progress_cb(2.0, "Calculando features lunares")

    features_list: list[dict] = []
    total = len(ordered)
    for i, draw in enumerate(ordered):
        if should_cancel and should_cancel():
            break
        astro = compute_astro_features(draw.draw_date)
        features_list.append(_as_lunar_view(astro))
        if progress_cb and i % 50 == 0:
            progress_cb(2.0 + 40.0 * i / max(total, 1), f"Luna {i + 1}/{total}")

    if should_cancel and should_cancel():
        return {"cancelled": True}, []

    if progress_cb:
        progress_cb(42.0, "Nulo lunar por permutación")
    pattern_null = estimate_pattern_null(ordered, features_list)

    if progress_cb:
        progress_cb(48.0, "Patrones lunares walk-forward")
    patterns = discover_lunar_patterns(
        ordered, features_list, permutation_null=pattern_null
    )
    pattern_null_public = {k: v for k, v in pattern_null.items() if k != "null_samples"}

    target_astro = compute_astro_features(target)
    target_features = _as_lunar_view(target_astro)
    # _as_lunar_view conserva bodies/lunar del payload completo → longitud exacta.
    number_scores, score_meta = build_lunar_number_scores(
        ordered, features_list, target_features, return_meta=True
    )

    if progress_cb:
        progress_cb(55.0, "Muestreando combinaciones lunares")

    def combo_progress(pct: float, message: str = "") -> None:
        if progress_cb:
            progress_cb(55.0 + 0.45 * pct, message)

    results = score_sample_lunar(
        number_scores,
        sample_size=sample_size,
        seed=seed,
        top_k=top_k,
        progress_cb=combo_progress,
        should_cancel=should_cancel,
    )

    ranked_numbers = sorted(
        (
            {"number": n, "lunar_score": round(number_scores[n - 1], 4)}
            for n in range(1, Config.POOL_SIZE + 1)
        ),
        key=lambda x: x["lunar_score"],
        reverse=True,
    )

    lunar = target_astro["lunar"]
    moon = target_astro["bodies"]["moon"]
    sync = history_sync or {}
    feature_keys = sorted(target_features["model_features"].keys())

    metrics = {
        "mode": "lunar",
        "disclaimer": (
            "Modelo lunar experimental. Usa solo variables de la Luna (fase, distancia, "
            "nodos, lunaciones, etc.). Los patrones son exploratorios: no demuestran "
            "causalidad ni garantizan resultados. Un mayor lunar_score no implica mayor "
            "probabilidad real: si el sorteo es uniforme, todas las combinaciones son "
            "equiprobables. El acoplamiento con la posición exacta de la Luna es una "
            "decisión de diseño del generador, no evidencia de influencia lunar."
        ),
        "audit_note": AUDIT_SUMMARY,
        "pattern_method": (
            "walk-forward lunar con hits y no-hits; solo valores de bin con soporte "
            "suficiente; contraste familiar por permutación (Westfall-Young sobre el "
            "|z| máximo) además del FDR Benjamini–Hochberg; dedupe por familia"
        ),
        "pattern_null": pattern_null_public,
        "feature_list": feature_keys,
        "feature_count": len(feature_keys),
        "score_formula": score_meta.get(
            "score_formula",
            "lunar_score[n] = Σ w_c * channel_c[n] (kernel longitud + sector + vector + bins)",
        ),
        "adaptive_weights": score_meta,
        "assumptions": {
            "draw_time_local": "22:30",
            "timezone": "America/Santiago",
            "location": target_astro["location"]["name"],
            "lat": target_astro["location"]["lat"],
            "lon": target_astro["location"]["lon"],
            "ephemeris": "Swiss Ephemeris Moshier",
            "synodic_month_days": SYNODIC_MONTH_DAYS,
            "geometric_tiebreak_weight": GEOMETRIC_TIEBREAK_WEIGHT,
            "moon_kernel_bandwidth_deg": MOON_KERNEL_BANDWIDTH_DEG,
            "channel_weighting": "igual entre los 4 canales (no adaptativo)",
            "min_draws_per_bin_value": MIN_DRAWS_PER_BIN_VALUE,
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
            "moon_sign": moon["sign_name"],
            "lunar_summary": {
                "phase_angle_deg": lunar["phase_angle_deg"],
                "illumination_pct": lunar["illumination_pct"],
                "age_days": lunar["age_days"],
                "distance_km": lunar["distance_km"],
                "distance_au": lunar["distance_au"],
                "near_perigee": lunar["near_perigee"],
                "near_apogee": lunar["near_apogee"],
                "apsis_index": lunar["apsis_index"],
                "sign_degree": f"{lunar['sign_name']} {lunar['degree_in_sign']:.2f}°",
                "declination_deg": lunar["declination_deg"],
                "phase_label": lunar["phase_label"],
                "speed_longitude_deg_per_day": moon["speed_longitude_deg_per_day"],
                "true_node": {
                    "longitude_deg": lunar["true_node"]["longitude_deg"],
                    "sign_degree": (
                        f"{lunar['true_node']['sign_name']} "
                        f"{lunar['true_node']['degree_in_sign']:.2f}°"
                    ),
                },
                "mean_node": {
                    "longitude_deg": lunar["mean_node"]["longitude_deg"],
                    "sign_degree": (
                        f"{lunar['mean_node']['sign_name']} "
                        f"{lunar['mean_node']['degree_in_sign']:.2f}°"
                    ),
                },
                "lunations": lunar["lunations"],
            },
        },
    }
    if progress_cb:
        progress_cb(100.0, "Generación lunar simple lista")
    return metrics, results
