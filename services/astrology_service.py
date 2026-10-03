from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from math import cos, degrees, radians, sin
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import swisseph as swe

from config import Config
from services.analysis_service import draws_matrix
from services.combination_service import sample_combinations
from services.scoring_service import _build_context, normalize_weights, score_combo

# Sorteos Kino: mié/vie/dom 22:30 America/Santiago. Coords desde Config.
DRAW_HOUR = Config.ASTRO_DRAW_HOUR
DRAW_MINUTE = Config.ASTRO_DRAW_MINUTE
DRAW_WEEKDAYS = tuple(Config.ASTRO_DRAW_WEEKDAYS)
TZ_CHILE = ZoneInfo(Config.ASTRO_TIMEZONE)
ASTRO_LAT = Config.ASTRO_LATITUDE
ASTRO_LON = Config.ASTRO_LONGITUDE
ASTRO_LOCATION_NAME = Config.ASTRO_LOCATION_NAME

WEEKDAY_NAMES_ES = {
    0: "lunes",
    1: "martes",
    2: "miércoles",
    3: "jueves",
    4: "viernes",
    5: "sábado",
    6: "domingo",
}

# Ephemeris Moshier: no requiere archivos externos de Swiss Ephemeris.
EPHE_FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED
EQ_FLAGS = swe.FLG_MOSEPH | swe.FLG_SPEED | swe.FLG_EQUATORIAL

# |dλ/dt| por debajo de este umbral (°/día) ⇒ estacionario (documentado).
STATIONARY_THRESHOLD_DEG_PER_DAY = 0.05
AU_TO_KM = 149_597_870.7
SYNODIC_MONTH_DAYS = 29.530588
MOON_MEAN_DISTANCE_KM = 384_400.0
MOON_PERIGEE_NEAR_KM = 370_000.0
MOON_APOGEE_NEAR_KM = 400_000.0
RADIAL_STATION_AU_PER_DAY = 1.5e-6

PLANETS = {
    "sun": swe.SUN,
    "moon": swe.MOON,
    "mercury": swe.MERCURY,
    "venus": swe.VENUS,
    "mars": swe.MARS,
    "jupiter": swe.JUPITER,
    "saturn": swe.SATURN,
    "uranus": swe.URANUS,
    "neptune": swe.NEPTUNE,
    "pluto": swe.PLUTO,
}
PLANET_ORDER = list(PLANETS.keys())

SIGN_NAMES = [
    "Aries",
    "Tauro",
    "Géminis",
    "Cáncer",
    "Leo",
    "Virgo",
    "Libra",
    "Escorpio",
    "Sagitario",
    "Capricornio",
    "Acuario",
    "Piscis",
]

ASTRO_BLEND_DEFAULT = 0.35

# --------------------------------------------------------------------------- #
# Parámetros fijados por la auditoría empírica (research/variable_audit.py).
#
# La auditoría midió 71 estrategias sobre 2.456 sorteos con walk-forward y nulos
# por permutación. Conclusiones que fijan estos valores:
#
#  - Ponderar canales por "precisión histórica" ajusta ruido: la correlación
#    entre rendir bien en un período y en el siguiente es NEGATIVA (-0.29).
#    Por eso los pesos son iguales y no se estiman de los datos.
#  - Un bin con pocos sorteos por valor solo puede producir falsos positivos.
#    Con 2.456 sorteos y tasa base 56%, hacen falta ~600 casos por valor para
#    detectar un efecto plausible tras corregir por multiplicidad.
#  - El nulo correcto no es 0.5 ni la tasa base: es la permutación, porque los
#    números más frecuentes del período dan ventaja estática sin predecir nada.
# --------------------------------------------------------------------------- #

# Peso igual entre canales: no se estima nada del histórico.
EQUAL_CHANNEL_WEIGHTS = True
# Mezcla fija similitud/bins. Ninguno mostró señal, así que no se privilegia uno.
SIM_BIN_BLEND = 0.5
# Sorteos mínimos por valor de bin para que ese valor sea utilizable.
MIN_DRAWS_PER_BIN_VALUE = 600
# Permutaciones para el nulo familiar de la búsqueda de patrones.
PATTERN_PERMUTATIONS = 200
# Nivel del contraste familiar (Westfall-Young sobre el estadístico máximo).
PATTERN_FWER_ALPHA = 0.05

# Resumen de la auditoría, mostrado en la UI para que el número que ve el
# usuario venga siempre acompañado de lo que ese número no significa.
AUDIT_SUMMARY = {
    "headline": (
        "Ninguna de las 156 variables mostró capacidad predictiva sobre 2.456 sorteos."
    ),
    "detail": (
        "Se compararon 71 formas de puntuar los números con walk-forward estricto "
        "(145.976 predicciones fuera de muestra) contra nulos por permutación. "
        "Cero superaron su nulo. La mejor alcanzó AUC 0.5061 cuando el máximo "
        "esperado por azar, probando 71 estrategias, es 0.5126."
    ),
    "what_the_score_means": (
        "El porcentaje mide afinidad con patrones históricos, no probabilidad de "
        "acertar. Si el sorteo es uniforme, las 4.457.400 combinaciones son "
        "igual de probables."
    ),
    "strategies_tested": 71,
    "out_of_sample_predictions": 145976,
    "significant_findings": 0,
    "best_auc": 0.5061,
    "chance_ceiling_auc": 0.5126,
    "reproduce_with": "python -m research.variable_audit",
}


def _angle_diff(a: float, b: float) -> float:
    """Separación circular mínima en [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _sign_index(longitude: float) -> int:
    return int(longitude % 360.0 // 30.0) % 12


def _degree_in_sign(longitude: float) -> float:
    return float(longitude % 360.0 % 30.0)


def _circular_encoding(longitude: float) -> tuple[float, float]:
    rad = radians(longitude % 360.0)
    return float(sin(rad)), float(cos(rad))


def circular_feature_distance(lon_a: float, lon_b: float) -> float:
    """Distancia euclídea entre encodings sin/cos (helper de tests / continuidad)."""
    sa, ca = _circular_encoding(lon_a)
    sb, cb = _circular_encoding(lon_b)
    return float(((sa - sb) ** 2 + (ca - cb) ** 2) ** 0.5)


def _motion_state(speed_lon: float, *, can_retrograde: bool) -> str:
    if abs(speed_lon) < STATIONARY_THRESHOLD_DEG_PER_DAY:
        return "stationary"
    if can_retrograde and speed_lon < 0:
        return "retrograde"
    return "direct"


def next_draw_datetime(now: datetime | None = None) -> datetime:
    """Próximo sorteo real: miércoles/viernes/domingo a las 22:30 America/Santiago."""
    if now is None:
        now = datetime.now(TZ_CHILE)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=TZ_CHILE)
    else:
        now = now.astimezone(TZ_CHILE)

    for offset in range(0, 8):
        day = (now + timedelta(days=offset)).date()
        if day.weekday() not in DRAW_WEEKDAYS:
            continue
        candidate = datetime(
            day.year,
            day.month,
            day.day,
            DRAW_HOUR,
            DRAW_MINUTE,
            tzinfo=TZ_CHILE,
        )
        if candidate > now:
            return candidate
    raise RuntimeError("No se pudo determinar el próximo sorteo")


def format_draw_label(dt: datetime) -> str:
    local = dt.astimezone(TZ_CHILE) if dt.tzinfo else dt.replace(tzinfo=TZ_CHILE)
    name = WEEKDAY_NAMES_ES[local.weekday()].capitalize()
    return (
        f"{name} {local.strftime('%Y-%m-%d')} "
        f"{local.strftime('%H:%M')} {Config.ASTRO_TIMEZONE}"
    )


def draw_local_datetime(draw_date: date) -> datetime:
    return datetime(
        draw_date.year,
        draw_date.month,
        draw_date.day,
        DRAW_HOUR,
        DRAW_MINUTE,
        tzinfo=TZ_CHILE,
    )


def _as_local_datetime(when: date | datetime) -> datetime:
    if isinstance(when, datetime):
        if when.tzinfo is None:
            return when.replace(tzinfo=TZ_CHILE)
        return when.astimezone(TZ_CHILE)
    return draw_local_datetime(when)


def julian_day_ut(when: date | datetime) -> float:
    utc = _as_local_datetime(when).astimezone(timezone.utc)
    hour_ut = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    return swe.julday(utc.year, utc.month, utc.day, hour_ut)


def _body_snapshot(jd: float, body: int, *, name: str) -> dict:
    ecl, _ = swe.calc_ut(jd, body, EPHE_FLAGS)
    equ, _ = swe.calc_ut(jd, body, EQ_FLAGS)
    lon = float(ecl[0]) % 360.0
    lat = float(ecl[1])
    dist_au = float(ecl[2])
    speed_lon = float(ecl[3])
    speed_lat = float(ecl[4])
    speed_dist = float(ecl[5])
    decl = float(equ[1])
    lon_sin, lon_cos = _circular_encoding(lon)
    can_retro = name not in ("sun", "moon")
    state = _motion_state(speed_lon, can_retrograde=can_retro)
    sign_idx = _sign_index(lon)
    out = {
        "name": name,
        "longitude_deg": round(lon, 6),  # solo UI/metadata; no usar lineal en el modelo
        "latitude_deg": round(lat, 6),
        "declination_deg": round(decl, 6),
        "speed_longitude_deg_per_day": round(speed_lon, 6),
        "speed_latitude_deg_per_day": round(speed_lat, 6),
        "speed_distance_au_per_day": round(speed_dist, 8),
        "distance_au": round(dist_au, 8),
        "motion_state": state,
        "is_retrograde": state == "retrograde",
        "is_stationary": state == "stationary",
        "sign_index": sign_idx,
        "sign_name": SIGN_NAMES[sign_idx],
        "degree_in_sign": round(_degree_in_sign(lon), 4),
        "longitude_sin": round(lon_sin, 8),
        "longitude_cos": round(lon_cos, 8),
    }
    if name == "moon":
        out["distance_km"] = round(dist_au * AU_TO_KM, 3)
    return out


def _node_snapshot(jd: float, body: int, *, name: str) -> dict:
    ecl, _ = swe.calc_ut(jd, body, EPHE_FLAGS)
    lon = float(ecl[0]) % 360.0
    lon_sin, lon_cos = _circular_encoding(lon)
    sign_idx = _sign_index(lon)
    return {
        "name": name,
        "longitude_deg": round(lon, 6),
        "longitude_sin": round(lon_sin, 8),
        "longitude_cos": round(lon_cos, 8),
        "sign_index": sign_idx,
        "sign_name": SIGN_NAMES[sign_idx],
        "degree_in_sign": round(_degree_in_sign(lon), 4),
        "speed_longitude_deg_per_day": round(float(ecl[3]), 6),
    }


def _moon_sun_elongation(jd: float) -> float:
    moon, _ = swe.calc_ut(jd, swe.MOON, EPHE_FLAGS)
    sun, _ = swe.calc_ut(jd, swe.SUN, EPHE_FLAGS)
    return (float(moon[0]) - float(sun[0])) % 360.0


def _refine_lunation_crossing(
    jd_lo: float,
    jd_hi: float,
    *,
    target_elongation: float,
    steps: int = 24,
) -> float:
    """Busca el JD en [jd_lo, jd_hi] donde la elongación cruza target (0 o 180)."""
    best_jd = jd_lo
    best_err = 1e9
    for i in range(steps + 1):
        jd = jd_lo + (jd_hi - jd_lo) * i / steps
        elong = _moon_sun_elongation(jd)
        err = _angle_diff(elong, target_elongation)
        if err < best_err:
            best_err = err
            best_jd = jd
    # Segunda pasada local
    span = (jd_hi - jd_lo) / max(steps, 1)
    lo = best_jd - span
    hi = best_jd + span
    for i in range(steps + 1):
        jd = lo + (hi - lo) * i / steps
        elong = _moon_sun_elongation(jd)
        err = _angle_diff(elong, target_elongation)
        if err < best_err:
            best_err = err
            best_jd = jd
    return best_jd


def _find_nearest_lunations(jd: float) -> dict:
    """Horas firmadas desde/hasta Luna nueva y llena más cercanas (reproducible)."""
    # Barrido ±18 días a paso de 6 h, luego refinamiento.
    sample_step = 0.25  # días
    window = 18.0
    samples = []
    n = int(window * 2 / sample_step) + 1
    for i in range(n):
        jdi = jd - window + i * sample_step
        samples.append((jdi, _moon_sun_elongation(jdi)))

    def crossings(target: float) -> list[float]:
        found: list[float] = []
        for (j0, e0), (j1, e1) in zip(samples, samples[1:]):
            # Detectar cruce del target midiendo salto de error / unwrap aproximado
            d0 = (e0 - target + 180.0) % 360.0 - 180.0
            d1 = (e1 - target + 180.0) % 360.0 - 180.0
            if d0 == 0 or d0 * d1 < 0:
                found.append(_refine_lunation_crossing(j0, j1, target_elongation=target))
        return found

    new_crossings = crossings(0.0)
    full_crossings = crossings(180.0)

    def nearest_pair(cross_list: list[float]) -> tuple[float | None, float | None, float | None]:
        if not cross_list:
            return None, None, None
        prev = max((c for c in cross_list if c <= jd), default=None)
        nxt = min((c for c in cross_list if c > jd), default=None)
        nearest = min(cross_list, key=lambda c: abs(c - jd))
        return prev, nxt, nearest

    prev_new, next_new, nearest_new = nearest_pair(new_crossings)
    prev_full, next_full, nearest_full = nearest_pair(full_crossings)

    def hours_signed(event_jd: float | None) -> float | None:
        if event_jd is None:
            return None
        return round((event_jd - jd) * 24.0, 4)

    return {
        "hours_since_previous_new_moon": (
            None if prev_new is None else round((jd - prev_new) * 24.0, 4)
        ),
        "hours_until_next_new_moon": (
            None if next_new is None else round((next_new - jd) * 24.0, 4)
        ),
        "hours_to_nearest_new_moon_signed": hours_signed(nearest_new),
        "hours_since_previous_full_moon": (
            None if prev_full is None else round((jd - prev_full) * 24.0, 4)
        ),
        "hours_until_next_full_moon": (
            None if next_full is None else round((next_full - jd) * 24.0, 4)
        ),
        "hours_to_nearest_full_moon_signed": hours_signed(nearest_full),
    }


def _build_lunar_block(jd: float, bodies: dict[str, dict]) -> dict:
    moon = bodies["moon"]
    sun = bodies["sun"]
    phase_angle = (moon["longitude_deg"] - sun["longitude_deg"]) % 360.0
    illumination_pct = float(50.0 * (1.0 - cos(radians(phase_angle))))
    illumination_pct = max(0.0, min(100.0, illumination_pct))
    age_days = phase_angle / 360.0 * SYNODIC_MONTH_DAYS
    waxing = phase_angle < 180.0
    distance_km = float(moon["distance_km"])
    radial = float(moon["speed_distance_au_per_day"])
    near_perigee = distance_km <= MOON_PERIGEE_NEAR_KM and abs(radial) <= RADIAL_STATION_AU_PER_DAY
    near_apogee = distance_km >= MOON_APOGEE_NEAR_KM and abs(radial) <= RADIAL_STATION_AU_PER_DAY
    # Índice [-1, 1] aprox. relativo a la media (negativo = más cerca / perigeo).
    apsis_index = (distance_km - MOON_MEAN_DISTANCE_KM) / 30_000.0
    apsis_index = float(max(-1.5, min(1.5, apsis_index)))

    true_node = _node_snapshot(jd, swe.TRUE_NODE, name="true_node")
    mean_node = _node_snapshot(jd, swe.MEAN_NODE, name="mean_node")

    separations: dict[str, float] = {}
    for name in PLANET_ORDER:
        if name == "moon":
            continue
        separations[name] = round(
            _angle_diff(moon["longitude_deg"], bodies[name]["longitude_deg"]),
            4,
        )

    lunations = _find_nearest_lunations(jd)

    return {
        "phase_angle_deg": round(phase_angle, 6),
        "phase_angle_sin": round(sin(radians(phase_angle)), 8),
        "phase_angle_cos": round(cos(radians(phase_angle)), 8),
        "illumination_pct": round(illumination_pct, 4),
        "age_days": round(age_days, 6),
        "distance_au": moon["distance_au"],
        "distance_km": distance_km,
        "radial_speed_au_per_day": moon["speed_distance_au_per_day"],
        "near_perigee": bool(near_perigee),
        "near_apogee": bool(near_apogee),
        "apsis_index": round(apsis_index, 6),
        "sign_name": moon["sign_name"],
        "sign_index": moon["sign_index"],
        "degree_in_sign": moon["degree_in_sign"],
        "declination_deg": moon["declination_deg"],
        "waxing": bool(waxing),
        "waning": bool(not waxing),
        "phase_label": "waxing" if waxing else "waning",
        "true_node": true_node,
        "mean_node": mean_node,
        "separations_to_planets_deg": separations,
        "lunations": lunations,
    }


def _model_feature_dict(bodies: dict[str, dict], lunar: dict) -> dict[str, float]:
    """Features numéricas para el modelo (sin longitud cruda ni strings)."""
    feats: dict[str, float] = {}
    for name, body in bodies.items():
        prefix = f"{name}."
        feats[prefix + "longitude_sin"] = float(body["longitude_sin"])
        feats[prefix + "longitude_cos"] = float(body["longitude_cos"])
        feats[prefix + "latitude_norm"] = float(body["latitude_deg"]) / 10.0
        feats[prefix + "declination_norm"] = float(body["declination_deg"]) / 30.0
        # Velocidad típica Luna ~13°/d; planetas exteriores << 1.
        scale = 13.0 if name == "moon" else 1.0
        feats[prefix + "speed_lon_norm"] = float(body["speed_longitude_deg_per_day"]) / scale
        feats[prefix + "retrograde"] = 1.0 if body["is_retrograde"] else 0.0
        feats[prefix + "stationary"] = 1.0 if body["is_stationary"] else 0.0
        # Distancia: Luna en fracción de AU media; exteriores log-escala suave.
        dist = float(body["distance_au"])
        if name == "moon":
            feats[prefix + "distance_norm"] = dist / 0.00257
        else:
            feats[prefix + "distance_norm"] = float(np.log1p(dist) / 4.0)

    feats["lunar.phase_sin"] = float(lunar["phase_angle_sin"])
    feats["lunar.phase_cos"] = float(lunar["phase_angle_cos"])
    feats["lunar.illumination_norm"] = float(lunar["illumination_pct"]) / 100.0
    feats["lunar.age_norm"] = float(lunar["age_days"]) / SYNODIC_MONTH_DAYS
    feats["lunar.apsis_index"] = float(lunar["apsis_index"])
    feats["lunar.waxing"] = 1.0 if lunar["waxing"] else 0.0
    feats["lunar.declination_norm"] = float(lunar["declination_deg"]) / 30.0
    feats["lunar.true_node_sin"] = float(lunar["true_node"]["longitude_sin"])
    feats["lunar.true_node_cos"] = float(lunar["true_node"]["longitude_cos"])
    feats["lunar.mean_node_sin"] = float(lunar["mean_node"]["longitude_sin"])
    feats["lunar.mean_node_cos"] = float(lunar["mean_node"]["longitude_cos"])
    feats["lunar.near_perigee"] = 1.0 if lunar["near_perigee"] else 0.0
    feats["lunar.near_apogee"] = 1.0 if lunar["near_apogee"] else 0.0

    for pname, sep in lunar["separations_to_planets_deg"].items():
        # Separación 0–180 → sin/cos de 2*sep para continuidad suave en oposición.
        feats[f"lunar.sep_{pname}_norm"] = float(sep) / 180.0
        feats[f"lunar.sep_{pname}_sin"] = float(sin(radians(sep)))
        feats[f"lunar.sep_{pname}_cos"] = float(cos(radians(sep)))

    lun = lunar["lunations"]
    for key in (
        "hours_to_nearest_new_moon_signed",
        "hours_to_nearest_full_moon_signed",
    ):
        val = lun.get(key)
        # Normalizar a ~±1 ciclo sinódico (±360 h aprox.).
        feats[f"lunar.{key}_norm"] = float(val or 0.0) / 360.0

    return feats


def _sep_distance_bin(sep_deg: float) -> str:
    if sep_deg < 30:
        return "near"
    if sep_deg < 90:
        return "moderate"
    if sep_deg < 150:
        return "wide"
    return "opposite"


def _phase_octile_bin(phase_angle: float) -> str:
    names = (
        "nueva",
        "creciente",
        "cuarto_creciente",
        "gibosa_creciente",
        "llena",
        "gibosa_menguante",
        "cuarto_menguante",
        "menguante",
    )
    return names[int(phase_angle % 360.0 // 45.0) % 8]


def _node_quadrant_bin(longitude: float) -> str:
    q = int(longitude % 360.0 // 90.0) % 4
    return ("q1_aries_cancer", "q2_cancer_libra", "q3_libra_cap", "q4_cap_aries")[q]


def _declination_bin(decl: float) -> str:
    if decl > 8:
        return "north"
    if decl < -8:
        return "south"
    return "equatorial"


def _nearest_lunation_bin(lunar: dict) -> str:
    lun = lunar.get("lunations") or {}
    new_h = lun.get("hours_to_nearest_new_moon_signed")
    full_h = lun.get("hours_to_nearest_full_moon_signed")
    if new_h is None and full_h is None:
        return "unknown"
    candidates = []
    if new_h is not None:
        candidates.append(("near_new", abs(float(new_h))))
    if full_h is not None:
        candidates.append(("near_full", abs(float(full_h))))
    label, hours = min(candidates, key=lambda x: x[1])
    if hours <= 48:
        return label
    return "mid_cycle"


def _binned_pattern_features(bodies: dict[str, dict], lunar: dict) -> dict[str, str]:
    """Bins discretos compactos para walk-forward (sin one-hot de signos)."""
    bins: dict[str, str] = {}
    # Motion: 10 cuerpos. Velocidad solo para clásicos (evita explosión).
    classical = ("sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn")
    for name, body in bodies.items():
        bins[f"{name}_motion"] = body["motion_state"]
        if name not in classical:
            continue
        spd = abs(float(body["speed_longitude_deg_per_day"]))
        if spd < STATIONARY_THRESHOLD_DEG_PER_DAY:
            bins[f"{name}_speed_bin"] = "stationary"
        elif spd < 0.2:
            bins[f"{name}_speed_bin"] = "slow"
        elif spd < 1.0:
            bins[f"{name}_speed_bin"] = "medium"
        else:
            bins[f"{name}_speed_bin"] = "fast"

    illum = float(lunar["illumination_pct"])
    if illum < 25:
        bins["lunar_illumination_bin"] = "dark"
    elif illum < 50:
        bins["lunar_illumination_bin"] = "crescent"
    elif illum < 75:
        bins["lunar_illumination_bin"] = "gibbous"
    else:
        bins["lunar_illumination_bin"] = "bright"
    bins["lunar_phase_dir"] = lunar["phase_label"]
    bins["lunar_phase_octile"] = _phase_octile_bin(float(lunar["phase_angle_deg"]))
    bins["lunar_apsis"] = (
        "perigee"
        if lunar["near_perigee"]
        else "apogee"
        if lunar["near_apogee"]
        else "mid"
    )
    bins["moon_declination_bin"] = _declination_bin(float(lunar["declination_deg"]))
    bins["lunar_true_node_quad"] = _node_quadrant_bin(
        float(lunar["true_node"]["longitude_deg"])
    )
    bins["lunar_mean_node_quad"] = _node_quadrant_bin(
        float(lunar["mean_node"]["longitude_deg"])
    )
    bins["lunar_nearest_lunation_bin"] = _nearest_lunation_bin(lunar)

    # Separaciones Luna–planeta discretizadas (subconjunto clásico).
    for planet in ("sun", "mercury", "venus", "mars", "jupiter"):
        sep = float(lunar["separations_to_planets_deg"].get(planet, 0.0))
        bins[f"lunar_{planet}_sep_bin"] = _sep_distance_bin(sep)
    return bins


PATTERN_FEATURE_FAMILIES = {
    "lunar_illumination_bin": "lunar_phase",
    "lunar_phase_dir": "lunar_phase",
    "lunar_phase_octile": "lunar_phase",
    "lunar_sun_sep_bin": "lunar_phase",
    "lunar_nearest_lunation_bin": "lunar_phase",
    "lunar_apsis": "lunar_distance",
    "moon_declination_bin": "moon_decl",
    "lunar_true_node_quad": "lunar_node",
    "lunar_mean_node_quad": "lunar_node",
    "moon_motion": "moon_motion",
    "moon_speed_bin": "moon_motion",
}


def _pattern_family(feature: str) -> str:
    if feature in PATTERN_FEATURE_FAMILIES:
        return PATTERN_FEATURE_FAMILIES[feature]
    if feature.endswith("_motion"):
        return f"motion:{feature.split('_', 1)[0]}"
    if feature.endswith("_speed_bin"):
        return f"speed:{feature.split('_', 1)[0]}"
    if feature.endswith("_sep_bin"):
        return "lunar_separation"
    return feature


def _binomial_pvalue_ge(successes: int, trials: int, p: float) -> float:
    """P(X >= successes) bajo Binomial(trials, p). Exacto si n<=200; si no, normal."""
    from math import erfc, exp, lgamma, log, sqrt

    if trials <= 0:
        return 1.0
    successes = max(0, min(int(successes), int(trials)))
    if successes <= 0:
        return 1.0
    p = min(max(float(p), 1e-15), 1.0 - 1e-15)
    if trials <= 200:
        log_p = log(p)
        log_q = log(1.0 - p)
        total = 0.0
        for k in range(successes, trials + 1):
            log_c = lgamma(trials + 1) - lgamma(k + 1) - lgamma(trials - k + 1)
            total += exp(log_c + k * log_p + (trials - k) * log_q)
        return float(min(1.0, max(0.0, total)))
    mean = trials * p
    var = trials * p * (1.0 - p)
    if var <= 1e-12:
        return 0.0 if successes > mean else 1.0
    z = (successes - 0.5 - mean) / sqrt(var)
    # survival = 1 - Phi(z)
    return float(max(0.0, min(1.0, 0.5 * erfc(z / sqrt(2.0)))))


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    """FDR Benjamini–Hochberg (ajustados monótonos, acotados a [0, 1])."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [1.0] * n
    prev = 1.0
    for rank_from_end, idx in enumerate(reversed(order)):
        rank = n - rank_from_end  # 1..n from smallest p
        raw = p_values[idx] * n / rank
        prev = min(prev, raw)
        adj[idx] = float(min(1.0, max(0.0, prev)))
    return adj


@lru_cache(maxsize=4096)
def _compute_astro_features_cached(local_iso: str) -> dict:
    local = datetime.fromisoformat(local_iso)
    if local.tzinfo is None:
        local = local.replace(tzinfo=TZ_CHILE)
    jd = julian_day_ut(local)

    bodies: dict[str, dict] = {}
    for name, body_id in PLANETS.items():
        bodies[name] = _body_snapshot(jd, body_id, name=name)

    lunar = _build_lunar_block(jd, bodies)
    model_features = _model_feature_dict(bodies, lunar)
    pattern_bins = _binned_pattern_features(bodies, lunar)

    # Compatibilidad con campos antiguos usados en UI/tests.
    phase_angle = lunar["phase_angle_deg"]
    phase_bin = int(phase_angle // 45.0) % 8
    phase_names = [
        "nueva",
        "creciente",
        "cuarto_creciente",
        "gibosa_creciente",
        "llena",
        "gibosa_menguante",
        "cuarto_menguante",
        "menguante",
    ]

    return {
        "draw_date": local.date().isoformat(),
        "local_datetime": local.isoformat(),
        "julian_day": jd,
        "location": {
            "lat": ASTRO_LAT,
            "lon": ASTRO_LON,
            "name": ASTRO_LOCATION_NAME,
        },
        "units": {
            "longitude": "deg ecliptic 0-360",
            "latitude": "deg ecliptic",
            "declination": "deg equatorial",
            "speed_longitude": "deg/day",
            "distance": "AU (Moon also km)",
            "illumination": "percent 0-100",
            "stationary_threshold_deg_per_day": STATIONARY_THRESHOLD_DEG_PER_DAY,
            "au_to_km": AU_TO_KM,
        },
        "bodies": bodies,
        "lunar": lunar,
        "model_features": model_features,
        "pattern_bins": pattern_bins,
        # aliases legacy
        "positions": {
            name: {
                "longitude": body["longitude_deg"],
                "sign": body["sign_index"],
                "sign_name": body["sign_name"],
                "retrograde": body["is_retrograde"],
                "speed": body["speed_longitude_deg_per_day"],
            }
            for name, body in bodies.items()
        },
        "moon_phase_bin": phase_bin,
        "moon_phase_name": phase_names[phase_bin],
        "elongation": phase_angle,
        "categorical": pattern_bins,
    }


def compute_astro_features(when: date | datetime) -> dict:
    """Variables planetarias + lunares reproducibles para un instante de sorteo."""
    local = _as_local_datetime(when)
    return _compute_astro_features_cached(local.isoformat())


def prior_draws_before_target(draws, target_dt: datetime):
    """Historial estricto: solo sorteos con fecha anterior al día del target."""
    target_date = _as_local_datetime(target_dt).date()
    return sorted(
        (d for d in draws if d.draw_date < target_date),
        key=lambda d: d.draw_date,
    )


def _feature_items(features: dict) -> list[tuple[str, str]]:
    bins = features.get("pattern_bins") or features.get("categorical") or {}
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


def _pearson_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = a - float(a.mean())
    b = b - float(b.mean())
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return abs(float(np.dot(a, b) / (da * db)))


def select_model_feature_keys(
    features_list: list[dict],
    *,
    min_std: float = 1e-4,
    max_abs_corr: float = 0.97,
) -> list[str]:
    """Filtra features de baja varianza y pares muy colineales (greedy)."""
    if not features_list:
        return []
    keys = sorted((features_list[0].get("model_features") or {}).keys())
    if len(features_list) < 5 or not keys:
        return keys
    mat = np.asarray(
        [
            [float((feats.get("model_features") or {}).get(k, 0.0)) for k in keys]
            for feats in features_list
        ],
        dtype=float,
    )
    stds = mat.std(axis=0)
    kept_idx = [i for i, s in enumerate(stds) if float(s) >= min_std]
    if not kept_idx:
        return keys
    keys = [keys[i] for i in kept_idx]
    mat = mat[:, kept_idx]
    selected: list[int] = []
    for i in range(len(keys)):
        col = mat[:, i]
        if any(_pearson_abs(col, mat[:, j]) >= max_abs_corr for j in selected):
            continue
        selected.append(i)
    return [keys[i] for i in selected] or keys


def _vectorize(
    features: dict,
    keys: list[str] | None = None,
    *,
    scales: dict[str, float] | None = None,
) -> np.ndarray:
    mf = features.get("model_features") or {}
    use_keys = keys if keys is not None else sorted(mf.keys())
    if not scales:
        return np.asarray([float(mf.get(k, 0.0)) for k in use_keys], dtype=float)
    return np.asarray(
        [float(mf.get(k, 0.0)) * float(scales.get(k, 1.0)) for k in use_keys],
        dtype=float,
    )


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def group_feature_channels(keys: list[str]) -> dict[str, list[str]]:
    """Agrupa variables en canales interpretables para ponderación adaptativa."""
    channels: dict[str, list[str]] = {}
    for key in keys:
        if key.startswith("lunar.") or key.startswith("moon."):
            if "longitude" in key or key.startswith("lunar.phase"):
                name = "lunar_position_phase"
            elif "node" in key:
                name = "lunar_nodes"
            elif "lunation" in key or "new_moon" in key or "full_moon" in key:
                name = "lunations"
            else:
                name = "lunar_other"
        elif key.endswith(".longitude_sin") or key.endswith(".longitude_cos"):
            body = key.split(".", 1)[0]
            name = f"{body}_longitude"
        elif "sep" in key or "separation" in key:
            name = "moon_planet_separations"
        else:
            body = key.split(".", 1)[0]
            name = f"{body}_kinematics"
        channels.setdefault(name, []).append(key)
    return channels


def _score_from_similarity(
    draws,
    features_list: list[dict],
    target_features: dict,
    *,
    feature_keys: list[str] | None = None,
    scales: dict[str, float] | None = None,
) -> list[float]:
    """Score por número ponderando sorteos previos por similitud del cielo (sin fuga)."""
    keys = feature_keys or select_model_feature_keys(features_list)
    target_vec = _vectorize(target_features, keys, scales=scales)
    scores = np.zeros(Config.POOL_SIZE + 1, dtype=float)
    weight_sum = 0.0
    for draw, feats in zip(draws, features_list):
        sim = _cosine_sim(_vectorize(feats, keys, scales=scales), target_vec)
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
    centered = lifts - float(lifts.mean()) if lifts.size else lifts
    scale = max(float(np.std(centered)), 1e-6) if centered.size else 1.0
    norm = 1.0 / (1.0 + np.exp(-centered / scale))
    return [float(x) for x in norm]


def _score_hit_advantage(scores: list[float], actual_numbers: list[int]) -> float:
    """Ventaja media de los números que salieron vs la media de los 25 scores."""
    if not scores or not actual_numbers:
        return 0.0
    arr = np.asarray(scores, dtype=float)
    hit = float(np.mean([arr[n - 1] for n in actual_numbers]))
    return hit - float(arr.mean())


def build_channel_scales(feature_keys: list[str]) -> tuple[dict[str, float], dict]:
    """Escalas por feature que dan el MISMO peso a cada canal.

    No se estima nada del histórico. El único ajuste es compensar que unos
    canales tienen muchas más variables que otros: sin esto, las 27 features de
    separaciones Luna-planeta aplastarían a las 2 de la longitud del Sol dentro
    del coseno. Cada key se divide por la raíz del tamaño de su canal, de modo
    que todos los canales aporten la misma varianza.

    La versión anterior ponderaba por precisión walk-forward. La auditoría
    mostró que esa precisión no persiste (correlación -0.29 entre períodos), o
    sea que ponderar por ella era ajustar ruido.
    """
    channels = group_feature_channels(feature_keys)
    scales: dict[str, float] = {}
    for keys in channels.values():
        if not keys:
            continue
        scale = 1.0 / float(np.sqrt(len(keys)))
        for key in keys:
            scales[key] = scale

    meta = {
        "mode": "equal_channel_weight",
        "channel_count": len(channels),
        "channel_weights": {
            name: round(1.0 / max(len(channels), 1), 6) for name in sorted(channels)
        },
        "channel_sizes": {name: len(keys) for name, keys in sorted(channels.items())},
        "discarded_channels": [],
        "rationale": (
            "Pesos iguales por decisión metodológica: la ponderación por precisión "
            "histórica correlacionó -0.29 con el rendimiento posterior en la "
            "auditoría, es decir, ajustaba ruido."
        ),
    }
    return scales, meta


def _score_from_assoc(
    assoc: dict,
    target_features: dict,
    *,
    min_support: int = MIN_DRAWS_PER_BIN_VALUE,
) -> list[float]:
    """Score por bins discretos (sin fuga si assoc solo tiene el pasado).

    Solo participan los valores de bin con soporte suficiente. Un valor visto en
    40 sorteos no puede sostener ninguna afirmación sobre 25 números: su única
    contribución posible es ruido.
    """
    scores = [0.0] * (Config.POOL_SIZE + 1)
    draw_count = max(assoc["draw_count"], 1)
    base_rate = Config.DRAW_SIZE / Config.POOL_SIZE
    # Con poco historial el umbral absoluto dejaría todo fuera; se relaja de
    # forma proporcional para que el módulo siga siendo utilizable.
    effective_support = min(min_support, max(3, draw_count // 4))

    for key, value in _feature_items(target_features):
        n_feat = assoc["feature_draw_counts"][key].get(value, 0)
        if n_feat < effective_support:
            continue
        for number in range(1, Config.POOL_SIZE + 1):
            hits = assoc["feature_number_hits"][key][value].get(number, 0)
            rate = hits / n_feat
            lift = rate / base_rate
            emp = assoc["number_hits"].get(number, 0) / draw_count
            emp = max(emp, 1e-6)
            relative = rate / emp
            scores[number] += 0.5 * (lift - 1.0) + 0.5 * (relative - 1.0)

    raw = scores[1:]
    arr = np.asarray(raw, dtype=float)
    if arr.size == 0:
        return [0.0] * Config.POOL_SIZE
    centered = arr - arr.mean()
    scale = max(float(np.std(centered)), 1e-6)
    norm = 1.0 / (1.0 + np.exp(-centered / scale))
    return [float(x) for x in norm]


def _hits_matrix(draws) -> np.ndarray:
    """Matriz (n_sorteos, 25) con 1 donde el número salió."""
    matrix = np.zeros((len(draws), Config.POOL_SIZE), dtype=np.float64)
    for i, draw in enumerate(draws):
        for number in draw.numbers():
            matrix[i, int(number) - 1] = 1.0
    return matrix


def _max_abs_z_over_cells(
    coded_bins: list[np.ndarray],
    hits: np.ndarray,
    *,
    min_support: int,
) -> float:
    """Estadístico máximo |z| sobre todas las celdas (bin, valor, número)."""
    base_rate = Config.DRAW_SIZE / Config.POOL_SIZE
    sd = float(np.sqrt(base_rate * (1.0 - base_rate)))
    best = 0.0
    for column in coded_bins:
        cardinality = int(column.max()) + 1
        one_hot = np.zeros((hits.shape[0], cardinality), dtype=np.float64)
        one_hot[np.arange(hits.shape[0]), column] = 1.0
        totals = one_hot.sum(axis=0)
        usable = totals >= min_support
        if not usable.any():
            continue
        rates = (one_hot.T @ hits)[usable] / totals[usable, None]
        z = np.abs(rates - base_rate) * np.sqrt(totals[usable, None]) / sd
        local = float(z.max())
        if local > best:
            best = local
    return best


def estimate_pattern_null(
    draws,
    features_list: list[dict],
    *,
    permutations: int = PATTERN_PERMUTATIONS,
    alpha: float = PATTERN_FWER_ALPHA,
    min_support: int = MIN_DRAWS_PER_BIN_VALUE,
    seed: int = 12345,
) -> dict:
    """Nulo familiar por permutación para la búsqueda de patrones.

    Baraja qué sorteo va con qué cielo y recalcula el |z| máximo sobre TODAS las
    celdas candidatas. La distribución resultante dice qué tan grande es el
    mayor patrón que aparece cuando, por construcción, no hay ninguno.

    Es el contraste que faltaba: comparar cada celda contra la tasa base 56% de
    forma aislada ignora que se están mirando cientos de celdas a la vez, y
    garantiza encontrar "patrones" que son el máximo de un montón de dados.
    """
    if not draws or not features_list:
        return {"available": False, "reason": "sin datos"}

    bin_keys = sorted({key for feats in features_list for key, _ in _feature_items(feats)})
    if not bin_keys:
        return {"available": False, "reason": "sin bins"}

    vocabularies: list[dict[str, int]] = []
    coded_bins: list[np.ndarray] = []
    for key in bin_keys:
        vocab: dict[str, int] = {}
        column = np.empty(len(features_list), dtype=np.int64)
        for i, feats in enumerate(features_list):
            value = str(dict(_feature_items(feats)).get(key, "unknown"))
            column[i] = vocab.setdefault(value, len(vocab))
        vocabularies.append(vocab)
        coded_bins.append(column)

    hits = _hits_matrix(draws)
    effective_support = min(min_support, max(5, len(draws) // 4))

    observed = _max_abs_z_over_cells(coded_bins, hits, min_support=effective_support)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for i in range(permutations):
        null[i] = _max_abs_z_over_cells(
            coded_bins, hits[rng.permutation(hits.shape[0])], min_support=effective_support
        )

    critical = float(np.quantile(null, 1.0 - alpha))
    usable_values = sum(
        int((np.bincount(col) >= effective_support).sum()) for col in coded_bins
    )
    return {
        "available": True,
        "permutations": permutations,
        "alpha": alpha,
        "min_support": effective_support,
        "critical_abs_z": round(critical, 4),
        "observed_max_abs_z": round(observed, 4),
        "null_max_abs_z_mean": round(float(null.mean()), 4),
        "family_wise_p_value": round(
            float((np.sum(null >= observed) + 1) / (permutations + 1)), 4
        ),
        "usable_bin_values": usable_values,
        "cells_tested": usable_values * Config.POOL_SIZE,
        "null_samples": null,
    }


def _dedupe_patterns(patterns: list[dict]) -> list[dict]:
    """Conserva el mejor patrón por (número, familia de feature)."""
    kept: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for pattern in patterns:
        family = _pattern_family(str(pattern["feature"]))
        key = (int(pattern["number"]), family)
        if key in seen:
            continue
        seen.add(key)
        pattern = dict(pattern)
        pattern["feature_family"] = family
        kept.append(pattern)
    return kept


def discover_patterns_walk_forward(
    draws,
    features_list: list[dict],
    *,
    min_support: int | None = None,
    min_tests: int = 12,
    top_n: int = 25,
    lift_floor: float = 1.10,
    max_p_adj: float = 0.15,
    permutation_null: dict | None = None,
) -> list[dict]:
    """Asociaciones bin→número con walk-forward real (cuenta hits y no-hits).

    En cada sorteo, si el bin está activo y tiene soporte previo suficiente, se evalúa
    cada número con lift previo interesante y se registra el outcome aunque el número
    NO salga. Así confirmation_rate / observed_rate no quedan sesgados solo a hits.

    Dos filtros de la auditoría, en este orden:

    1. Solo participan valores de bin con al menos MIN_DRAWS_PER_BIN_VALUE casos.
       Debajo de eso la celda no tiene resolución para sostener nada.
    2. Cada candidato debe superar el nulo familiar por permutación, no la tasa
       base. Comparar contra 56% de forma aislada ignora que se miran cientos de
       celdas y produce "patrones" que son el máximo de un montón de dados.

    El FDR de Benjamini–Hochberg se conserva como filtro previo, pero ya no es el
    criterio final: manda el contraste familiar.
    """
    base_rate = Config.DRAW_SIZE / Config.POOL_SIZE
    if min_support is None:
        min_support = min(MIN_DRAWS_PER_BIN_VALUE, max(10, len(draws) // 4))
    assoc = _empty_assoc()
    accum: dict[tuple[str, str, int], dict] = {}
    mid = max(1, len(draws) // 2)

    for idx, (draw, feats) in enumerate(zip(draws, features_list)):
        numbers_set = {int(n) for n in draw.numbers()}
        if assoc["draw_count"] >= min_support:
            for key, value in _feature_items(feats):
                n_feat = assoc["feature_draw_counts"][key].get(value, 0)
                if n_feat < min_support:
                    continue
                for number in range(1, Config.POOL_SIZE + 1):
                    prior_hits = assoc["feature_number_hits"][key][value].get(number, 0)
                    prior_rate = prior_hits / n_feat
                    prior_lift = prior_rate / base_rate
                    triple = (key, value, number)
                    # Solo inicia tracking de asociaciones con lift previo elevado.
                    if prior_lift < 1.02 and triple not in accum:
                        continue
                    bucket = accum.setdefault(
                        triple,
                        {
                            "tests": 0,
                            "observed_hits": 0,
                            "prior_lift_sum": 0.0,
                            "elevated_tests": 0,
                            "elevated_hits": 0,
                            "early_tests": 0,
                            "early_hits": 0,
                            "late_tests": 0,
                            "late_hits": 0,
                        },
                    )
                    appeared = 1 if number in numbers_set else 0
                    bucket["tests"] += 1
                    bucket["observed_hits"] += appeared
                    bucket["prior_lift_sum"] += prior_lift
                    if idx < mid:
                        bucket["early_tests"] += 1
                        bucket["early_hits"] += appeared
                    else:
                        bucket["late_tests"] += 1
                        bucket["late_hits"] += appeared
                    if prior_lift >= 1.05:
                        bucket["elevated_tests"] += 1
                        bucket["elevated_hits"] += appeared
        _update_assoc(assoc, feats, list(numbers_set))

    candidates: list[dict] = []
    for (key, value, number), stats in accum.items():
        tests = int(stats["tests"])
        if tests < min_tests:
            continue
        observed_hits = int(stats["observed_hits"])
        observed_rate = observed_hits / tests
        expected_rate = base_rate
        lift = observed_rate / max(expected_rate, 1e-9)
        avg_prior_lift = float(stats["prior_lift_sum"]) / tests
        elevated_tests = int(stats["elevated_tests"])
        if elevated_tests >= 5:
            confirmation_rate = float(stats["elevated_hits"]) / elevated_tests
            confirmations = int(stats["elevated_hits"])
            confirmation_denom = elevated_tests
        else:
            confirmation_rate = observed_rate
            confirmations = observed_hits
            confirmation_denom = tests

        early_rate = (
            float(stats["early_hits"]) / stats["early_tests"]
            if stats["early_tests"] >= 5
            else None
        )
        late_rate = (
            float(stats["late_hits"]) / stats["late_tests"]
            if stats["late_tests"] >= 5
            else None
        )
        if early_rate is not None and late_rate is not None:
            stability = 1.0 - abs(early_rate - late_rate) / max(early_rate, late_rate, 1e-6)
            stability = float(max(0.0, min(1.0, stability)))
        else:
            stability = 0.5

        p_value = _binomial_pvalue_ge(observed_hits, tests, expected_rate)

        # Prefiltro anti-ruido antes de la corrección familiar.
        if lift < lift_floor and confirmation_rate < 0.62:
            continue
        if p_value > 0.15 and lift < 1.18:
            continue
        if stability < 0.30 and tests < 30:
            continue

        candidates.append(
            {
                "feature": key,
                "value": value,
                "number": number,
                "lift": float(lift),
                "avg_lift": float(avg_prior_lift),
                "observed_rate": float(observed_rate),
                "expected_rate": float(expected_rate),
                "support": tests,
                "tests": tests,
                "observed_hits": observed_hits,
                "confirmations": confirmations,
                "confirmation_tests": confirmation_denom,
                "confirmation_rate": float(confirmation_rate),
                "stability": float(stability),
                "p_value": float(p_value),
            }
        )

    null_info = permutation_null
    if null_info is None:
        null_info = estimate_pattern_null(draws, features_list, min_support=min_support)
    critical_z = (
        float(null_info.get("critical_abs_z", 0.0)) if null_info.get("available") else 0.0
    )
    null_samples = null_info.get("null_samples")
    sd_base = float(np.sqrt(base_rate * (1.0 - base_rate)))

    adj_values = _benjamini_hochberg([c["p_value"] for c in candidates])
    patterns: list[dict] = []
    for cand, p_adj in zip(candidates, adj_values):
        # Filtro estricto post-FDR (sigue siendo exploratorio, no causal).
        if p_adj > max_p_adj and cand["lift"] < 1.25:
            continue
        if p_adj > 0.25:
            continue

        # Contraste familiar: ¿supera al mayor patrón que produce el puro azar?
        abs_z = (
            abs(cand["observed_rate"] - base_rate)
            * float(np.sqrt(cand["tests"]))
            / sd_base
        )
        if null_samples is not None and len(null_samples):
            p_fwer = float(
                (np.sum(np.asarray(null_samples) >= abs_z) + 1) / (len(null_samples) + 1)
            )
        else:
            p_fwer = 1.0
        if critical_z > 0.0 and abs_z < critical_z:
            continue

        cand["abs_z"] = abs_z
        cand["p_value_family_wise"] = p_fwer
        rank_score = (
            float(cand["lift"])
            * float(cand["confirmation_rate"])
            * float(np.log1p(cand["tests"]))
            * float(max(cand["stability"], 0.15))
            * float(max(0.05, 1.0 - min(p_adj, 1.0)))
        )
        patterns.append(
            {
                "feature": cand["feature"],
                "value": cand["value"],
                "number": cand["number"],
                "lift": round(float(cand["lift"]), 4),
                "avg_lift": round(float(cand["avg_lift"]), 4),
                "observed_rate": round(float(cand["observed_rate"]), 4),
                "expected_rate": round(float(cand["expected_rate"]), 4),
                "support": cand["support"],
                "tests": cand["tests"],
                "observed_hits": cand["observed_hits"],
                "confirmations": cand["confirmations"],
                "confirmation_tests": cand["confirmation_tests"],
                "confirmation_rate": round(float(cand["confirmation_rate"]), 4),
                "stability": round(float(cand["stability"]), 4),
                "p_value": round(float(cand["p_value"]), 4),
                "p_value_adj": round(float(p_adj), 4),
                "abs_z": round(float(cand["abs_z"]), 3),
                "p_value_family_wise": round(float(cand["p_value_family_wise"]), 4),
                "fdr_method": "benjamini_hochberg + permutación familiar",
                "rank_score": round(float(rank_score), 6),
            }
        )

    patterns.sort(
        key=lambda p: (
            p["rank_score"],
            p["lift"],
            p["confirmation_rate"],
            p["support"],
            -p["p_value_adj"],
            -p["p_value"],
        ),
        reverse=True,
    )
    return _dedupe_patterns(patterns)[:top_n]


def build_number_astro_scores(
    draws,
    features_list: list[dict],
    target_features: dict,
    *,
    return_meta: bool = False,
):
    """Entrena con historial previo al target y puntúa el cielo del próximo sorteo.

    Todos los canales pesan igual y la mezcla similitud/bins es fija en 50/50.
    Nada se estima del histórico: la auditoría mostró que la "precisión" medida
    en un período no persiste en el siguiente (correlación -0.29), así que
    ajustar pesos con ella empeoraba el resultado en vez de mejorarlo.
    """
    if not draws:
        scores = [0.5] * Config.POOL_SIZE
        meta = {
            "sim_weight": SIM_BIN_BLEND,
            "bin_weight": 1.0 - SIM_BIN_BLEND,
            "channel_weights": {},
            "discarded_channels": [],
            "mode": "empty",
        }
        return (scores, meta) if return_meta else scores

    feature_keys = select_model_feature_keys(features_list)
    scales, channel_meta = build_channel_scales(feature_keys)

    sim_scores = _score_from_similarity(
        draws,
        features_list,
        target_features,
        feature_keys=feature_keys,
        scales=scales,
    )
    assoc = _empty_assoc()
    for draw, feats in zip(draws, features_list):
        _update_assoc(assoc, feats, [int(n) for n in draw.numbers()])
    bin_scores = _score_from_assoc(assoc, target_features)

    sim_weight = SIM_BIN_BLEND
    bin_weight = 1.0 - SIM_BIN_BLEND
    blended = [
        sim_weight * float(sim_scores[i]) + bin_weight * float(bin_scores[i])
        for i in range(Config.POOL_SIZE)
    ]

    usable_values = sum(
        1
        for key in assoc["feature_draw_counts"]
        for count in assoc["feature_draw_counts"][key].values()
        if count >= min(MIN_DRAWS_PER_BIN_VALUE, max(3, len(draws) // 4))
    )
    total_values = sum(
        len(values) for values in assoc["feature_draw_counts"].values()
    )
    meta = {
        **channel_meta,
        "sim_weight": round(sim_weight, 4),
        "bin_weight": round(bin_weight, 4),
        "selected_feature_count": len(feature_keys),
        "active_feature_count": len(feature_keys),
        "bin_values_usable": usable_values,
        "bin_values_total": total_values,
        "min_draws_per_bin_value": MIN_DRAWS_PER_BIN_VALUE,
    }
    return (blended, meta) if return_meta else blended


def assert_no_future_leak_on_split(
    draws,
    features_list: list[dict],
    split_index: int,
) -> list[float]:
    """Utilidad de prueba: scores en split_index usan solo draws[:split_index]."""
    if split_index <= 0 or split_index >= len(draws):
        raise ValueError("split_index fuera de rango")
    prior_draws = draws[:split_index]
    prior_feats = features_list[:split_index]
    target = features_list[split_index]
    return build_number_astro_scores(prior_draws, prior_feats, target)


def score_sample_hybrid(
    matrix: np.ndarray,
    number_astro_scores: list[float],
    *,
    sample_size: int,
    seed: int | None,
    weights: dict[str, float] | None = None,
    blend: float = ASTRO_BLEND_DEFAULT,
    top_k: int = 20,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict]:
    weights = normalize_weights(weights)
    ctx = _build_context(matrix)
    blend = float(min(1.0, max(0.0, blend)))
    by_number = {i + 1: float(number_astro_scores[i]) for i in range(Config.POOL_SIZE)}
    best: list[dict] = []
    processed = 0

    for batch in sample_combinations(sample_size, seed=seed):
        if should_cancel and should_cancel():
            break
        for row in batch:
            scored = score_combo(row, ctx, weights)
            nums = scored["numbers"]
            astro = sum(by_number[n] for n in nums) / Config.DRAW_SIZE
            hybrid = (1.0 - blend) * scored["score"] + blend * astro
            scored["astro_score"] = float(astro)
            scored["stat_score"] = float(scored["score"])
            scored["score"] = float(hybrid)
            scored["components"] = {
                **scored.get("components", {}),
                "astrology": float(astro),
                "statistical": float(scored["stat_score"]),
            }
            best.append(scored)
            processed += 1
        best.sort(key=lambda x: x["score"], reverse=True)
        best = best[: max(top_k * 3, top_k)]
        if progress_cb:
            progress_cb(
                min(99.0, 100.0 * processed / max(sample_size, 1)),
                f"{processed:,} combinaciones híbridas",
            )

    best = sorted(best, key=lambda x: x["score"], reverse=True)[:top_k]
    for i, item in enumerate(best, start=1):
        item["rank"] = i
    if progress_cb:
        progress_cb(100.0, f"Listo: top {len(best)}")
    return best


def _target_sky_summary(features: dict) -> dict:
    bodies = features["bodies"]
    lunar = features["lunar"]
    return {
        "local_datetime": features["local_datetime"],
        "moon_phase": features["moon_phase_name"],
        "moon_sign": bodies["moon"]["sign_name"],
        "sun_sign": bodies["sun"]["sign_name"],
        "planets_table": [
            {
                "body": name,
                "longitude_deg": bodies[name]["longitude_deg"],
                "sign_degree": (
                    f"{bodies[name]['sign_name']} "
                    f"{bodies[name]['degree_in_sign']:.2f}°"
                ),
                "latitude_deg": bodies[name]["latitude_deg"],
                "declination_deg": bodies[name]["declination_deg"],
                "speed_longitude_deg_per_day": bodies[name][
                    "speed_longitude_deg_per_day"
                ],
                "distance_au": bodies[name]["distance_au"],
                "distance_km": bodies[name].get("distance_km"),
                "motion_state": bodies[name]["motion_state"],
                "longitude_sin": bodies[name]["longitude_sin"],
                "longitude_cos": bodies[name]["longitude_cos"],
            }
            for name in PLANET_ORDER
        ],
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
            "true_node": {
                "longitude_deg": lunar["true_node"]["longitude_deg"],
                "sign_degree": (
                    f"{lunar['true_node']['sign_name']} "
                    f"{lunar['true_node']['degree_in_sign']:.2f}°"
                ),
                "longitude_sin": lunar["true_node"]["longitude_sin"],
                "longitude_cos": lunar["true_node"]["longitude_cos"],
            },
            "mean_node": {
                "longitude_deg": lunar["mean_node"]["longitude_deg"],
                "sign_degree": (
                    f"{lunar['mean_node']['sign_name']} "
                    f"{lunar['mean_node']['degree_in_sign']:.2f}°"
                ),
                "longitude_sin": lunar["mean_node"]["longitude_sin"],
                "longitude_cos": lunar["mean_node"]["longitude_cos"],
            },
            "separations_to_planets_deg": lunar["separations_to_planets_deg"],
            "lunations": lunar["lunations"],
        },
        "units": features["units"],
    }


def run_astrology_experiment(
    draws,
    *,
    now: datetime | None = None,
    target_dt: datetime | None = None,
    history_sync: dict | None = None,
    sample_size: int = 20_000,
    seed: int = 42,
    blend: float = ASTRO_BLEND_DEFAULT,
    weights: dict[str, float] | None = None,
    top_k: int = 20,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[dict, list[dict]]:
    target = target_dt or next_draw_datetime(now)
    target = _as_local_datetime(target)
    ordered = prior_draws_before_target(draws, target)
    if len(ordered) < 20:
        raise ValueError(
            "Se necesitan al menos 20 sorteos históricos anteriores al próximo sorteo"
        )

    if progress_cb:
        progress_cb(2.0, "Calculando efemérides planetarias/lunares")

    features_list: list[dict] = []
    total = len(ordered)
    for i, draw in enumerate(ordered):
        if should_cancel and should_cancel():
            break
        features_list.append(compute_astro_features(draw.draw_date))
        if progress_cb and i % 50 == 0:
            progress_cb(2.0 + 40.0 * i / max(total, 1), f"Efemérides {i + 1}/{total}")

    if should_cancel and should_cancel():
        return {"cancelled": True}, []

    if progress_cb:
        progress_cb(42.0, "Calculando nulo por permutación")
    pattern_null = estimate_pattern_null(ordered, features_list)

    if progress_cb:
        progress_cb(48.0, "Buscando patrones walk-forward")
    patterns = discover_patterns_walk_forward(
        ordered, features_list, permutation_null=pattern_null
    )
    # El array de permutaciones es interno; no viaja al reporte.
    pattern_null_public = {k: v for k, v in pattern_null.items() if k != "null_samples"}

    target_features = compute_astro_features(target)
    number_scores, score_meta = build_number_astro_scores(
        ordered, features_list, target_features, return_meta=True
    )
    # Blend fijo: ajustarlo a la "precisión histórica" era ajustar ruido.
    effective_blend = float(min(1.0, max(0.0, blend)))

    if progress_cb:
        progress_cb(55.0, "Combinando con modelo estadístico")

    matrix = draws_matrix(ordered)

    def combo_progress(pct: float, message: str = "") -> None:
        if progress_cb:
            progress_cb(55.0 + 0.45 * pct, message)

    results = score_sample_hybrid(
        matrix,
        number_scores,
        sample_size=sample_size,
        seed=seed,
        weights=weights,
        blend=effective_blend,
        top_k=top_k,
        progress_cb=combo_progress,
        should_cancel=should_cancel,
    )

    ranked_numbers = sorted(
        (
            {"number": n, "astro_score": round(number_scores[n - 1], 4)}
            for n in range(1, Config.POOL_SIZE + 1)
        ),
        key=lambda x: x["astro_score"],
        reverse=True,
    )

    sync = history_sync or {}
    metrics = {
        "disclaimer": (
            "Módulo experimental (planetas + Luna). Los patrones son exploratorios y "
            "descriptivos: no demuestran causalidad ni garantizan resultados. Un mayor "
            "score del modelo no implica mayor probabilidad real; si el sorteo es uniforme, "
            "todas las combinaciones son equiprobables. Longitudes crudas solo se muestran "
            "en UI; el modelo usa sin/cos y variables numéricas escaladas/filtradas. "
            "Todos los canales pesan igual: una auditoría sobre 2.456 sorteos mostró que "
            "ponderarlos por su precisión histórica ajustaba ruido."
        ),
        "audit_note": AUDIT_SUMMARY,
        "pattern_method": (
            "walk-forward con tests que cuentan aparición y no-aparición; "
            "solo valores de bin con soporte suficiente; contraste familiar por "
            "permutación (Westfall-Young sobre el |z| máximo) además del FDR "
            "Benjamini–Hochberg; dedupe por familia de feature"
        ),
        "score_method": (
            "similitud coseno con peso igual por canal (normalizado por tamaño de canal); "
            "mezcla fija 50/50 similitud/bins; blend astro/estadístico fijo"
        ),
        "adaptive_weights": score_meta,
        "pattern_null": pattern_null_public,
        "assumptions": {
            "draw_time_local": f"{DRAW_HOUR:02d}:{DRAW_MINUTE:02d}",
            "timezone": Config.ASTRO_TIMEZONE,
            "location": ASTRO_LOCATION_NAME,
            "lat": ASTRO_LAT,
            "lon": ASTRO_LON,
            "ephemeris": "Swiss Ephemeris Moshier (sin archivos externos)",
            "draw_weekdays": ["miércoles", "viernes", "domingo"],
            "stationary_threshold_deg_per_day": STATIONARY_THRESHOLD_DEG_PER_DAY,
            "distance_unit": "AU (Luna también en km)",
            "bodies": PLANET_ORDER,
            "channel_weighting": "igual por canal (no adaptativo)",
            "min_draws_per_bin_value": MIN_DRAWS_PER_BIN_VALUE,
            "pattern_permutations": PATTERN_PERMUTATIONS,
        },
        "target_draw_datetime": target.isoformat(),
        "target_draw_date": target.date().isoformat(),
        "target_draw_time": f"{DRAW_HOUR:02d}:{DRAW_MINUTE:02d}",
        "timezone": Config.ASTRO_TIMEZONE,
        "target_draw_label": format_draw_label(target),
        "history_as_of": ordered[-1].draw_date.isoformat(),
        "remote_latest_date": sync.get("remote_latest_date"),
        "remote_latest_draw_number": sync.get("remote_latest_draw_number"),
        "history_freshness": sync.get("history_freshness", "already_current"),
        "history_sync_message": sync.get("message"),
        "experiment_date": target.date().isoformat(),
        "draw_count": len(ordered),
        "blend": effective_blend,
        "blend_requested": blend,
        "blend_is_fixed": True,
        "sample_size": sample_size,
        "seed": seed,
        "selected_feature_count": score_meta.get(
            "selected_feature_count", len(select_model_feature_keys(features_list))
        ),
        "active_feature_count": score_meta.get("active_feature_count"),
        "target_sky": _target_sky_summary(target_features),
        "target_model_feature_count": len(target_features.get("model_features") or {}),
        "number_scores": ranked_numbers,
        "patterns": patterns,
        "top_k": len(results),
        "candidate_count": sample_size,
    }
    if progress_cb:
        progress_cb(100.0, "Predicción astrológica experimental lista")
    return metrics, results
