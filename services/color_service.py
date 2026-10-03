"""Cada combinación 14/25 (o 15/25) se pinta de un color.

No es un oráculo: es una traducción geométrica. Cada número 1–25 ocupa un
arco de 14.4° en el círculo cromático. La combinación es el promedio circular
de esos arcos (tono) más qué tan juntos están (saturación).

También se parte el 1–25 en cinco pigmentos de 5 números. El color dominante
es el pigmento con más bolillas en ese sorteo. Eso ya tiene distribución
exacta bajo un sorteo justo (hipergeométrica multivariada): cada pigmento
espera 14×5/25 = 2.8 números.

Si los sorteos reales se agrupan en un tono, o un pigmento gana más de lo
que esa cuenta permite, hay patrón. Si no, el color es solo una etiqueta
bonita de la combinación.
"""

from __future__ import annotations

from colorsys import hsv_to_rgb
from typing import Callable

import numpy as np

from config import Config
from services.combination_service import features_for_combo, sample_combinations
from services.lunar_service import GEOMETRIC_TIEBREAK_WEIGHT, _geometric_tiebreak

PIGMENTS: tuple[tuple[str, range, str], ...] = (
    ("rojo", range(1, 6), "#e23d3d"),
    ("ámbar", range(6, 11), "#f0a202"),
    ("verde", range(11, 16), "#2f9e44"),
    ("azul", range(16, 21), "#3d8bfd"),
    ("violeta", range(21, 26), "#9b5de5"),
)
PIGMENT_NAMES = tuple(name for name, _range, _hex in PIGMENTS)
NUMBER_TO_PIGMENT = {
    n: name for name, span, _hex in PIGMENTS for n in span
}
HUE_STEP_DEG = 360.0 / Config.POOL_SIZE


def number_hue_deg(number: int) -> float:
    return float((int(number) - 1) % Config.POOL_SIZE) * HUE_STEP_DEG


def combo_color(numbers: list[int] | np.ndarray) -> dict:
    """Color de una combinación: tono circular + saturación + pigmento dominante."""
    nums = [int(n) for n in numbers]
    if not nums:
        raise ValueError("combinación vacía")

    hues = np.asarray([number_hue_deg(n) for n in nums], dtype=float)
    rads = np.deg2rad(hues)
    mean_sin = float(np.sin(rads).mean())
    mean_cos = float(np.cos(rads).mean())
    hue = float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0)
    # R ∈ [0, 1]: 1 = todos los números del mismo arco; ~0 = repartidos.
    radius = float(np.hypot(mean_sin, mean_cos))
    saturation = float(min(1.0, max(0.0, radius)))
    # Claridad según la media de los números (bajos=oscuros, altos=claros).
    lightness = float(0.28 + 0.50 * ((np.mean(nums) - 1.0) / (Config.POOL_SIZE - 1)))

    counts = {name: 0 for name in PIGMENT_NAMES}
    for n in nums:
        counts[NUMBER_TO_PIGMENT[n]] += 1
    dominant = max(PIGMENT_NAMES, key=lambda name: (counts[name], -PIGMENT_NAMES.index(name)))
    ties = [name for name in PIGMENT_NAMES if counts[name] == counts[dominant]]
    mixed = len(ties) > 1

    r, g, b = hsv_to_rgb(hue / 360.0, 0.35 + 0.55 * saturation, 0.45 + 0.45 * lightness)
    hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    return {
        "hue_deg": round(hue, 3),
        "saturation": round(saturation, 4),
        "lightness": round(lightness, 4),
        "hex": hex_color,
        "pigment_counts": counts,
        "dominant_pigment": dominant,
        "dominant_is_tie": mixed,
        "hue_bin_12": int(hue // 30.0) % 12,
    }


def colors_for_draws(number_rows: list[list[int]]) -> list[dict]:
    return [combo_color(row) for row in number_rows]


def _hue_to_hex(hue_deg: float, saturation: float = 0.75, value: float = 0.72) -> str:
    r, g, b = hsv_to_rgb((hue_deg % 360.0) / 360.0, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def build_color_arc_payload(draws) -> dict:
    """Datos para el aro de colores del sitio."""
    number_marks = []
    for n in range(1, Config.POOL_SIZE + 1):
        hue = number_hue_deg(n)
        pigment = NUMBER_TO_PIGMENT[n]
        pigment_hex = next(hex_ for name, _span, hex_ in PIGMENTS if name == pigment)
        number_marks.append(
            {
                "number": n,
                "hue_deg": round(hue, 3),
                "hex": _hue_to_hex(hue),
                "pigment": pigment,
                "pigment_hex": pigment_hex,
            }
        )

    points: list[dict] = []
    pigment_totals = {name: 0 for name in PIGMENT_NAMES}
    for draw in draws:
        nums = list(draw.numbers()) if hasattr(draw, "numbers") else list(draw["numbers"])
        color = combo_color(nums)
        draw_date = draw.draw_date if hasattr(draw, "draw_date") else draw.get("draw_date")
        points.append(
            {
                "date": draw_date.isoformat() if hasattr(draw_date, "isoformat") else str(draw_date),
                "numbers": nums,
                "hue_deg": color["hue_deg"],
                "saturation": color["saturation"],
                "hex": color["hex"],
                "dominant": color["dominant_pigment"],
                "pigment_counts": color["pigment_counts"],
            }
        )
        pigment_totals[color["dominant_pigment"]] += 1

    last = points[-1] if points else None
    return {
        "draw_count": len(points),
        "numbers": number_marks,
        "pigments": [
            {
                "name": name,
                "hex": hex_,
                "numbers": f"{min(span)}–{max(span)}",
                "dominant_count": pigment_totals[name],
            }
            for name, span, hex_ in PIGMENTS
        ],
        "points": points,
        "last": last,
        "note": (
            "El aro es el círculo cromático de los 25 números. Cada punto es un "
            "sorteo, colocado en el tono de su combinación. Más afuera = más reciente. "
            "No hay patrón estadístico: el color es una etiqueta del conjunto."
        ),
    }


def circular_hue_distance(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def resolve_target_hue(
    draws,
    *,
    mode: str = "continue",
    target_hue: float | None = None,
) -> dict:
    """Elige el tono objetivo del generador. No es una predicción del cielo."""
    if draws:
        last_nums = list(draws[-1].numbers()) if hasattr(draws[-1], "numbers") else list(draws[-1]["numbers"])
        last_color = combo_color(last_nums)
        last_hue = float(last_color["hue_deg"])
    else:
        last_color = None
        last_hue = 0.0

    mode = (mode or "continue").strip().lower()
    if mode == "pick" and target_hue is not None:
        hue = float(target_hue) % 360.0
        rationale = "tono elegido en el aro"
    elif mode == "complement":
        hue = (last_hue + 180.0) % 360.0
        rationale = "complemento del último sorteo (opuesto en el aro)"
    else:
        hue = last_hue
        rationale = "continuar el tono del último sorteo"
        mode = "continue"

    return {
        "mode": mode,
        "hue_deg": round(hue, 3),
        "hex": _hue_to_hex(hue),
        "last_hue_deg": round(last_hue, 3),
        "rationale": rationale,
        "last_color": last_color,
    }


def run_color_experiment(
    draws,
    *,
    mode: str = "continue",
    target_hue: float | None = None,
    sample_size: int = 20_000,
    seed: int = 42,
    top_k: int = 20,
    progress_cb: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[dict, list[dict]]:
    """Genera combinaciones cercanas a un tono. No cambia la equiprobabilidad."""
    from services.astrology_service import format_draw_label, next_draw_datetime

    target = resolve_target_hue(draws, mode=mode, target_hue=target_hue)
    hue = float(target["hue_deg"])
    if progress_cb:
        progress_cb(5.0, f"Buscando combinaciones cerca de {hue:.0f}°")

    best: list[dict] = []
    processed = 0
    for batch in sample_combinations(sample_size, seed=seed):
        if should_cancel and should_cancel():
            break
        for row in batch:
            feats = features_for_combo(row)
            nums = feats["numbers"]
            color = combo_color(nums)
            dist = circular_hue_distance(color["hue_deg"], hue)
            color_score = 1.0 - dist / 180.0
            tie = _geometric_tiebreak(feats)
            total = float(color_score + GEOMETRIC_TIEBREAK_WEIGHT * tie)
            best.append(
                {
                    **feats,
                    "color_score": float(color_score),
                    "hue_distance_deg": round(dist, 3),
                    "score": total,
                    "hex": color["hex"],
                    "hue_deg": color["hue_deg"],
                    "dominant": color["dominant_pigment"],
                    "saturation": color["saturation"],
                    "components": {
                        "color": float(color_score),
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
                f"{processed:,} combinaciones de color",
            )

    results = sorted(best, key=lambda x: x["score"], reverse=True)[:top_k]
    for i, item in enumerate(results, start=1):
        item["rank"] = i

    top = results[0] if results else None
    next_dt = next_draw_datetime()
    metrics = {
        "mode": "color",
        "target": target,
        "prediction": {
            "numbers": top["numbers"] if top else [],
            "numbers_csv": top["numbers_csv"] if top else "",
            "hex": top["hex"] if top else target["hex"],
            "hue_deg": top["hue_deg"] if top else target["hue_deg"],
            "dominant": top["dominant"] if top else None,
            "saturation": top["saturation"] if top else None,
            "score": top["score"] if top else None,
            "hue_distance_deg": top["hue_distance_deg"] if top else None,
        } if top else None,
        "disclaimer": (
            "Generador de color: busca combinaciones cuyo tono se parece al objetivo. "
            "El histórico no mostró que el color prediga el Kino (el tono de un sorteo "
            "no se parece al del siguiente). Un mayor color_score no es probabilidad."
        ),
        "target_draw_label": format_draw_label(next_dt),
        "target_draw_date": next_dt.date().isoformat(),
        "draw_count": len(draws),
        "sample_size": sample_size,
        "seed": seed,
        "top_k": len(results),
    }
    if progress_cb:
        progress_cb(100.0, "Predicción de color lista")
    return metrics, results
