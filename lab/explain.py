from __future__ import annotations

from typing import Any


def explain(question: str, context: dict[str, Any]) -> dict:
    """Respuestas locales por plantilla. No inventa cifras ausentes."""
    q = (question or "").strip().lower()
    kind = context.get("kind", "")
    missing = []

    def need(*keys):
        for key in keys:
            if key not in context:
                missing.append(key)

    if "significa" in q or "gráfico" in q:
        if kind == "calculo_teorico":
            return _ok(
                "Es un cálculo teórico (hipergeométrica), no un histórico ni una predicción. "
                f"Cada barra es P(obtener exactamente j aciertos) con N={context.get('n')}, "
                f"k={context.get('k')}, r={context.get('r')}. La suma debe ser ≈ 1."
            )
        if kind == "simulacion":
            need("n_completed", "hits")
            if missing:
                return _lack(missing)
            return _ok(
                f"Compara frecuencias observadas en {context.get('n_completed')} sorteos "
                "simulados contra la probabilidad teórica. Si un evento raro sale 0 veces, "
                "su probabilidad teórica no se vuelve cero."
            )
        if kind == "datos_historicos":
            return _ok(
                "Describe lo que ya ocurrió en las filas filtradas. No cambia la "
                "probabilidad del próximo sorteo si el mecanismo es uniforme."
            )
        return _ok(
            "No hay un gráfico asociado a esta vista. Elige Probabilidades, Simular "
            "o Explorar y vuelve a preguntar."
        )

    if "cambia" in q or "ejecuciones" in q or "semilla" in q:
        return _ok(
            "Otra semilla produce otra muestra. La media debería acercarse a la "
            "teoría al subir n; la dispersión entre semillas estima incertidumbre. "
            "No uses una sola corrida afortunada como prueba."
        )

    if "concluir" in q or "compar" in q:
        if kind == "resultado_experimental":
            ref = context.get("results", {}).get("uniform", {})
            other = {k: v for k, v in context.get("results", {}).items() if k != "uniform"}
            if not ref:
                return _lack(["results.uniform"])
            bits = [
                f"Referencia uniforme: media de mejores aciertos = {ref.get('mean_best_hits')} "
                f"en {ref.get('evaluated')} sorteos."
            ]
            for name, payload in other.items():
                bits.append(
                    f"{name}: media = {payload.get('mean_best_hits')} "
                    f"(misma cantidad de boletos y fechas)."
                )
            bits.append(
                "Si la diferencia es chica o cambia de signo por período, no hay "
                "ventaja demostrada. El tramo final de holdout es el único que cuenta "
                "como prueba."
            )
            return _ok(" ".join(bits))
        return _ok(
            "Para concluir una comparación necesitas referencia uniforme, mismas "
            "fechas y la misma cantidad de boletos. Aquí no está ese experimento."
        )

    if "falta" in q or "evidencia" in q:
        return _ok(
            "Falta evidencia si no hay período de prueba, si se usó el futuro para "
            "elegir la estrategia, o si solo se muestra el tramo favorable."
        )

    return _ok(
        "Preguntas que sí puedo responder sin red: qué significa el gráfico, "
        "por qué cambia entre ejecuciones, y qué se puede concluir. "
        "Los números salen del motor, no de este texto."
    )


def _ok(text: str) -> dict:
    return {"source": "plantilla_local", "text": text, "online": False}


def _lack(keys: list[str]) -> dict:
    return {
        "source": "plantilla_local",
        "text": "No hay evidencia suficiente para esa pregunta. Falta: " + ", ".join(keys) + ".",
        "online": False,
        "missing": keys,
    }
