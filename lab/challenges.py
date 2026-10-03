from __future__ import annotations

CHALLENGES = [
    {
        "id": "independencia",
        "title": "¿El siguiente sorteo 'debe' compensar?",
        "prompt": "Salieron muchos impares. ¿El próximo sorteo tiene más chance de pares?",
        "options": ["Sí, para equilibrar", "No, si cada sorteo es independiente"],
        "answer": 1,
        "explain": (
            "Si cada sorteo es independiente y uniforme, el pasado no cambia "
            "las probabilidades del siguiente. La sensación de 'compensar' es la falacia del jugador."
        ),
        "demo": "bias_uniform_vs_small_n",
    },
    {
        "id": "rachas",
        "title": "Rachas",
        "prompt": "Un número no sale en 40 sorteos. ¿Está 'atrasado' de forma útil?",
        "options": ["Sí, hay que jugarlo", "Es normal; el atraso no predice"],
        "answer": 1,
        "explain": (
            "Con 25 números y 14 extraídos, algunos atrasos largos ocurren por azar. "
            "El atraso describe el pasado; no modifica P(salga mañana)."
        ),
        "demo": "explore_gaps",
    },
    {
        "id": "muestra",
        "title": "Tamaño de muestra",
        "prompt": "En 20 sorteos sintéticos un número salió 16 veces. ¿El bombo está cargado?",
        "options": ["Sí, es evidencia fuerte", "Aún no: 20 sorteos mienten mucho"],
        "answer": 1,
        "explain": (
            "Muestras chicas fabrican 'ganadores'. El laboratorio de sesgos muestra "
            "cómo el mismo generador uniforme se ve raro con n=20 y aburrido con n=2000."
        ),
        "demo": "bias_lab",
    },
    {
        "id": "frecuencias",
        "title": "Frecuencias históricas",
        "prompt": "Jugar los 14 más frecuentes del año pasado, ¿sube P(pleno)?",
        "options": ["Sí", "No: cada combinación válida sigue equiprobable"],
        "answer": 1,
        "explain": (
            "En un sorteo uniforme, P(pleno) = 1/C(N,k) para cualquier boleto válido. "
            "Filtrar por frecuencia cambia qué boleto eliges, no esa probabilidad."
        ),
        "demo": "probability",
    },
    {
        "id": "coincidencias",
        "title": "Coincidencias dentro del sorteo",
        "prompt": "Si ya salió el 1, ¿cambia la chance del 2 en el MISMO sorteo?",
        "options": ["No cambia", "Sí: sin reposición, las bolillas dependen"],
        "answer": 1,
        "explain": (
            "Independencia entre sorteos ≠ independencia entre extracciones. "
            "Sin reposición, sacar un número reduce el universo del mismo sorteo."
        ),
        "demo": "probability",
    },
    {
        "id": "incertidumbre",
        "title": "Ceros en la simulación",
        "prompt": "Simulé 1.000 sorteos y nunca vi 14 aciertos. ¿Es imposible?",
        "options": ["Sí, probabilidad 0", "No: 0 observados ≠ probabilidad 0"],
        "answer": 1,
        "explain": (
            "P(14 aciertos) = 1/4.457.400. En 1.000 simulaciones casi nunca aparece. "
            "La app etiqueta '0 casos observados' sin apagar la probabilidad teórica."
        ),
        "demo": "simulate",
    },
    {
        "id": "comparar",
        "title": "Comparar estrategias",
        "prompt": "Una estrategia ganó en 80 sorteos y perdió en 80. ¿Qué debes ver?",
        "options": ["Solo el tramo bueno", "Ambos tramos y la referencia uniforme"],
        "answer": 1,
        "explain": (
            "Elegir el tramo favorable es sobreajuste. El comparador muestra la "
            "referencia uniforme y los períodos malos."
        ),
        "demo": "compare",
    },
]


def get_challenge(cid: str) -> dict:
    for item in CHALLENGES:
        if item["id"] == cid:
            return item
    raise KeyError(cid)


def grade(cid: str, choice: int) -> dict:
    item = get_challenge(cid)
    ok = int(choice) == int(item["answer"])
    return {
        "id": cid,
        "correct": ok,
        "explain": item["explain"],
        "demo": item["demo"],
    }
