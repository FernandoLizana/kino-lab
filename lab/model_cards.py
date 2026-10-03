from __future__ import annotations

CARDS = [
    {
        "id": "statistical_seed",
        "name": "Semilla estadística SHA-256",
        "objective": "Combinaciones deterministas a partir del histórico local.",
        "inputs": "Frecuencias, atrasos, últimos 20 sorteos.",
        "training": "No entrena; hashea el estado del histórico.",
        "output_type": "puntuacion / ranking",
        "not": "No es P(ganar) ni la semilla del operador.",
        "baseline": "Referencia uniforme: cualquier boleto válido tiene la misma P(pleno).",
        "limitations": "Reproduce una huella del pasado. No demuestra ventaja.",
        "available": True,
        "optional_ml": False,
    },
    {
        "id": "frequency_rank",
        "name": "Ranking por frecuencia",
        "objective": "Educativo: ilustrar una tabla de conteos por número.",
        "inputs": "Conteos 1–N en el período de entrenamiento permitido.",
        "training": "Solo sorteos anteriores al corte.",
        "output_type": "puntuacion_marginal",
        "not": "No multiplicar marginales como si las bolillas fueran independientes.",
        "baseline": "Uniforme (esperanza k/N por número).",
        "limitations": "Un buen ajuste in-sample no es capacidad predictiva.",
        "available": True,
        "optional_ml": False,
    },
    {
        "id": "legacy_random_forest",
        "name": "Random Forest legado",
        "objective": "Script histórico generar_data_random_forest*.py",
        "inputs": "Features agregadas de frecuencia (no auditadas aquí).",
        "training": "Desconocido / sin partición temporal en el script original.",
        "output_type": "puntuacion",
        "not": "No integrado al producto. Dependencias ML opcionales no instaladas por defecto.",
        "baseline": "No evaluado en este motor.",
        "limitations": "No disponible. Causa: extra de ML no cargado y script desconectado.",
        "available": False,
        "optional_ml": True,
        "unavailable_reason": "Módulo legado fuera de requirements.txt. El resto del laboratorio funciona sin él.",
    },
]


def list_cards() -> list[dict]:
    return CARDS


def get_card(cid: str) -> dict:
    for card in CARDS:
        if card["id"] == cid:
            return card
    raise KeyError(cid)
