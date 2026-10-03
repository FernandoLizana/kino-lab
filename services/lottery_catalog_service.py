from __future__ import annotations

import json
from typing import Any

from models import LotteryGame, db

# Reglas verificadas a julio 2026 (fuentes oficiales / operadores).
# sync_adapter: solo cuando hay API pública estable (NY Open Data).
# Resto: importación CSV manual desde fuentes oficiales.

CATALOG_SEED: list[dict[str, Any]] = [
    {
        "slug": "powerball",
        "name": "Powerball",
        "country": "Estados Unidos (multi-estado)",
        "sort_order": 1,
        "official_url": "https://www.powerball.com/",
        "history_url": "https://data.ny.gov/resource/d6yy-54nr.json",
        "sync_adapter": "ny_powerball",
        "rules": {
            "main": {"count": 5, "min": 1, "max": 69, "name": "White balls"},
            "bonus": [
                {
                    "count": 1,
                    "min": 1,
                    "max": 26,
                    "name": "Powerball",
                    "pool": "separate",
                }
            ],
            "draw_days": ["monday", "wednesday", "saturday"],
            "currency": "USD",
            "ticket_price_hint": 2,
            "rules_effective": "2015-10-07",
            "rules_verified": "2026-07",
            "notes": (
                "Matriz 5/69 + Powerball 1/26. Sorteos lun/mié/sáb ~22:59 ET. "
                "Power Play es multiplicador opcional (no es bola del juego base). "
                "Fuente histórica: NY Open Data (SODA)."
            ),
            "sources": [
                "https://www.powerball.com/",
                "https://data.ny.gov/d/d6yy-54nr",
            ],
        },
    },
    {
        "slug": "mega-millions",
        "name": "Mega Millions",
        "country": "Estados Unidos (multi-estado)",
        "sort_order": 2,
        "official_url": "https://www.megamillions.com/",
        "history_url": "https://data.ny.gov/resource/5xaw-6ayf.json",
        "sync_adapter": "ny_mega_millions",
        "rules": {
            "main": {"count": 5, "min": 1, "max": 70, "name": "White balls"},
            "bonus": [
                {
                    "count": 1,
                    "min": 1,
                    "max": 24,
                    "name": "Mega Ball",
                    "pool": "separate",
                }
            ],
            "draw_days": ["tuesday", "friday"],
            "currency": "USD",
            "ticket_price_hint": 5,
            "rules_effective": "2025-04-08",
            "rules_verified": "2026-07",
            "notes": (
                "Cambio de reglas abril 2025: matriz 5/70 + Mega Ball 1/24 "
                "(antes Mega Ball 1-25), boleto $5, multiplicador integrado 2x–10x "
                "(no es bola elegible). Primer sorteo nueva matriz: 2025-04-08. "
                "Históricos previos al cambio pueden tener Mega Ball 25."
            ),
            "legacy_bonus_max": 25,
            "legacy_until": "2025-04-07",
            "sources": [
                "https://www.megamillions.com/",
                "https://data.ny.gov/d/5xaw-6ayf",
            ],
        },
    },
    {
        "slug": "euromillions",
        "name": "EuroMillions",
        "country": "Europa (multi-país)",
        "sort_order": 3,
        "official_url": "https://www.euro-millions.com/",
        "history_url": None,
        "sync_adapter": None,
        "rules": {
            "main": {"count": 5, "min": 1, "max": 50, "name": "Main numbers"},
            "bonus": [
                {
                    "count": 2,
                    "min": 1,
                    "max": 12,
                    "name": "Lucky Stars",
                    "pool": "separate",
                }
            ],
            "draw_days": ["tuesday", "friday"],
            "currency": "EUR",
            "ticket_price_hint": None,
            "rules_effective": "2016-09-24",
            "rules_verified": "2026-07",
            "notes": (
                "5/50 + 2 Lucky Stars 1-12. Sin API pública unificada estable; "
                "importar CSV desde el operador nacional (p. ej. National Lottery UK, FDJ, etc.)."
            ),
            "sources": [
                "https://www.euro-millions.com/",
                "https://www.national-lottery.co.uk/games/euromillions",
            ],
        },
    },
    {
        "slug": "eurojackpot",
        "name": "EuroJackpot",
        "country": "Europa (multi-país)",
        "sort_order": 4,
        "official_url": "https://www.eurojackpot.com/",
        "history_url": None,
        "sync_adapter": None,
        "rules": {
            "main": {"count": 5, "min": 1, "max": 50, "name": "Main numbers"},
            "bonus": [
                {
                    "count": 2,
                    "min": 1,
                    "max": 12,
                    "name": "Euro numbers",
                    "pool": "separate",
                }
            ],
            "draw_days": ["tuesday", "friday"],
            "currency": "EUR",
            "ticket_price_hint": 2,
            "rules_effective": "2022-03-25",
            "rules_verified": "2026-07",
            "notes": (
                "5/50 + 2 Euro numbers 1-12 (pools independientes; puede repetirse "
                "el mismo dígito en main y euro). CSV manual desde operadores participantes."
            ),
            "sources": ["https://www.eurojackpot.com/"],
        },
    },
    {
        "slug": "uk-lotto",
        "name": "UK Lotto",
        "country": "Reino Unido",
        "sort_order": 5,
        "official_url": "https://www.national-lottery.co.uk/games/lotto",
        "history_url": None,
        "sync_adapter": None,
        "rules": {
            "main": {"count": 6, "min": 1, "max": 59, "name": "Main numbers"},
            "bonus": [
                {
                    "count": 1,
                    "min": 1,
                    "max": 59,
                    "name": "Bonus Ball",
                    "pool": "same_remaining",
                }
            ],
            "draw_days": ["wednesday", "saturday"],
            "currency": "GBP",
            "ticket_price_hint": 2,
            "rules_effective": "2015-10-10",
            "rules_verified": "2026-07",
            "notes": (
                "6/59 + Bonus Ball del resto del mismo bombo (no se elige; "
                "sirve para ciertos tramos). CSV manual desde National Lottery UK."
            ),
            "sources": ["https://www.national-lottery.co.uk/games/lotto"],
        },
    },
    {
        "slug": "thunderball",
        "name": "Thunderball",
        "country": "Reino Unido",
        "sort_order": 6,
        "official_url": "https://www.national-lottery.co.uk/games/thunderball",
        "history_url": None,
        "sync_adapter": None,
        "rules": {
            "main": {"count": 5, "min": 1, "max": 39, "name": "Main numbers"},
            "bonus": [
                {
                    "count": 1,
                    "min": 1,
                    "max": 14,
                    "name": "Thunderball",
                    "pool": "separate",
                }
            ],
            "draw_days": ["tuesday", "wednesday", "friday", "saturday"],
            "currency": "GBP",
            "ticket_price_hint": 1,
            "rules_effective": "2010-05-12",
            "rules_verified": "2026-07",
            "notes": "5/39 + Thunderball 1-14. Cuatro sorteos semanales. CSV manual.",
            "sources": ["https://www.national-lottery.co.uk/games/thunderball"],
        },
    },
    {
        "slug": "set-for-life",
        "name": "Set For Life",
        "country": "Reino Unido",
        "sort_order": 7,
        "official_url": "https://www.national-lottery.co.uk/games/set-for-life",
        "history_url": None,
        "sync_adapter": None,
        "rules": {
            "main": {"count": 5, "min": 1, "max": 47, "name": "Main numbers"},
            "bonus": [
                {
                    "count": 1,
                    "min": 1,
                    "max": 10,
                    "name": "Life Ball",
                    "pool": "separate",
                }
            ],
            "draw_days": ["monday", "thursday"],
            "currency": "GBP",
            "ticket_price_hint": 1.5,
            "rules_effective": "2019-03-18",
            "rules_verified": "2026-07",
            "notes": "5/47 + Life Ball 1-10. Lun/jue. CSV manual.",
            "sources": ["https://www.national-lottery.co.uk/games/set-for-life"],
        },
    },
    {
        "slug": "irish-lotto",
        "name": "Irish Lotto",
        "country": "Irlanda",
        "sort_order": 8,
        "official_url": "https://www.lottery.ie/draw-games/lotto",
        "history_url": None,
        "sync_adapter": None,
        "rules": {
            "main": {"count": 6, "min": 1, "max": 47, "name": "Main numbers"},
            "bonus": [
                {
                    "count": 1,
                    "min": 1,
                    "max": 47,
                    "name": "Bonus Ball",
                    "pool": "same_remaining",
                }
            ],
            "draw_days": ["wednesday", "saturday"],
            "currency": "EUR",
            "ticket_price_hint": None,
            "rules_effective": "2015-09-05",
            "rules_verified": "2026-07",
            "notes": (
                "6/47 + Bonus Ball del resto del bombo. CSV manual desde lottery.ie."
            ),
            "sources": ["https://www.lottery.ie/draw-games/lotto"],
        },
    },
    {
        "slug": "eurodreams",
        "name": "EuroDreams",
        "country": "Europa (multi-país)",
        "sort_order": 9,
        "official_url": "https://www.lottery.ie/game-information/eurodreams",
        "history_url": None,
        "sync_adapter": None,
        "rules": {
            "main": {"count": 6, "min": 1, "max": 40, "name": "Main numbers"},
            "bonus": [
                {
                    "count": 1,
                    "min": 1,
                    "max": 5,
                    "name": "Dream Number",
                    "pool": "separate",
                }
            ],
            "draw_days": ["monday", "thursday"],
            "currency": "EUR",
            "ticket_price_hint": None,
            "rules_effective": "2023-11-06",
            "rules_verified": "2026-07",
            "notes": (
                "6/40 + Dream Number 1-5. Premio principal tipo renta. "
                "CSV manual desde operadores participantes."
            ),
            "sources": [
                "https://www.lottery.ie/game-information/eurodreams",
            ],
        },
    },
    {
        "slug": "loto-chile",
        "name": "Loto Chile",
        "country": "Chile",
        "sort_order": 10,
        "official_url": "https://www.pollachilena.cl/nuestros-juegos/loto/",
        "history_url": None,
        "sync_adapter": None,
        "rules": {
            "main": {"count": 6, "min": 1, "max": 41, "name": "Números principales"},
            "bonus": [
                {
                    "count": 1,
                    "min": 1,
                    "max": 41,
                    "name": "Comodín",
                    "pool": "same_remaining",
                }
            ],
            "draw_days": ["tuesday", "thursday", "sunday"],
            "currency": "CLP",
            "ticket_price_hint": 1000,
            "rules_effective": "1989-01-01",
            "rules_verified": "2026-07",
            "notes": (
                "6/41 + Comodín (7.º número del mismo bombo). "
                "Sorteos mar/jue/dom 21:00 (hora Chile). "
                "Modalidades Recargado/Revancha/Desquite/Jubilazo son sorteos "
                "adicionales; este catálogo modela el sorteo Loto principal + comodín. "
                "Sin API pública estable; CSV manual desde Polla Chilena."
            ),
            "sources": ["https://www.pollachilena.cl/nuestros-juegos/loto/"],
        },
    },
]

def expected_slugs() -> list[str]:
    return [item["slug"] for item in CATALOG_SEED]


def get_game_by_slug(slug: str) -> LotteryGame | None:
    return LotteryGame.query.filter_by(slug=slug).one_or_none()


def seed_lottery_catalog(*, force_update_rules: bool = True) -> dict[str, int]:
    """Inserta o actualiza el catálogo de forma idempotente."""
    inserted = 0
    updated = 0
    for item in CATALOG_SEED:
        existing = LotteryGame.query.filter_by(slug=item["slug"]).one_or_none()
        payload = {
            "name": item["name"],
            "country": item["country"],
            "rules_json": json.dumps(item["rules"], ensure_ascii=False),
            "official_url": item.get("official_url"),
            "history_url": item.get("history_url"),
            "sync_adapter": item.get("sync_adapter"),
            "enabled": True,
            "sort_order": int(item.get("sort_order") or 0),
        }
        if existing is None:
            db.session.add(LotteryGame(slug=item["slug"], **payload))
            inserted += 1
        else:
            if force_update_rules:
                for key, value in payload.items():
                    setattr(existing, key, value)
                updated += 1
    db.session.commit()
    return {
        "inserted": inserted,
        "updated": updated,
        "total": LotteryGame.query.count(),
    }


def list_games(enabled_only: bool = True) -> list[LotteryGame]:
    q = LotteryGame.query
    if enabled_only:
        q = q.filter_by(enabled=True)
    return q.order_by(LotteryGame.sort_order.asc(), LotteryGame.name.asc()).all()


def game_rules_summary(game: LotteryGame) -> str:
    rules = game.rules()
    main = rules.get("main") or {}
    parts = [
        f"{main.get('count')} de {main.get('min')}–{main.get('max')}",
    ]
    for b in rules.get("bonus") or []:
        parts.append(
            f"+ {b.get('count')} {b.get('name')} ({b.get('min')}–{b.get('max')})"
        )
    return " ".join(parts)
