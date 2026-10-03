"""Adaptadores remotos opt-in solo para APIs públicas estables.

Actualmente:
- Powerball / Mega Millions vía NY Open Data (Socrata / SODA).
Otros juegos del catálogo: CSV manual (sin endpoints inventados).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import requests

from services.lottery_csv_service import validate_draw_numbers

NY_POWERBALL_URL = "https://data.ny.gov/resource/d6yy-54nr.json"
NY_MEGA_MILLIONS_URL = "https://data.ny.gov/resource/5xaw-6ayf.json"

ADAPTERS = {
    "ny_powerball": {
        "url": NY_POWERBALL_URL,
        "label": "NY Open Data — Powerball",
        "page": "https://data.ny.gov/d/d6yy-54nr",
    },
    "ny_mega_millions": {
        "url": NY_MEGA_MILLIONS_URL,
        "label": "NY Open Data — Mega Millions",
        "page": "https://data.ny.gov/d/5xaw-6ayf",
    },
}


class LotteryRemoteError(RuntimeError):
    pass


def adapter_info(adapter: str | None) -> dict | None:
    if not adapter:
        return None
    return ADAPTERS.get(adapter)


def _parse_iso_date(value: str) -> date:
    text = str(value).strip()
    if "T" in text:
        text = text.split("T", 1)[0]
    return datetime.strptime(text, "%Y-%m-%d").date()


def _fetch_soda(url: str, *, limit: int = 5000, timeout: int = 30) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    page = min(1000, limit)
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": "KinoLab-LotteryModule/1.0"})
    while offset < limit:
        params = {
            "$limit": page,
            "$offset": offset,
            "$order": "draw_date DESC",
        }
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            batch = resp.json()
        except requests.RequestException as exc:
            raise LotteryRemoteError(f"Error al consultar {url}: {exc}") from exc
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return rows


def _parse_powerball_row(item: dict, rules: dict) -> dict | None:
    try:
        d = _parse_iso_date(item["draw_date"])
        nums = [int(x) for x in str(item.get("winning_numbers", "")).split()]
        if len(nums) < 6:
            return None
        main, bonus = validate_draw_numbers(
            rules, nums[:5], nums[5:6], draw_date=d
        )
        return {
            "draw_date": d,
            "draw_number": None,
            "main_numbers": main,
            "bonus_numbers": bonus,
            "source": "ny_open_data",
        }
    except Exception:
        return None


def _parse_mega_millions_row(item: dict, rules: dict) -> dict | None:
    try:
        d = _parse_iso_date(item["draw_date"])
        mains = [int(x) for x in str(item.get("winning_numbers", "")).split()]
        mega = item.get("mega_ball")
        if mega is None:
            return None
        bonus = [int(mega)]
        main, bonus_c = validate_draw_numbers(
            rules, mains, bonus, draw_date=d
        )
        return {
            "draw_date": d,
            "draw_number": None,
            "main_numbers": main,
            "bonus_numbers": bonus_c,
            "source": "ny_open_data",
        }
    except Exception:
        return None


def fetch_remote_draws(
    adapter: str,
    rules: dict,
    *,
    limit: int = 2000,
) -> list[dict]:
    meta = ADAPTERS.get(adapter)
    if not meta:
        raise LotteryRemoteError(f"Adaptador remoto desconocido: {adapter}")

    raw = _fetch_soda(meta["url"], limit=limit)
    parsed: list[dict] = []
    if adapter == "ny_powerball":
        for item in raw:
            row = _parse_powerball_row(item, rules)
            if row:
                parsed.append(row)
    elif adapter == "ny_mega_millions":
        for item in raw:
            row = _parse_mega_millions_row(item, rules)
            if row:
                parsed.append(row)
    else:  # pragma: no cover
        raise LotteryRemoteError(f"Sin parser para {adapter}")

    if not parsed:
        raise LotteryRemoteError(
            "La API respondió pero no se pudo parsear ningún sorteo válido."
        )
    return parsed


def sync_status_message(adapter: str | None) -> str:
    info = adapter_info(adapter)
    if not info:
        return "Sin sync remoto: cargar CSV manual desde la fuente oficial."
    return f"Sync opt-in disponible vía {info['label']}."
