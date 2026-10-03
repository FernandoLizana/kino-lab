"""Cliente remoto inspirado en Blank2D/datos-de-azar, sin MySQL.

Fuente principal de actualización: API pública de kinohistorico.cl
(https://kinohistorico.cl/kino-api/draws), útil porque el histórico
oficial de Lotería ya no está disponible de forma usable.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from .validate import Draw, DrawValidationError, make_draw

API_BASE = "https://kinohistorico.cl/kino-api"
USER_AGENT = (
    "Mozilla/5.0 (compatible; LoteriaLab/0.1; "
    "+https://github.com/Blank2D/datos-de-azar)"
)


class RemoteError(RuntimeError):
    pass


class KinoHistoricClient:
    def __init__(
        self,
        *,
        delay_seconds: float = 0.35,
        timeout: int = 30,
        max_retries: int = 3,
        page_size: int = 50,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self.page_size = page_size
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            }
        )

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{API_BASE}{path}"
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                time.sleep(self.delay_seconds)
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(min(2 ** attempt, 8))
        raise RemoteError(f"Fallo al pedir {url}: {last_error}") from last_error

    @staticmethod
    def _extract_kino_numbers(item: dict[str, Any]) -> list[int]:
        games = item.get("games") or item.get("game") or []
        for game in games:
            code = str(game.get("game_code", "")).upper()
            if code == "KINO":
                return list(game.get("numbers") or [])
        raise DrawValidationError(
            f"Sorteo {item.get('draw_number')} sin bloque KINO"
        )

    def fetch_draw(self, draw_number: int) -> Draw:
        payload = self._get_json(f"/draws/{draw_number}")
        item = payload.get("data") or {}
        return make_draw(
            item["draw_date"],
            self._extract_kino_numbers(item),
            draw_number=int(item["draw_number"]),
            source="kinohistorico.cl",
        )

    def fetch_latest(self) -> Draw:
        payload = self._get_json("/draws", params={"page": 1, "limit": 1})
        items = payload.get("data") or []
        if not items:
            raise RemoteError("La API no devolvió sorteos")
        item = items[0]
        return make_draw(
            item["draw_date"],
            self._extract_kino_numbers(item),
            draw_number=int(item["draw_number"]),
            source="kinohistorico.cl",
        )

    def fetch_all(self) -> list[Draw]:
        first = self._get_json(
            "/draws", params={"page": 1, "limit": self.page_size}
        )
        meta = first.get("meta") or {}
        total_pages = int(meta.get("total_pages") or 1)
        total = int(meta.get("total") or 0)
        print(f"API kinohistorico.cl: {total} sorteos en {total_pages} páginas")

        draws: list[Draw] = []
        pages = [first]
        for page in range(2, total_pages + 1):
            pages.append(
                self._get_json(
                    "/draws",
                    params={"page": page, "limit": self.page_size},
                )
            )
            if page % 10 == 0 or page == total_pages:
                print(f"  descargadas {page}/{total_pages} páginas")

        skipped = 0
        for payload in pages:
            for item in payload.get("data") or []:
                try:
                    draws.append(
                        make_draw(
                            item["draw_date"],
                            self._extract_kino_numbers(item),
                            draw_number=int(item["draw_number"]),
                            source="kinohistorico.cl",
                        )
                    )
                except (DrawValidationError, KeyError, TypeError, ValueError):
                    skipped += 1
        print(f"API: {len(draws)} sorteos válidos ({skipped} omitidos)")
        return draws
