from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

from . import DRAW_SIZE, POOL_SIZE


class DrawValidationError(ValueError):
    """Sorteo inválido según las reglas modernas de Kino."""


@dataclass(frozen=True, slots=True)
class Draw:
    draw_date: date
    numbers: tuple[int, ...]
    draw_number: int | None = None
    source: str = "local"

    def sorted_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self.numbers))


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise DrawValidationError(f"Fecha no reconocida: {value!r}")


def validate_numbers(numbers: Sequence[int]) -> tuple[int, ...]:
    if len(numbers) != DRAW_SIZE:
        raise DrawValidationError(
            f"Se esperaban {DRAW_SIZE} números, se obtuvieron {len(numbers)}: {numbers}"
        )
    cleaned = tuple(int(n) for n in numbers)
    if any(n < 1 or n > POOL_SIZE for n in cleaned):
        raise DrawValidationError(f"Números fuera de rango 1-{POOL_SIZE}: {cleaned}")
    if len(set(cleaned)) != DRAW_SIZE:
        raise DrawValidationError(f"Números duplicados: {cleaned}")
    return tuple(sorted(cleaned))


def make_draw(
    draw_date: str | date,
    numbers: Iterable[int],
    *,
    draw_number: int | None = None,
    source: str = "local",
) -> Draw:
    return Draw(
        draw_date=parse_date(draw_date),
        numbers=validate_numbers(list(numbers)),
        draw_number=draw_number,
        source=source,
    )
