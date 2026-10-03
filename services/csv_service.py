from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from config import Config


class CsvValidationError(ValueError):
    pass


@dataclass
class ParsedDraw:
    draw_date: date
    numbers: list[int]
    draw_number: int | None = None


def _parse_date(value: Any) -> date:
    if pd.isna(value):
        raise CsvValidationError("Fecha vacía")
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # pandas fallback
    ts = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(ts):
        raise CsvValidationError(f"Fecha no reconocida: {value!r}")
    return ts.date()


def _validate_numbers(nums: list[int]) -> list[int]:
    if len(nums) != Config.DRAW_SIZE:
        raise CsvValidationError(
            f"Se esperaban {Config.DRAW_SIZE} números, se obtuvieron {len(nums)}: {nums}"
        )
    cleaned = [int(n) for n in nums]
    if any(n < 1 or n > Config.POOL_SIZE for n in cleaned):
        raise CsvValidationError(f"Números fuera de rango 1-{Config.POOL_SIZE}: {cleaned}")
    if len(set(cleaned)) != Config.DRAW_SIZE:
        raise CsvValidationError(f"Números duplicados: {cleaned}")
    return sorted(cleaned)


def detect_number_columns(frame: pd.DataFrame) -> list[str]:
    """Detecta automáticamente 14 columnas numéricas con valores 1-25."""
    candidates: list[str] = []
    for col in frame.columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        if valid.min() >= 1 and valid.max() <= Config.POOL_SIZE:
            # Prefer columns that look like lottery balls
            if (valid == valid.round()).all():
                candidates.append(col)

    # Prefer N1..N14 / n1..n14 style
    preferred = [
        c for c in candidates
        if str(c).lower().replace(" ", "").startswith("n")
        and str(c).lower().replace("n", "").isdigit()
    ]
    if len(preferred) >= Config.DRAW_SIZE:
        return preferred[: Config.DRAW_SIZE]

    if len(candidates) >= Config.DRAW_SIZE:
        return candidates[: Config.DRAW_SIZE]

    # Raw CSV without headers: try all but first column
    if frame.shape[1] >= Config.DRAW_SIZE + 1:
        return list(frame.columns[1 : 1 + Config.DRAW_SIZE])

    raise CsvValidationError(
        "No se pudieron detectar 14 columnas de números entre 1 y 25"
    )


def detect_date_column(frame: pd.DataFrame, number_cols: list[str]) -> str | None:
    for col in frame.columns:
        if col in number_cols:
            continue
        sample = frame[col].dropna().astype(str).head(20)
        ok = 0
        for value in sample:
            try:
                _parse_date(value)
                ok += 1
            except CsvValidationError:
                pass
        if ok >= max(1, len(sample) // 2):
            return col
    return None


def parse_csv_bytes(content: bytes, source: str = "upload") -> list[dict]:
    # Try with header first
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    # Detect if first row looks like header
    sample = text.splitlines()[:2]
    has_header = False
    if sample:
        first = sample[0].lower()
        has_header = any(token in first for token in ("fecha", "date", "n1", "numero"))

    if has_header:
        frame = pd.read_csv(io.StringIO(text))
    else:
        frame = pd.read_csv(io.StringIO(text), header=None)

    if frame.empty:
        raise CsvValidationError("El CSV está vacío")

    # Drop fully empty trailing columns (common with trailing commas)
    frame = frame.dropna(axis=1, how="all")

    number_cols = detect_number_columns(frame)
    date_col = detect_date_column(frame, number_cols)

    draws: list[dict] = []
    errors: list[str] = []
    for idx, row in frame.iterrows():
        try:
            nums = _validate_numbers([int(float(row[c])) for c in number_cols])
            if date_col is not None and not pd.isna(row[date_col]):
                draw_date = _parse_date(row[date_col])
            else:
                draw_date = date.fromordinal(date(1990, 1, 1).toordinal() + int(idx))
            draws.append(
                {
                    "draw_date": draw_date,
                    "draw_number": None,
                    "numbers": nums,
                    "source": source,
                }
            )
        except (CsvValidationError, ValueError, TypeError) as exc:
            errors.append(f"Fila {int(idx) + 1}: {exc}")
            if len(errors) > 30:
                break

    if not draws:
        detail = "; ".join(errors[:5]) if errors else "sin detalle"
        raise CsvValidationError(f"Ningún sorteo válido. {detail}")

    return draws


def draws_to_csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["rank", "score", "numbers", "even", "odd", "sum", "consecutive", "coverage", "similarity"]
    )
    for row in rows:
        writer.writerow(
            [
                row.get("rank"),
                f"{row.get('score', 0):.6f}",
                row.get("numbers_csv") or "-".join(f"{n:02d}" for n in row.get("numbers", [])),
                row.get("even_count"),
                row.get("odd_count"),
                row.get("total_sum"),
                row.get("consecutive"),
                row.get("coverage"),
                row.get("similarity"),
            ]
        )
    return buffer.getvalue().encode("utf-8")
