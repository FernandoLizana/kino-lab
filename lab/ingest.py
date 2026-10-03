from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .profiles import GameProfile, get_profile, profile_for_draw_size


MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 50_000


class IngestError(ValueError):
    pass


def _decode(content: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise IngestError("No se pudo decodificar el archivo (prueba UTF-8).")


def _parse_date(value: str) -> date | None:
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def sniff(content: bytes) -> dict:
    if len(content) > MAX_BYTES:
        raise IngestError(f"El archivo supera {MAX_BYTES // (1024 * 1024)} MB.")
    text = _decode(content)
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        raise IngestError("El CSV está vacío.")
    headerish = any(
        token in ",".join(rows[0]).lower()
        for token in ("fecha", "date", "n1", "numero")
    )
    preview = rows[:8]
    return {
        "delimiter": delimiter,
        "has_header": headerish,
        "row_count": len(rows) - (1 if headerish else 0),
        "column_count": max(len(r) for r in rows),
        "preview": preview,
        "fingerprint": hashlib.sha256(content).hexdigest(),
        "encoding_used": "utf-8-sig/utf-8/latin-1",
    }


def _extract_numbers(cells: list[str], number_idxs: list[int] | None) -> list[int]:
    values: list[int] = []
    if number_idxs:
        source = [cells[i] for i in number_idxs if i < len(cells)]
    else:
        source = cells
    for cell in source:
        cell = str(cell).strip()
        if not cell:
            continue
        if "," in cell and cell.replace(",", "").replace(" ", "").isdigit():
            for part in cell.split(","):
                part = part.strip()
                if part:
                    values.append(int(part))
            continue
        try:
            values.append(int(float(cell)))
        except ValueError:
            continue
    return values


@dataclass
class IngestReport:
    source_name: str
    fingerprint: str
    imported_at: str
    rows_read: int
    accepted: int
    duplicates: int
    conflicts: int
    excluded: int
    by_profile: dict[str, int]
    period: dict[str, str | None]
    exclusions: list[dict] = field(default_factory=list)
    draws: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "fingerprint": self.fingerprint,
            "imported_at": self.imported_at,
            "rows_read": self.rows_read,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "conflicts": self.conflicts,
            "excluded": self.excluded,
            "by_profile": self.by_profile,
            "period": self.period,
            "exclusions": self.exclusions[:200],
            "notes": self.notes,
            "reconciliation": (
                self.accepted + self.duplicates + self.conflicts + self.excluded
            ),
        }


def parse_draws(
    content: bytes,
    source_name: str = "upload",
    *,
    has_header: bool | None = None,
    delimiter: str | None = None,
    date_index: int | None = None,
    id_index: int | None = None,
    number_indexes: list[int] | None = None,
    default_profile: str | None = None,
) -> IngestReport:
    meta = sniff(content)
    text = _decode(content)
    delim = delimiter or meta["delimiter"]
    header = meta["has_header"] if has_header is None else has_header
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    raw_rows = list(reader)
    if header and raw_rows:
        raw_rows = raw_rows[1:]
    if len(raw_rows) > MAX_ROWS:
        raise IngestError(f"Más de {MAX_ROWS} filas; parte el archivo.")

    report = IngestReport(
        source_name=source_name,
        fingerprint=meta["fingerprint"],
        imported_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        rows_read=len(raw_rows),
        accepted=0,
        duplicates=0,
        conflicts=0,
        excluded=0,
        by_profile={},
        period={"first_draw": None, "last_draw": None},
    )
    seen: dict[tuple[str, str], list[int]] = {}

    for idx, cells in enumerate(raw_rows, start=1):
        if not any(str(c).strip() for c in cells):
            report.excluded += 1
            report.exclusions.append({"row": idx, "reason": "fila vacía"})
            continue
        date_cell = cells[date_index] if date_index is not None and date_index < len(cells) else cells[0]
        draw_date = _parse_date(str(date_cell))
        if draw_date is None:
            report.excluded += 1
            report.exclusions.append({"row": idx, "reason": f"fecha ambigua o inválida: {date_cell!r}"})
            continue
        id_val = None
        if id_index is not None and id_index < len(cells):
            try:
                id_val = int(float(cells[id_index]))
            except ValueError:
                id_val = None
        number_cells = cells[1:] if date_index is None and number_indexes is None else cells
        nums = _extract_numbers(number_cells if number_indexes is None else cells, number_indexes)
        if not nums:
            report.excluded += 1
            report.exclusions.append({"row": idx, "reason": "sin números"})
            continue
        if len(set(nums)) != len(nums):
            report.excluded += 1
            report.exclusions.append({"row": idx, "reason": f"números repetidos: {nums}"})
            continue
        profile = None
        if default_profile:
            profile = get_profile(default_profile)
            if len(nums) != profile.k:
                report.excluded += 1
                report.exclusions.append(
                    {
                        "row": idx,
                        "reason": (
                            f"el perfil {profile.slug} espera {profile.k} números, "
                            f"la fila tiene {len(nums)}. No se recorta."
                        ),
                    }
                )
                continue
        else:
            profile = profile_for_draw_size(len(nums))
            if profile is None:
                report.excluded += 1
                report.exclusions.append(
                    {"row": idx, "reason": f"sin perfil para {len(nums)} números (no se convierte)"}
                )
                continue
        if any(n < 1 or n > profile.n for n in nums):
            report.excluded += 1
            report.exclusions.append({"row": idx, "reason": f"fuera de rango 1–{profile.n}: {nums}"})
            continue
        nums = sorted(nums)
        key = (profile.slug, draw_date.isoformat())
        if key in seen:
            if seen[key] == nums:
                report.duplicates += 1
                report.exclusions.append({"row": idx, "reason": "duplicado misma fecha y perfil"})
            else:
                report.conflicts += 1
                report.exclusions.append({"row": idx, "reason": "conflicto: misma fecha, números distintos"})
            continue
        seen[key] = nums
        report.draws.append(
            {
                "draw_date": draw_date,
                "draw_number": id_val,
                "numbers": nums,
                "profile": profile.slug,
                "source": source_name,
                "synthetic": profile.synthetic,
            }
        )
        report.accepted += 1
        report.by_profile[profile.slug] = report.by_profile.get(profile.slug, 0) + 1

    if report.draws:
        dates = [d["draw_date"] for d in report.draws]
        report.period["first_draw"] = min(dates).isoformat()
        report.period["last_draw"] = max(dates).isoformat()
    report.notes.append(
        "La fecha de importación no es la del último sorteo. "
        "Los sorteos de 15 números no se convierten a 14."
    )
    return report


def report_to_csv(report: IngestReport) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["row", "reason"])
    for item in report.exclusions:
        writer.writerow([item.get("row"), item.get("reason")])
    return buf.getvalue().encode("utf-8")


def fingerprint_draws(draws: list[dict]) -> str:
    payload = json.dumps(
        [
            {
                "d": d["draw_date"].isoformat() if hasattr(d["draw_date"], "isoformat") else str(d["draw_date"]),
                "p": d.get("profile"),
                "n": d.get("numbers"),
            }
            for d in sorted(draws, key=lambda x: str(x.get("draw_date")))
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
