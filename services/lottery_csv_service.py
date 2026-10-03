from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime
from typing import Any

import pandas as pd

from models import LotteryDraw, LotteryGame, db


class LotteryCsvValidationError(ValueError):
    pass


DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y")


def _parse_date(value: Any) -> date:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise LotteryCsvValidationError("Fecha vacía")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        raise LotteryCsvValidationError("Fecha vacía")
    # ISO with time
    if "T" in text:
        text = text.split("T", 1)[0]
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    ts = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(ts):
        raise LotteryCsvValidationError(f"Fecha no reconocida: {value!r}")
    return ts.date()


def _to_ints(values: list[Any]) -> list[int]:
    out: list[int] = []
    for v in values:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            raise LotteryCsvValidationError("Número vacío")
        text = str(v).strip()
        if not text:
            raise LotteryCsvValidationError("Número vacío")
        # Allow "01", "1.0"
        try:
            num = int(float(text))
        except ValueError as exc:
            raise LotteryCsvValidationError(f"Número inválido: {v!r}") from exc
        out.append(num)
    return out


def _validate_pool(
    nums: list[int],
    *,
    count: int,
    min_n: int,
    max_n: int,
    label: str,
    allow_duplicates: bool = False,
) -> list[int]:
    if len(nums) != count:
        raise LotteryCsvValidationError(
            f"{label}: se esperaban {count} números, se obtuvieron {len(nums)}: {nums}"
        )
    if any(n < min_n or n > max_n for n in nums):
        raise LotteryCsvValidationError(
            f"{label}: fuera de rango {min_n}-{max_n}: {nums}"
        )
    if not allow_duplicates and len(set(nums)) != len(nums):
        raise LotteryCsvValidationError(f"{label}: números duplicados: {nums}")
    return sorted(nums)


def validate_draw_numbers(
    rules: dict,
    main: list[int],
    bonus: list[int],
    *,
    draw_date: date | None = None,
) -> tuple[list[int], list[int]]:
    main_spec = rules.get("main") or {}
    main_clean = _validate_pool(
        main,
        count=int(main_spec["count"]),
        min_n=int(main_spec["min"]),
        max_n=int(main_spec["max"]),
        label=str(main_spec.get("name") or "Principales"),
    )

    bonus_specs = rules.get("bonus") or []
    expected_bonus = sum(int(b["count"]) for b in bonus_specs)
    if len(bonus) != expected_bonus:
        raise LotteryCsvValidationError(
            f"Bonus: se esperaban {expected_bonus} números, se obtuvieron {len(bonus)}: {bonus}"
        )

    cleaned_bonus: list[int] = []
    offset = 0
    for spec in bonus_specs:
        count = int(spec["count"])
        chunk = bonus[offset : offset + count]
        offset += count
        max_n = int(spec["max"])
        # Mega Millions legado: Mega Ball 1-25 antes del 2025-04-08
        if (
            draw_date is not None
            and rules.get("legacy_until")
            and rules.get("legacy_bonus_max")
            and draw_date <= date.fromisoformat(str(rules["legacy_until"]))
        ):
            max_n = max(max_n, int(rules["legacy_bonus_max"]))
        cleaned = _validate_pool(
            chunk,
            count=count,
            min_n=int(spec["min"]),
            max_n=max_n,
            label=str(spec.get("name") or "Bonus"),
        )
        pool_mode = (spec.get("pool") or "separate").lower()
        if pool_mode == "same_remaining":
            overlap = set(cleaned) & set(main_clean)
            if overlap:
                raise LotteryCsvValidationError(
                    f"{spec.get('name')}: no puede repetir principales ({sorted(overlap)})"
                )
        cleaned_bonus.extend(cleaned)

    return main_clean, cleaned_bonus


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _find_date_column(frame: pd.DataFrame) -> str | None:
    preferred = {
        "fecha",
        "date",
        "drawdate",
        "draw_date",
        "sorteo",
        "fechadesorteo",
    }
    for col in frame.columns:
        norm = _normalize_header(col)
        if norm in preferred or "date" in norm or "fecha" in norm:
            return col
    for col in frame.columns:
        sample = frame[col].dropna().astype(str).head(15)
        ok = 0
        for value in sample:
            try:
                _parse_date(value)
                ok += 1
            except LotteryCsvValidationError:
                pass
        if ok >= max(1, len(sample) // 2):
            return col
    return None


def _column_groups(frame: pd.DataFrame, rules: dict) -> tuple[list[str], list[str]]:
    """Detecta columnas main/bonus por headers típicos o posición."""
    main_count = int((rules.get("main") or {})["count"])
    bonus_count = sum(int(b["count"]) for b in (rules.get("bonus") or []))

    cols = list(frame.columns)
    norms = {_normalize_header(c): c for c in cols}

    main_cols: list[str] = []
    for i in range(1, main_count + 1):
        for key in (f"n{i}", f"main{i}", f"ball{i}", f"numero{i}", f"num{i}", f"b{i}"):
            if key in norms and norms[key] not in main_cols:
                main_cols.append(norms[key])
                break
    if len(main_cols) < main_count:
        # Prefer columns named main / white / principal
        named = [
            c
            for c in cols
            if any(
                tok in _normalize_header(c)
                for tok in ("main", "white", "principal", "bola")
            )
            and "bonus" not in _normalize_header(c)
            and "star" not in _normalize_header(c)
            and "power" not in _normalize_header(c)
            and "mega" not in _normalize_header(c)
            and "thunder" not in _normalize_header(c)
            and "life" not in _normalize_header(c)
            and "dream" not in _normalize_header(c)
            and "comodin" not in _normalize_header(c)
            and "euro" not in _normalize_header(c)
            and "lucky" not in _normalize_header(c)
        ]
        if len(named) >= main_count:
            main_cols = named[:main_count]

    bonus_cols: list[str] = []
    bonus_keys = [
        "bonus",
        "powerball",
        "megaball",
        "thunderball",
        "lifeball",
        "dreamball",
        "dreamnumber",
        "comodin",
        "luckystar",
        "star",
        "euro",
        "bonusball",
        "bb",
        "pb",
        "mb",
    ]
    for i in range(1, bonus_count + 1):
        for key in (
            f"bonus{i}",
            f"b{i}",
            f"star{i}",
            f"luckystar{i}",
            f"euro{i}",
            f"s{i}",
        ):
            if key in norms and norms[key] not in bonus_cols:
                bonus_cols.append(norms[key])
                break
    if len(bonus_cols) < bonus_count:
        named_bonus = [
            c
            for c in cols
            if any(tok in _normalize_header(c) for tok in bonus_keys)
            and c not in main_cols
        ]
        # Prefer ordered luckystar1, luckystar2 etc.
        if len(named_bonus) >= bonus_count:
            bonus_cols = named_bonus[:bonus_count]

    # Normalized single-column formats handled in parse_lottery_csv_bytes
    # Positional fallback: date + mains + bonuses
    date_col = _find_date_column(frame)
    remaining = [c for c in cols if c != date_col]
    if len(main_cols) < main_count or len(bonus_cols) < bonus_count:
        if len(remaining) >= main_count + bonus_count:
            main_cols = remaining[:main_count]
            bonus_cols = remaining[main_count : main_count + bonus_count]

    if len(main_cols) != main_count:
        raise LotteryCsvValidationError(
            f"No se detectaron {main_count} columnas principales "
            f"(encontradas: {main_cols})"
        )
    if bonus_count and len(bonus_cols) != bonus_count:
        raise LotteryCsvValidationError(
            f"No se detectaron {bonus_count} columnas bonus "
            f"(encontradas: {bonus_cols})"
        )
    return main_cols, bonus_cols


def _expand_cell_numbers(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    # Space / dash / semicolon / comma separated
    if re.search(r"[\s,;|/\\-]", text) and not re.fullmatch(r"\d+", text):
        parts = re.split(r"[\s,;|/\\-]+", text)
        parts = [p for p in parts if p]
        return _to_ints(parts)
    return _to_ints([text])


def parse_lottery_csv_bytes(
    content: bytes,
    rules: dict,
    *,
    source: str = "upload",
) -> list[dict]:
    """Parsea CSV genérico según reglas del juego. Devuelve filas válidas."""
    if not content or not content.strip():
        raise LotteryCsvValidationError("Archivo vacío")

    # Try utf-8 then latin-1
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise LotteryCsvValidationError("No se pudo decodificar el CSV")

    frame = pd.read_csv(io.StringIO(text))
    if frame.empty:
        raise LotteryCsvValidationError("CSV sin filas")

    frame.columns = [str(c).strip() for c in frame.columns]
    date_col = _find_date_column(frame)
    if date_col is None:
        raise LotteryCsvValidationError("No se detectó columna de fecha")

    main_count = int((rules.get("main") or {})["count"])
    bonus_count = sum(int(b["count"]) for b in (rules.get("bonus") or []))

    # Special: single winning_numbers column with all balls
    norms = {_normalize_header(c): c for c in frame.columns}
    combined_col = None
    for key in ("winningnumbers", "numeros", "numbers", "result"):
        if key in norms:
            combined_col = norms[key]
            break

    rows: list[dict] = []
    errors: list[str] = []

    if combined_col and combined_col != date_col:
        # Optional separate mega_ball etc.
        extra_bonus_cols = [
            c
            for c in frame.columns
            if c not in {date_col, combined_col}
            and any(
                tok in _normalize_header(c)
                for tok in (
                    "megaball",
                    "powerball",
                    "bonus",
                    "thunder",
                    "life",
                    "dream",
                    "comodin",
                    "star",
                    "euro",
                )
            )
        ]
        for idx, series in frame.iterrows():
            try:
                d = _parse_date(series[date_col])
                all_nums = _expand_cell_numbers(series[combined_col])
                extra: list[int] = []
                for bc in extra_bonus_cols:
                    extra.extend(_expand_cell_numbers(series[bc]))
                if extra:
                    main = all_nums[:main_count]
                    bonus = extra if len(extra) == bonus_count else all_nums[main_count:] + extra
                    if len(bonus) > bonus_count:
                        bonus = bonus[:bonus_count]
                else:
                    main = all_nums[:main_count]
                    bonus = all_nums[main_count : main_count + bonus_count]
                main_c, bonus_c = validate_draw_numbers(
                    rules, main, bonus, draw_date=d
                )
                draw_number = None
                for key in ("drawnumber", "sorteo", "draw", "numero"):
                    if key in norms:
                        raw = series[norms[key]]
                        if pd.notna(raw) and str(raw).strip():
                            draw_number = int(float(raw))
                        break
                rows.append(
                    {
                        "draw_date": d,
                        "draw_number": draw_number,
                        "main_numbers": main_c,
                        "bonus_numbers": bonus_c,
                        "source": source,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Fila {int(idx) + 2}: {exc}")
    else:
        main_cols, bonus_cols = _column_groups(frame, rules)
        for idx, series in frame.iterrows():
            try:
                d = _parse_date(series[date_col])
                main: list[int] = []
                for c in main_cols:
                    main.extend(_expand_cell_numbers(series[c]))
                bonus: list[int] = []
                for c in bonus_cols:
                    bonus.extend(_expand_cell_numbers(series[c]))
                # If one main column held all numbers
                if len(main_cols) == 1 and len(main) == main_count + bonus_count and not bonus:
                    bonus = main[main_count:]
                    main = main[:main_count]
                main_c, bonus_c = validate_draw_numbers(
                    rules, main, bonus, draw_date=d
                )
                draw_number = None
                for key in ("drawnumber", "sorteo", "draw", "numero"):
                    if key in norms:
                        raw = series[norms[key]]
                        if pd.notna(raw) and str(raw).strip():
                            draw_number = int(float(raw))
                        break
                rows.append(
                    {
                        "draw_date": d,
                        "draw_number": draw_number,
                        "main_numbers": main_c,
                        "bonus_numbers": bonus_c,
                        "source": source,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Fila {int(idx) + 2}: {exc}")

    if not rows:
        detail = "; ".join(errors[:5]) if errors else "sin detalle"
        raise LotteryCsvValidationError(
            f"Ningún sorteo válido. Errores: {detail}"
        )
    return rows


def persist_lottery_draws(
    game: LotteryGame,
    rows: list[dict],
    *,
    replace: bool = False,
) -> dict:
    if replace:
        LotteryDraw.query.filter_by(game_id=game.id).delete()
        db.session.commit()

    inserted = 0
    updated = 0
    for row in rows:
        existing = LotteryDraw.query.filter_by(
            game_id=game.id, draw_date=row["draw_date"]
        ).one_or_none()
        fields = {
            "draw_number": row.get("draw_number"),
            "main_numbers": json.dumps(row["main_numbers"]),
            "bonus_numbers": json.dumps(row.get("bonus_numbers") or []),
            "source": row.get("source") or "upload",
        }
        if existing:
            for k, v in fields.items():
                setattr(existing, k, v)
            updated += 1
        else:
            db.session.add(
                LotteryDraw(
                    game_id=game.id,
                    draw_date=row["draw_date"],
                    **fields,
                )
            )
            inserted += 1
    db.session.commit()
    total = LotteryDraw.query.filter_by(game_id=game.id).count()
    return {
        "inserted": inserted,
        "updated": updated,
        "skipped_invalid": 0,
        "total": total,
        "accepted": inserted + updated,
    }


def draws_to_normalized_csv(game: LotteryGame, draws: list[LotteryDraw]) -> bytes:
    rules = game.rules()
    main_count = int((rules.get("main") or {}).get("count") or 0)
    bonus_specs = rules.get("bonus") or []
    bonus_count = sum(int(b["count"]) for b in bonus_specs)

    buf = io.StringIO()
    headers = ["draw_date", "draw_number"]
    headers += [f"n{i}" for i in range(1, main_count + 1)]
    headers += [f"bonus{i}" for i in range(1, bonus_count + 1)]
    headers += ["source"]
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    for d in draws:
        main = d.main()
        bonus = d.bonus()
        row = {
            "draw_date": d.draw_date.isoformat(),
            "draw_number": d.draw_number if d.draw_number is not None else "",
            "source": d.source or "",
        }
        for i in range(main_count):
            row[f"n{i + 1}"] = main[i] if i < len(main) else ""
        for i in range(bonus_count):
            row[f"bonus{i + 1}"] = bonus[i] if i < len(bonus) else ""
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def sample_csv_bytes(rules: dict, *, rows: int = 2) -> bytes:
    """CSV de ejemplo válido según reglas del juego (números ficticios)."""
    main_spec = rules.get("main") or {}
    main_count = int(main_spec.get("count") or 0)
    main_min = int(main_spec.get("min") or 1)
    main_max = int(main_spec.get("max") or main_count)
    bonus_specs = rules.get("bonus") or []
    bonus_count = sum(int(b["count"]) for b in bonus_specs)

    headers = ["draw_date", "draw_number"]
    headers += [f"n{i}" for i in range(1, main_count + 1)]
    headers += [f"bonus{i}" for i in range(1, bonus_count + 1)]
    headers += ["source"]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()

    base_dates = (date(2026, 7, 18), date(2026, 7, 15), date(2026, 7, 11))
    for i in range(max(1, rows)):
        # Principales: secuencia creciente válida dentro del rango.
        span = max(1, main_max - main_min + 1)
        start = main_min + (i * 2) % max(1, span - main_count)
        main_nums = []
        cursor = start
        while len(main_nums) < main_count:
            if cursor > main_max:
                cursor = main_min
            if cursor not in main_nums:
                main_nums.append(cursor)
            cursor += 1
        main_nums = sorted(main_nums)

        bonus_nums: list[int] = []
        used_main = set(main_nums)
        for spec in bonus_specs:
            b_count = int(spec["count"])
            b_min = int(spec["min"])
            b_max = int(spec["max"])
            pool = str(spec.get("pool") or "separate")
            for j in range(b_count):
                candidate = b_min + ((i + j) % (b_max - b_min + 1))
                if pool == "same_remaining":
                    # Comodín / bonus del mismo bombo: no puede repetir un principal.
                    while candidate in used_main or candidate in bonus_nums:
                        candidate += 1
                        if candidate > b_max:
                            candidate = b_min
                    used_main.add(candidate)
                else:
                    while candidate in bonus_nums and b_count > 1:
                        candidate += 1
                        if candidate > b_max:
                            candidate = b_min
                bonus_nums.append(candidate)

        row: dict[str, Any] = {
            "draw_date": base_dates[i % len(base_dates)].isoformat(),
            "draw_number": 1000 + i,
            "source": "sample",
        }
        for idx, n in enumerate(main_nums):
            row[f"n{idx + 1}"] = n
        for idx, n in enumerate(bonus_nums):
            row[f"bonus{idx + 1}"] = n
        writer.writerow(row)

    return buf.getvalue().encode("utf-8")
