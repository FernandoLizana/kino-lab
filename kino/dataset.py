from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import DRAW_SIZE, MODERN_START
from .validate import Draw, DrawValidationError, make_draw, parse_date

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "kino_draws.csv"
HISTORICO_PATH = ROOT / "historico.csv"
KINO_PATH = ROOT / "kino.csv"

COLUMNS = ["draw_date", "draw_number", "n1", "n2", "n3", "n4", "n5", "n6", "n7",
           "n8", "n9", "n10", "n11", "n12", "n13", "n14", "source"]


def _number_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("n") and c[1:].isdigit()]


def draws_to_frame(draws: list[Draw]) -> pd.DataFrame:
    rows = []
    for draw in draws:
        nums = draw.sorted_numbers()
        row = {
            "draw_date": draw.draw_date.isoformat(),
            "draw_number": draw.draw_number,
            "source": draw.source,
        }
        for i, value in enumerate(nums, start=1):
            row[f"n{i}"] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=COLUMNS)


def frame_to_draws(frame: pd.DataFrame) -> list[Draw]:
    draws: list[Draw] = []
    cols = _number_cols(frame)
    for _, row in frame.iterrows():
        numbers = [int(row[c]) for c in cols]
        draw_number = row.get("draw_number")
        if pd.isna(draw_number):
            draw_number = None
        else:
            draw_number = int(draw_number)
        draws.append(
            make_draw(
                row["draw_date"],
                numbers,
                draw_number=draw_number,
                source=str(row.get("source", "local")),
            )
        )
    return draws


def load_historico(path: Path = HISTORICO_PATH) -> list[Draw]:
    """Carga historico.csv tolerando filas de 15 números (pre-2006) y comas finales."""
    if not path.exists():
        return []

    draws: list[Draw] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            if len(parts) < 1 + DRAW_SIZE:
                skipped += 1
                continue
            date_raw, *rest = parts
            # Era moderna: exactamente 14 números. Era antigua: 15 → se omite.
            if len(rest) != DRAW_SIZE:
                skipped += 1
                continue
            try:
                draw = make_draw(date_raw, rest, source="historico.csv")
            except DrawValidationError:
                skipped += 1
                continue
            if draw.draw_date < parse_date(MODERN_START):
                skipped += 1
                continue
            draws.append(draw)
    print(f"historico.csv: {len(draws)} sorteos modernos válidos ({skipped} omitidos)")
    return draws


def load_kino_csv(path: Path = KINO_PATH) -> list[Draw]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    date_col = frame.columns[0]
    num_cols = list(frame.columns[1 : 1 + DRAW_SIZE])
    draws: list[Draw] = []
    skipped = 0
    for _, row in frame.iterrows():
        try:
            draws.append(
                make_draw(
                    row[date_col],
                    [row[c] for c in num_cols],
                    source="kino.csv",
                )
            )
        except DrawValidationError:
            skipped += 1
    print(f"kino.csv: {len(draws)} sorteos válidos ({skipped} omitidos)")
    return draws


def merge_draws(*groups: list[Draw]) -> list[Draw]:
    """Fusiona por fecha. Prioridad: draw_number presente > fuente remota > local."""
    priority = {
        "kinohistorico.cl": 3,
        "kino.csv": 2,
        "historico.csv": 1,
        "local": 0,
    }
    best: dict[str, Draw] = {}
    for group in groups:
        for draw in group:
            key = draw.draw_date.isoformat()
            current = best.get(key)
            if current is None:
                best[key] = draw
                continue
            cur_score = (
                10 if current.draw_number is not None else 0
            ) + priority.get(current.source, 0)
            new_score = (
                10 if draw.draw_number is not None else 0
            ) + priority.get(draw.source, 0)
            if new_score >= cur_score:
                # Si los números difieren y ambas tienen score alto, conservar remota.
                best[key] = draw
    return sorted(best.values(), key=lambda d: (d.draw_date, d.draw_number or 0))


def save_draws(draws: list[Draw], path: Path = DEFAULT_OUTPUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = draws_to_frame(draws)
    frame.to_csv(path, index=False)
    return path


def load_canonical(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero: python -m kino build"
        )
    frame = pd.read_csv(path)
    return frame


def build_local_dataset(output: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    draws = merge_draws(load_historico(), load_kino_csv())
    path = save_draws(draws, output)
    frame = pd.read_csv(path)
    print(f"Dataset local: {len(frame)} sorteos -> {path}")
    if len(frame):
        print(f"Rango: {frame['draw_date'].iloc[0]} .. {frame['draw_date'].iloc[-1]}")
    return frame


def frequency_table(frame: pd.DataFrame) -> pd.Series:
    cols = _number_cols(frame)
    values = frame[cols].to_numpy().ravel()
    return pd.Series(values).value_counts().sort_index()
