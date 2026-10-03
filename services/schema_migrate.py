from __future__ import annotations

from models import db


def ensure_columns() -> None:
    """Añade columnas nuevas en SQLite existente sin borrar datos."""
    engine = db.engine
    with engine.begin() as conn:
        rows = conn.exec_driver_sql("PRAGMA table_info(jobs)").fetchall()
        if not rows:
            return
        existing = {row[1] for row in rows}
        if "pause_requested" not in existing:
            conn.exec_driver_sql(
                "ALTER TABLE jobs ADD COLUMN pause_requested BOOLEAN DEFAULT 0"
            )
        if "checkpoint_json" not in existing:
            conn.exec_driver_sql(
                "ALTER TABLE jobs ADD COLUMN checkpoint_json TEXT DEFAULT '{}'"
            )
