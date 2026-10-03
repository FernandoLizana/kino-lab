from __future__ import annotations

import json
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Draw(db.Model):
    __tablename__ = "draws"

    id = db.Column(db.Integer, primary_key=True)
    draw_date = db.Column(db.Date, nullable=False, unique=True, index=True)
    draw_number = db.Column(db.Integer, nullable=True, index=True)
    n1 = db.Column(db.Integer, nullable=False)
    n2 = db.Column(db.Integer, nullable=False)
    n3 = db.Column(db.Integer, nullable=False)
    n4 = db.Column(db.Integer, nullable=False)
    n5 = db.Column(db.Integer, nullable=False)
    n6 = db.Column(db.Integer, nullable=False)
    n7 = db.Column(db.Integer, nullable=False)
    n8 = db.Column(db.Integer, nullable=False)
    n9 = db.Column(db.Integer, nullable=False)
    n10 = db.Column(db.Integer, nullable=False)
    n11 = db.Column(db.Integer, nullable=False)
    n12 = db.Column(db.Integer, nullable=False)
    n13 = db.Column(db.Integer, nullable=False)
    n14 = db.Column(db.Integer, nullable=False)
    source = db.Column(db.String(64), default="upload")
    created_at = db.Column(db.DateTime, default=utcnow)

    def numbers(self) -> list[int]:
        return [
            self.n1, self.n2, self.n3, self.n4, self.n5, self.n6, self.n7,
            self.n8, self.n9, self.n10, self.n11, self.n12, self.n13, self.n14,
        ]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "draw_date": self.draw_date.isoformat(),
            "draw_number": self.draw_number,
            "numbers": self.numbers(),
            "source": self.source,
        }


class Experiment(db.Model):
    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    kind = db.Column(db.String(40), nullable=False)  # score | backtest | portfolio
    seed = db.Column(db.Integer, nullable=True)
    sample_size = db.Column(db.Integer, nullable=True)
    ticket_count = db.Column(db.Integer, nullable=True)
    max_overlap = db.Column(db.Integer, nullable=True)
    weights_json = db.Column(db.Text, nullable=False, default="{}")
    metrics_json = db.Column(db.Text, nullable=False, default="{}")
    status = db.Column(db.String(20), default="done")
    created_at = db.Column(db.DateTime, default=utcnow)

    results = db.relationship(
        "ExperimentResult",
        backref="experiment",
        cascade="all, delete-orphan",
        lazy=True,
    )


class ExperimentResult(db.Model):
    __tablename__ = "experiment_results"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey("experiments.id"), nullable=False)
    rank = db.Column(db.Integer, nullable=False)
    numbers_csv = db.Column(db.String(80), nullable=False)
    score = db.Column(db.Float, nullable=False)
    even_count = db.Column(db.Integer, nullable=False)
    odd_count = db.Column(db.Integer, nullable=False)
    total_sum = db.Column(db.Integer, nullable=False)
    consecutive = db.Column(db.Integer, nullable=False)
    coverage = db.Column(db.Float, nullable=True)
    similarity = db.Column(db.Float, nullable=True)
    extras_json = db.Column(db.Text, default="{}")


class Job(db.Model):
    __tablename__ = "jobs"

    id = db.Column(db.Integer, primary_key=True)
    kind = db.Column(db.String(40), nullable=False)
    status = db.Column(db.String(20), default="pending")  # pending|running|done|error|cancelled
    progress = db.Column(db.Float, default=0.0)
    message = db.Column(db.String(255), default="")
    result_json = db.Column(db.Text, default="{}")
    cancel_requested = db.Column(db.Boolean, default=False)
    pause_requested = db.Column(db.Boolean, default=False)
    checkpoint_json = db.Column(db.Text, default="{}")
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


# ---------------------------------------------------------------------------
# Módulo "Otras loterías" — tablas propias, sin mezclar con Kino (Draw).
# ---------------------------------------------------------------------------


class LotteryGame(db.Model):
    __tablename__ = "lottery_games"

    id = db.Column(db.Integer, primary_key=True)
    slug = db.Column(db.String(64), nullable=False, unique=True, index=True)
    name = db.Column(db.String(120), nullable=False)
    country = db.Column(db.String(80), nullable=False)
    rules_json = db.Column(db.Text, nullable=False, default="{}")
    official_url = db.Column(db.String(255), nullable=True)
    history_url = db.Column(db.String(255), nullable=True)
    sync_adapter = db.Column(db.String(64), nullable=True)  # None = solo CSV
    enabled = db.Column(db.Boolean, default=True)
    sort_order = db.Column(db.Integer, default=0)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)
    created_at = db.Column(db.DateTime, default=utcnow)

    draws = db.relationship(
        "LotteryDraw",
        backref="game",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )

    def rules(self) -> dict:
        try:
            return json.loads(self.rules_json or "{}")
        except json.JSONDecodeError:
            return {}

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "slug": self.slug,
            "name": self.name,
            "country": self.country,
            "rules": self.rules(),
            "official_url": self.official_url,
            "history_url": self.history_url,
            "sync_adapter": self.sync_adapter,
            "enabled": self.enabled,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class LotteryDraw(db.Model):
    __tablename__ = "lottery_draws"
    __table_args__ = (
        db.UniqueConstraint("game_id", "draw_date", name="uq_lottery_draw_game_date"),
    )

    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey("lottery_games.id"), nullable=False, index=True)
    draw_date = db.Column(db.Date, nullable=False, index=True)
    draw_number = db.Column(db.Integer, nullable=True)
    main_numbers = db.Column(db.Text, nullable=False)  # JSON list
    bonus_numbers = db.Column(db.Text, nullable=False, default="[]")  # JSON list
    source = db.Column(db.String(64), default="upload")
    created_at = db.Column(db.DateTime, default=utcnow)

    def main(self) -> list[int]:
        try:
            return [int(x) for x in json.loads(self.main_numbers or "[]")]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def bonus(self) -> list[int]:
        try:
            return [int(x) for x in json.loads(self.bonus_numbers or "[]")]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "game_id": self.game_id,
            "draw_date": self.draw_date.isoformat(),
            "draw_number": self.draw_number,
            "main_numbers": self.main(),
            "bonus_numbers": self.bonus(),
            "source": self.source,
        }


class LotteryAnalysisSnapshot(db.Model):
    __tablename__ = "lottery_analysis_snapshots"

    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.Integer, db.ForeignKey("lottery_games.id"), nullable=False, index=True)
    draw_count = db.Column(db.Integer, nullable=False, default=0)
    date_from = db.Column(db.Date, nullable=True)
    date_to = db.Column(db.Date, nullable=True)
    summary_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, default=utcnow)


class LabDraw(db.Model):
    """Sorteo normalizado por perfil (14 o 15 bolillas). No recorta números."""

    __tablename__ = "lab_draws"
    __table_args__ = (
        db.UniqueConstraint("profile_slug", "draw_date", name="uq_lab_draw_profile_date"),
    )

    id = db.Column(db.Integer, primary_key=True)
    profile_slug = db.Column(db.String(64), nullable=False, index=True)
    draw_date = db.Column(db.Date, nullable=False, index=True)
    draw_number = db.Column(db.Integer, nullable=True)
    numbers_json = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(80), default="upload")
    synthetic = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    def numbers(self) -> list[int]:
        try:
            return [int(x) for x in json.loads(self.numbers_json or "[]")]
        except (json.JSONDecodeError, TypeError, ValueError):
            return []

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "profile": self.profile_slug,
            "draw_date": self.draw_date.isoformat(),
            "draw_number": self.draw_number,
            "numbers": self.numbers(),
            "source": self.source,
            "synthetic": self.synthetic,
        }


class DatasetImport(db.Model):
    __tablename__ = "dataset_imports"

    id = db.Column(db.Integer, primary_key=True)
    source_name = db.Column(db.String(200), nullable=False)
    fingerprint = db.Column(db.String(64), nullable=False)
    data_fingerprint = db.Column(db.String(64), nullable=True)
    report_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, default=utcnow)


class SavedCombo(db.Model):
    __tablename__ = "saved_combos"

    id = db.Column(db.Integer, primary_key=True)
    profile_slug = db.Column(db.String(64), nullable=False)
    numbers_csv = db.Column(db.String(120), nullable=False)
    label = db.Column(db.String(120), default="")
    favorite = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utcnow)


class Notebook(db.Model):
    __tablename__ = "notebooks"

    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.String(64), unique=True, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    document_json = db.Column(db.Text, nullable=False)
    archived = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class ChallengeProgress(db.Model):
    __tablename__ = "challenge_progress"

    id = db.Column(db.Integer, primary_key=True)
    challenge_id = db.Column(db.String(64), unique=True, nullable=False)
    correct = db.Column(db.Boolean, default=False)
    attempts = db.Column(db.Integer, default=0)
    updated_at = db.Column(db.DateTime, default=utcnow, onupdate=utcnow)


class AppSetting(db.Model):
    __tablename__ = "app_settings"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(80), unique=True, nullable=False)
    value_json = db.Column(db.Text, default="{}")
