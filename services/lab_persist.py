from __future__ import annotations

import json
from datetime import date

from lab.ingest import IngestReport, fingerprint_draws
from lab.profiles import get_profile
from models import DatasetImport, Draw, LabDraw, db


def persist_ingest(report: IngestReport, *, sync_legacy_draw: bool = True) -> dict:
    inserted = 0
    for row in report.draws:
        existing = LabDraw.query.filter_by(
            profile_slug=row["profile"], draw_date=row["draw_date"]
        ).one_or_none()
        payload = json.dumps(row["numbers"])
        if existing:
            existing.numbers_json = payload
            existing.source = row.get("source", "upload")
            existing.synthetic = bool(row.get("synthetic"))
            existing.draw_number = row.get("draw_number")
        else:
            db.session.add(
                LabDraw(
                    profile_slug=row["profile"],
                    draw_date=row["draw_date"],
                    draw_number=row.get("draw_number"),
                    numbers_json=payload,
                    source=row.get("source", "upload"),
                    synthetic=bool(row.get("synthetic")),
                )
            )
            inserted += 1
        if sync_legacy_draw and row["profile"] == "kino_moderno" and len(row["numbers"]) == 14:
            nums = row["numbers"]
            legacy = Draw.query.filter_by(draw_date=row["draw_date"]).one_or_none()
            fields = {f"n{i}": nums[i - 1] for i in range(1, 15)}
            fields["source"] = row.get("source", "upload")
            fields["draw_number"] = row.get("draw_number")
            if legacy:
                for k, v in fields.items():
                    setattr(legacy, k, v)
            else:
                db.session.add(Draw(draw_date=row["draw_date"], **fields))
    rec = DatasetImport(
        source_name=report.source_name,
        fingerprint=report.fingerprint,
        data_fingerprint=fingerprint_draws(report.draws),
        report_json=json.dumps(report.as_dict(), ensure_ascii=False, default=str),
    )
    db.session.add(rec)
    db.session.commit()
    return {"inserted": inserted, "import_id": rec.id, "report": report.as_dict()}


def lab_draws_as_dicts(profile_slug: str, date_from: date | None = None, date_to: date | None = None) -> list[dict]:
    q = LabDraw.query.filter_by(profile_slug=profile_slug)
    if date_from:
        q = q.filter(LabDraw.draw_date >= date_from)
    if date_to:
        q = q.filter(LabDraw.draw_date <= date_to)
    rows = q.order_by(LabDraw.draw_date.asc()).all()
    return [
        {
            "draw_date": r.draw_date,
            "numbers": r.numbers(),
            "profile": r.profile_slug,
            "source": r.source,
            "synthetic": r.synthetic,
        }
        for r in rows
    ]


def explore_payload(profile_slug: str, date_from: date | None, date_to: date | None, number: int | None) -> dict:
    profile = get_profile(profile_slug)
    rows = lab_draws_as_dicts(profile_slug, date_from, date_to)
    if number:
        rows = [r for r in rows if number in r["numbers"]]
    n = len(rows)
    freq = [0] * (profile.n + 1)
    last_seen = {i: None for i in range(1, profile.n + 1)}
    gaps = {i: n for i in range(1, profile.n + 1)}
    for idx, row in enumerate(rows):
        for ball in row["numbers"]:
            if 1 <= ball <= profile.n:
                freq[ball] += 1
                last_seen[ball] = idx
    for i in range(1, profile.n + 1):
        if last_seen[i] is not None:
            gaps[i] = n - 1 - last_seen[i]
    expected = n * profile.k / profile.n if n else 0
    return {
        "kind": "datos_historicos",
        "profile": profile.slug,
        "draw_count": n,
        "period": {
            "from": rows[0]["draw_date"].isoformat() if rows else None,
            "to": rows[-1]["draw_date"].isoformat() if rows else None,
        },
        "expected_frequency": expected,
        "frequency": [
            {
                "n": i,
                "count": freq[i],
                "relative": (freq[i] / n) if n else 0,
                "gap": gaps[i],
            }
            for i in range(1, profile.n + 1)
        ],
        "draws": [
            {
                "date": r["draw_date"].isoformat(),
                "numbers": r["numbers"],
                "synthetic": r["synthetic"],
            }
            for r in rows[-80:]
        ],
        "empty_reason": None if n else "No hay sorteos para ese perfil y filtro.",
        "denominator_note": f"Frecuencia relativa = count / {n} sorteos filtrados.",
    }
