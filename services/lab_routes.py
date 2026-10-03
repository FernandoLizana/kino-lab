from __future__ import annotations

import json
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

from flask import (
    Blueprint,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from config import Config
from lab.backup import make_backup, restore_backup
from lab.bias import generate_synthetic
from lab.budget import run_budget
from lab.challenges import CHALLENGES, grade
from lab.combinations import ComboError, generate_many, validate_combo
from lab.explain import explain
from lab.export import experiment_json, report_pdf, table_csv
from lab.ingest import IngestError, parse_draws, report_to_csv, sniff
from lab.model_cards import list_cards
from lab.notebook import new_experiment, new_run, validate_import
from lab.probability import ProbabilityError, build_probability_table
from lab.profiles import duplicate_profile, get_profile, list_profiles
from lab.simulate import DETAILED_N, MAX_N, QUICK_N, run_monte_carlo
from lab.strategies import compare_strategies
from models import (
    ChallengeProgress,
    DatasetImport,
    LabDraw,
    Notebook,
    SavedCombo,
    db,
)
from services.job_runner import job_runner
from services.lab_persist import explore_payload, lab_draws_as_dicts, persist_ingest

lab_bp = Blueprint("lab", __name__)

DEMO_DIR = Config.BASE_DIR / "samples" / "demo"


def _profile():
    slug = request.values.get("profile") or session.get("lab_profile") or "kino_moderno"
    try:
        profile = get_profile(slug)
    except KeyError:
        profile = get_profile("kino_moderno")
    session["lab_profile"] = profile.slug
    return profile


def _parse_date(name: str) -> date | None:
    raw = (request.values.get(name) or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _parse_nums(raw: str) -> list[int]:
    out = []
    for part in (raw or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


@lab_bp.route("/inicio")
def home():
    profile = _profile()
    counts = {
        p.slug: LabDraw.query.filter_by(profile_slug=p.slug).count()
        for p in list_profiles()
    }
    last_import = DatasetImport.query.order_by(DatasetImport.created_at.desc()).first()
    tour = request.cookies.get("kino_tour") != "1"
    return render_template(
        "lab/home.html",
        profile=profile,
        profiles=list_profiles(),
        counts=counts,
        last_import=last_import,
        show_tour=tour,
        demo_ready=DEMO_DIR.exists(),
    )


@lab_bp.route("/demo/<kind>", methods=["POST"])
def load_demo(kind: str):
    mapping = {
        "historico": DEMO_DIR / "kino_moderno_sintetico.csv",
        "mini": DEMO_DIR / "demo_6_3.csv",
        "probabilidades": None,
        "simular": None,
        "comparar": None,
    }
    if kind == "probabilidades":
        return redirect(url_for("lab.probabilities", profile="kino_moderno"))
    if kind == "simular":
        return redirect(url_for("lab.simulate", profile="kino_moderno"))
    if kind == "comparar":
        return redirect(url_for("lab.compare", profile="kino_moderno"))
    path = mapping.get(kind)
    if not path or not path.exists():
        flash("Ejemplo no encontrado.", "danger")
        return redirect(url_for("lab.home"))
    report = parse_draws(path.read_bytes(), source_name=f"demo:{path.name}")
    persist_ingest(report)
    flash(
        f"Demostración cargada ({report.accepted} sorteos sintéticos). No son resultados reales.",
        "success",
    )
    return redirect(url_for("lab.explore", profile=report.draws[0]["profile"] if report.draws else "demo_uniforme"))


@lab_bp.route("/explorar")
def explore():
    profile = _profile()
    data = explore_payload(profile.slug, _parse_date("from"), _parse_date("to"), request.args.get("number", type=int))
    return render_template("lab/explore.html", profile=profile, profiles=list_profiles(), analysis=data)


@lab_bp.route("/probabilidades")
def probabilities():
    profile = _profile()
    table = None
    error = None
    try:
        table = build_probability_table(profile).as_dict()
    except ProbabilityError as exc:
        error = str(exc)
    return render_template(
        "lab/probability.html",
        profile=profile,
        profiles=list_profiles(),
        table=table,
        error=error,
    )


@lab_bp.route("/combinaciones", methods=["GET", "POST"])
def combos():
    profile = _profile()
    generated = None
    if request.method == "POST":
        try:
            include = _parse_nums(request.form.get("include", ""))
            exclude = _parse_nums(request.form.get("exclude", ""))
            even = request.form.get("even_count", type=int)
            generated = generate_many(
                profile,
                request.form.get("count", type=int) or 5,
                request.form.get("seed", type=int) or 42,
                include=include,
                exclude=exclude,
                even_count=even if even not in (None, 0) else None,
            )
            if request.form.get("save"):
                for ticket in generated["tickets"]:
                    db.session.add(
                        SavedCombo(
                            profile_slug=profile.slug,
                            numbers_csv="-".join(f"{n:02d}" for n in ticket),
                            label=request.form.get("label") or "",
                            favorite=True,
                        )
                    )
                db.session.commit()
                flash("Combinaciones guardadas en favoritos.", "success")
        except (ComboError, ValueError) as exc:
            flash(str(exc), "danger")
    saved = SavedCombo.query.filter_by(profile_slug=profile.slug).order_by(SavedCombo.id.desc()).all()
    return render_template(
        "lab/combos.html",
        profile=profile,
        profiles=list_profiles(),
        generated=generated,
        saved=saved,
    )


@lab_bp.route("/combinaciones/<int:combo_id>/borrar", methods=["POST"])
def delete_combo(combo_id: int):
    row = SavedCombo.query.get_or_404(combo_id)
    db.session.delete(row)
    db.session.commit()
    return redirect(url_for("lab.combos"))


@lab_bp.route("/simular", methods=["GET", "POST"])
def simulate():
    profile = _profile()
    if request.method == "POST":
        mode = request.form.get("mode") or "quick"
        n = {"quick": QUICK_N, "detailed": DETAILED_N}.get(mode, request.form.get("n", type=int) or QUICK_N)
        n = min(max(int(n), 1), MAX_N)
        seed = request.form.get("seed", type=int) or 42
        raw = request.form.get("ticket") or ""
        try:
            ticket = validate_combo(profile, _parse_nums(raw)) if raw.strip() else None
        except ComboError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("lab.simulate", profile=profile.slug))
        if ticket is None:
            from lab.combinations import sample_uniform
            import random as _rnd
            ticket = sample_uniform(profile, _rnd.Random(seed))

        def work(job, progress_cb, should_cancel, should_pause=None):
            return run_monte_carlo(
                profile,
                n,
                seed,
                [ticket],
                progress_cb=progress_cb,
                should_cancel=should_cancel,
                should_pause=should_pause,
            )

        from flask import current_app

        job = job_runner.create_job("montecarlo")
        job_runner.run(current_app._get_current_object(), job.id, work)
        return redirect(url_for("job_status", job_id=job.id))
    return render_template(
        "lab/simulate.html",
        profile=profile,
        profiles=list_profiles(),
        limits={"quick": QUICK_N, "detailed": DETAILED_N, "max": MAX_N},
    )


@lab_bp.route("/comparar", methods=["GET", "POST"])
def compare():
    profile = _profile()
    result = None
    if request.method == "POST":
        draws = lab_draws_as_dicts(profile.slug, _parse_date("from"), _parse_date("to"))
        holdout = _parse_date("holdout")
        cutoff = _parse_date("cutoff")
        strategies = request.form.getlist("strategies") or ["uniform", "frequency"]
        tickets = request.form.get("tickets", type=int) or 1
        seed = request.form.get("seed", type=int) or 7
        if len(draws) < 40:
            flash("Se necesitan al menos 40 sorteos del perfil para comparar.", "warning")
        else:
            result = compare_strategies(
                profile,
                draws,
                strategies,
                tickets_per_draw=tickets,
                seed=seed,
                cutoff=cutoff,
                holdout_from=holdout,
            )
    return render_template(
        "lab/compare.html",
        profile=profile,
        profiles=list_profiles(),
        result=result,
        cards=list_cards(),
    )


@lab_bp.route("/aprender")
def learn():
    progress = {p.challenge_id: p for p in ChallengeProgress.query.all()}
    return render_template("lab/learn.html", challenges=CHALLENGES, progress=progress)


@lab_bp.route("/aprender/<cid>", methods=["GET", "POST"])
def challenge(cid: str):
    from lab.challenges import get_challenge

    item = get_challenge(cid)
    graded = None
    if request.method == "POST":
        graded = grade(cid, request.form.get("choice", type=int))
        row = ChallengeProgress.query.filter_by(challenge_id=cid).one_or_none()
        if not row:
            row = ChallengeProgress(challenge_id=cid, attempts=0)
            db.session.add(row)
        row.attempts += 1
        row.correct = bool(graded["correct"])
        db.session.commit()
    return render_template("lab/challenge.html", item=item, graded=graded)


@lab_bp.route("/sesgos", methods=["GET", "POST"])
def bias():
    profile = _profile()
    payload = None
    if request.method == "POST":
        payload = generate_synthetic(
            profile,
            request.form.get("n", type=int) or 200,
            request.form.get("seed", type=int) or 1,
            biased=request.form.get("biased") == "on",
            intensity=float(request.form.get("intensity") or 0),
        )
    return render_template("lab/bias.html", profile=profile, profiles=list_profiles(), payload=payload)


@lab_bp.route("/gastos", methods=["GET", "POST"])
def budget():
    profile = _profile()
    payload = None
    if request.method == "POST":
        prizes = {}
        for j in range(0, profile.r + 1):
            val = request.form.get(f"prize_{j}", type=float)
            if val:
                prizes[j] = val
        payload = run_budget(
            profile,
            cost=float(request.form.get("cost") or 1000),
            tickets=int(request.form.get("tickets") or 2),
            draws=int(request.form.get("draws") or 52),
            seed=int(request.form.get("seed") or 3),
            prizes=prizes or {profile.r: 1_000_000, profile.r - 1: 10_000},
            currency=request.form.get("currency") or "CLP",
        )
    return render_template("lab/budget.html", profile=profile, profiles=list_profiles(), payload=payload)


@lab_bp.route("/modelos")
def models_page():
    return render_template("lab/models.html", cards=list_cards())


@lab_bp.route("/experimentos", methods=["GET", "POST"])
def notebooks():
    if request.method == "POST":
        doc = new_experiment(
            {
                "title": request.form.get("title"),
                "question": request.form.get("question"),
                "hypothesis": request.form.get("hypothesis"),
                "notes": request.form.get("notes"),
                "profile": _profile().slug,
            },
            Config.BASE_DIR,
        )
        row = Notebook(experiment_id=doc["experiment_id"], title=doc["title"], document_json=json.dumps(doc))
        db.session.add(row)
        db.session.commit()
        return redirect(url_for("lab.notebook_detail", experiment_id=doc["experiment_id"]))
    rows = Notebook.query.filter_by(archived=False).order_by(Notebook.updated_at.desc()).all()
    return render_template("lab/notebooks.html", rows=rows)


@lab_bp.route("/experimentos/<experiment_id>", methods=["GET", "POST"])
def notebook_detail(experiment_id: str):
    row = Notebook.query.filter_by(experiment_id=experiment_id).first_or_404()
    doc = json.loads(row.document_json)
    if request.method == "POST":
        action = request.form.get("action")
        if action == "save":
            doc["notes"] = request.form.get("notes") or ""
            doc["conclusion"] = request.form.get("conclusion") or ""
            doc["question"] = request.form.get("question") or doc.get("question")
            row.title = request.form.get("title") or row.title
            doc["title"] = row.title
            row.document_json = json.dumps(doc, default=str)
            db.session.commit()
            flash("Guardado en este equipo.", "success")
        elif action == "run_prob":
            table = build_probability_table(get_profile(doc.get("profile") or "kino_moderno")).as_dict()
            new_run(doc, {"action": "probability"}, table)
            row.document_json = json.dumps(doc, default=str)
            db.session.commit()
        elif action == "duplicate":
            copy = new_experiment({**doc, "experiment_id": None, "title": doc["title"] + " (copia)"}, Config.BASE_DIR)
            db.session.add(Notebook(experiment_id=copy["experiment_id"], title=copy["title"], document_json=json.dumps(copy)))
            db.session.commit()
            return redirect(url_for("lab.notebook_detail", experiment_id=copy["experiment_id"]))
        elif action == "archive":
            row.archived = True
            db.session.commit()
            return redirect(url_for("lab.notebooks"))
        elif action == "delete":
            db.session.delete(row)
            db.session.commit()
            flash("Eliminado.", "warning")
            return redirect(url_for("lab.notebooks"))
    return render_template("lab/notebook.html", row=row, doc=doc)


@lab_bp.route("/experimentos/<experiment_id>/export.json")
def export_json(experiment_id: str):
    row = Notebook.query.filter_by(experiment_id=experiment_id).first_or_404()
    return Response(row.document_json, mimetype="application/json", headers={
        "Content-Disposition": f"attachment; filename=experimento_{experiment_id}.json"
    })


@lab_bp.route("/experimentos/<experiment_id>/export.pdf")
def export_pdf(experiment_id: str):
    row = Notebook.query.filter_by(experiment_id=experiment_id).first_or_404()
    doc = json.loads(row.document_json)
    pdf = report_pdf(
        doc.get("title", "Experimento"),
        {
            "Pregunta": doc.get("question") or "—",
            "Hipotesis": doc.get("hypothesis") or "—",
            "Perfil": str(doc.get("profile")),
            "Resultados": json.dumps(doc.get("runs", [])[-1] if doc.get("runs") else {}, ensure_ascii=False)[:1500],
            "Limitaciones": "Cálculo teórico, histórico y simulación no son predicción. Notas personales no son conclusiones del motor.",
            "Reproduccion": f"engine={doc.get('engine')} commit={doc.get('git', {}).get('commit')}",
        },
    )
    return send_file(BytesIO(pdf), mimetype="application/pdf", as_attachment=True, download_name=f"experimento_{experiment_id}.pdf")


@lab_bp.route("/experimentos/importar", methods=["POST"])
def import_notebook():
    file = request.files.get("file")
    if not file:
        flash("Elige un JSON.", "danger")
        return redirect(url_for("lab.notebooks"))
    try:
        doc = validate_import(file.read())
    except Exception as exc:
        flash(f"JSON inválido: {exc}", "danger")
        return redirect(url_for("lab.notebooks"))
    if Notebook.query.filter_by(experiment_id=doc["experiment_id"]).first():
        flash("Ya existe ese experimento; no se sobrescribió.", "warning")
        return redirect(url_for("lab.notebook_detail", experiment_id=doc["experiment_id"]))
    db.session.add(Notebook(experiment_id=doc["experiment_id"], title=doc["title"], document_json=json.dumps(doc)))
    db.session.commit()
    flash("Importado y validado.", "success")
    return redirect(url_for("lab.notebook_detail", experiment_id=doc["experiment_id"]))


@lab_bp.route("/importar", methods=["GET", "POST"])
def importer():
    preview = None
    report = None
    if request.method == "POST":
        file = request.files.get("file")
        stored = Config.UPLOAD_FOLDER / "_ingest_preview.csv"
        if file and file.filename:
            content = file.read()
            stored.write_bytes(content)
            session["ingest_name"] = file.filename
        elif request.form.get("confirm") == "1" and stored.exists():
            content = stored.read_bytes()
        else:
            flash("Selecciona un CSV.", "danger")
            return redirect(url_for("lab.importer"))
        try:
            if request.form.get("confirm") == "1":
                report_obj = parse_draws(
                    content,
                    source_name=session.get("ingest_name") or "upload.csv",
                    has_header=request.form.get("has_header") == "1",
                    delimiter=request.form.get("delimiter") or None,
                    date_index=request.form.get("date_index", type=int),
                )
                persist_ingest(report_obj)
                flash(
                    f"Importados {report_obj.accepted}. Excluidos {report_obj.excluded}. "
                    f"Duplicados {report_obj.duplicates}. Conflictos {report_obj.conflicts}. "
                    "El archivo original no se modificó.",
                    "success",
                )
                return redirect(url_for("lab.explore"))
            preview = sniff(content)
        except IngestError as exc:
            flash(str(exc), "danger")
    last = DatasetImport.query.order_by(DatasetImport.created_at.desc()).first()
    return render_template("lab/import.html", preview=preview, last=last, report=report)


@lab_bp.route("/importar/errores.csv")
def importer_errors():
    last = DatasetImport.query.order_by(DatasetImport.created_at.desc()).first()
    if not last:
        flash("No hay informe.", "warning")
        return redirect(url_for("lab.importer"))
    data = json.loads(last.report_json)
    from lab.ingest import IngestReport

    fake = IngestReport(
        source_name=data["source_name"],
        fingerprint=data["fingerprint"],
        imported_at=data["imported_at"],
        rows_read=data["rows_read"],
        accepted=data["accepted"],
        duplicates=data["duplicates"],
        conflicts=data["conflicts"],
        excluded=data["excluded"],
        by_profile=data.get("by_profile", {}),
        period=data.get("period", {}),
        exclusions=data.get("exclusions", []),
    )
    return Response(report_to_csv(fake), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=informe_importacion.csv"
    })


@lab_bp.route("/asistente", methods=["GET", "POST"])
def assistant():
    answer = None
    if request.method == "POST":
        ctx = {}
        kind = request.form.get("kind") or ""
        if kind == "calculo_teorico":
            ctx = build_probability_table(_profile()).as_dict()
        elif kind == "datos_historicos":
            ctx = explore_payload(_profile().slug, None, None, None)
        answer = explain(request.form.get("question") or "", ctx)
    return render_template("lab/assistant.html", answer=answer)


@lab_bp.route("/ajustes", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "backup":
            dest = Config.BASE_DIR / "data" / f"respaldo-{datetime.utcnow().strftime('%Y%m%d-%H%M')}.zip"
            make_backup(Config.BASE_DIR / "data", {"note": "local"}, dest)
            return send_file(dest, as_attachment=True)
        if action == "restore":
            file = request.files.get("file")
            if not file:
                flash("Elige un zip de respaldo.", "danger")
            else:
                tmp = Config.BASE_DIR / "data" / "_restore.zip"
                tmp.write_bytes(file.read())
                restore_backup(tmp, Config.BASE_DIR / "data")
                flash("Restaurado. Reinicia la app si la base estaba abierta.", "success")
        if action == "wipe":
            LabDraw.query.delete()
            DatasetImport.query.delete()
            Notebook.query.delete()
            SavedCombo.query.delete()
            db.session.commit()
            flash("Datos del laboratorio borrados de este equipo.", "warning")
        if action == "copy_profile":
            try:
                duplicate_profile(
                    request.form.get("source") or "kino_moderno",
                    request.form.get("new_slug") or "copia",
                    request.form.get("new_name") or "Copia",
                )
                flash("Perfil copiado en memoria (sesión). Los experimentos viejos no cambian.", "success")
            except Exception as exc:
                flash(str(exc), "danger")
    return render_template("lab/settings.html", profiles=list_profiles())


@lab_bp.route("/export/explorar.csv")
def export_explore_csv():
    profile = _profile()
    data = explore_payload(profile.slug, _parse_date("from"), _parse_date("to"), request.args.get("number", type=int))
    rows = [[r["n"], r["count"], f"{r['relative']:.4f}", r["gap"]] for r in data["frequency"]]
    return Response(table_csv(["numero", "count", "relativa", "atraso"], rows), mimetype="text/csv", headers={
        "Content-Disposition": "attachment; filename=explorar.csv"
    })


@lab_bp.route("/api/jobs/<int:job_id>/pause", methods=["POST"])
def pause_job(job_id: int):
    job_runner.request_pause(job_id)
    return jsonify({"ok": True})
