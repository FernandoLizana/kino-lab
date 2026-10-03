from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from sqlalchemy import desc

from config import Config
from models import (
    Draw,
    Experiment,
    ExperimentResult,
    Job,
    LabDraw,
    LotteryAnalysisSnapshot,  # noqa: F401 — registered for create_all
    LotteryDraw,  # noqa: F401
    LotteryGame,  # noqa: F401
    db,
)
from services.analysis_service import build_analysis, draws_matrix
from services.color_service import build_color_arc_payload, run_color_experiment
from services.backtesting_service import run_walk_forward
from services.csv_service import CsvValidationError, draws_to_csv_bytes, parse_csv_bytes
from services.job_runner import job_runner
from services.logging_setup import setup_logging
from services.portfolio_service import build_portfolio
from services.scoring_service import normalize_weights, score_sample
from services.astrology_service import (
    format_draw_label,
    next_draw_datetime,
    run_astrology_experiment,
)
from services.lunar_service import run_lunar_experiment
from services.tarot_service import run_tarot_experiment
from services.statistical_seed_service import generate_seed_candidates
from services.lottery_catalog_service import seed_lottery_catalog
from services.lottery_routes import lottery_bp
from services.lab_routes import lab_bp
from services.schema_migrate import ensure_columns

logger = setup_logging()


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config_overrides:
        app.config.update(config_overrides)
    Config.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (Config.BASE_DIR / "data").mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    setup_logging(app)

    with app.app_context():
        db.create_all()
        try:
            ensure_columns()
        except Exception as exc:
            logger.warning("schema migrate: %s", exc)
        if not app.config.get("TESTING"):
            _maybe_seed_from_existing()
            _maybe_seed_lab_draws()
        try:
            seed_lottery_catalog(force_update_rules=True)
        except Exception as exc:
            logger.warning("Could not seed lottery catalog: %s", exc)

    register_routes(app)
    app.register_blueprint(lottery_bp)
    app.register_blueprint(lab_bp)
    return app


def _maybe_seed_from_existing() -> None:
    if Draw.query.count() > 0:
        return
    candidates = [
        Config.BASE_DIR / "data" / "kino_draws.csv",
        Config.BASE_DIR / "kino.csv",
        Config.BASE_DIR / "historico.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            content = path.read_bytes()
            rows = parse_csv_bytes(content, source=path.name)
            _persist_draws(rows, replace=False)
            logger.info("Seeded %s draws from %s", len(rows), path.name)
            return
        except Exception as exc:
            logger.warning("Could not seed from %s: %s", path, exc)


def _maybe_seed_lab_draws() -> None:
    if LabDraw.query.count() > 0:
        return
    path = Config.BASE_DIR / "historico.csv"
    if not path.exists():
        return
    try:
        from lab.ingest import parse_draws
        from services.lab_persist import persist_ingest

        report = parse_draws(path.read_bytes(), source_name="historico.csv")
        persist_ingest(report, sync_legacy_draw=Draw.query.count() == 0)
        logger.info(
            "LabDraw seed: %s aceptados, perfiles=%s",
            report.accepted,
            report.by_profile,
        )
    except Exception as exc:
        logger.warning("Could not seed LabDraw: %s", exc)


def _persist_draws(rows: list[dict], replace: bool = False) -> dict:
    if replace:
        Draw.query.delete()
        db.session.commit()

    inserted = 0
    updated = 0
    for row in rows:
        existing = Draw.query.filter_by(draw_date=row["draw_date"]).one_or_none()
        nums = row["numbers"]
        fields = {
            "draw_number": row.get("draw_number"),
            "source": row.get("source", "upload"),
            **{f"n{i}": nums[i - 1] for i in range(1, 15)},
        }
        if existing:
            for k, v in fields.items():
                setattr(existing, k, v)
            updated += 1
        else:
            db.session.add(Draw(draw_date=row["draw_date"], **fields))
            inserted += 1
    db.session.commit()
    return {"inserted": inserted, "updated": updated, "total": Draw.query.count()}


def _all_draws_ordered():
    return Draw.query.order_by(Draw.draw_date.asc()).all()


def _sync_remote_latest_draw() -> dict:
    """Consulta kinohistorico.cl y actualiza SQLite si el último sorteo remoto es nuevo."""
    try:
        from kino.remote import KinoHistoricClient
    except Exception as exc:  # pragma: no cover - import guard
        return {
            "history_freshness": "api_unavailable",
            "message": f"Cliente remoto no disponible: {exc}",
            "remote_latest_date": None,
            "remote_latest_draw_number": None,
        }

    try:
        remote = KinoHistoricClient(timeout=20, max_retries=2, delay_seconds=0.1).fetch_latest()
    except Exception as exc:
        logger.warning("API kinohistorico.cl no disponible: %s", exc)
        return {
            "history_freshness": "api_unavailable",
            "message": (
                "No se pudo consultar kinohistorico.cl; "
                "se continúa con el histórico local."
            ),
            "remote_latest_date": None,
            "remote_latest_draw_number": None,
            "error": str(exc),
        }

    remote_nums = list(remote.numbers)
    existing = Draw.query.filter_by(draw_date=remote.draw_date).one_or_none()
    local_latest = Draw.query.order_by(Draw.draw_date.desc()).first()

    same_as_existing = (
        existing is not None
        and existing.draw_number == remote.draw_number
        and existing.numbers() == remote_nums
    )
    same_as_latest = (
        local_latest is not None
        and local_latest.draw_date == remote.draw_date
        and local_latest.draw_number == remote.draw_number
        and local_latest.numbers() == remote_nums
    )

    if same_as_existing and same_as_latest:
        return {
            "history_freshness": "already_current",
            "message": "Histórico local ya estaba al día con kinohistorico.cl.",
            "remote_latest_date": remote.draw_date.isoformat(),
            "remote_latest_draw_number": remote.draw_number,
        }

    _persist_draws(
        [
            {
                "draw_date": remote.draw_date,
                "draw_number": remote.draw_number,
                "numbers": remote_nums,
                "source": remote.source or "kinohistorico.cl",
            }
        ],
        replace=False,
    )
    return {
        "history_freshness": "updated",
        "message": (
            f"Histórico actualizado con sorteo remoto "
            f"{remote.draw_date.isoformat()} (#{remote.draw_number})."
        ),
        "remote_latest_date": remote.draw_date.isoformat(),
        "remote_latest_draw_number": remote.draw_number,
    }


def _weights_from_form(form) -> dict[str, float]:
    weights = {}
    for key in Config.DEFAULT_WEIGHTS:
        raw = form.get(f"w_{key}")
        if raw is not None and raw != "":
            weights[key] = float(raw)
    return normalize_weights(weights)


def _save_experiment(
    *,
    name: str,
    kind: str,
    seed: int | None,
    sample_size: int | None,
    ticket_count: int | None,
    max_overlap: int | None,
    weights: dict,
    metrics: dict,
    results: list[dict],
) -> Experiment:
    exp = Experiment(
        name=name,
        kind=kind,
        seed=seed,
        sample_size=sample_size,
        ticket_count=ticket_count,
        max_overlap=max_overlap,
        weights_json=json.dumps(weights),
        metrics_json=json.dumps(metrics, default=str),
        status="done",
    )
    db.session.add(exp)
    db.session.flush()
    for item in results:
        db.session.add(
            ExperimentResult(
                experiment_id=exp.id,
                rank=int(item.get("rank", 0)),
                numbers_csv=item.get("numbers_csv")
                or "-".join(f"{n:02d}" for n in item.get("numbers", [])),
                score=float(item.get("score", 0)),
                even_count=int(item.get("even_count", 0)),
                odd_count=int(item.get("odd_count", 0)),
                total_sum=int(item.get("total_sum", 0)),
                consecutive=int(item.get("consecutive", 0)),
                coverage=item.get("coverage"),
                similarity=item.get("similarity"),
                extras_json=json.dumps(
                    {"components": item.get("components", {})}, default=str
                ),
            )
        )
    db.session.commit()
    return exp


def register_routes(app: Flask) -> None:
    @app.context_processor
    def inject_globals():
        return {
            "disclaimer": (
                "Advertencia: estos análisis son históricos y exploratorios. "
                "No garantizan resultados futuros. Si el sorteo es uniforme, "
                "cada combinación de 14 números entre 1 y 25 tiene exactamente "
                "la misma probabilidad teórica (1 / 4.457.400)."
            ),
            "default_weights": Config.DEFAULT_WEIGHTS,
            "default_astro_lat": Config.ASTRO_LATITUDE,
            "default_astro_lon": Config.ASTRO_LONGITUDE,
            "next_draw_label": format_draw_label(next_draw_datetime()),
            "astro_draw_time": f"{Config.ASTRO_DRAW_HOUR:02d}:{Config.ASTRO_DRAW_MINUTE:02d}",
        }

    @app.route("/")
    def dashboard():
        draws = _all_draws_ordered()
        experiments = Experiment.query.order_by(desc(Experiment.created_at)).limit(8).all()
        best = (
            ExperimentResult.query.order_by(desc(ExperimentResult.score))
            .limit(10)
            .all()
        )
        last_date = draws[-1].draw_date.isoformat() if draws else None
        return render_template(
            "dashboard.html",
            draw_count=len(draws),
            last_date=last_date,
            experiment_count=Experiment.query.count(),
            experiments=experiments,
            best_results=best,
        )

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "POST":
            file = request.files.get("file")
            replace = request.form.get("replace") == "on"
            if not file or not file.filename:
                flash("Selecciona un archivo CSV.", "danger")
                return redirect(url_for("upload"))
            if not file.filename.lower().endswith(".csv"):
                flash("Solo se aceptan archivos .csv", "danger")
                return redirect(url_for("upload"))
            try:
                content = file.read()
                rows = parse_csv_bytes(content, source=file.filename)
                stats = _persist_draws(rows, replace=replace)
                flash(
                    f"Carga OK: {stats['inserted']} nuevos, {stats['updated']} actualizados. "
                    f"Total en BD: {stats['total']}.",
                    "success",
                )
                return redirect(url_for("analysis"))
            except CsvValidationError as exc:
                flash(str(exc), "danger")
            except Exception as exc:
                logger.exception("upload failed")
                flash(f"Error al procesar el CSV: {exc}", "danger")
        return render_template("upload.html", draw_count=Draw.query.count())

    @app.route("/analysis")
    def analysis():
        draws = _all_draws_ordered()
        data = build_analysis(draws) if draws else None
        return render_template("analysis.html", analysis=data, draw_count=len(draws))

    @app.route("/colores")
    def colors():
        draws = _all_draws_ordered()
        arc = build_color_arc_payload(draws) if draws else None
        pred_id = request.args.get("pred", type=int)
        prediction = None
        if pred_id:
            selected = db.session.get(Experiment, pred_id)
            if selected and selected.kind == "color":
                prediction = json.loads(selected.metrics_json or "{}")
                prediction["experiment_id"] = selected.id
        elif draws:
            latest = (
                Experiment.query.filter_by(kind="color")
                .order_by(desc(Experiment.created_at))
                .first()
            )
            if latest:
                prediction = json.loads(latest.metrics_json or "{}")
                prediction["experiment_id"] = latest.id
        return render_template("colors.html", arc=arc, prediction=prediction)

    @app.route("/colores/predecir", methods=["POST"])
    def color_predict():
        draws = _all_draws_ordered()
        if len(draws) < 5:
            flash("Necesitas al menos 5 sorteos cargados para predecir por color.", "warning")
            return redirect(url_for("upload"))

        sample_size = int(request.form.get("sample_size") or 20_000)
        sample_size = max(1_000, min(sample_size, 200_000))
        seed = int(request.form.get("seed") or 42)
        mode = (request.form.get("mode") or "continue").strip().lower()
        raw_hue = request.form.get("target_hue")
        target_hue = float(raw_hue) if raw_hue not in (None, "") else None
        target = next_draw_datetime()
        job = job_runner.create_job("color")

        def work(job_obj, progress_cb, should_cancel):
            history = _all_draws_ordered()
            metrics, results = run_color_experiment(
                history,
                mode=mode,
                target_hue=target_hue,
                sample_size=sample_size,
                seed=seed,
                top_k=20,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
            )
            exp = _save_experiment(
                name=f"Color próximo sorteo {target.date().isoformat()}",
                kind="color",
                seed=seed,
                sample_size=sample_size,
                ticket_count=None,
                max_overlap=None,
                weights={"mode": mode, "target_hue": target_hue},
                metrics=metrics,
                results=results,
            )
            return {
                "experiment_id": exp.id,
                "count": len(results),
                "mode": "color",
                "redirect": f"/colores?pred={exp.id}",
            }

        job_runner.run(app, job.id, work)
        return redirect(url_for("job_status", job_id=job.id))

    @app.route("/score", methods=["GET", "POST"])
    def score():
        if request.method == "POST":
            draws = _all_draws_ordered()
            if len(draws) < 10:
                flash("Necesitas al menos 10 sorteos cargados.", "warning")
                return redirect(url_for("upload"))

            sample_size = int(request.form.get("sample_size") or Config.DEFAULT_SAMPLE_SIZE)
            seed = int(request.form.get("seed") or 42)
            top_k = int(request.form.get("top_k") or 50)
            weights = _weights_from_form(request.form)
            sample_size = max(100, min(sample_size, 200_000))
            top_k = max(5, min(top_k, 200))

            job = job_runner.create_job("score")

            def work(job_obj, progress_cb, should_cancel):
                matrix = draws_matrix(_all_draws_ordered())
                results = score_sample(
                    matrix,
                    sample_size=sample_size,
                    seed=seed,
                    weights=weights,
                    top_k=top_k,
                    progress_cb=progress_cb,
                    should_cancel=should_cancel,
                )
                exp = _save_experiment(
                    name=f"Score seed={seed} n={sample_size}",
                    kind="score",
                    seed=seed,
                    sample_size=sample_size,
                    ticket_count=None,
                    max_overlap=None,
                    weights=weights,
                    metrics={"top_k": top_k, "count": len(results)},
                    results=results,
                )
                return {"experiment_id": exp.id, "count": len(results)}

            job_runner.run(app, job.id, work)
            return redirect(url_for("job_status", job_id=job.id))

        return render_template("score.html")

    @app.route("/statistical-seed", methods=["POST"])
    def statistical_seed():
        draws = _all_draws_ordered()
        if len(draws) < 10:
            flash("Necesitas al menos 10 sorteos cargados.", "warning")
            return redirect(url_for("upload"))

        sample_size = int(request.form.get("sample_size") or 20_000)
        sample_size = max(1_000, min(sample_size, 200_000))
        experiment_date = date.today()
        weights = normalize_weights(None)
        job = job_runner.create_job("statistical_seed")

        def work(job_obj, progress_cb, should_cancel):
            history = _all_draws_ordered()
            seed_info, results = generate_seed_candidates(
                history,
                experiment_date=experiment_date,
                sample_size=sample_size,
                weights=weights,
                top_k=20,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
            )
            payload = seed_info["payload"]
            metrics = {
                "seed_sha256": seed_info["sha256"],
                "seed_integer": seed_info["numeric_seed"],
                "experiment_date": payload["experiment_date"],
                "history_as_of": payload["history_as_of"],
                "draw_count": payload["draw_count"],
                "recent_draw_count": len(payload["recent_draws"]),
                "candidate_count": sample_size,
                "top_k": len(results),
            }
            exp = _save_experiment(
                name=f"Semilla estadística {seed_info['sha256'][:12]}",
                kind="statistical_seed",
                seed=seed_info["numeric_seed"],
                sample_size=sample_size,
                ticket_count=None,
                max_overlap=None,
                weights=weights,
                metrics=metrics,
                results=results,
            )
            return {
                "experiment_id": exp.id,
                "count": len(results),
                "seed_sha256": seed_info["sha256"],
            }

        job_runner.run(app, job.id, work)
        return redirect(url_for("job_status", job_id=job.id))

    @app.route("/astrology-predict", methods=["POST"])
    def astrology_predict():
        draws = _all_draws_ordered()
        if len(draws) < 20:
            flash("Necesitas al menos 20 sorteos cargados para la predicción astrológica.", "warning")
            return redirect(url_for("upload"))

        sample_size = int(request.form.get("sample_size") or 20_000)
        sample_size = max(1_000, min(sample_size, 200_000))
        blend = float(request.form.get("blend") or 0.35)
        blend = max(0.0, min(blend, 1.0))
        seed = int(request.form.get("seed") or 42)
        target = next_draw_datetime()
        weights = normalize_weights(None)
        job = job_runner.create_job("astrology")

        def work(job_obj, progress_cb, should_cancel):
            progress_cb(1.0, "Consultando histórico remoto (kinohistorico.cl)")
            sync_info = _sync_remote_latest_draw()
            if should_cancel():
                return {"cancelled": True}

            history = _all_draws_ordered()
            metrics, results = run_astrology_experiment(
                history,
                target_dt=target,
                history_sync=sync_info,
                sample_size=sample_size,
                seed=seed,
                blend=blend,
                weights=weights,
                top_k=20,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
            )
            exp = _save_experiment(
                name=f"Astrología próximo sorteo {target.date().isoformat()}",
                kind="astrology",
                seed=seed,
                sample_size=sample_size,
                ticket_count=None,
                max_overlap=None,
                weights=weights,
                metrics=metrics,
                results=results,
            )
            return {
                "experiment_id": exp.id,
                "count": len(results),
                "pattern_count": len(metrics.get("patterns", [])),
                "target_draw_datetime": metrics.get("target_draw_datetime"),
                "history_freshness": metrics.get("history_freshness"),
            }

        job_runner.run(app, job.id, work)
        return redirect(url_for("job_status", job_id=job.id))

    @app.route("/lunar-predict", methods=["POST"])
    def lunar_predict():
        draws = _all_draws_ordered()
        if len(draws) < 20:
            flash("Necesitas al menos 20 sorteos cargados para la generación lunar.", "warning")
            return redirect(url_for("upload"))

        sample_size = int(request.form.get("sample_size") or 20_000)
        sample_size = max(1_000, min(sample_size, 200_000))
        seed = int(request.form.get("seed") or 42)
        target = next_draw_datetime()
        job = job_runner.create_job("lunar")

        def work(job_obj, progress_cb, should_cancel):
            progress_cb(1.0, "Consultando histórico remoto (kinohistorico.cl)")
            sync_info = _sync_remote_latest_draw()
            if should_cancel():
                return {"cancelled": True}

            history = _all_draws_ordered()
            metrics, results = run_lunar_experiment(
                history,
                target_dt=target,
                history_sync=sync_info,
                sample_size=sample_size,
                seed=seed,
                top_k=20,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
            )
            exp = _save_experiment(
                name=f"Lunar simple {target.date().isoformat()}",
                kind="lunar",
                seed=seed,
                sample_size=sample_size,
                ticket_count=None,
                max_overlap=None,
                weights={"mode": "lunar"},
                metrics=metrics,
                results=results,
            )
            return {
                "experiment_id": exp.id,
                "count": len(results),
                "pattern_count": len(metrics.get("patterns", [])),
                "target_draw_datetime": metrics.get("target_draw_datetime"),
                "history_freshness": metrics.get("history_freshness"),
                "mode": "lunar",
            }

        job_runner.run(app, job.id, work)
        return redirect(url_for("job_status", job_id=job.id))

    @app.route("/tarot-predict", methods=["POST"])
    def tarot_predict():
        draws = _all_draws_ordered()
        if len(draws) < 20:
            flash("Necesitas al menos 20 sorteos cargados para el ejercicio tarot.", "warning")
            return redirect(url_for("upload"))

        sample_size = int(request.form.get("sample_size") or 20_000)
        sample_size = max(1_000, min(sample_size, 200_000))
        seed = int(request.form.get("seed") or 42)
        target = next_draw_datetime()
        job = job_runner.create_job("tarot")

        def work(job_obj, progress_cb, should_cancel):
            progress_cb(1.0, "Consultando histórico remoto (kinohistorico.cl)")
            sync_info = _sync_remote_latest_draw()
            if should_cancel():
                return {"cancelled": True}

            history = _all_draws_ordered()
            metrics, results = run_tarot_experiment(
                history,
                target_dt=target,
                history_sync=sync_info,
                sample_size=sample_size,
                seed=seed,
                top_k=20,
                progress_cb=progress_cb,
                should_cancel=should_cancel,
            )
            exp = _save_experiment(
                name=f"Tarot–astro {target.date().isoformat()}",
                kind="tarot",
                seed=seed,
                sample_size=sample_size,
                ticket_count=None,
                max_overlap=None,
                weights={"mode": "tarot"},
                metrics=metrics,
                results=results,
            )
            return {
                "experiment_id": exp.id,
                "count": len(results),
                "pattern_count": len(metrics.get("patterns", [])),
                "target_draw_datetime": metrics.get("target_draw_datetime"),
                "history_freshness": metrics.get("history_freshness"),
                "mode": "tarot",
            }

        job_runner.run(app, job.id, work)
        return redirect(url_for("job_status", job_id=job.id))

    @app.route("/backtest", methods=["GET", "POST"])
    def backtest():
        if request.method == "POST":
            draws = _all_draws_ordered()
            if len(draws) < 60:
                flash("Para un backtesting mínimo se recomiendan 60+ sorteos.", "warning")
                return redirect(url_for("upload"))

            sample_size = int(request.form.get("sample_size") or 2000)
            top_k = int(request.form.get("top_k") or 5)
            seed = int(request.form.get("seed") or 42)
            min_history = int(request.form.get("min_history") or 50)
            max_steps = int(request.form.get("max_steps") or 20)
            weights = _weights_from_form(request.form)

            sample_size = max(200, min(sample_size, 20_000))
            max_steps = max(5, min(max_steps, 100))

            job = job_runner.create_job("backtest")

            def work(job_obj, progress_cb, should_cancel):
                metrics = run_walk_forward(
                    _all_draws_ordered(),
                    sample_size=sample_size,
                    top_k=top_k,
                    seed=seed,
                    weights=weights,
                    min_history=min_history,
                    max_steps=max_steps,
                    progress_cb=progress_cb,
                    should_cancel=should_cancel,
                )
                exp = _save_experiment(
                    name=f"Backtest steps={metrics['steps']}",
                    kind="backtest",
                    seed=seed,
                    sample_size=sample_size,
                    ticket_count=top_k,
                    max_overlap=None,
                    weights=weights,
                    metrics=metrics,
                    results=[],
                )
                return {"experiment_id": exp.id, "metrics": metrics}

            job_runner.run(app, job.id, work)
            return redirect(url_for("job_status", job_id=job.id))

        return render_template("backtest.html")

    @app.route("/portfolio", methods=["GET", "POST"])
    def portfolio():
        if request.method == "POST":
            draws = _all_draws_ordered()
            if len(draws) < 10:
                flash("Necesitas al menos 10 sorteos cargados.", "warning")
                return redirect(url_for("upload"))

            ticket_count = int(request.form.get("ticket_count") or 5)
            sample_size = int(request.form.get("sample_size") or 20000)
            seed = int(request.form.get("seed") or 42)
            max_overlap = int(request.form.get("max_overlap") or 9)
            weights = _weights_from_form(request.form)

            ticket_count = max(1, min(ticket_count, 50))
            sample_size = max(500, min(sample_size, 200_000))
            max_overlap = max(0, min(max_overlap, 13))

            job = job_runner.create_job("portfolio")

            def work(job_obj, progress_cb, should_cancel):
                matrix = draws_matrix(_all_draws_ordered())
                results = build_portfolio(
                    matrix,
                    ticket_count=ticket_count,
                    sample_size=sample_size,
                    seed=seed,
                    max_overlap=max_overlap,
                    weights=weights,
                    progress_cb=progress_cb,
                    should_cancel=should_cancel,
                )
                metrics = {
                    "ticket_count": len(results),
                    "max_overlap": max_overlap,
                    "coverage": results[0]["coverage"] if results else 0,
                }
                exp = _save_experiment(
                    name=f"Portfolio x{ticket_count}",
                    kind="portfolio",
                    seed=seed,
                    sample_size=sample_size,
                    ticket_count=ticket_count,
                    max_overlap=max_overlap,
                    weights=weights,
                    metrics=metrics,
                    results=results,
                )
                return {"experiment_id": exp.id, "count": len(results)}

            job_runner.run(app, job.id, work)
            return redirect(url_for("job_status", job_id=job.id))

        return render_template("portfolio.html")

    @app.route("/results")
    def results():
        experiments = Experiment.query.order_by(desc(Experiment.created_at)).all()
        selected_id = request.args.get("id", type=int)
        selected = None
        rows = []
        metrics = {}
        if selected_id:
            selected = db.session.get(Experiment, selected_id)
        elif experiments:
            selected = experiments[0]
        if selected:
            rows = (
                ExperimentResult.query.filter_by(experiment_id=selected.id)
                .order_by(ExperimentResult.rank.asc())
                .all()
            )
            metrics = json.loads(selected.metrics_json or "{}")
        return render_template(
            "results.html",
            experiments=experiments,
            selected=selected,
            rows=rows,
            metrics=metrics,
        )

    @app.route("/results/<int:experiment_id>/download")
    def download_results(experiment_id: int):
        exp = Experiment.query.get_or_404(experiment_id)
        rows = (
            ExperimentResult.query.filter_by(experiment_id=exp.id)
            .order_by(ExperimentResult.rank.asc())
            .all()
        )
        payload = [
            {
                "rank": r.rank,
                "score": r.score,
                "numbers_csv": r.numbers_csv,
                "even_count": r.even_count,
                "odd_count": r.odd_count,
                "total_sum": r.total_sum,
                "consecutive": r.consecutive,
                "coverage": r.coverage,
                "similarity": r.similarity,
            }
            for r in rows
        ]
        data = draws_to_csv_bytes(payload)
        out = Config.BASE_DIR / "data" / f"experiment_{exp.id}.csv"
        out.write_bytes(data)
        return send_file(out, as_attachment=True, download_name=out.name)

    @app.route("/jobs/<int:job_id>")
    def job_status(job_id: int):
        job = Job.query.get_or_404(job_id)
        return render_template("job.html", job=job)

    @app.route("/api/jobs/<int:job_id>")
    def api_job(job_id: int):
        job = Job.query.get_or_404(job_id)
        result = {}
        try:
            result = json.loads(job.result_json or "{}")
        except json.JSONDecodeError:
            result = {}
        return jsonify(
            {
                "id": job.id,
                "kind": job.kind,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "result": result,
            }
        )

    @app.route("/api/jobs/<int:job_id>/cancel", methods=["POST"])
    def api_cancel_job(job_id: int):
        job_runner.request_cancel(job_id)
        return jsonify({"ok": True})

    @app.route("/api/analysis.json")
    def api_analysis():
        draws = _all_draws_ordered()
        return jsonify(build_analysis(draws) if draws else {})


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
