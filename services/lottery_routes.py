from __future__ import annotations

from pathlib import Path

from flask import (
    Blueprint,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from sqlalchemy import desc

from models import LotteryDraw, LotteryGame, db
from services.lottery_analysis_service import (
    build_lottery_analysis,
    compare_games,
    save_analysis_snapshot,
)
from services.lottery_catalog_service import (
    game_rules_summary,
    get_game_by_slug,
    list_games,
)
from services.lottery_csv_service import (
    LotteryCsvValidationError,
    draws_to_normalized_csv,
    parse_lottery_csv_bytes,
    persist_lottery_draws,
    sample_csv_bytes,
)
from services.lottery_remote_service import (
    LotteryRemoteError,
    adapter_info,
    fetch_remote_draws,
    sync_status_message,
)

lottery_bp = Blueprint("lotteries", __name__)

LOTTERY_DISCLAIMER = (
    "Patrones históricos descriptivos. No cambian las probabilidades futuras "
    "si el mecanismo de sorteo es justo e independiente. Este módulo no afirma "
    "predicción ni ventaja."
)


def _ordered_draws(game: LotteryGame) -> list[LotteryDraw]:
    return (
        LotteryDraw.query.filter_by(game_id=game.id)
        .order_by(LotteryDraw.draw_date.asc(), LotteryDraw.draw_number.asc())
        .all()
    )


def _draw_counts() -> dict[int, int]:
    rows = (
        db.session.query(LotteryDraw.game_id, db.func.count(LotteryDraw.id))
        .group_by(LotteryDraw.game_id)
        .all()
    )
    return {int(gid): int(cnt) for gid, cnt in rows}


@lottery_bp.route("/lotteries")
def catalog():
    games = list_games()
    counts = _draw_counts()
    cards = []
    for g in games:
        cards.append(
            {
                "game": g,
                "rules_summary": game_rules_summary(g),
                "draw_count": counts.get(g.id, 0),
                "has_sync": bool(g.sync_adapter),
                "sync_label": sync_status_message(g.sync_adapter),
            }
        )
    return render_template(
        "lotteries/catalog.html",
        cards=cards,
        lottery_disclaimer=LOTTERY_DISCLAIMER,
    )


@lottery_bp.route("/lotteries/compare")
def compare():
    games = list_games()
    pairs = []
    for g in games:
        draws = _ordered_draws(g)
        if draws:
            pairs.append((g, draws))
    # Also allow empty games for matrix comparison
    all_pairs = [(g, _ordered_draws(g)) for g in games]
    data = compare_games(all_pairs)
    return render_template(
        "lotteries/compare.html",
        comparison=data,
        games_with_data=len(pairs),
        lottery_disclaimer=LOTTERY_DISCLAIMER,
    )


@lottery_bp.route("/lotteries/<slug>")
def detail(slug: str):
    game = get_game_by_slug(slug)
    if game is None:
        flash("Juego no encontrado.", "danger")
        return redirect(url_for("lotteries.catalog"))
    draws = (
        LotteryDraw.query.filter_by(game_id=game.id)
        .order_by(desc(LotteryDraw.draw_date), desc(LotteryDraw.draw_number))
        .limit(30)
        .all()
    )
    total = LotteryDraw.query.filter_by(game_id=game.id).count()
    info = adapter_info(game.sync_adapter)
    return render_template(
        "lotteries/detail.html",
        game=game,
        rules=game.rules(),
        rules_summary=game_rules_summary(game),
        draws=draws,
        draw_count=total,
        sync_info=info,
        sync_message=sync_status_message(game.sync_adapter),
        lottery_disclaimer=LOTTERY_DISCLAIMER,
    )


@lottery_bp.route("/lotteries/<slug>/upload", methods=["POST"])
def upload(slug: str):
    game = get_game_by_slug(slug)
    if game is None:
        flash("Juego no encontrado.", "danger")
        return redirect(url_for("lotteries.catalog"))

    file = request.files.get("file")
    replace = request.form.get("replace") == "on"
    if not file or not file.filename:
        flash("Selecciona un archivo CSV.", "danger")
        return redirect(url_for("lotteries.detail", slug=slug))
    if not file.filename.lower().endswith(".csv"):
        flash("Solo se aceptan archivos .csv", "danger")
        return redirect(url_for("lotteries.detail", slug=slug))
    try:
        content = file.read()
        rows = parse_lottery_csv_bytes(
            content, game.rules(), source=file.filename
        )
        stats = persist_lottery_draws(game, rows, replace=replace)
        flash(
            f"CSV OK: {stats['inserted']} nuevos, {stats['updated']} actualizados. "
            f"Total local: {stats['total']}.",
            "success",
        )
    except LotteryCsvValidationError as exc:
        flash(str(exc), "danger")
    except Exception as exc:  # noqa: BLE001
        flash(f"Error al procesar CSV: {exc}", "danger")
    return redirect(url_for("lotteries.detail", slug=slug))


@lottery_bp.route("/lotteries/<slug>/sync", methods=["POST"])
def sync(slug: str):
    game = get_game_by_slug(slug)
    if game is None:
        flash("Juego no encontrado.", "danger")
        return redirect(url_for("lotteries.catalog"))
    if not game.sync_adapter:
        flash("Este juego no tiene sync remoto; usa carga CSV.", "warning")
        return redirect(url_for("lotteries.detail", slug=slug))
    try:
        limit = int(request.form.get("limit") or 2000)
        limit = max(50, min(limit, 5000))
        rows = fetch_remote_draws(game.sync_adapter, game.rules(), limit=limit)
        stats = persist_lottery_draws(game, rows, replace=False)
        flash(
            f"Sync OK ({game.sync_adapter}): {stats['inserted']} nuevos, "
            f"{stats['updated']} actualizados. Total: {stats['total']}.",
            "success",
        )
    except LotteryRemoteError as exc:
        flash(str(exc), "danger")
    except Exception as exc:  # noqa: BLE001
        flash(f"Error de sync: {exc}", "danger")
    return redirect(url_for("lotteries.detail", slug=slug))


@lottery_bp.route("/lotteries/<slug>/analysis")
def analysis(slug: str):
    game = get_game_by_slug(slug)
    if game is None:
        flash("Juego no encontrado.", "danger")
        return redirect(url_for("lotteries.catalog"))
    draws = _ordered_draws(game)
    data = build_lottery_analysis(game, draws) if draws else None
    if data and draws:
        save_analysis_snapshot(game, data)
    return render_template(
        "lotteries/analysis.html",
        game=game,
        analysis=data,
        draw_count=len(draws),
        lottery_disclaimer=LOTTERY_DISCLAIMER,
    )


@lottery_bp.route("/lotteries/<slug>/export.csv")
def export_csv(slug: str):
    game = get_game_by_slug(slug)
    if game is None:
        flash("Juego no encontrado.", "danger")
        return redirect(url_for("lotteries.catalog"))
    draws = _ordered_draws(game)
    if not draws:
        flash("No hay sorteos locales para exportar.", "warning")
        return redirect(url_for("lotteries.detail", slug=slug))
    payload = draws_to_normalized_csv(game, draws)
    out = Path(__file__).resolve().parent.parent / "data" / f"lottery_{slug}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)
    return send_file(out, as_attachment=True, download_name=out.name)


@lottery_bp.route("/lotteries/<slug>/sample.csv")
def sample_csv(slug: str):
    """Plantilla CSV de ejemplo (números ficticios) según reglas del juego."""
    game = get_game_by_slug(slug)
    if game is None:
        flash("Juego no encontrado.", "danger")
        return redirect(url_for("lotteries.catalog"))
    from io import BytesIO

    payload = sample_csv_bytes(game.rules(), rows=2)
    return send_file(
        BytesIO(payload),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"sample_{slug}.csv",
    )


@lottery_bp.route("/lotteries/<slug>/analysis.json")
def analysis_json(slug: str):
    game = get_game_by_slug(slug)
    if game is None:
        return jsonify({"error": "not_found"}), 404
    draws = _ordered_draws(game)
    return jsonify(build_lottery_analysis(game, draws) if draws else {})
