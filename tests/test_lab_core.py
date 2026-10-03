from __future__ import annotations

from datetime import date

from lab.combinations import ComboError, generate_many, validate_combo
from lab.ingest import parse_draws
from lab.notebook import validate_import
from lab.probability import build_probability_table, comb, expected_hits, hit_support
from lab.profiles import get_profile
from lab.strategies import evaluate_walk_forward


def test_kino_references():
    p = get_profile("kino_moderno")
    table = build_probability_table(p)
    assert table.total_combinations == 4_457_400
    assert abs(table.expected_hits - 196 / 25) < 1e-12
    assert table.support == (3, 14)
    assert abs(sum(table.pmf.values()) - 1) < 1e-12


def test_tiny_game_matches_enumeration():
    p = get_profile("demo_6_3")
    table = build_probability_table(p)
    assert table.total_combinations == 20
    assert table.support == (0, 3)
    # Enumeración: C(3,j)*C(3,3-j)/20
    assert abs(table.pmf[0] - 1 / 20) < 1e-12
    assert abs(table.pmf[3] - 1 / 20) < 1e-12
    assert abs(sum(table.pmf.values()) - 1) < 1e-12
    assert expected_hits(p) == 1.5


def test_impossible_filters():
    p = get_profile("demo_6_3")
    try:
        generate_many(p, 1, 1, include=[1, 2, 3, 4])
        assert False
    except ComboError:
        pass


def test_ingest_keeps_15_and_14():
    raw = (
        "19-09-1990,01,02,03,04,05,08,09,11,14,17,20,21,22,23,24\n"
        "08-01-2006,01,02,03,04,05,06,07,08,09,10,11,12,13,14\n"
    ).encode()
    report = parse_draws(raw, source_name="mix.csv")
    assert report.accepted == 2
    assert report.by_profile["kino_clasico"] == 1
    assert report.by_profile["kino_moderno"] == 1
    assert report.excluded == 0
    assert all(len(d["numbers"]) in (14, 15) for d in report.draws)
    assert not any(len(d["numbers"]) == 14 and d["profile"] == "kino_clasico" for d in report.draws)


def test_no_future_leak_frequency():
    p = get_profile("demo_6_3")
    draws = [
        {"draw_date": date(2020, 1, 1), "numbers": [1, 2, 3]},
        {"draw_date": date(2020, 1, 2), "numbers": [4, 5, 6]},
        {"draw_date": date(2020, 1, 3), "numbers": [1, 2, 6]},
    ]
    # Rellenar para min_train=2
    more = [{"draw_date": date(2020, 1, 4 + i), "numbers": [1, 2, 3]} for i in range(10)]
    future = {"draw_date": date(2021, 1, 1), "numbers": [4, 5, 6]}
    base = evaluate_walk_forward(p, draws + more, strategy="frequency", tickets_per_draw=1, seed=1, min_train=2)
    leaked = evaluate_walk_forward(
        p, draws + more + [future], strategy="frequency", tickets_per_draw=1, seed=1, min_train=2
    )
    # Las decisiones hasta 2020-01-13 no deben cambiar al añadir 2021
    assert base["all_best"] == leaked["all_best"][: len(base["all_best"])]


def test_json_import_rejects_bad_schema():
    try:
        validate_import('{"schema":"nope","title":"x"}')
        assert False
    except ValueError:
        pass


def test_comb_zero():
    assert comb(5, 9) == 0
    assert hit_support(get_profile("kino_moderno")) == (3, 14)


def test_validate_combo():
    p = get_profile("kino_moderno")
    out = validate_combo(p, list(range(1, 15)))
    assert out == list(range(1, 15))


def test_vendor_assets_are_local():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "static" / "vendor"
    for name in ("bootstrap.min.css", "bootstrap.bundle.min.js", "chart.umd.min.js"):
        assert (root / name).is_file()
        assert (root / name).stat().st_size > 1000


def test_lab_routes_smoke(tmp_path):
    from app import create_app

    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'lab.db'}",
            "SECRET_KEY": "test",
        }
    )
    client = app.test_client()
    assert client.get("/inicio").status_code == 200
    assert client.get("/probabilidades").status_code == 200
    assert b"4.457.400" in client.get("/probabilidades?profile=kino_moderno").data or b"4457400" in client.get("/probabilidades").data
    assert client.get("/aprender").status_code == 200
    assert client.get("/ajustes").status_code == 200
    demo = client.post("/demo/historico", follow_redirects=True)
    assert demo.status_code == 200
