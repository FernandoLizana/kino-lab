from __future__ import annotations

import json
from datetime import date, timedelta
from io import BytesIO

import pytest

from services.lottery_analysis_service import build_lottery_analysis, compare_games
from services.lottery_catalog_service import (
    CATALOG_SEED,
    expected_slugs,
    seed_lottery_catalog,
)
from services.lottery_csv_service import (
    LotteryCsvValidationError,
    parse_lottery_csv_bytes,
    persist_lottery_draws,
    sample_csv_bytes,
    validate_draw_numbers,
)


EXPECTED_RULES = {
    "powerball": {
        "main": (5, 1, 69),
        "bonus": [("Powerball", 1, 1, 26)],
        "days": ["monday", "wednesday", "saturday"],
        "currency": "USD",
        "sync": "ny_powerball",
    },
    "mega-millions": {
        "main": (5, 1, 70),
        "bonus": [("Mega Ball", 1, 1, 24)],
        "days": ["tuesday", "friday"],
        "currency": "USD",
        "sync": "ny_mega_millions",
        "effective": "2025-04-08",
    },
    "euromillions": {
        "main": (5, 1, 50),
        "bonus": [("Lucky Stars", 2, 1, 12)],
        "days": ["tuesday", "friday"],
        "currency": "EUR",
        "sync": None,
    },
    "eurojackpot": {
        "main": (5, 1, 50),
        "bonus": [("Euro numbers", 2, 1, 12)],
        "days": ["tuesday", "friday"],
        "currency": "EUR",
        "sync": None,
    },
    "uk-lotto": {
        "main": (6, 1, 59),
        "bonus": [("Bonus Ball", 1, 1, 59)],
        "days": ["wednesday", "saturday"],
        "currency": "GBP",
        "sync": None,
    },
    "thunderball": {
        "main": (5, 1, 39),
        "bonus": [("Thunderball", 1, 1, 14)],
        "days": ["tuesday", "wednesday", "friday", "saturday"],
        "currency": "GBP",
        "sync": None,
    },
    "set-for-life": {
        "main": (5, 1, 47),
        "bonus": [("Life Ball", 1, 1, 10)],
        "days": ["monday", "thursday"],
        "currency": "GBP",
        "sync": None,
    },
    "irish-lotto": {
        "main": (6, 1, 47),
        "bonus": [("Bonus Ball", 1, 1, 47)],
        "days": ["wednesday", "saturday"],
        "currency": "EUR",
        "sync": None,
    },
    "eurodreams": {
        "main": (6, 1, 40),
        "bonus": [("Dream Number", 1, 1, 5)],
        "days": ["monday", "thursday"],
        "currency": "EUR",
        "sync": None,
    },
    "loto-chile": {
        "main": (6, 1, 41),
        "bonus": [("Comodín", 1, 1, 41)],
        "days": ["tuesday", "thursday", "sunday"],
        "currency": "CLP",
        "sync": None,
    },
}


@pytest.fixture()
def app(tmp_path):
    db_path = tmp_path / "test_lotteries.db"
    from app import create_app

    application = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
            "SECRET_KEY": "test",
        }
    )
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


def test_catalog_seed_has_ten_exact_rules(app):
    assert len(expected_slugs()) == 10
    assert set(expected_slugs()) == set(EXPECTED_RULES)

    with app.app_context():
        from models import LotteryGame

        # Idempotent re-seed
        first = seed_lottery_catalog()
        second = seed_lottery_catalog()
        assert LotteryGame.query.count() == 10
        assert first["total"] == 10
        assert second["inserted"] == 0

        for item in CATALOG_SEED:
            game = LotteryGame.query.filter_by(slug=item["slug"]).one()
            exp = EXPECTED_RULES[item["slug"]]
            rules = game.rules()
            main = rules["main"]
            assert (main["count"], main["min"], main["max"]) == exp["main"]
            assert len(rules["bonus"]) == len(exp["bonus"])
            for bonus_rule, expected in zip(rules["bonus"], exp["bonus"]):
                name, count, bmin, bmax = expected
                assert bonus_rule["name"] == name
                assert bonus_rule["count"] == count
                assert bonus_rule["min"] == bmin
                assert bonus_rule["max"] == bmax
            assert rules["draw_days"] == exp["days"]
            assert rules["currency"] == exp["currency"]
            assert game.sync_adapter == exp["sync"]
            assert game.official_url
            if "effective" in exp:
                assert rules["rules_effective"] == exp["effective"]

        # Mega Millions post-2025 matrix note
        mm = LotteryGame.query.filter_by(slug="mega-millions").one()
        assert mm.rules()["bonus"][0]["max"] == 24
        assert "2025" in mm.rules()["notes"]


def test_csv_powerball_euromillions_loto_chile():
    pb_rules = next(x["rules"] for x in CATALOG_SEED if x["slug"] == "powerball")
    raw_pb = (
        b"draw_date,n1,n2,n3,n4,n5,bonus1\n"
        b"2026-07-18,9,14,44,50,56,3\n"
        b"2026-07-15,2,7,18,29,38,16\n"
    )
    rows = parse_lottery_csv_bytes(raw_pb, pb_rules)
    assert len(rows) == 2
    assert rows[0]["main_numbers"] == [9, 14, 44, 50, 56]
    assert rows[0]["bonus_numbers"] == [3]

    em_rules = next(x["rules"] for x in CATALOG_SEED if x["slug"] == "euromillions")
    raw_em = (
        b"Fecha,N1,N2,N3,N4,N5,Star1,Star2\n"
        b"01/07/2026,3,11,19,27,44,2,8\n"
    )
    rows_em = parse_lottery_csv_bytes(raw_em, em_rules)
    assert rows_em[0]["main_numbers"] == [3, 11, 19, 27, 44]
    assert rows_em[0]["bonus_numbers"] == [2, 8]

    lc_rules = next(x["rules"] for x in CATALOG_SEED if x["slug"] == "loto-chile")
    raw_lc = (
        b"fecha,n1,n2,n3,n4,n5,n6,comodin\n"
        b"2026-07-14,1,5,12,20,33,41,7\n"
    )
    rows_lc = parse_lottery_csv_bytes(raw_lc, lc_rules)
    assert rows_lc[0]["main_numbers"] == [1, 5, 12, 20, 33, 41]
    assert rows_lc[0]["bonus_numbers"] == [7]

    # Comodín no puede repetir principal
    bad = b"fecha,n1,n2,n3,n4,n5,n6,comodin\n2026-07-14,1,5,12,20,33,41,1\n"
    with pytest.raises(LotteryCsvValidationError):
        parse_lottery_csv_bytes(bad, lc_rules)


def test_validate_mega_millions_legacy_ball():
    rules = next(x["rules"] for x in CATALOG_SEED if x["slug"] == "mega-millions")
    # Current rules reject 25
    with pytest.raises(LotteryCsvValidationError):
        validate_draw_numbers(
            rules, [1, 2, 3, 4, 5], [25], draw_date=date(2026, 7, 1)
        )
    # Legacy allowed before 2025-04-08
    main, bonus = validate_draw_numbers(
        rules, [1, 2, 3, 4, 5], [25], draw_date=date(2025, 4, 4)
    )
    assert bonus == [25]


def test_generic_analysis_synthetic(app):
    with app.app_context():
        from models import LotteryGame

        game = LotteryGame.query.filter_by(slug="powerball").one()
        rows = []
        for i in range(20):
            d = date(2026, 1, 1) + timedelta(days=i * 2)
            main = sorted(
                {
                    1 + (i % 10),
                    11 + (i % 10),
                    21 + (i % 10),
                    31 + (i % 10),
                    41 + (i % 10),
                }
            )
            bonus = [1 + (i % 26)]
            rows.append(
                {
                    "draw_date": d,
                    "draw_number": None,
                    "main_numbers": main,
                    "bonus_numbers": bonus,
                    "source": "test",
                }
            )
        persist_lottery_draws(game, rows)
        from models import LotteryDraw

        draws = LotteryDraw.query.filter_by(game_id=game.id).all()
        analysis = build_lottery_analysis(game, draws)
        assert analysis["quality"]["draw_count"] == 20
        assert "frequency" in analysis["main"]
        assert analysis["main"]["pairs"]
        assert analysis["main"]["chi_square"]["statistic"] is not None
        assert analysis["bonus"]
        assert analysis["calendar"]["by_weekday"]

        cmp = compare_games([(game, draws)])
        assert cmp["games"][0]["draw_count"] == 20


def test_lottery_routes(client, app):
    assert client.get("/lotteries").status_code == 200
    assert client.get("/lotteries/compare").status_code == 200

    home = client.get("/")
    assert home.status_code == 200
    assert b"Otras loter" in home.data

    for slug in expected_slugs():
        resp = client.get(f"/lotteries/{slug}")
        assert resp.status_code == 200, slug
        assert client.get(f"/lotteries/{slug}/analysis").status_code == 200

    from io import BytesIO

    raw = (
        b"draw_date,n1,n2,n3,n4,n5,bonus1\n"
        b"2026-07-18,9,14,44,50,56,3\n"
    )
    resp = client.post(
        "/lotteries/powerball/upload",
        data={"file": (BytesIO(raw), "pb.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"CSV OK" in resp.data or b"nuevos" in resp.data

    with app.app_context():
        from models import LotteryDraw, LotteryGame

        game = LotteryGame.query.filter_by(slug="powerball").one()
        assert LotteryDraw.query.filter_by(game_id=game.id).count() >= 1

    export = client.get("/lotteries/powerball/export.csv")
    assert export.status_code == 200
    assert b"draw_date" in export.data

    sample = client.get("/lotteries/powerball/sample.csv")
    assert sample.status_code == 200
    assert b"draw_date" in sample.data
    assert b"bonus1" in sample.data

    aj = client.get("/lotteries/powerball/analysis.json")
    assert aj.status_code == 200
    payload = json.loads(aj.data)
    assert payload["quality"]["draw_count"] >= 1

    missing = client.get("/lotteries/no-existe", follow_redirects=True)
    assert missing.status_code == 200
    assert b"no encontrado" in missing.data.lower() or b"Otras loter" in missing.data


def test_sample_csv_valid_for_all_catalog_games():
    for item in CATALOG_SEED:
        raw = sample_csv_bytes(item["rules"], rows=2)
        rows = parse_lottery_csv_bytes(raw, item["rules"], source="sample")
        assert len(rows) == 2
        for row in rows:
            validate_draw_numbers(
                item["rules"],
                row["main_numbers"],
                row.get("bonus_numbers") or [],
                draw_date=row["draw_date"],
            )
