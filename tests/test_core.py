from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from services.combination_service import combo_to_mask, features_for_combo, mask_to_combo, overlap_count
from services.csv_service import parse_csv_bytes
from services.scoring_service import normalize_weights, score_combo
from services.astrology_service import (
    PLANET_ORDER,
    TZ_CHILE,
    assert_no_future_leak_on_split,
    build_number_astro_scores,
    circular_feature_distance,
    compute_astro_features,
    next_draw_datetime,
    prior_draws_before_target,
    run_astrology_experiment,
)
from services.statistical_seed_service import build_statistical_seed


class FakeDraw:
    def __init__(self, draw_date, numbers, draw_number):
        self.draw_date = draw_date
        self._numbers = numbers
        self.draw_number = draw_number

    def numbers(self):
        return self._numbers


def test_parse_kino_header_csv():
    raw = b"Fecha,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14\n01-01-2023,01,02,03,04,05,06,09,11,13,18,19,22,24,25\n"
    rows = parse_csv_bytes(raw)
    assert len(rows) == 1
    assert rows[0]["numbers"] == [1, 2, 3, 4, 5, 6, 9, 11, 13, 18, 19, 22, 24, 25]


def test_reject_duplicates():
    raw = b"Fecha,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14\n01-01-2023,01,01,03,04,05,06,09,11,13,18,19,22,24,25\n"
    try:
        parse_csv_bytes(raw)
        assert False, "should fail"
    except Exception as exc:
        assert "Ningún sorteo válido" in str(exc) or "duplicad" in str(exc).lower()


def test_mask_roundtrip():
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    mask = combo_to_mask(nums)
    assert mask_to_combo(mask) == nums
    assert overlap_count(mask, mask) == 14


def test_features_and_score():
    import numpy as np

    matrix = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ],
        dtype=np.int16,
    )
    from services.scoring_service import _build_context

    ctx = _build_context(matrix)
    weights = normalize_weights(None)
    scored = score_combo([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], ctx, weights)
    assert 0 <= scored["score"] <= 2
    feats = features_for_combo(scored["numbers"])
    assert feats["total_sum"] == sum(range(1, 15))


def test_statistical_seed_is_reproducible_and_history_sensitive():
    draws = [
        FakeDraw(
            date(2026, 7, 1) + timedelta(days=i),
            list(range(1 + (i % 2), 15 + (i % 2))),
            3200 + i,
        )
        for i in range(4)
    ]
    run_date = date(2026, 7, 19)

    first = build_statistical_seed(draws, experiment_date=run_date)
    second = build_statistical_seed(list(reversed(draws)), experiment_date=run_date)

    assert first["sha256"] == second["sha256"]
    assert first["numeric_seed"] == second["numeric_seed"]
    assert len(first["sha256"]) == 64
    assert 0 <= first["numeric_seed"] < 2**63

    changed = build_statistical_seed(
        draws + [
            FakeDraw(
                date(2026, 7, 20),
                list(range(3, 17)),
                3204,
            )
        ],
        experiment_date=run_date,
    )
    assert changed["sha256"] != first["sha256"]


def test_astro_features_reproducible_for_fixed_date():
    first = compute_astro_features(date(2024, 6, 15))
    second = compute_astro_features(date(2024, 6, 15))
    assert first["julian_day"] == second["julian_day"]
    assert first["moon_phase_name"] == second["moon_phase_name"]
    assert first["bodies"]["moon"]["sign_index"] == second["bodies"]["moon"]["sign_index"]
    assert first["lunar"]["phase_angle_deg"] == second["lunar"]["phase_angle_deg"]
    assert 0 <= first["moon_phase_bin"] <= 7
    assert first["bodies"]["moon"]["longitude_deg"] == second["bodies"]["moon"]["longitude_deg"]
    from config import Config

    assert first["location"]["lat"] == Config.ASTRO_LATITUDE
    assert first["location"]["lon"] == Config.ASTRO_LONGITUDE


def test_astro_ten_bodies_and_circular_encodings():
    feats = compute_astro_features(date(2024, 6, 15))
    assert list(feats["bodies"].keys()) == PLANET_ORDER
    assert len(PLANET_ORDER) == 10
    for name in PLANET_ORDER:
        body = feats["bodies"][name]
        assert -1.0 <= body["longitude_sin"] <= 1.0
        assert -1.0 <= body["longitude_cos"] <= 1.0
        assert body["motion_state"] in {"direct", "stationary", "retrograde"}
        assert 0.0 <= body["longitude_deg"] < 360.0
        assert body["distance_au"] > 0


def test_circular_encoding_continuity_near_0_360():
    # 359° y 1° deben estar cerca en sin/cos; 0° y 180° lejos.
    near = circular_feature_distance(359.0, 1.0)
    far = circular_feature_distance(0.0, 180.0)
    assert near < 0.1
    assert far > 1.5
    assert near < far / 5


def test_lunar_block_ranges_and_nodes():
    lunar = compute_astro_features(date(2024, 6, 15))["lunar"]
    assert 0.0 <= lunar["illumination_pct"] <= 100.0
    assert lunar["distance_km"] > 0
    assert lunar["distance_au"] > 0
    assert 0.0 <= lunar["phase_angle_deg"] < 360.0
    assert lunar["phase_label"] in {"waxing", "waning"}
    for node_key in ("true_node", "mean_node"):
        node = lunar[node_key]
        assert 0.0 <= node["longitude_deg"] < 360.0
        assert -1.0 <= node["longitude_sin"] <= 1.0
        assert -1.0 <= node["longitude_cos"] <= 1.0
    seps = lunar["separations_to_planets_deg"]
    assert "sun" in seps and "pluto" in seps
    for sep in seps.values():
        assert 0.0 <= sep <= 180.0
    lun = lunar["lunations"]
    assert lun["hours_to_nearest_new_moon_signed"] is not None
    assert lun["hours_to_nearest_full_moon_signed"] is not None


def test_planetary_motion_states_valid():
    feats = compute_astro_features(datetime(2024, 6, 15, 22, 30, tzinfo=TZ_CHILE))
    for name, body in feats["bodies"].items():
        assert body["motion_state"] in {"direct", "stationary", "retrograde"}
        if name in {"sun", "moon"}:
            assert body["motion_state"] != "retrograde"


def test_astro_number_scores_length_and_no_future_leak():
    draws = []
    features = []
    for i in range(40):
        d = date(2024, 1, 3) + timedelta(days=i * 2)
        nums = [((i + k * 3) % 25) + 1 for k in range(14)]
        nums = sorted(dict.fromkeys(nums))
        filler = 1
        while len(nums) < 14:
            if filler not in nums:
                nums.append(filler)
            filler += 1
        nums = sorted(nums)[:14]
        draws.append(FakeDraw(d, nums, 3000 + i))
        features.append(compute_astro_features(d))

    scores = build_number_astro_scores(draws, features, features[-1])
    assert len(scores) == 25
    assert all(0.0 <= s <= 1.0 for s in scores)

    split = 25
    leak_safe = assert_no_future_leak_on_split(draws, features, split)
    prior_only = build_number_astro_scores(
        draws[:split], features[:split], features[split]
    )
    assert leak_safe == prior_only

    with_current = build_number_astro_scores(
        draws[: split + 1],
        features[: split + 1],
        features[split],
    )
    assert with_current != prior_only


def test_next_draw_datetime_calendar_cases():
    # Sábado mañana -> domingo 22:30
    sat_morning = datetime(2026, 7, 18, 10, 0, tzinfo=TZ_CHILE)  # sábado
    nxt = next_draw_datetime(sat_morning)
    assert nxt == datetime(2026, 7, 19, 22, 30, tzinfo=TZ_CHILE)
    assert nxt.weekday() == 6

    # Domingo mañana -> domingo 22:30
    sun_morning = datetime(2026, 7, 19, 9, 15, tzinfo=TZ_CHILE)
    nxt = next_draw_datetime(sun_morning)
    assert nxt == datetime(2026, 7, 19, 22, 30, tzinfo=TZ_CHILE)

    # Domingo 22:31 -> miércoles 22:30
    sun_late = datetime(2026, 7, 19, 22, 31, tzinfo=TZ_CHILE)
    nxt = next_draw_datetime(sun_late)
    assert nxt == datetime(2026, 7, 22, 22, 30, tzinfo=TZ_CHILE)
    assert nxt.weekday() == 2

    # Miércoles 22:29 -> miércoles 22:30
    wed_early = datetime(2026, 7, 22, 22, 29, tzinfo=TZ_CHILE)
    nxt = next_draw_datetime(wed_early)
    assert nxt == datetime(2026, 7, 22, 22, 30, tzinfo=TZ_CHILE)


def test_astrology_uses_future_target_and_excludes_later_draws():
    draws = []
    cursor = date(2026, 7, 17)  # viernes
    while len(draws) < 25:
        if cursor.weekday() in (2, 4, 6):
            draws.append(FakeDraw(cursor, list(range(1, 15)), 4000 + len(draws)))
        cursor -= timedelta(days=1)

    future_same_day = FakeDraw(date(2026, 7, 19), list(range(2, 16)), 9999)
    all_draws = draws + [future_same_day]

    now = datetime(2026, 7, 19, 10, 0, tzinfo=TZ_CHILE)  # domingo mañana
    target = next_draw_datetime(now)
    assert target.date() == date(2026, 7, 19)

    prior = prior_draws_before_target(all_draws, target)
    assert all(d.draw_date < date(2026, 7, 19) for d in prior)
    assert all(d.draw_number != 9999 for d in prior)

    metrics, results = run_astrology_experiment(
        all_draws,
        now=now,
        history_sync={"history_freshness": "already_current"},
        sample_size=200,
        seed=7,
        top_k=5,
    )
    assert metrics["target_draw_time"] == "22:30"
    assert metrics["timezone"] == "America/Santiago"
    assert metrics["target_draw_date"] == "2026-07-19"
    assert "22:30" in metrics["target_draw_datetime"]
    assert date.fromisoformat(metrics["history_as_of"]) < target.date()
    assert len(results) == 5
    assert len(results[0]["numbers"]) == 14
    assert metrics["target_sky"]["local_datetime"].startswith("2026-07-19T22:30")
    assert len(metrics["target_sky"]["planets_table"]) == 10
    assert "lunar_summary" in metrics["target_sky"]
    assert metrics["target_sky"]["lunar_summary"]["illumination_pct"] >= 0
    assert "adaptive_weights" in metrics
    assert "channel_weights" in metrics["adaptive_weights"]

    # Pesos iguales: nada se estima del histórico (hallazgo de la auditoría).
    weights = metrics["adaptive_weights"]["channel_weights"]
    assert metrics["adaptive_weights"]["mode"] == "equal_channel_weight"
    assert len(set(round(w, 6) for w in weights.values())) == 1
    assert metrics["adaptive_weights"]["sim_weight"] == 0.5
    assert metrics["adaptive_weights"]["bin_weight"] == 0.5

    # Blend fijo, no adaptativo.
    assert metrics["blend_is_fixed"] is True
    assert "blend_adaptive" not in metrics
    assert metrics["blend"] == metrics["blend_requested"]

    # Contraste familiar por permutación disponible y sin filtrar el array crudo.
    assert "pattern_null" in metrics
    assert "null_samples" not in metrics["pattern_null"]
    assert "audit_note" in metrics
    assert metrics["audit_note"]["significant_findings"] == 0


def test_astrology_results_copy_ui_contract():
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "results.html"
    ).read_text(encoding="utf-8")

    assert "Combinación con mayor puntaje del modelo experimental" in template
    assert "Copiar 14 números" in template
    assert "navigator.clipboard.writeText" in template
    assert 'document.execCommand("copy")' in template
    assert "selected.kind in ['astrology', 'lunar', 'tarot', 'color'] and rows" in template
    assert "mayor score del modelo no implica una mayor probabilidad real" in template
    assert "todas las combinaciones son equiprobables" in template
    assert "Cielo del target — planetas" in template
    assert "Resumen lunar" in template
    assert "Pesos por canal" in template
    assert "Correlación lunar posicional" in template

    # Los pesos adaptativos ya no existen: la UI no debe prometerlos.
    assert "Pesos por precisión histórica" not in template
    assert "channel_precisions" not in template

    # Aviso de la auditoría, incluido en astro, lunar y tarot.
    assert template.count('{% include "_audit_notice.html" %}') == 3
    notice = (
        Path(__file__).resolve().parents[1] / "templates" / "_audit_notice.html"
    ).read_text(encoding="utf-8")
    assert "Qué significa este porcentaje" in notice
    assert "what_the_score_means" in notice
    assert "Contraste de los patrones" in notice
    assert "family_wise_p_value" in notice

    # Columna del contraste familiar en las tres tablas de patrones.
    assert template.count("p familiar") == 3
    # Tres tablas × (condición + formato) = 6 menciones.
    assert template.count("p_value_family_wise") == 6
    assert template.count("superó el contraste familiar por permutación") == 3

    dashboard = (
        Path(__file__).resolve().parents[1] / "templates" / "dashboard.html"
    ).read_text(encoding="utf-8")
    assert "Generación lunar simple" in dashboard
    assert "lunar_predict" in dashboard
    assert "tarot_predict" in dashboard
    assert "Tirada tarot–astro" in dashboard
    assert "url_for('colors')" in dashboard
    assert "Arco de colores" in dashboard
    assert "selected.kind == 'lunar'" in template
    assert "selected.kind == 'tarot'" in template


def test_lunar_vector_excludes_planets_and_separations():
    from services.astrology_service import compute_astro_features
    from services.lunar_service import (
        assert_lunar_feature_keys_clean,
        extract_lunar_model_features,
    )

    feats = extract_lunar_model_features(compute_astro_features(date(2024, 6, 15)))
    assert_lunar_feature_keys_clean(feats)
    assert "moon.longitude_sin" in feats
    assert "lunar.phase_sin" in feats
    assert "lunar.true_node_sin" in feats
    assert all(not k.startswith("mars.") for k in feats)
    assert all("sep_" not in k for k in feats)
    assert all("sun." not in k for k in feats)


def test_lunar_features_reproducible_and_scores():
    from services.lunar_service import (
        build_lunar_number_scores,
        extract_lunar_model_features,
        run_lunar_experiment,
    )
    from services.astrology_service import compute_astro_features
    from services.lunar_service import _as_lunar_view

    a = extract_lunar_model_features(compute_astro_features(date(2024, 6, 15)))
    b = extract_lunar_model_features(compute_astro_features(date(2024, 6, 15)))
    assert a == b

    draws = []
    features = []
    cursor = date(2026, 7, 17)
    while len(draws) < 30:
        if cursor.weekday() in (2, 4, 6):
            draws.append(FakeDraw(cursor, list(range(1, 15)), 5000 + len(draws)))
            features.append(_as_lunar_view(compute_astro_features(cursor)))
        cursor -= timedelta(days=1)

    scores = build_lunar_number_scores(draws, features, features[0])
    assert len(scores) == 25
    assert all(0.0 <= s <= 1.0 for s in scores)

    _, meta = build_lunar_number_scores(draws, features, features[0], return_meta=True)
    assert "channel_weights" in meta
    assert "moon_longitude_deg" in meta
    assert 1 <= int(meta["moon_sector"]) <= 25
    # Los 4 canales pesan exactamente igual.
    assert meta["mode"] == "equal_channel_weight"
    assert sorted(meta["channel_weights"]) == [
        "lunar_bins",
        "lunar_vector_similarity",
        "moon_longitude_kernel",
        "moon_sector",
    ]
    assert set(round(w, 6) for w in meta["channel_weights"].values()) == {0.25}

    now = datetime(2026, 7, 19, 10, 0, tzinfo=TZ_CHILE)
    metrics, results = run_lunar_experiment(
        draws,
        now=now,
        history_sync={"history_freshness": "already_current"},
        sample_size=300,
        seed=11,
        top_k=20,
    )
    assert metrics["mode"] == "lunar"
    assert "adaptive_weights" in metrics
    assert metrics["target_draw_time"] == "22:30"
    assert len(results) == 20
    assert "lunar_score" in results[0]
    assert len(results[0]["numbers"]) == 14
    assert all("sep_" not in k for k in metrics["feature_list"])
    assert all(not k.startswith("mars.") for k in metrics["feature_list"])
    assert metrics["number_scores"][0]["lunar_score"] >= metrics["number_scores"][-1]["lunar_score"]


def test_bins_without_resolution_are_discarded():
    """Un valor de bin con poquísimos casos no debe influir en el score."""
    from services.astrology_service import (
        MIN_DRAWS_PER_BIN_VALUE,
        _empty_assoc,
        _score_from_assoc,
        _update_assoc,
    )

    assert MIN_DRAWS_PER_BIN_VALUE >= 600

    assoc = _empty_assoc()
    # 400 sorteos con un bin frecuente y 2 con un bin rarísimo que "predice" el 1.
    for i in range(400):
        feats = {"pattern_bins": {"comun": "a"}}
        _update_assoc(assoc, feats, list(range(1, 15)))
    for _ in range(2):
        feats = {"pattern_bins": {"raro": "z"}}
        _update_assoc(assoc, feats, [1] * 14)

    scores = _score_from_assoc(assoc, {"pattern_bins": {"raro": "z"}})
    # El bin raro tiene soporte 2 y el umbral efectivo es 402//4 = 100: se ignora,
    # así que el score queda plano en vez de disparar el número 1.
    assert len(scores) == 25
    assert max(scores) - min(scores) < 1e-9


def test_pattern_null_is_permutation_based_and_family_wise():
    """El nulo debe venir de permutar, no de comparar contra la tasa base."""
    from services.astrology_service import compute_astro_features, estimate_pattern_null

    draws = []
    features = []
    cursor = date(2026, 7, 17)
    while len(draws) < 120:
        if cursor.weekday() in (2, 4, 6):
            numbers = sorted(((cursor.toordinal() * 7 + k * 3) % 25) + 1 for k in range(40))
            unique = sorted(set(numbers))[:14]
            while len(unique) < 14:
                candidate = (unique[-1] % 25) + 1
                if candidate in unique:
                    candidate = max(set(range(1, 26)) - set(unique))
                unique.append(candidate)
                unique = sorted(set(unique))
            draws.append(FakeDraw(cursor, unique, 7000 + len(draws)))
            features.append(compute_astro_features(cursor))
        cursor -= timedelta(days=1)

    null = estimate_pattern_null(draws, features, permutations=25, seed=3)
    assert null["available"] is True
    assert null["permutations"] == 25
    # El |z| máximo por azar es claramente mayor que el umbral ingenuo de 1.96,
    # que es justamente el punto: mirar cientos de celdas infla el máximo.
    assert null["critical_abs_z"] > 1.96
    assert null["null_max_abs_z_mean"] > 0
    assert 0.0 < null["family_wise_p_value"] <= 1.0
    assert null["cells_tested"] > 0


def test_lunar_route_registered():
    from app import create_app

    app = create_app()
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/lunar-predict" in rules
    assert "/tarot-predict" in rules


def test_tarot_spread_is_sky_determined_and_reproducible():
    from services.astrology_service import compute_astro_features
    from services.tarot_service import (
        as_tarot_view,
        build_day_spread,
        decan_from_longitude,
        run_tarot_experiment,
    )

    # Book T: 0° Aries = 2 de Bastos; 10° Tauro = 6 de Oros; 25° Géminis = 10 de Espadas.
    aries = decan_from_longitude(0.0)
    assert aries["name"] == "2 de Bastos"
    assert aries["ruler"] == "mars"
    taurus = decan_from_longitude(40.0)
    assert taurus["name"] == "6 de Oros"
    gemini = decan_from_longitude(85.0)
    assert gemini["name"] == "10 de Espadas"

    a = compute_astro_features(date(2024, 6, 15))
    b = compute_astro_features(date(2024, 6, 15))
    assert build_day_spread(a)["signature"] == build_day_spread(b)["signature"]

    view = as_tarot_view(a)
    assert view["mode"] == "tarot"
    from services.tarot_service import ALL_TAROT_BIN_KEYS

    assert set(view["pattern_bins"]) == set(ALL_TAROT_BIN_KEYS)
    assert "saturn_major" in view["pattern_bins"]
    assert "slow_climate" in view["pattern_bins"]
    # El ancla es un menor: cambia con la Luna, no es un sorteo de mazo.
    assert view["tarot"]["pattern"]["kind"] == "minor"
    assert view["tarot"]["energy"]["kind"] == "major"

    draws = []
    cursor = date(2026, 7, 17)
    while len(draws) < 40:
        if cursor.weekday() in (2, 4, 6):
            draws.append(FakeDraw(cursor, list(range(1, 15)), 8000 + len(draws)))
        cursor -= timedelta(days=1)

    now = datetime(2026, 7, 19, 10, 0, tzinfo=TZ_CHILE)
    metrics, results = run_tarot_experiment(
        draws,
        now=now,
        history_sync={"history_freshness": "already_current"},
        sample_size=300,
        seed=11,
        top_k=5,
    )
    assert metrics["mode"] == "tarot"
    assert "saturn_major" in as_tarot_view(compute_astro_features(date(1990, 9, 19)))["pattern_bins"]
    assert "target_spread" in metrics
    assert metrics["target_spread"]["pattern"]["kind"] == "minor"
    assert "synchronicity" in metrics
    assert "resolution" in metrics
    assert "pattern_null" in metrics
    assert len(results) == 5


def test_pattern_discovery_counts_non_hits_and_exposes_metrics():
    from services.astrology_service import (
        _compute_astro_features_cached,
        discover_patterns_walk_forward,
        select_model_feature_keys,
    )

    _compute_astro_features_cached.cache_clear()
    draws = []
    features = []
    cursor = date(2024, 1, 3)
    while len(draws) < 80:
        if cursor.weekday() in (2, 4, 6):
            # números rotativos para variar Y
            offset = len(draws) % 12
            nums = [((offset + k) % 25) + 1 for k in range(14)]
            nums = sorted(set(nums))
            filler = 1
            while len(nums) < 14:
                if filler not in nums:
                    nums.append(filler)
                filler += 1
            nums = sorted(nums)[:14]
            draws.append(FakeDraw(cursor, nums, 6000 + len(draws)))
            features.append(compute_astro_features(cursor))
        cursor += timedelta(days=1)

    patterns = discover_patterns_walk_forward(
        draws,
        features,
        min_support=8,
        min_tests=8,
        top_n=20,
        lift_floor=1.0,
    )
    assert isinstance(patterns, list)
    if patterns:
        p = patterns[0]
        for key in (
            "feature",
            "value",
            "number",
            "lift",
            "observed_rate",
            "expected_rate",
            "support",
            "confirmation_rate",
            "stability",
            "p_value",
            "p_value_adj",
            "confirmations",
            "tests",
        ):
            assert key in p
        # Soporte = tests walk-forward; confirmation_tests cuenta outcomes elevados
        # incluyendo no-aparición (confirmation_tests >= confirmations).
        assert p["support"] == p["tests"]
        assert p["confirmation_tests"] >= p["confirmations"]
        assert 0.0 <= p["observed_rate"] <= 1.0
        assert abs(p["expected_rate"] - 14 / 25) < 1e-6
        assert 0.0 <= p["p_value"] <= 1.0
        assert 0.0 <= p["p_value_adj"] <= 1.0
        assert p["p_value_adj"] + 1e-9 >= p["p_value"]
        # Dedupe deja familias únicas por número
        pairs = {(x["number"], x.get("feature_family", x["feature"])) for x in patterns}
        assert len(pairs) == len(patterns)

    keys = select_model_feature_keys(features)
    assert keys
    assert "moon.longitude_sin" in keys or "lunar.phase_sin" in keys


def test_benjamini_hochberg_monotonic():
    from services.astrology_service import _benjamini_hochberg

    raw = [0.01, 0.04, 0.03, 0.20]
    adj = _benjamini_hochberg(raw)
    assert len(adj) == 4
    assert all(0.0 <= a <= 1.0 for a in adj)
    # El menor p crudo no puede quedar con adj mayor que el siguiente en orden BH.
    order = sorted(range(4), key=lambda i: raw[i])
    for i in range(len(order) - 1):
        assert adj[order[i]] <= adj[order[i + 1]] + 1e-12


def test_pattern_ui_shows_new_metrics():
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "results.html"
    ).read_text(encoding="utf-8")
    assert "observed_rate" in template
    assert "stability" in template
    assert "p_value" in template
    assert "p_value_adj" in template
    assert "equiprobabilidad" in template.lower() or "equiprobables" in template


def test_color_arc_payload_and_route():
    from app import create_app
    from services.color_service import build_color_arc_payload

    draws = [
        FakeDraw(date(2026, 7, 15), list(range(1, 15)), 1),
        FakeDraw(date(2026, 7, 17), list(range(12, 26)), 2),
    ]
    payload = build_color_arc_payload(draws)
    assert payload["draw_count"] == 2
    assert len(payload["numbers"]) == 25
    assert payload["numbers"][0]["number"] == 1
    assert payload["last"]["date"] == "2026-07-17"
    assert payload["points"][0]["hex"] != payload["points"][1]["hex"]
    assert sum(p["dominant_count"] for p in payload["pigments"]) == 2

    app = create_app()
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/colores" in rules
    assert "/colores/predecir" in rules
    template = (
        Path(__file__).resolve().parents[1] / "templates" / "colors.html"
    ).read_text(encoding="utf-8")
    assert "colorArcSvg" in template
    assert "Predecir próximo sorteo" in template
    assert "color_predict" in template
    assert "conic-gradient" in (
        Path(__file__).resolve().parents[1] / "static" / "css" / "app.css"
    ).read_text(encoding="utf-8")
    navbar = (
        Path(__file__).resolve().parents[1] / "templates" / "base.html"
    ).read_text(encoding="utf-8")
    assert "url_for('colors')" in navbar


def test_combo_color_is_deterministic_and_uses_all_five_pigments():
    from services.color_service import PIGMENT_NAMES, combo_color, number_hue_deg

    a = combo_color([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    b = combo_color([14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    assert a["hex"] == b["hex"]
    assert a["dominant_pigment"] == "rojo"
    assert a["pigment_counts"]["rojo"] == 5
    assert number_hue_deg(1) == 0.0
    assert abs(number_hue_deg(26) - 0.0) < 1e-9
    # Un bloque alto se va al violeta y a un tono distinto.
    high = combo_color([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    assert high["hex"] != a["hex"]
    assert high["hue_deg"] > a["hue_deg"]
    assert combo_color(list(range(12, 26)))["pigment_counts"]["violeta"] == 5
    assert set(a["pigment_counts"]) == set(PIGMENT_NAMES)


def test_color_prediction_prefers_target_hue():
    from services.color_service import combo_color, run_color_experiment

    draws = [
        FakeDraw(date(2026, 7, 15), list(range(1, 15)), 1),
        FakeDraw(date(2026, 7, 17), list(range(2, 16)), 2),
    ]
    metrics, results = run_color_experiment(
        draws, mode="pick", target_hue=0.0, sample_size=400, seed=3, top_k=5
    )
    assert metrics["mode"] == "color"
    assert metrics["target"]["hue_deg"] == 0.0
    assert len(results) == 5
    top = metrics["prediction"]
    assert len(top["numbers"]) == 14
    # Cerca del rojo (0°) — más cerca que una cartilla alta/violeta.
    far = combo_color(list(range(12, 26)))
    assert top["hue_distance_deg"] < min(far["hue_deg"], 360 - far["hue_deg"]) - 10
