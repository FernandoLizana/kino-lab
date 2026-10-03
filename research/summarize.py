"""Resumen legible del reporte de auditoría (research/audit_report.json)."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

report = json.loads(
    (Path(__file__).resolve().parent / "audit_report.json").read_text(encoding="utf-8")
)

d = report["dataset"]
print("=" * 74)
print(f"DATOS: {d['draws']} sorteos  {d['first_draw']} → {d['last_draw']}")
print(f"       {d['numeric_features']} variables numéricas, {d['categorical_features']} categóricas")
print("=" * 74)

m = report["marginal_associations"]
print("\n[A] ASOCIACIÓN MARGINAL variable ↔ número")
print(f"    pruebas realizadas          : {m['tests_run']}")
print(f"    descubrimientos FDR 5%      : {m['discoveries_fdr05']}")
print(f"    esperados por puro azar     : {m['null_discoveries_mean']:.2f} (p95 = {m['null_discoveries_p95']:.0f})")
print(f"    correlación máxima |r|      : {m['max_abs_correlation']:.4f}")
print(f"    |r| máx. esperado por azar  : {m['null_max_abs_correlation_mean']:.4f}")
print(f"    p permutado del máximo      : {m['max_correlation_p_value']:.4f}")
print("    pares más fuertes:")
for pair in m["strongest_pairs"][:5]:
    print(
        f"      {pair['feature']:<44} n={pair['number']:<3} r={pair['r']:+.4f} p={pair['p_value']:.5f}"
    )

c = report["categorical_associations"]
print("\n[B] ASOCIACIÓN CATEGÓRICA (bin ↔ número)")
best = c["strongest_bin_association"]
if best:
    print(
        f"    bin más fuerte              : {best['bin_variable']} → n{best['number']}"
        f" (chi2={best['chi2']}, p nominal={best['nominal_p']:.5f})"
    )
print(f"    p permutado (multiplicidad) : {c['permutation_p_value']:.4f}")
print(f"    chi2 máx. esperado por azar : {c['null_max_chi2_mean']:.2f} (p95 {c['null_max_chi2_p95']:.2f})")

ws = report["walk_forward_strategies"]
s = report["walk_forward_summary"]
print("\n[C] WALK-FORWARD")
print(f"    estrategias × pasos         : {s['n_strategies']} × {s['n_steps']} = {s['n_strategies']*s['n_steps']:,}")
print(f"    AUC medio de las {s['n_strategies']} estrategias : {statistics.mean(x['auc'] for x in ws):.5f}")
print(f"    rango AUC                   : {min(x['auc'] for x in ws):.4f} – {max(x['auc'] for x in ws):.4f}")
print(f"    mejor AUC                   : {s['best_auc']:.4f} ({s['best_strategy']})")
print(f"    máx. esperado por azar p95  : {s['null_max_auc_p95']:.4f}")
print(f"    p familiar del mejor        : {s['best_p_value_family_wise']:.4f}")
print(f"    significativas (FWER 5%)    : {s['strategies_significant_fwer05']} de {s['n_strategies']}")

print("\n    Aciertos en el top-14 (azar = 7.840):")
for row in ws[:5]:
    print(f"      {row['strategy']:<40} {row['top14_hits']:.4f}  ({row['top14_hits_vs_expected']:+.4f})")
ctrl = next(x for x in ws if x["strategy"] == "control_random")
print(f"      {'control_random (ruido puro)':<40} {ctrl['top14_hits']:.4f}  ({ctrl['top14_hits_vs_expected']:+.4f})")
best_hits = max(ws, key=lambda x: x["top14_hits"])
print(f"      {'MEJOR en hits: ' + best_hits['strategy']:<40} {best_hits['top14_hits']:.4f}  ({best_hits['top14_hits_vs_expected']:+.4f})")

st = report["selection_stability"]
print("\n[D] ¿SE PUEDE ELEGIR LA MEJOR VARIABLE?")
print(f"    correlación selección↔holdout    : {st['selection_vs_holdout_correlation']:+.3f}")
print(f"    correlación de rangos            : {st['selection_vs_holdout_rank_correlation']:+.3f}")
print(f"    ganadores distintos en 200 cortes: {st['distinct_winners_across_splits']}")
print(f"    el ganador supera al azar después: {st['winner_beat_chance_rate']:.0%} de las veces")
print(f"    encogimiento medio del ganador   : {st['mean_shrinkage_random_splits']:+.4f} AUC")

cv = report["composite_vs_single"]
print("\n[E] FORMA DE EVALUAR (AUC holdout, 100 reparticiones)")
for name, row in sorted(cv["methods"].items(), key=lambda kv: -kv[1]["holdout_auc_mean"]):
    print(
        f"    {name:<32} {row['holdout_auc_mean']:.4f} ± {row['holdout_auc_sd']:.4f}"
        f"   supera al azar {row['beats_chance_rate']:.0%}"
    )
print(f"    diferencia entre métodos: {cv['spread_between_methods']:.4f}")
print(f"    ruido dentro de un método: {cv['typical_sd_within_method']:.4f}")
print(f"    → {'INDISTINGUIBLE del ruido' if cv['difference_is_within_noise'] else 'diferencia real'}")

res = report["statistical_resolution"]
print("\n[F] RESOLUCIÓN ESTADÍSTICA (cuánto efecto haría falta para verlo)")
print(f"    {'variable':<34}{'valores':>8}{'n/celda':>9}{'lift mín':>10}{'forzado':>9}")
for row in res["variables"][:8]:
    print(
        f"    {row['variable']:<34}{row['distinct_values']:>8}{row['median_draws_per_value']:>9.0f}"
        f"{row['min_detectable_lift_pp']:>9.1f}pp{row['equivalent_forcing_strength']:>9.0%}"
    )
print("    ...")
for row in res["variables"][-6:]:
    print(
        f"    {row['variable']:<34}{row['distinct_values']:>8}{row['median_draws_per_value']:>9.0f}"
        f"{row['min_detectable_lift_pp']:>9.1f}pp{row['equivalent_forcing_strength']:>9.0%}"
    )

pw = report["power_check"]
print("\n[G] CONTROL DE POTENCIA (señal luna→número inyectada a propósito)")
print(f"    {'forzado':>8}{'AUC oráculo':>13}{'AUC aprendido':>15}{'patrón hallado':>16}{'regla correcta':>16}")
for run in pw["runs"]:
    det = run["detectors"]
    print(
        f"    {run['planted_strength']:>7.0%}"
        f"{det['oracle_knows_the_rule']['auc']:>13.4f}"
        f"{det['learned_lunar_sector25']['auc']:>15.4f}"
        f"{('sí' if run['pattern_search']['detected'] else 'no'):>15}"
        f"{('sí' if run['pattern_search']['recovered_planted_rule'] else 'no'):>16}"
    )
print(f"    señal mínima detectable: {pw['minimum_detectable_strength']:.0%}")
print(f"\n    Tiempo de cómputo: {report['runtime_seconds']}s")
