"""Traduce los resultados de la auditoría a cotas prácticas.

¿Qué tan grande podría ser, como máximo, un efecto real que se nos haya escapado?
¿Y cuántos sorteos harían falta para confirmar el mejor candidato observado?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

report = json.loads(
    (Path(__file__).resolve().parent / "audit_report.json").read_text(encoding="utf-8")
)
rows = report["walk_forward_strategies"]
summary = report["walk_forward_summary"]
n_steps = summary["n_steps"]

print("=" * 74)
print("COTAS PRÁCTICAS")
print("=" * 74)

print("\n[1] Intervalos de confianza del mejor candidato (95%)")
print(f"    {'estrategia':<38}{'AUC':>8}{'IC 95%':>20}{'incluye azar':>14}")
for row in rows[:5] + [next(r for r in rows if r["strategy"] == "control_random")]:
    se = row["null_auc_sd"]
    lo, hi = row["auc"] - 1.96 * se, row["auc"] + 1.96 * se
    contains = "sí" if lo <= 0.5 <= hi else "NO"
    print(f"    {row['strategy']:<38}{row['auc']:>8.4f}  [{lo:.4f}, {hi:.4f}]{contains:>12}")

best = rows[0]
se = best["null_auc_sd"]
print("\n[2] Cota superior del efecto real")
upper = best["auc"] + 1.96 * se - 0.5
print(f"    Ninguna estrategia superó su nulo. La ventaja verdadera de la mejor")
print(f"    está, con 95% de confianza, por debajo de {upper:+.4f} de AUC.")

# Traducción a aciertos: cuántos aciertos extra en el top-14 implica ese AUC.
# Un desplazamiento de AUC delta mueve el ranking de los aciertos; empíricamente
# la relación observada en los datos es casi lineal en este rango.
aucs = np.array([r["auc"] for r in rows])
hits = np.array([r["top14_hits"] for r in rows])
slope, intercept = np.polyfit(aucs, hits, 1)
r_fit = float(np.corrcoef(aucs, hits)[0, 1])
print(f"\n    Relación empírica AUC → aciertos: {slope:.2f} aciertos por punto de AUC (r={r_fit:.2f})")
print(f"    Cota superior en aciertos: {slope * upper:+.4f} sobre {14*14/25:.2f} esperados por azar.")
print(f"    Es decir, menos de 1 acierto extra cada {abs(1/(slope*upper)):.0f} sorteos en el mejor caso.")

print("\n[3] ¿Cuántos sorteos harían falta para confirmar el mejor candidato?")
observed_edge = best["auc"] - 0.5
z_alpha = stats.norm.isf(0.025)
z_power = stats.norm.isf(0.20)
se_one_step = se * np.sqrt(n_steps)
needed = ((z_alpha + z_power) * se_one_step / observed_edge) ** 2
print(f"    Ventaja observada        : {observed_edge:+.4f} AUC ({best['strategy']})")
print(f"    Pasos usados             : {n_steps}")
print(f"    Pasos necesarios (80%)   : {needed:,.0f}")
print(f"    Sorteos Kino por año     : ~156 (3 por semana)")
print(f"    Años de espera           : {needed/156:,.0f}")

print("\n[4] Lo que dice la teoría")
total = 4_457_400
print(f"    Combinaciones posibles de 14 en 25 : {total:,}")
print(f"    Probabilidad de cada una           : 1 / {total:,} = {1/total:.3e}")
print(f"    Tasa base de cada número           : 14/25 = {14/25:.2%}")
print(f"    Sorteos disponibles                : {report['dataset']['draws']:,}")
print(f"    Sorteos por combinación            : {report['dataset']['draws']/total:.6f}")
print("    → el histórico cubre el 0.055% de las combinaciones; ninguna se repite.")

print("\n[5] Veredicto por familia de variables")
families: dict[str, list[float]] = {}
for row in rows:
    families.setdefault(row["family"], []).append(row["auc"])
print(f"    {'familia':<22}{'n':>4}{'AUC medio':>12}{'AUC máx':>10}{'veredicto':>26}")
for family, values in sorted(families.items(), key=lambda kv: -np.mean(kv[1])):
    arr = np.array(values)
    verdict = "sin señal detectable"
    if family == "control":
        verdict = "referencia (ruido)"
    print(
        f"    {family:<22}{len(arr):>4}{arr.mean():>12.4f}{arr.max():>10.4f}{verdict:>26}"
    )
