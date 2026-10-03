# Fase A — Diagnóstico (contrastado con archivos reales)

Informe histórico `analisis-proyecto-loteria.canvas.tsx` = revisión estática antigua.
No se reutilizan sus cifras como mediciones propias.

| Hallazgo del informe | Estado | Evidencia actual |
|---|---|---|
| Credenciales en `scrap.py` | Ya corregido en el árbol; residual en historial git | Working tree usa `LOTERIA_RUT`/`LOTERIA_CLAVE`. Commit inicial aún las contiene. No se reescribe historial. |
| `historico.csv` 15 y 14 números | Confirmado | 2.950 filas: 799 con 15 bolillas (19-09-1990 … 01-01-2006) y 2.151 con 14 (08-01-2006 … 04-08-2024). Fecha de cambio = observada en el archivo, no oficial. |
| Carga conserva 798 y descarta 2.151 | No aplicable al Flask actual | `csv_service._validate_numbers` exige 14. Hoy se descartan las filas de 15 (hasta 30 errores) y se aceptan las de 14. El problema real es la exclusión silenciosa del período clásico. |
| 4.000 millones de iteraciones | Confirmado solo en legado | `max_rendimiento.py`. Flask no hereda ese default. |
| Combinaciones con repetidos en `montecarlo.c` | Confirmado | `rand() % n` independiente por posición; puede repetir. Motor canónico nuevo no usa este C. |
| Error `s` aislada en `max_rendimiento.py` | No verificable / no presente | No hay una `s` suelta en el archivo actual. |
| Variantes ML duplicadas | Confirmado (legado) | `generar_data_*.py`. Fuera del producto Flask. |
| Modelos solo de frecuencia, sin eval temporal | Parcialmente ya corregido | Flask tiene walk-forward. Scripts legado no. |
| Resultados sobrescritos | Parcialmente ya corregido | `Experiment` versiona corridas. Scripts legado escriben PNG fijos. |
| Sin pruebas | Ya corregido | 32 tests pytest (núcleo + loterías). |

## Recorrido real hoy

`python app.py` → Flask 127.0.0.1:5000 → seed `data/kino_draws.csv` / `kino.csv` / `historico.csv` (solo filas de 14) → dashboard, upload, analysis, score, backtest, labs experimentales, otras loterías.

`scrap.py` no se importa. CDN Bootstrap/Chart.js (offline incompleto). Identidad visual: fondo `#0f1419`, acento `#3d8bfd`, cards 12px.

## Decisiones

- Conservar Flask + plantillas actuales.
- Motor canónico nuevo en `lab/` (perfiles, hipergeométrica, ingestión, simulación).
- No convertir 15→14. Segmentar por perfil observado.
- Defaults de simulación: 10.000 (rápido) / 100.000 (detallado) / tope 1.000.000.
- `montecarlo.c` y `scrap.py` quedan como legado desconectado.
