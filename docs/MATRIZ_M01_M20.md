# Matriz M01–M20

| ID | Estado | Implementación | Evidencia |
|---|---|---|---|
| M01 | Implementado y verificado | `/inicio`, demos, tour omitible | `test_lab_routes_smoke` + UI `/inicio` |
| M02 | Implementado y verificado | Ingestión + `LabDraw`; no recorta 15→14 | `test_ingest_keeps_15_and_14` |
| M03 | Implementado y verificado | `/importar` preview + confirm + CSV de errores | UI importar + seed historico |
| M04 | Implementado y verificado | Perfiles versionados | `test_kino_references` |
| M05 | Implementado y verificado | `/explorar` filtros, tabla, gráfico, CSV | UI explorar |
| M06 | Implementado y verificado | Hipergeométrica | C(25,14)=4457400, E=7.84, soporte 3–14 |
| M07 | Implementado y verificado | Combinaciones válidas | `test_impossible_filters` |
| M08 | Implementado y verificado | Monte Carlo + job pausa/cancel | `/simular` + `job_runner` |
| M09 | Implementado y verificado | Comparador vs uniforme | `lab/strategies.py` |
| M10 | Implementado y verificado | Walk-forward / no-fuga | `test_no_future_leak_frequency` |
| M11 | Implementado y verificado | Fichas de modelos | `/modelos` |
| M12 | Implementado y verificado | Sesgos sintéticos | `/sesgos` |
| M13 | Implementado y verificado | 7 desafíos | `/aprender` |
| M14 | Implementado y verificado | Gastos ficticios | `/gastos` |
| M15 | Implementado y verificado | Cuaderno + ejecuciones | `/experimentos` |
| M16 | Implementado y verificado | CSV / JSON / PDF | export routes |
| M17 | Implementado y verificado | Límites, pausa, checkpoint | modos 10k/100k/1e6 |
| M18 | Implementado y verificado | Asistente local | `/asistente` |
| M19 | Implementado y verificado | Vendor local + loopback + backup | `static/vendor/`, `/ajustes` |
| M20 | Implementado pendiente viewports en navegador | skip-link, foco, reduced-motion | `app.css`; viewports se comprueban en esta sesión |

Este repositorio público no incluye historial previo ni credenciales.
