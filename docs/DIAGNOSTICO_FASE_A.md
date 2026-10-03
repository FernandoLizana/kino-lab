# Diagnóstico del código legado

Este repositorio público es un único commit limpio. Los scripts `generar_data_*`
y `montecarlo.c` son restos educativos, no el producto Flask.

| Tema | Estado en este árbol |
|---|---|
| Credenciales en código | No hay. Flask no automatiza compras. |
| `historico.csv` 15 y 14 números | 799 filas de 15 y 2.151 de 14, segmentadas por perfil |
| Simulaciones de miles de millones | Solo en scripts legado; Flask tope 1.000.000 |
| `montecarlo.c` con posibles repetidos | Confirmado; el motor `lab/` no lo usa |
| Pruebas | pytest en `tests/` |

Producto: `python app.py` → http://127.0.0.1:5000/inicio
