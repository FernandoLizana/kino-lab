# analizar_resultados.py
import os
import pandas as pd
import h5py
from collections import defaultdict

# Directorio donde se almacenan los archivos de bloques
block_dir = 'blocks'

def combine_blocks_and_get_top_combinations(block_dir, top_n=100):
    combinacion_frecuencia = defaultdict(int)

    # Leer cada archivo de bloque y actualizar el diccionario de frecuencias
    for filename in os.listdir(block_dir):
        if filename.endswith('.h5'):
            block_file = os.path.join(block_dir, filename)
            with h5py.File(block_file, 'r') as f:
                simulaciones = f['simulaciones'][:]
                for comb in simulaciones:
                    comb_tuple = tuple(comb)
                    combinacion_frecuencia[comb_tuple] += 1

    # Convertir el resultado en un DataFrame
    combinaciones = list(combinacion_frecuencia.keys())
    counts = list(combinacion_frecuencia.values())

    comparacion_combinaciones = pd.DataFrame(combinaciones, columns=[f'Número_{i+1}' for i in range(14)])
    comparacion_combinaciones['Frecuencia'] = counts

    # Ordenar las combinaciones por frecuencia en orden descendente
    combinaciones_recomendables = comparacion_combinaciones.sort_values(by='Frecuencia', ascending=False)

    # Obtener las mejores combinaciones
    mejores_combinaciones = combinaciones_recomendables.head(top_n)

    return mejores_combinaciones

# Obtener las 100 mejores combinaciones a partir de los bloques
mejores_100_combinaciones = combine_blocks_and_get_top_combinations(block_dir, top_n=100)

# Mostrar las 100 mejores combinaciones
print("Las 100 combinaciones más recomendables para jugar son:")
print(mejores_100_combinaciones)

# Guardar las 100 mejores combinaciones en un archivo CSV
mejores_100_combinaciones.to_csv('mejores_100_combinaciones_kino_numba.csv', index=False)
