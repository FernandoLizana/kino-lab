import pandas as pd
import numpy as np
import os
from numba import njit
import h5py
from collections import defaultdict

# Crear directorio para almacenar archivos de bloques
block_dir = 'blocks'
os.makedirs(block_dir, exist_ok=True)

# Leer el archivo CSV
data = pd.read_csv('historico.csv')

# Asegurarse de que los datos se leen correctamente
print(data.head())

# Convertir solo las columnas con números a enteros
numeros = data.columns[1:]  # Omitir la primera columna que es la fecha
data[numeros] = data[numeros].apply(pd.to_numeric, errors='coerce')

# Asegurarse de que todos los valores son enteros
data = data.dropna().astype({col: int for col in numeros})

# Obtener los números históricos
historicos = data[numeros].values.flatten()

# Definir el número total de simulaciones
num_simulaciones_total = 2_000_000_000
block_size = 20_000_000  # Tamaño del bloque para la simulación
num_blocks = num_simulaciones_total // block_size

# Verificar el rango de números posibles
rango_numeros = np.arange(1, historicos.max() + 1)

@njit
def monte_carlo_simulation(rango_numeros, num_simulaciones):
    simulaciones = np.zeros((num_simulaciones, 14), dtype=np.uint8)  # Uso de uint8 para reducir tamaño
    for i in range(num_simulaciones):
        simulaciones[i, :] = np.sort(np.random.choice(rango_numeros, 14, replace=False))
    return simulaciones

def process_blocks_and_save(num_blocks, block_size, rango_numeros, block_dir):
    for block in range(num_blocks):
        simulaciones = monte_carlo_simulation(rango_numeros, block_size)
        block_file = os.path.join(block_dir, f'simulaciones_block_{block}.h5')
        with h5py.File(block_file, 'w') as f:
            f.create_dataset('simulaciones', data=simulaciones, dtype=np.uint8, compression='gzip', compression_opts=9)

# Realizar la simulación en bloques y guardar los resultados
process_blocks_and_save(num_blocks, block_size, rango_numeros, block_dir)

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
