import numpy as np
import os
from numba import njit
import h5py
from collections import defaultdict
import pandas as pd

# Crear directorio para almacenar archivos codificados
encoded_block_dir = 'encoded_blocks'
os.makedirs(encoded_block_dir, exist_ok=True)

# Definir el número total de simulaciones
num_simulaciones_total = 4_000_000_000
block_size = 20_000_000  # Tamaño del bloque para la simulación
num_blocks = num_simulaciones_total // block_size

# Verificar el rango de números posibles
rango_numeros = np.arange(1, 26)  # Números del 1 al 25

@njit
def monte_carlo_simulation(rango_numeros, num_simulaciones):
    simulaciones = np.zeros((num_simulaciones, 14), dtype=np.uint8)  # Uso de uint8 para reducir tamaño
    for i in range(num_simulaciones):
        simulaciones[i, :] = np.sort(np.random.choice(rango_numeros, 14, replace=False))
    return simulaciones

@njit
def encode_combination(combination):
    bitmask = np.uint64(0)
    for number in combination:
        bitmask |= (np.uint64(1) << (number - 1))
    return bitmask

@njit
def decode_combination(bitmask, num_elements):
    combination = []
    for i in range(num_elements):
        if bitmask & (np.uint64(1) << i):
            combination.append(i + 1)
    return combination

def process_blocks_and_save(num_blocks, block_size, rango_numeros, block_dir):
    for block in range(num_blocks):
        simulaciones = monte_carlo_simulation(rango_numeros, block_size)
        encoded_simulations = np.empty(block_size, dtype=np.uint64)
        for i, comb in enumerate(simulaciones):
            encoded_simulations[i] = encode_combination(comb)
        block_file = os.path.join(block_dir, f'encoded_simulaciones_block_{block}.h5')
        with h5py.File(block_file, 'w') as f:
            f.create_dataset('simulaciones', data=encoded_simulations, dtype=np.uint64, compression='gzip', compression_opts=9)

# Realizar la simulación en bloques y guardar los resultados
process_blocks_and_save(num_blocks, block_size, rango_numeros, encoded_block_dir)

def combine_blocks_and_get_top_combinations(encoded_block_dir, top_n=100):
    combinacion_frecuencia = defaultdict(int)

    # Leer cada archivo de bloque y actualizar el diccionario de frecuencias
    for filename in os.listdir(encoded_block_dir):
        if filename.endswith('.h5'):
            block_file = os.path.join(encoded_block_dir, filename)
            with h5py.File(block_file, 'r') as f:
                encoded_simulations = f['simulaciones'][:]
                for encoded_comb in encoded_simulations:
                    combinacion_frecuencia[encoded_comb] += 1

    # Convertir el resultado en un DataFrame
    combinaciones = []
    counts = []
    for enc_comb, freq in combinacion_frecuencia.items():
        dec_comb = decode_combination(enc_comb, 25)
        combinaciones.append(dec_comb)
        counts.append(freq)

    comparacion_combinaciones = pd.DataFrame(combinaciones, columns=[f'Número_{i+1}' for i in range(14)])
    comparacion_combinaciones['Frecuencia'] = counts

    # Ordenar las combinaciones por frecuencia en orden descendente
    combinaciones_recomendables = comparacion_combinaciones.sort_values(by='Frecuencia', ascending=False)

    # Obtener las mejores combinaciones
    mejores_combinaciones = combinaciones_recomendables.head(top_n)

    return mejores_combinaciones

# Obtener las 100 mejores combinaciones a partir de los bloques
mejores_100_combinaciones = combine_blocks_and_get_top_combinations(encoded_block_dir, top_n=100)

# Mostrar las 100 mejores combinaciones
print("Las 100 combinaciones más recomendables para jugar son:")
print(mejores_100_combinaciones)

# Guardar las 100 mejores combinaciones en un archivo CSV
mejores_100_combinaciones.to_csv('mejores_100_combinaciones_kino_final_final.csv', index=False)
s
