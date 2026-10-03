# generar_simulaciones.py
import pandas as pd
import numpy as np
import os
from numba import njit
import h5py

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
