import ctypes
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# Cargar la librería DLL
dll_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(dll_directory)

# Ahora cargar la DLL
lib = ctypes.CDLL('montecarlo.dll')

# Definir los argumentos y tipos de retorno de la función
lib.generar_simulaciones.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.generar_simulaciones.restype = None

# Leer el archivo CSV
data = pd.read_csv('historico.csv')

# Asegurarse de que los datos se leen correctamente
print(data.head())

# Convertir solo las columnas con números a enteros
numeros = data.columns[1:]  # Omitir la primera columna que es la fecha
data[numeros] = data[numeros].apply(pd.to_numeric, errors='coerce')

# Asegurarse de que todos los valores son enteros
data = data.dropna().astype({col: int for col in numeros})

# Definir el número de simulaciones
num_simulaciones = 20000000
num_numeros = data[numeros].values.max()
num_selecciones = 14

# Crear un arreglo para almacenar las simulaciones
resultado = np.zeros((num_simulaciones * num_selecciones,), dtype=np.int32)

# Llamar a la función C para generar simulaciones
lib.generar_simulaciones(num_simulaciones, num_numeros, num_selecciones, resultado.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

# Convertir el resultado en un DataFrame
simulaciones = np.array(resultado).reshape((num_simulaciones, num_selecciones))

# Procesar simulaciones y contar frecuencias
frecuencias_simuladas = pd.Series(simulaciones.flatten()).value_counts().sort_index()

# Crear un DataFrame para comparar las frecuencias
rango_numeros = np.arange(1, num_numeros + 1)
comparacion = pd.DataFrame({
    'Número': rango_numeros,
    'Frecuencia Simulada': frecuencias_simuladas.reindex(rango_numeros, fill_value=0),
})

# Guardar el resultado en un archivo CSV
comparacion.to_csv('resultados_simulaciones.csv', index=False)

# Graficar las frecuencias simuladas de cada número con un gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(comparacion['Número'], comparacion['Frecuencia Simulada'], color='orange')
plt.xlabel('Número')
plt.ylabel('Frecuencia Simulada')
plt.title('Frecuencia de cada número en las Simulaciones')
plt.grid(True)
plt.savefig('frecuencia_numeros_simulaciones.png')
plt.show()
