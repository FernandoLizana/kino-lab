import pandas as pd
import numpy as np

# Leer el archivo CSV
data = pd.read_csv('historico.csv')

# Convertir solo las columnas con números a enteros
numeros = data.columns[1:]  # Omitir la primera columna que es la fecha
data[numeros] = data[numeros].apply(pd.to_numeric, errors='coerce')

# Asegurarse de que todos los valores son enteros
data = data.dropna().astype({col: int for col in numeros})

# Obtener los números históricos
historicos = data[numeros].values.flatten()

# Verificar el rango de números posibles
rango_numeros = np.arange(1, historicos.max() + 1)

# Crear un vector con los números y sus frecuencias históricas
frecuencias_hist = np.bincount(historicos, minlength=rango_numeros.max() + 1)[1:]

# Crear un DataFrame para mostrar frecuencias históricas
frecuencias_df = pd.DataFrame({
    'Número': rango_numeros,
    'Frecuencia': frecuencias_hist
})

# Mostrar frecuencias
print(frecuencias_df)

# Puedes utilizar esta información para hacer predicciones basadas en la frecuencia histórica.
