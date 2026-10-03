import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
num_simulaciones = 1000000

# Obtener los números históricos
historicos = data[numeros].values.flatten()

# Verificar el rango de números posibles
rango_numeros = np.arange(1, historicos.max() + 1)

# Crear un vector con los números y sus frecuencias históricas
X = np.arange(1, rango_numeros.max() + 1).reshape(-1, 1)
y = np.bincount(historicos, minlength=rango_numeros.max() + 1)[1:]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de Random Forest
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Predecir las frecuencias de los números con el modelo
frecuencias_predichas = modelo_rf.predict(X)
frecuencias_predichas = np.clip(frecuencias_predichas, 0, None)  # Asegurar que las frecuencias sean no negativas

# Evaluar el modelo
mse = mean_squared_error(y_test, modelo_rf.predict(X_test))
print(f"Error cuadrático medio del modelo Random Forest: {mse}")

# Realizar la simulación de Monte Carlo ponderada por las frecuencias predichas
probabilidades = frecuencias_predichas / frecuencias_predichas.sum()
simulaciones = np.array([
    np.sort(np.random.choice(rango_numeros, 14, replace=False, p=probabilidades)) 
    for _ in range(num_simulaciones)
])

# Contar la frecuencia de cada número en las simulaciones
frecuencias_simuladas = np.bincount(simulaciones.flatten(), minlength=rango_numeros.max() + 1)[1:]

# Crear un DataFrame para comparar las frecuencias
comparacion = pd.DataFrame({
    'Número': np.arange(1, rango_numeros.max() + 1),
    'Frecuencia Histórica': y,
    'Frecuencia Simulada': frecuencias_simuladas,
    'Frecuencia Predicha': frecuencias_predichas
})

# Guardar el resultado en un archivo CSV
comparacion.to_csv('resultados_comparacion_kino_rf.csv', index=False)

# Graficar las frecuencias históricas vs simuladas vs predichas
"""
plt.figure(figsize=(10, 6))
plt.plot(comparacion['Número'], comparacion['Frecuencia Histórica'], marker='o', label='Histórica')
plt.plot(comparacion['Número'], comparacion['Frecuencia Simulada'], marker='x', label='Simulada')
plt.plot(comparacion['Número'], comparacion['Frecuencia Predicha'], marker='s', label='Predicha')
plt.xlabel('Número')
plt.ylabel('Frecuencia')
plt.title('Comparación de Frecuencias: Históricas vs Simuladas vs Predichas (Random Forest)')
plt.legend()
plt.grid(True)
plt.savefig('comparacion_frecuencias_kino_rf.png')
plt.show()
"""
# Contar la frecuencia de cada combinación en las simulaciones
combinaciones, counts = np.unique(simulaciones, axis=0, return_counts=True)

# Crear un DataFrame para comparar las combinaciones y sus frecuencias
comparacion_combinaciones = pd.DataFrame(combinaciones, columns=[f'Número_{i+1}' for i in range(14)])
comparacion_combinaciones['Frecuencia'] = counts

# Ordenar las combinaciones por frecuencia en orden descendente
combinaciones_recomendables = comparacion_combinaciones.sort_values(by='Frecuencia', ascending=False)

# Obtener las 100 mejores combinaciones
mejores_100_combinaciones = combinaciones_recomendables.head(100)

# Mostrar las 100 mejores combinaciones
print("Las 100 combinaciones más recomendables para jugar son:")
print(mejores_100_combinaciones)

# Guardar las 100 mejores combinaciones en un archivo CSV
mejores_100_combinaciones.to_csv('mejores_100_combinaciones_kino_rf.csv', index=False)
"""
# Graficar las frecuencias simuladas de cada número con un gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(comparacion['Número'], comparacion['Frecuencia Simulada'], color='orange')
plt.xlabel('Número')
plt.ylabel('Frecuencia Simulada')
plt.title('Frecuencia de cada número en las Simulaciones (Random Forest)')
plt.grid(True)
plt.savefig('frecuencia_numeros_simulaciones_rf.png')
plt.show()
"""