import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar los datos
data = pd.read_csv('ruta_datos_transporte.csv')

# Preprocesamiento
data = data.dropna()  # Eliminar valores nulos
X = data[['hora', 'ubicacion', 'temperatura']]  # Variables independientes
y = data['demanda']  # Variable dependiente

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
predicciones = modelo.predict(X_test)
