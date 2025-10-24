import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


# Se utiliza el archivo CSV generado en Extraccion.py
df = pd.read_csv('datos_entrenamiento.csv', header=None) 

# La última columna (índice -1) es la etiqueta (letra)
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]  

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Entrenar el modelo ---
print("Entrenando el modelo Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación
score = model.score(X_test, y_test)
print(f"Precisión del modelo en el set de prueba: {score*100:.2f}%")

# --- 3. Guardar el modelo entrenado ---
data = {"model": model}
with open('clasificador_letras.pkl', 'wb') as file:
    pickle.dump(data, file)

print("Modelo guardado exitosamente como 'clasificador_letras.pkl'")