import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

# Rutas de carpetas
base_dir = "landmarks_capturados"
train_dir = os.path.join(base_dir, "Entrenamiento")
val_dir   = os.path.join(base_dir, "Validacion")

# Función para cargar todos los CSV de una carpeta
def cargar_datos(carpeta):
    dfs = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".csv"):
            df = pd.read_csv(os.path.join(carpeta, archivo))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Cargar datasets
train_df = cargar_datos(train_dir)
val_df   = cargar_datos(val_dir)

# Separar features y etiquetas
X_train = train_df.drop("clase", axis=1).values
y_train = train_df["clase"].values
X_val   = val_df.drop("clase", axis=1).values
y_val   = val_df["clase"].values

# Codificar etiquetas
encoder = LabelEncoder()
encoder.fit(np.concatenate([y_train, y_val]))
y_train_enc = encoder.transform(y_train)
y_val_enc   = encoder.transform(y_val)

# Crear modelo MLP
modelo = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(encoder.classes_), activation='softmax')
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar
hist = modelo.fit(X_train, y_train_enc, validation_data=(X_val, y_val_enc),
                  epochs=50, batch_size=16)

# Guardar el modelo
modelo.save("modelo_landmarksB.keras")

print("Entrenamiento finalizado y modelo guardado como 'modelo_landmarksB.keras'")
