import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Cargar los datos
train_df = pd.read_csv("landmarks_entrenamiento.csv")
val_df = pd.read_csv("landmarks_validacion.csv")

# Separar features y etiquetas
X_train = train_df.drop("clase", axis=1).values
y_train = train_df["clase"].values
X_val = val_df.drop("clase", axis=1).values
y_val = val_df["clase"].values

# y_train, y_val son arrays de strings con etiquetas
all_labels = np.concatenate([y_train, y_val])
encoder = LabelEncoder()
encoder.fit(all_labels)

y_train_enc = encoder.transform(y_train)
y_val_enc = encoder.transform(y_val)

# Codificar etiquetas
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_val = encoder.transform(y_val)

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
hist = modelo.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

# Guardar el modelo
modelo.save("modelo_landmarks.keras")
