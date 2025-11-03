import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json # Necesario para guardar las clases
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

K.clear_session()

# Rutas de carpetas
base_dir = "C:/Users/elies/Documents/Projects/HandsCAKE/data/Landmarks"
train_dir = os.path.join(base_dir, "C:/Users/elies/Documents/Projects/HandsCAKE/data/Landmarks/Entrenamiento")
val_dir = os.path.join(base_dir, "C:/Users/elies/Documents/Projects/HandsCAKE/data/Landmarks/Validacion")
input_shape = 63 # 21 puntos * 3 coordenadas (x, y, z)

# --- Funciones de Carga y Preprocesamiento ---
def cargar_datos(carpeta):
    """Carga todos los CSV de una carpeta y los concatena."""
    dfs = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".csv"):
            df = pd.read_csv(os.path.join(carpeta, archivo))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

TEST_SIZE = 0.2
BATCH_SIZE = 64
EPOCHS = 200
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# Cargar datasets
train_df = cargar_datos(train_dir)
val_df = cargar_datos(val_dir)

# Separar features y etiquetas
X_train = train_df.drop("clase", axis=1).values
y_train = train_df["clase"].values
X_val = val_df.drop("clase", axis=1).values
y_val = val_df["clase"].values

# Codificar etiquetas
encoder = LabelEncoder()
encoder.fit(np.concatenate([y_train, y_val]))
y_train_enc = encoder.transform(y_train)
y_val_enc = encoder.transform(y_val)
num_classes = len(encoder.classes_)

# Guardar las clases para inferencia futura
clases_indices = {
    clase: int(indice)  # <-- La clave es usar int() aquí
    for clase, indice in zip(encoder.classes_, encoder.transform(encoder.classes_))
}

with open('clases_landmarks.json', 'w') as f:
    json.dump(clases_indices, f) # La línea 53 ahora debería funcionar
    
x_train

# --- Callbacks ---
# Usaremos los mismos callbacks de tu código de imágenes
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# --- Definición del Modelo MLP con Nombres de Capa ---
def create_mlp_model(input_shape, num_classes):
    """Crea y devuelve el modelo MLP."""
    modelo = models.Sequential([
        layers.Input(shape=(input_shape,), name='input_layer'),
        layers.Dense(128, activation='relu', name='hidden_1'),
        layers.Dropout(0.3, name='dropout_1'),
        layers.Dense(64, activation='relu', name='hidden_2'),
        layers.Dropout(0.3, name='dropout_2'),
        layers.Dense(num_classes, activation='softmax', name='output_layer') # Esta es la "cabeza"
    ], name='Landmark_MLP')
    return modelo

cnn = create_mlp_model(input_shape, num_classes)

# =======================================================
## FASE 1: ENTRENAR SOLO LA CAPA DE SALIDA ("CABEZA")
# =======================================================
# Congelar todas las capas excepto la capa de salida
for layer in cnn.layers:
    if layer.name != 'output_layer':
        layer.trainable = False
    else:
        layer.trainable = True

lr_fase1 = 0.001
checkpoint_fase1 = ModelCheckpoint(
    filepath='pesos_fase1_landmarks.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Compilar
cnn.compile(
    optimizer=optimizers.Adam(learning_rate=lr_fase1),
    loss='sparse_categorical_crossentropy', # Usamos sparse porque y_train_enc no está en one-hot
    metrics=['accuracy']
)

print("\n===== FASE 1: ENTRENANDO SOLO LA CAPA DE SALIDA =====")
history1 = cnn.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=20, # Menos épocas, ya que solo es la cabeza
    batch_size=16,
    callbacks=[checkpoint_fase1, early_stop, reduce_lr]
)

# =======================================================
## FASE 2: FINE-TUNING DE TODO EL MODELO
# =======================================================
# Descongelar todas las capas
for layer in cnn.layers:
    layer.trainable = True

lr_fase2 = lr_fase1 / 10 # Tasa de aprendizaje más baja para fine-tuning

checkpoint_fase2 = ModelCheckpoint(
    filepath='pesos_finetuned_landmarks.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Compilar de nuevo con tasa de aprendizaje más baja
cnn.compile(
    optimizer=optimizers.Adam(learning_rate=lr_fase2),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n===== FASE 2: FINE-TUNING DE TODO EL MODELO =====")
history2 = cnn.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),
    epochs=30, # Más épocas, pero el EarlyStopping lo detendrá
    batch_size=16,
    callbacks=[checkpoint_fase2, early_stop, reduce_lr]
)

# --- Gráficos combinados (Mantenido la estructura de tu función) ---
def plot_history(hist1, hist2):
    acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, label='Entrenamiento')
    plt.plot(epochs, val_acc, label='Validación')
    plt.title('Precisión del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label='Pérdida Entrenamiento')
    plt.plot(epochs, val_loss, label='Pérdida Validación')
    plt.title('Pérdida del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Requiere matplotlib
import matplotlib.pyplot as plt
plot_history(history1, history2)

cnn.load_weights('pesos_finetuned_landmarks.weights.h5')

# Guardamos el modelo final
cnn.save("Modelo_Landmarks_FineTuned.keras")

print("Entrenamiento finalizado y modelo guardado como 'Modelo_Landmarks_FineTuned.keras'")