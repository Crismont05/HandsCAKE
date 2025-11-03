import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os, json

# ======================================================
# CONFIGURACIÓN
# ======================================================
DATA_DIR_TRAIN = "C:/Users/elies/Documents/Projects/HandsCAKE/data/Landmarks/Entrenamiento"
DATA_DIR_VAL   = "C:/Users/elies/Documents/Projects/HandsCAKE/data/Landmarks/Validacion"
MODEL_SAVE_PATH = "Modelo_Landmarks_FineTuned_V2.keras"
CLASSES_JSON = "clases_landmarks.json"

BATCH_SIZE = 64
EPOCHS = 200
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================================================
# 1. FUNCIÓN PARA CARGAR LOS CSV
# ======================================================
def cargar_dataset_desde_carpeta(carpeta):
    dataframes = []
    clases = []

    for file in os.listdir(carpeta):
        if file.endswith(".csv"):
            clase = os.path.splitext(file)[0].upper()
            df = pd.read_csv(os.path.join(carpeta, file))
            
            # Si no hay columna "clase", se agrega
            if "clase" not in df.columns:
                df["clase"] = clase
            
            dataframes.append(df)
            clases.append(clase)

    data_total = pd.concat(dataframes, ignore_index=True)
    print(f"Carpeta {carpeta} cargada con {len(data_total)} muestras, clases: {sorted(set(clases))}")
    return data_total

print("Cargando datos de entrenamiento y validación...")
df_train = cargar_dataset_desde_carpeta(DATA_DIR_TRAIN)
df_val   = cargar_dataset_desde_carpeta(DATA_DIR_VAL)

# ======================================================
# 2. PROCESAMIENTO Y CODIFICACIÓN
# ======================================================
X_train = df_train.drop(columns=["clase"]).values.astype(np.float32)
y_train = df_train["clase"].values

X_val = df_val.drop(columns=["clase"]).values.astype(np.float32)
y_val = df_val["clase"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
label_encoder.fit(np.unique(np.concatenate([y_train, y_val])))

y_train_enc = label_encoder.transform(y_train)
y_val_enc = label_encoder.transform(y_val)
num_classes = len(label_encoder.classes_)

y_train_cat = to_categorical(y_train_enc, num_classes)
y_val_cat = to_categorical(y_val_enc, num_classes)

# Guardar diccionario de clases
with open(CLASSES_JSON, "w") as f:
    json.dump({cls: int(idx) for idx, cls in enumerate(label_encoder.classes_)}, f, indent=4)

print(f"Total de clases: {num_classes}")

# ======================================================
# 3. NORMALIZACIÓN DE LANDMARKS
# ======================================================
def normalize_landmarks(x):
    coords = tf.reshape(x, (-1, 3))
    wrist = coords[0]
    coords -= wrist
    norm = tf.norm(coords, axis=1)
    max_val = tf.reduce_max(norm)
    coords /= max_val
    return tf.reshape(coords, (-1,))

# ======================================================
# 4. DATA AUGMENTATION (solo en entrenamiento)
# ======================================================
def augment(x, y):
    x = tf.cast(x, tf.float32)
    x = normalize_landmarks(x)
    x = tf.reshape(x, (-1, 3))

    angle = tf.random.uniform([], minval=-10, maxval=10) * np.pi / 180
    rotation = tf.stack([
        [tf.cos(angle), -tf.sin(angle), 0.0],
        [tf.sin(angle),  tf.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    x = tf.matmul(x, rotation)

    scale = tf.random.uniform([], 0.95, 1.05)
    x *= scale
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.005)
    x += noise

    x = tf.reshape(x, (-1,))
    return x, y

def preprocess_val(x, y):
    x = normalize_landmarks(x)
    return x, y

# ======================================================
# 5. CREAR DATASETS DE TENSORFLOW
# ======================================================
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))

train_ds = (train_ds
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(4096)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (val_ds
          .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(tf.data.AUTOTUNE))

# ======================================================
# 6. MODELO NEURONAL
# ======================================================
def build_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

cnn = build_model(input_dim=63, num_classes=num_classes)
cnn.summary()

# ======================================================
# 7. CALLBACKS
# ======================================================
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr  = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
checkpoint = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max')

# ======================================================
# 8. ENTRENAMIENTO
# ======================================================
history = cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ======================================================
# 9. GUARDAR MODELO FINAL
# ======================================================
cnn.save(MODEL_SAVE_PATH)
print(f"\n Modelo guardado en: {MODEL_SAVE_PATH}")

# ======================================================
# 10. GRAFICAR ENTRENAMIENTO
# ======================================================
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("historial_entrenamiento.png", dpi=120)
plt.show()

print(" Entrenamiento finalizado y gráfica guardada en 'historial_entrenamiento.png'")
