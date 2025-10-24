# importamos librerías 
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import backend as K
import json

K.clear_session()

# Directorios de datos
entrenamiento_data = 'C:/Users/elies/Documents/Projects/HandsCAKE/data/Entrenamiento'
validacion_data = 'C:/Users/elies/Documents/Projects/HandsCAKE/data/Validacion'

# Parámetros
iteraciones = 20
altura, longitud = 200, 200
batch_size = 32
pasos = 300
pasos_validacion = 300
lr = 0.0001

# --- Preprocesamiento de imágenes ---
preprocesamiento_entrenamiento = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

preprocesamiento_validacion = ImageDataGenerator(rescale=1./255)

imagen_entreno = preprocesamiento_entrenamiento.flow_from_directory(
    entrenamiento_data,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

clases = imagen_entreno.num_classes
clases_indices = imagen_entreno.class_indices
print(clases_indices)

with open('clases.json', 'w') as f:
    json.dump(clases_indices, f)

imagen_validacion = preprocesamiento_validacion.flow_from_directory(
    validacion_data,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# --- MODELO BASE (MobileNetV2) ---
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(altura, longitud, 3)
)

# Fase 1: ENTRENAR SOLO CAPAS SUPERIORES
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
x = Dropout(0.5)(x)
salida = Dense(clases, activation='softmax')(x)

cnn = Model(inputs=base_model.input, outputs=salida)

# Compilar
cnn.compile(
    optimizer=optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint(
    filepath='pesos_preentrenado.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

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

print("\n===== FASE 1: ENTRENANDO SOLO CAPAS SUPERIORES =====")
history1 = cnn.fit(
    imagen_entreno,
    steps_per_epoch=pasos,
    epochs=iteraciones,
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# --- Fase 2: FINE-TUNING AUTOMÁTICO ---
# Descongelamos las últimas capas de MobileNetV2 (por ejemplo, las últimas 40)
for layer in base_model.layers[-40:]:
    layer.trainable = True

# Compilamos de nuevo con tasa de aprendizaje más baja
cnn.compile(
    optimizer=optimizers.Adam(learning_rate=lr / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint_finetune = ModelCheckpoint(
    filepath='pesos_finetuned.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

early_stop_finetune = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\n===== FASE 2: FINE-TUNING DE LAS ÚLTIMAS CAPAS =====")
history2 = cnn.fit(
    imagen_entreno,
    steps_per_epoch=pasos,
    epochs=10,  # menos épocas para fine-tuning
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion,
    callbacks=[checkpoint_finetune, early_stop_finetune, reduce_lr]
)

# --- Gráficos combinados ---
def plot_history(hist1, hist2):
    acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']

    plt.figure(figsize=(8, 5))
    plt.plot(acc, label='Entrenamiento')
    plt.plot(val_acc, label='Validación')
    plt.title('Precisión del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label='Pérdida Entrenamiento')
    plt.plot(val_loss, label='Pérdida Validación')
    plt.title('Pérdida del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history1, history2)

# Guardamos el modelo final
cnn.save('Modelo_FineTuned.keras')
