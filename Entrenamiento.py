# importamos librerías 
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # preprocesamiento de imágenes
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

K.clear_session()  #Limpiamos cualquier modelo que haya quedado en memoria

entrenamiento_data = 'C:/Users/elies/Documents/Projects/HandsCAKE/data/Entrenamiento'
validacion_data = 'C:/Users/elies/Documents/Projects/HandsCAKE/data/Validacion'

#Parametros
iteraciones = 20 #Numero de veces que se va a entrenar el modelo
altura, longitud = 200, 200 #Dimensiones de las imagenes
batch_size = 32 #Numero de imagenes que se van a procesar al mismo tiempo
pasos = 300 // 1 # Numero de veces que se va a actualizar el modelo por cada epoca
pasos_validacion = 300 // 1 # Numero de veces que se va a actualizar el modelo por cada epoca de validacion
filtrosconv1 = 32 #Numero de filtros para la primera capa de convolucion
filtrosconv2 = 64 #Numero de filtros para la segunda capa de convolucion
# filtrosconv3 = 128 #Numero de filtros para la tercera capa de convolucion
tam_filtro1 = (3,3) #Tamaño del filtro para la primera capa de convolucion
tam_filtro2 = (3,3) #Tamaño del filtro para la segunda capa de convolucion
tam_pool = (2,2) #Tamaño del area de max pooling
lr = 0.0001 #Learning rate

# Preprocesamiento de las imagenes
preprocesamiento_entrenamiento = ImageDataGenerator(
    rescale=1./255, # Normalizamos los valores de los pixeles entre 0
    rotation_range=25, # Rotacion aleatoria de las imagenes
    width_shift_range=0.3, # Desplazamiento horizontal aleatorio
    height_shift_range=0.3, # Desplazamiento vertical aleatorio
    zoom_range=0.3, # Zoom aleatorio
    brightness_range=[0.7, 1.3], # Rango de brillo
    shear_range=0.2, # Cizallamiento aleatorio
    horizontal_flip=True, # Volteo horizontal aleatorio
    fill_mode='nearest' # Modo de relleno
)

preprocesamiento_validacion = ImageDataGenerator(
    rescale=1./255 # Normalizamos los valores de los pixeles entre 0
)

# Preparamos las imagenes de entrenamiento
imagen_entreno = preprocesamiento_entrenamiento.flow_from_directory(
    entrenamiento_data,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

clases = imagen_entreno.num_classes #Numero de clases o categorias que va a predecir el modelo

# Guardamos el diccionario de clases
clases_indices = imagen_entreno.class_indices
print(clases_indices)

import json
with open('clases.json', 'w') as f:
    json.dump(clases_indices, f)

 # Preparamos las imagenes de validacion
imagen_validacion = preprocesamiento_validacion.flow_from_directory(
    validacion_data,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# Construcción CNN
cnn = Sequential()

# Primera capa
cnn.add(Conv2D(
    filtrosconv1, 
    tam_filtro1, 
    padding='same', 
    activation='relu', 
    input_shape=(altura,longitud,3), 
    kernel_regularizer=regularizers.l2(0.0001)
    )
)
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Dropout(0.25))

# Segunda capa
cnn.add(Conv2D(filtrosconv2, tam_filtro2, padding='same', activation='relu',
               kernel_regularizer=regularizers.l2(0.0001))
               )
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Dropout(0.25))

# Pooling global + capas densas
cnn.add(GlobalAveragePooling2D())
cnn.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

# Compilación
optimizar = optimizers.Adam(learning_rate=lr)
cnn.compile(loss='categorical_crossentropy', optimizer=optimizar, metrics=['accuracy'])

#Guarda solo pesos como checkpoint
checkpoint = ModelCheckpoint(
    filepath='pesos.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

#Early stopping para evitar sobreentrenamiento
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)   

# Entrenamiento
history = cnn.fit(
    imagen_entreno, 
    steps_per_epoch=pasos, 
    epochs=iteraciones, 
    validation_data=imagen_validacion, 
    validation_steps=pasos_validacion,
    callbacks=[checkpoint, early_stop]
)

#  Graficar accuracy y val_accuracy
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#  Graficar pérdida (loss)
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Guardamos el modelo
cnn.save('Modelo.keras')