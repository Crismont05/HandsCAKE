# importamos librerías 
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # preprocesamiento de imágenes
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

K.clear_session()  #Limpiamos cualquier modelo que haya quedado en memoria

entrenamiento_data = 'C:/Users/elies/Documents/Projects/HandsCAKE/data/Entrenamiento'
validacion_data = 'C:/Users/elies/Documents/Projects/HandsCAKE/data/Validacion'

#Parametros
iteraciones = 20 #Numero de veces que se va a entrenar el modelo
altura, longitud = 200, 200 #Dimensiones de las imagenes
batch_size = 16 #Numero de imagenes que se van a procesar al mismo tiempo
pasos = 300 // 1 # Numero de veces que se va a actualizar el modelo por cada epoca
pasos_validacion = 300 // 1 # Numero de veces que se va a actualizar el modelo por cada epoca de validacion
filtrosconv1 = 32 #Numero de filtros para la primera capa de convolucion
filtrosconv2 = 64 #Numero de filtros para la segunda capa de convolucion
filtrosconv3 = 128 #Numero de filtros para la tercera capa de convolucion
tam_filtro1 = (3,3) #Tamaño del filtro para la primera capa de convolucion
tam_filtro2 = (3,3) #Tamaño del filtro para la segunda capa de convolucion
tam_filtro3 = (3,3) #Tamaño del filtro para la tercera capa de convolucion
tam_pool = (2,2) #Tamaño del area de max pooling
lr = 0.0005 #Learning rate

# Preprocesamiento de las imagenes
preprocesamiento_entrenamiento = ImageDataGenerator(
    rescale=1./255, # Normalizamos los valores de los pixeles entre 0 y 1
    shear_range=0.2, # Aplicamos transformaciones aleatorias a las imagenes
    zoom_range=0.2, # Genera imagenes con zoom aleatorio
    rotation_range=20, # Rota las imagenes aleatoriamente
    width_shift_range=0.2, # Desplaza las imagenes horizontalmente
    height_shift_range=0.2, # Desplaza las imagenes verticalmente
    brightness_range=[0.7,1.3], # Cambia el brillo de las imagenes
    horizontal_flip=True, # Voltea las imagenes horizontalmente para entrenar mejor
    fill_mode='nearest' # Rellena los pixeles que quedan vacios tras una transformacion
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
cnn.add(Conv2D(filtrosconv1, tam_filtro1, padding='same', activation='relu', input_shape=(altura,longitud,3)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Dropout(0.25))

# Segunda capa
cnn.add(Conv2D(filtrosconv2, tam_filtro2, padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Dropout(0.25))

# Tercera capa
cnn.add(Conv2D(filtrosconv3, tam_filtro3, padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=tam_pool))
cnn.add(Dropout(0.25))

# Pooling global + capas densas
cnn.add(GlobalAveragePooling2D())
cnn.add(Dense(256, activation='relu'))
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

#Entrenaremos nuestra red
cnn.fit(
    imagen_entreno, 
    steps_per_epoch=pasos, 
    epochs= iteraciones, 
    validation_data= imagen_validacion, 
    validation_steps=pasos_validacion,
    callbacks=[checkpoint]
)

#Guardamos el modelo
cnn.save('Modelo.keras')