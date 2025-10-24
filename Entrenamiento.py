# importamos librerías 
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # preprocesamiento de imágenes
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D
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
tam_filtro1 = (3,3) #Tamaño del filtro para la primera capa de convolucion
tam_filtro2 = (2,2) #Tamaño del filtro para la segunda capa de convolucion
tam_pool = (2,2) #Tamaño del area de max pooling
lr = 0.0005 #Learning rate

# Preprocesamiento de las imagenes
preprocesamiento_entrenamiento = ImageDataGenerator(
    rescale=1./255, # Normalizamos los valores de los pixeles entre 0 y 1
    shear_range=0.2, # Aplicamos transformaciones aleatorias a las imagenes
    zoom_range=0.2, # Genera imagenes con zoom aleatorio
    horizontal_flip=True # Voltea las imagenes horizontalmente para entrenar mejor
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

# Creacion de red neuronal convolucional (CNN)
cnn = Sequential()
#Agregamos filtros con el fin de volver nuestra imagen muy profunda pero pequeña
cnn.add(Conv2D(filtrosconv1, tam_filtro1, padding = 'same', input_shape=(altura,longitud,3), activation = 'relu')) #Agregamos la primera capa
         #Es una convolucion y realizamos config
cnn.add(MaxPooling2D(pool_size=tam_pool)) #Despues de la primera capa vamos a tener una capa de max pooling y asignamos el tamaño

cnn.add(Conv2D(filtrosconv2, tam_filtro2, padding = 'same', activation='relu')) #Agregamos nueva capa

cnn.add(MaxPooling2D(pool_size=tam_pool))

#Ahora vamos a convertir esa imagen profunda a una plana, para tener 1 dimension con toda la info
cnn.add(Flatten())  #Aplanamos la imagen
cnn.add(Dense(256,activation='relu'))  #Asignamos 256 neuronas
cnn.add(Dropout(0.5)) #Apagamos el 50% de las neuronas en la funcion anterior para no sobreajustar la red
cnn.add(Dense(clases, activation='softmax'))  #Es nuestra ultima capa, es la que nos dice la probabilidad de que sea alguna de las clases

#Agregamos parametros para optimizar el modelo
#Durante el entrenamiento tenga una autoevalucion, que se optimice con Adam, y la metrica sera accuracy
optimizar = optimizers.Adam(learning_rate= lr)
cnn.compile(loss = 'categorical_crossentropy', optimizer= optimizar, metrics=['accuracy'])

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
cnn.save_weights('pesos.weights.h5')