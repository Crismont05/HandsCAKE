import cv2
import mediapipe as mp
import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model

modelo = 'C:/Users/elies/Documents/Projects/HandsCAKE/Modelo.keras'
peso =  'C:/Users/elies/Documents/Projects/HandsCAKE/pesos.weights.h5'
cnn = load_model(modelo)  #Cargamos el modelo
cnn.load_weights(peso)  #Cargamos los pesos

direccion = 'C:/Users/elies/Documents/Projects/HandsCAKE/data/Validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

import json

# Cargar el diccionario de clases
with open('clases.json', 'r') as f:
    clases_dict = json.load(f)  # {'A': 0, 'E': 1, 'I': 2, 'O': 3, 'U': 4}

# Invertir para que podamos hacer indice -> nombre
indice_a_nombre = {v: k for k, v in clases_dict.items()}

#Leemos la camara
cap = cv2.VideoCapture(0)

#----------------------------Creamos un obejto que va almacenar la deteccion y el seguimiento de las manos------------
clase_manos  =  mp.solutions.hands
manos = clase_manos.Hands() #Primer parametro, FALSE para que no haga la deteccion 24/7
                            #Solo hara deteccion cuando hay una confianza alta
                            #Segundo parametro: numero maximo de manos
                            #Tercer parametro: confianza minima de deteccion
                            #Cuarto parametro: confianza minima de seguimiento

#----------------------------------Metodo para dibujar las manos---------------------------
dibujo = mp.solutions.drawing_utils #Con este metodo dibujamos 21 puntos criticos de la mano


while True:
    ret,frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []  # En esta lista vamos a almcenar las coordenadas de los puntos
    #print(resultado.multi_hand_landmarks) #Si queremos ver si existe la deteccion

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            alto, ancho, _ = frame.shape
            coords = [(int(lm.x*ancho), int(lm.y*alto)) for lm in mano.landmark]

            x_min = max(0, min([x for x,_ in coords]) - 20)
            y_min = max(0, min([y for _,y in coords]) - 20)
            x_max = min(ancho, max([x for x,_ in coords]) + 20)
            y_max = min(alto, max([y for _,y in coords]) + 20)

            mano_crop = frame[y_min:y_max, x_min:x_max]
            mano_crop = cv2.resize(mano_crop, (200,200))
            mano_crop_norm = mano_crop / 255.0
            mano_crop_norm = np.expand_dims(mano_crop_norm, axis=0)

            pred = cnn.predict(mano_crop_norm)
            clase = np.argmax(pred)
            nombre_clase = indice_a_nombre[clase]

            cv2.putText(frame, f"Clase: {nombre_clase}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()