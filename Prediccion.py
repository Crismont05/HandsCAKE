import cv2
import mediapipe as mp
import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model

modelo = 'C:/Users/elies/Documents/Projects/HandsCAKE/Modelo.h5'
peso =  'C:/Users/elies/Documents/Projects/HandsCAKE/pesos.h5'
cnn = load_model(modelo)  #Cargamos el modelo
cnn.load_weights(peso)  #Cargamos los pesos

direccion = 'C:/Users/elies/Documents/Projects/HandsCAKE/data/Validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

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


while (1):
    ret,frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []  # En esta lista vamos a almcenar las coordenadas de los puntos
    #print(resultado.multi_hand_landmarks) #Si queremos ver si existe la deteccion

    if resultado.multi_hand_landmarks:
        for mano in resultado.multi_hand_landmarks:
            dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            # Convertir landmarks a coordenadas en píxeles
            alto, ancho, _ = frame.shape
            coords = []
            for lm in mano.landmark:
                coords.append((int(lm.x * ancho), int(lm.y * alto)))

            # Calcular caja delimitadora
            x_min = min([x for (x, y) in coords]) - 20
            y_min = min([y for (x, y) in coords]) - 20
            x_max = max([x for (x, y) in coords]) + 20
            y_max = max([y for (x, y) in coords]) + 20

    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()