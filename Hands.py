import cv2
import mediapipe as mp
import os

# Configuración de la carpeta para guardar imágenes
name = "A" #cambiar nombre en funcion de la clase que se quiera agregar
address = "C:/Users/elies/Documents/Projects/HandsCAKE/data/Validacion" #Cambiar ruta cada ve se deba agregar una nueva clase de imagen
directory = address + '/' + name
if not os.path.exists(directory):
    print("Carpeta creada: ", directory)
    os.makedirs(directory)

# Inicializar contador 
cont = 0

# Captura de video
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()    #Primer parametro, FALSE para que no haga la deteccion 24/7
                            #Solo hara deteccion cuando hay una confianza alta
                            #Segundo parametro: numero maximo de manos
                            #Tercer parametro: confianza minima de deteccion
                            #Cuarto parametro: confianza minima de seguimiento

# Utilidades para dibujar
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB porque MediaPipe lo necesita
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    copy = frame.copy()

    # Procesar la imagen para detectar manos
    results = hands.process(frame_rgb)

    positions = []

    # Dibujar las manos detectadas
    if results.multi_hand_landmarks:
        for mano in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)
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

            # Asegurar que los límites estén dentro del frame
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(ancho, x_max)
            y_max = min(alto, y_max)

            # Dibujar caja y recortar
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            dedos_reg = copy[y_min:y_max, x_min:x_max]

            # Redimensionar y guardar
            dedos_reg = cv2.resize(dedos_reg, (200,200), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{directory}/{name}_{cont}.jpg", dedos_reg)
            cont += 1


    # Mostrar el video
    cv2.imshow("Manos", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()