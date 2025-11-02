import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import json

# Cargar modelo de landmarks
modelo = 'C:/Users/elies/Documents/Projects/HandsCAKE/Modelo_Landmarks_FineTuned.keras'
cnn = load_model(modelo)

# Cargar diccionario de clases
with open('clases_landmarks.json', 'r') as f:
    clases_dict = json.load(f)  # {'A':0, 'E':1, ...}
indice_a_nombre = {v:k for k,v in clases_dict.items()}

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,         # número máximo de manos a detectar
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar la mano
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibujar landmarks en la imagen
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer los 63 valores (x, y, z de 21 puntos)
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            
            # Convertir a numpy array con forma (1,63)
            landmarks_array = np.array(coords).reshape(1,63)

            # Predecir clase
            pred = cnn.predict(landmarks_array)
            clase_idx = np.argmax(pred)
            nombre_clase = indice_a_nombre[clase_idx]

            # Mostrar resultado sobre la mano
            alto, ancho, _ = frame.shape
            x_min = max(0, min([int(lm.x*ancho) for lm in hand_landmarks.landmark]) - 20)
            y_min = max(0, min([int(lm.y*alto) for lm in hand_landmarks.landmark]) - 20)
            cv2.putText(frame, f"Clase: {nombre_clase}", (x_min, y_min-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Mostrar la imagen
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()