import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Carpeta donde se guardarán los CSV por clase
output_dir = "landmarks_capturados/Validacion" # cambiar nombre de carpeta dependiendo de si son para entrenamiento o validacion
os.makedirs(output_dir, exist_ok=True)

# Clases
clases = ["A", "B", "E", "I", "O", "U"]

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Inicializar cámara
cap = cv2.VideoCapture(0)

# Contador total y límite
total_registros = 0
max_registros = 600

# Variable para indicar la clase activa (modo captura automática)
clase_activa = None
print("Instrucciones:")
print("Presiona A, B, E, I, O, U para activar captura automática para esa clase.")
print("Presiona ESC para salir.")
print("Presiona la misma tecla de clase para desactivar la captura automática.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Si hay clase activa, guardar automáticamente
            if clase_activa:
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y, lm.z])

                if all(v == 0 for v in coords):
                    continue  # Ignorar landmarks vacíos

                df_new = pd.DataFrame([coords + [clase_activa]],
                                      columns=[f"x{i}" for i in range(21)] +
                                              [f"y{i}" for i in range(21)] +
                                              [f"z{i}" for i in range(21)] +
                                              ["clase"])
                csv_path = os.path.join(output_dir, f"landmarks_{clase_activa}.csv")
                if os.path.exists(csv_path):
                    df_new.to_csv(csv_path, mode='a', index=False, header=False)
                else:
                    df_new.to_csv(csv_path, index=False)

                total_registros += 1
                cv2.putText(frame, f"Clase activa: {clase_activa} | Total: {total_registros}/{max_registros}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if total_registros >= max_registros:
                    print("Se alcanzó el límite de 600 registros. Cerrando la captura.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    # Mostrar la imagen
    cv2.imshow("Captura automática de landmarks", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif chr(key).upper() in clases:
        tecla = chr(key).upper()
        if clase_activa == tecla:
            clase_activa = None  # Desactivar captura
            print(f"Captura automática desactivada para clase {tecla}")
        else:
            clase_activa = tecla  # Activar captura
            print(f"Captura automática activada para clase {tecla}")

cap.release()
cv2.destroyAllWindows()
