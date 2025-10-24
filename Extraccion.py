import mediapipe as mp
import cv2
import os
import pandas as pd
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extraer_landmarks(carpeta_base, salida_csv, augmentation=True):
    data = []
    manos = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    for clase in os.listdir(carpeta_base):
        ruta_clase = os.path.join(carpeta_base, clase)
        if not os.path.isdir(ruta_clase):
            continue

        archivos = os.listdir(ruta_clase)
        for archivo in archivos:
            ruta_imagen = os.path.join(ruta_clase, archivo)
            img = cv2.imread(ruta_imagen)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resultados = manos.process(img_rgb)

            # Inicializar fila con ceros
            fila = [0.0] * 63

            # Si se detecta mano, rellenar con landmarks reales
            if resultados.multi_hand_landmarks:
                landmarks = resultados.multi_hand_landmarks[0]
                fila = []
                for punto in landmarks.landmark:
                    fila.extend([punto.x, punto.y, punto.z])
                # Data augmentation: pequeño ruido gaussiano
                if augmentation:
                    fila = np.array(fila) + np.random.normal(0, 0.01, size=(63,))
                    fila = fila.tolist()

            # Agregar etiqueta
            fila.append(clase)
            data.append(fila)

    # Columnas
    columnas = [f"x{i}" for i in range(21)] + \
               [f"y{i}" for i in range(21)] + \
               [f"z{i}" for i in range(21)] + ["clase"]

    df = pd.DataFrame(data, columns=columnas)
    df.to_csv(salida_csv, index=False)
    print(f"Dataset guardado en: {salida_csv}")
    print(df['clase'].value_counts())

# Ejecutar para entrenamiento y validación
extraer_landmarks("data/Entrenamiento", "landmarks_entrenamiento.csv")
extraer_landmarks("data/Validacion", "landmarks_validacion.csv")
